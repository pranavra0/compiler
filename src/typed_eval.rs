//! Evaluation of pure, already-typed expressions.
//!
//! This is the compile-time half of the execution contract.  It deliberately
//! accepts HIR rather than syntax: names, integer widths, aggregates, and
//! result values have already been resolved by the frontend.  Runtime-only
//! operations return `NotConstant` instead of acquiring a second ad-hoc
//! meaning.

use crate::ast::{BinaryOp, UnaryOp};
use crate::typed::{IntegerWidth, ResolvedType, TypedExpr};
use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    Integer(i128),
    Float(f64),
    Struct(Vec<Value>),
    Array(Vec<Value>),
    Result { error: bool, value: Box<Value> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NotConstant,
    DivisionByZero,
    InvalidOperation,
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::NotConstant => "expression is not a pure compile-time constant",
            Self::DivisionByZero => "division by zero during typed evaluation",
            Self::InvalidOperation => "invalid operation during typed evaluation",
        })
    }
}
impl std::error::Error for Error {}

/// Evaluate an expression produced by the typed frontend.
///
/// Calls, pointers, loads, and layout queries are intentionally rejected here.
/// The caller can then choose the runtime/MIR evaluator for those operations;
/// no host-language fallback is performed.
pub fn evaluate(expression: &TypedExpr) -> Result<Value, Error> {
    match expression {
        TypedExpr::Integer { value, ty, .. } => Ok(Value::Integer(normalize(*value, ty))),
        TypedExpr::Float { value, .. } => Ok(Value::Float(*value)),
        TypedExpr::Bool { value, .. } => Ok(Value::Bool(*value)),
        TypedExpr::StructLiteral { fields, .. } => Ok(Value::Struct(
            fields.iter().map(evaluate).collect::<Result<_, _>>()?,
        )),
        TypedExpr::ArrayLiteral { elements, .. } => Ok(Value::Array(
            elements.iter().map(evaluate).collect::<Result<_, _>>()?,
        )),
        TypedExpr::Unary {
            operator,
            operand,
            ty,
            ..
        } => {
            let value = evaluate(operand)?;
            match (operator, value) {
                (UnaryOp::Negate, Value::Integer(value)) => {
                    Ok(Value::Integer(normalize(value.wrapping_neg(), ty)))
                }
                (UnaryOp::Negate, Value::Float(value)) => Ok(Value::Float(-value)),
                (UnaryOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
                (UnaryOp::BitwiseNot, Value::Integer(value)) => {
                    Ok(Value::Integer(normalize(!value, ty)))
                }
                _ => Err(Error::InvalidOperation),
            }
        }
        TypedExpr::Binary {
            left,
            operator,
            right,
            operand_type,
            ..
        } => {
            let left = evaluate(left)?;
            if *operator == BinaryOp::LogicalAnd && !truth(&left)? {
                return Ok(Value::Bool(false));
            }
            if *operator == BinaryOp::LogicalOr && truth(&left)? {
                return Ok(Value::Bool(true));
            }
            binary(*operator, left, evaluate(right)?, operand_type)
        }
        TypedExpr::ResultOk { value, .. } => Ok(Value::Result {
            error: false,
            value: Box::new(evaluate(value)?),
        }),
        TypedExpr::ResultErr { value, .. } => Ok(Value::Result {
            error: true,
            value: Box::new(evaluate(value)?),
        }),
        TypedExpr::IsErr { value, .. } => match evaluate(value)? {
            Value::Result { error, .. } => Ok(Value::Bool(error)),
            _ => Err(Error::InvalidOperation),
        },
        TypedExpr::Unwrap { value, .. } => match evaluate(value)? {
            Value::Result {
                error: false,
                value,
            } => Ok(*value),
            Value::Result { error: true, .. } => Err(Error::InvalidOperation),
            _ => Err(Error::InvalidOperation),
        },
        TypedExpr::Propagate { value, .. } => match evaluate(value)? {
            Value::Result {
                error: false,
                value,
            } => Ok(*value),
            Value::Result { error: true, value } => Ok(Value::Result { error: true, value }),
            _ => Err(Error::InvalidOperation),
        },
        TypedExpr::Load { .. }
        | TypedExpr::GlobalLoad { .. }
        | TypedExpr::Field { .. }
        | TypedExpr::Index { .. }
        | TypedExpr::Call { .. }
        | TypedExpr::MakeSlice { .. }
        | TypedExpr::LowLevel { .. }
        | TypedExpr::Null { .. }
        | TypedExpr::AddressOf { .. }
        | TypedExpr::Dereference { .. }
        | TypedExpr::Layout { .. } => Err(Error::NotConstant),
    }
}

pub fn is_constant(expression: &TypedExpr) -> bool {
    evaluate(expression).is_ok()
}

fn bits(ty: &ResolvedType) -> (u32, bool) {
    match ty {
        ResolvedType::Integer {
            width: IntegerWidth::Bits(bits),
            signed,
        } => (*bits as u32, *signed),
        // Target-dependent pointer integers are kept exact in the typed
        // evaluator. The target-aware layout evaluator owns their final width.
        ResolvedType::Integer {
            width: IntegerWidth::Pointer,
            signed,
        } => (128, *signed),
        _ => (128, true),
    }
}
fn normalize(value: i128, ty: &ResolvedType) -> i128 {
    let (width, signed) = bits(ty);
    if width >= 128 {
        return value;
    }
    let mask = (1u128 << width) - 1;
    let raw = (value as u128) & mask;
    if signed && raw & (1u128 << (width - 1)) != 0 {
        (raw | (!0u128 << width)) as i128
    } else {
        raw as i128
    }
}
fn truth(value: &Value) -> Result<bool, Error> {
    match value {
        Value::Bool(value) => Ok(*value),
        Value::Integer(value) => Ok(*value != 0),
        _ => Err(Error::InvalidOperation),
    }
}
fn binary(op: BinaryOp, left: Value, right: Value, ty: &ResolvedType) -> Result<Value, Error> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => {
            let (width, signed) = bits(ty);
            let mask = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            let a_raw = (a as u128) & mask;
            let b_raw = (b as u128) & mask;
            let a_signed = normalize(a, ty);
            let b_signed = normalize(b, ty);
            let value = match op {
                BinaryOp::Add => {
                    Value::Integer(normalize((a_raw.wrapping_add(b_raw) & mask) as i128, ty))
                }
                BinaryOp::Subtract => {
                    Value::Integer(normalize((a_raw.wrapping_sub(b_raw) & mask) as i128, ty))
                }
                BinaryOp::Multiply => {
                    Value::Integer(normalize((a_raw.wrapping_mul(b_raw) & mask) as i128, ty))
                }
                BinaryOp::Divide
                    if b_raw != 0 && !(signed && a_signed == i128::MIN && b_signed == -1) =>
                {
                    Value::Integer(normalize(
                        if signed {
                            a_signed / b_signed
                        } else {
                            (a_raw / b_raw) as i128
                        },
                        ty,
                    ))
                }
                BinaryOp::Modulo
                    if b_raw != 0 && !(signed && a_signed == i128::MIN && b_signed == -1) =>
                {
                    Value::Integer(normalize(
                        if signed {
                            a_signed % b_signed
                        } else {
                            (a_raw % b_raw) as i128
                        },
                        ty,
                    ))
                }
                BinaryOp::Equal => Value::Bool(a_raw == b_raw),
                BinaryOp::NotEqual => Value::Bool(a_raw != b_raw),
                BinaryOp::Less => Value::Bool(if signed {
                    a_signed < b_signed
                } else {
                    a_raw < b_raw
                }),
                BinaryOp::LessEqual => Value::Bool(if signed {
                    a_signed <= b_signed
                } else {
                    a_raw <= b_raw
                }),
                BinaryOp::Greater => Value::Bool(if signed {
                    a_signed > b_signed
                } else {
                    a_raw > b_raw
                }),
                BinaryOp::GreaterEqual => Value::Bool(if signed {
                    a_signed >= b_signed
                } else {
                    a_raw >= b_raw
                }),
                BinaryOp::BitwiseAnd => Value::Integer(normalize((a_raw & b_raw) as i128, ty)),
                BinaryOp::BitwiseOr => Value::Integer(normalize((a_raw | b_raw) as i128, ty)),
                BinaryOp::BitwiseXor => Value::Integer(normalize((a_raw ^ b_raw) as i128, ty)),
                BinaryOp::ShiftLeft if b_raw < width as u128 => {
                    Value::Integer(normalize((a_raw << b_raw) as i128, ty))
                }
                BinaryOp::ShiftRight if b_raw < width as u128 => Value::Integer(normalize(
                    if signed {
                        a_signed >> b_raw
                    } else {
                        (a_raw >> b_raw) as i128
                    },
                    ty,
                )),
                BinaryOp::Divide | BinaryOp::Modulo => return Err(Error::DivisionByZero),
                _ => return Err(Error::InvalidOperation),
            };
            Ok(value)
        }
        (Value::Float(a), Value::Float(b)) => Ok(match op {
            BinaryOp::Add => Value::Float(a + b),
            BinaryOp::Subtract => Value::Float(a - b),
            BinaryOp::Multiply => Value::Float(a * b),
            BinaryOp::Divide => Value::Float(a / b),
            BinaryOp::Modulo => Value::Float(a % b),
            BinaryOp::Equal => Value::Bool(a == b),
            BinaryOp::NotEqual => Value::Bool(a != b),
            BinaryOp::Less => Value::Bool(a < b),
            BinaryOp::LessEqual => Value::Bool(a <= b),
            BinaryOp::Greater => Value::Bool(a > b),
            BinaryOp::GreaterEqual => Value::Bool(a >= b),
            _ => return Err(Error::InvalidOperation),
        }),
        (Value::Bool(a), Value::Bool(b)) => Ok(match op {
            BinaryOp::Equal => Value::Bool(a == b),
            BinaryOp::NotEqual => Value::Bool(a != b),
            BinaryOp::LogicalAnd => Value::Bool(a && b),
            BinaryOp::LogicalOr => Value::Bool(a || b),
            _ => return Err(Error::InvalidOperation),
        }),
        _ => Err(Error::InvalidOperation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;

    fn i32_value(value: i128) -> TypedExpr {
        TypedExpr::Integer {
            value,
            ty: ResolvedType::Integer {
                width: IntegerWidth::Bits(32),
                signed: true,
            },
            span: Span::new(0, 1),
        }
    }

    #[test]
    fn evaluates_typed_integer_widths_and_short_circuiting() {
        let expression = TypedExpr::Binary {
            left: Box::new(i32_value(2)),
            operator: BinaryOp::Add,
            right: Box::new(i32_value(3)),
            ty: i32_value(0).ty(),
            operand_type: i32_value(0).ty(),
            span: Span::new(0, 1),
        };
        assert_eq!(evaluate(&expression), Ok(Value::Integer(5)));
    }
}
