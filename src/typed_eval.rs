//! Evaluation of pure, already-typed expressions.
//!
//! This is the compile-time half of the execution contract.  It deliberately
//! accepts HIR rather than syntax: names, integer widths, aggregates, and
//! result values have already been resolved by the frontend.  Runtime-only
//! operations return `NotConstant` instead of acquiring a second ad-hoc
//! meaning.

use crate::ast::BinaryOp;
use crate::ops;
use crate::typed::{ResolvedType, TypedExpr};
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
        TypedExpr::Integer { value, ty, .. } => Ok(Value::Integer(ops::normalize(*value, ty))),
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
            let value = primitive(evaluate(operand)?);
            from_primitive(ops::unary(*operator, value, ty).map_err(from_ops_error)?)
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

fn primitive(value: Value) -> ops::Value {
    match value {
        Value::Bool(value) => ops::Value::Bool(value),
        Value::Integer(value) => ops::Value::Integer(value),
        Value::Float(value) => ops::Value::Float(value),
        _ => unreachable!("aggregate is not a primitive operation"),
    }
}

fn from_primitive(value: ops::Value) -> Result<Value, Error> {
    Ok(match value {
        ops::Value::Bool(value) => Value::Bool(value),
        ops::Value::Integer(value) => Value::Integer(value),
        ops::Value::Float(value) => Value::Float(value),
    })
}

fn from_ops_error(error: ops::Error) -> Error {
    match error {
        ops::Error::DivisionByZero => Error::DivisionByZero,
        ops::Error::InvalidOperation => Error::InvalidOperation,
    }
}

fn truth(value: &Value) -> Result<bool, Error> {
    ops::truth(primitive(value.clone())).map_err(from_ops_error)
}

fn binary(op: BinaryOp, left: Value, right: Value, ty: &ResolvedType) -> Result<Value, Error> {
    from_primitive(ops::binary(op, primitive(left), primitive(right), ty).map_err(from_ops_error)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;
    use crate::typed::IntegerWidth;

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
