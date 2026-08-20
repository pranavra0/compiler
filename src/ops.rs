//! Primitive operator semantics shared by compile-time and runtime evaluators.
use crate::ast::{BinaryOp, UnaryOp};
use crate::typed::{IntegerWidth, ResolvedType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Value {
    Bool(bool),
    Integer(i128),
    Float(f64),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    DivisionByZero,
    InvalidOperation,
}

pub fn truth(value: Value) -> Result<bool, Error> {
    match value {
        Value::Bool(value) => Ok(value),
        Value::Integer(value) => Ok(value != 0),
        _ => Err(Error::InvalidOperation),
    }
}

pub fn unary(op: UnaryOp, value: Value, ty: &ResolvedType) -> Result<Value, Error> {
    unary_with_pointer_width(op, value, ty, 128)
}

pub fn unary_with_pointer_width(
    op: UnaryOp,
    value: Value,
    ty: &ResolvedType,
    pointer_width: u32,
) -> Result<Value, Error> {
    match (op, value) {
        (UnaryOp::Negate, Value::Integer(value)) => Ok(Value::Integer(
            normalize_with_pointer_width(value.wrapping_neg(), ty, pointer_width),
        )),
        (UnaryOp::Negate, Value::Float(value)) => Ok(Value::Float(-value)),
        (UnaryOp::Not, Value::Bool(value)) => Ok(Value::Bool(!value)),
        (UnaryOp::BitwiseNot, Value::Integer(value)) => Ok(Value::Integer(
            normalize_with_pointer_width(!value, ty, pointer_width),
        )),
        _ => Err(Error::InvalidOperation),
    }
}

pub fn binary(op: BinaryOp, left: Value, right: Value, ty: &ResolvedType) -> Result<Value, Error> {
    binary_with_pointer_width(op, left, right, ty, 128)
}

pub fn binary_with_pointer_width(
    op: BinaryOp,
    left: Value,
    right: Value,
    ty: &ResolvedType,
    pointer_width: u32,
) -> Result<Value, Error> {
    match (left, right) {
        (Value::Integer(a), Value::Integer(b)) => {
            let (width, signed) = integer_info(ty, pointer_width);
            let mask = if width >= 128 {
                u128::MAX
            } else {
                (1u128 << width) - 1
            };
            let raw_a = (a as u128) & mask;
            let raw_b = (b as u128) & mask;
            let signed_a = normalize_with_pointer_width(a, ty, pointer_width);
            let signed_b = normalize_with_pointer_width(b, ty, pointer_width);
            let result = match op {
                BinaryOp::Add => Value::Integer(normalize_with_pointer_width(
                    (raw_a.wrapping_add(raw_b) & mask) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::Subtract => Value::Integer(normalize_with_pointer_width(
                    (raw_a.wrapping_sub(raw_b) & mask) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::Multiply => Value::Integer(normalize_with_pointer_width(
                    (raw_a.wrapping_mul(raw_b) & mask) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::Divide
                    if raw_b != 0 && !(signed && signed_a == i128::MIN && signed_b == -1) =>
                {
                    Value::Integer(normalize_with_pointer_width(
                        if signed {
                            signed_a / signed_b
                        } else {
                            (raw_a / raw_b) as i128
                        },
                        ty,
                        pointer_width,
                    ))
                }
                BinaryOp::Modulo
                    if raw_b != 0 && !(signed && signed_a == i128::MIN && signed_b == -1) =>
                {
                    Value::Integer(normalize_with_pointer_width(
                        if signed {
                            signed_a % signed_b
                        } else {
                            (raw_a % raw_b) as i128
                        },
                        ty,
                        pointer_width,
                    ))
                }
                BinaryOp::Equal => Value::Bool(raw_a == raw_b),
                BinaryOp::NotEqual => Value::Bool(raw_a != raw_b),
                BinaryOp::Less => Value::Bool(if signed {
                    signed_a < signed_b
                } else {
                    raw_a < raw_b
                }),
                BinaryOp::LessEqual => Value::Bool(if signed {
                    signed_a <= signed_b
                } else {
                    raw_a <= raw_b
                }),
                BinaryOp::Greater => Value::Bool(if signed {
                    signed_a > signed_b
                } else {
                    raw_a > raw_b
                }),
                BinaryOp::GreaterEqual => Value::Bool(if signed {
                    signed_a >= signed_b
                } else {
                    raw_a >= raw_b
                }),
                BinaryOp::BitwiseAnd => Value::Integer(normalize_with_pointer_width(
                    (raw_a & raw_b) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::BitwiseOr => Value::Integer(normalize_with_pointer_width(
                    (raw_a | raw_b) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::BitwiseXor => Value::Integer(normalize_with_pointer_width(
                    (raw_a ^ raw_b) as i128,
                    ty,
                    pointer_width,
                )),
                BinaryOp::ShiftLeft if raw_b < width as u128 => Value::Integer(
                    normalize_with_pointer_width((raw_a << raw_b) as i128, ty, pointer_width),
                ),
                BinaryOp::ShiftRight if raw_b < width as u128 => {
                    Value::Integer(normalize_with_pointer_width(
                        if signed {
                            signed_a >> raw_b
                        } else {
                            (raw_a >> raw_b) as i128
                        },
                        ty,
                        pointer_width,
                    ))
                }
                BinaryOp::Divide | BinaryOp::Modulo => return Err(Error::DivisionByZero),
                _ => return Err(Error::InvalidOperation),
            };
            Ok(result)
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

fn integer_info(ty: &ResolvedType, pointer_width: u32) -> (u32, bool) {
    match ty {
        ResolvedType::Integer {
            width: IntegerWidth::Bits(bits),
            signed,
        } => (*bits as u32, *signed),
        ResolvedType::Integer {
            width: IntegerWidth::Pointer,
            signed,
        } => (pointer_width, *signed),
        _ => (128, true),
    }
}

pub fn normalize(value: i128, ty: &ResolvedType) -> i128 {
    normalize_with_pointer_width(value, ty, 128)
}

pub fn normalize_with_pointer_width(value: i128, ty: &ResolvedType, pointer_width: u32) -> i128 {
    let (width, signed) = integer_info(ty, pointer_width);
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
