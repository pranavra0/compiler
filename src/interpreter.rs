//! A small, deliberately explicit interpreter for the typed IR.
//!
//! The interpreter is independent of LLVM and is intended for semantic tests and
//! quick experiments. Operations which need a native address or an FFI symbol
//! are rejected instead of being silently emulated.
use crate::ast::{BinaryOp, UnaryOp};
use crate::typed::{
    DefId, FunctionId, IntegerWidth, LocalId, ResolvedType, TypedBlock, TypedExpr, TypedPlace,
    TypedProgram, TypedStmt,
};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

static POINTER_BITS: AtomicU32 = AtomicU32::new(usize::BITS);
static STEPS: AtomicU64 = AtomicU64::new(0);
static CALL_DEPTH: AtomicU64 = AtomicU64::new(0);
const MAX_STEPS: u64 = 1_000_000;
const MAX_CALL_DEPTH: u64 = 1_024;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Bool(bool),
    Int(i128),
    Float(f64),
    Struct(Vec<Value>),
    Array(Vec<Value>),
    Result { error: bool, value: Box<Value> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

enum Flow {
    Normal,
    Return(Value),
    Break,
    Continue,
}

type Env = HashMap<LocalId, Value>;
type Globals = HashMap<DefId, Value>;

pub fn run(program: &TypedProgram) -> Result<i32, Error> {
    run_with_pointer_width(program, usize::BITS)
}

pub fn run_with_pointer_width(program: &TypedProgram, pointer_bits: u32) -> Result<i32, Error> {
    POINTER_BITS.store(pointer_bits, Ordering::Relaxed);
    STEPS.store(0, Ordering::Relaxed);
    CALL_DEPTH.store(0, Ordering::Relaxed);
    let mut globals = HashMap::new();
    for global in &program.globals {
        let value = eval_expr(program, &mut globals, &mut HashMap::new(), &global.value)?;
        globals.insert(global.id, value);
    }
    for constant in &program.constants {
        let value = eval_expr(program, &mut globals, &mut HashMap::new(), &constant.value)?;
        globals.insert(constant.id, value);
    }
    let main = program
        .functions
        .iter()
        .find(|f| f.name == "main" && !f.is_extern)
        .ok_or_else(|| Error("interpreter requires a source `main` function".into()))?;
    let value = call(program, &mut globals, main.id, Vec::new())?;
    match value {
        Value::Unit => Ok(0),
        Value::Int(x) => Ok(x as i32),
        _ => Err(Error("main must return an integer or void".into())),
    }
}

fn function(program: &TypedProgram, id: FunctionId) -> Result<&crate::typed::TypedFunction, Error> {
    program
        .functions
        .iter()
        .find(|function| function.id == id)
        .ok_or_else(|| Error(format!("unknown function id {id}")))
}
struct CallGuard;
impl Drop for CallGuard {
    fn drop(&mut self) {
        CALL_DEPTH.fetch_sub(1, Ordering::Relaxed);
    }
}

fn call(
    program: &TypedProgram,
    globals: &mut Globals,
    id: FunctionId,
    args: Vec<Value>,
) -> Result<Value, Error> {
    if CALL_DEPTH.fetch_add(1, Ordering::Relaxed) >= MAX_CALL_DEPTH {
        CALL_DEPTH.fetch_sub(1, Ordering::Relaxed);
        return Err(Error("interpreter call-depth limit exceeded".into()));
    }
    let _guard = CallGuard;
    let f = function(program, id)?;
    if f.is_extern {
        return Err(Error(format!(
            "interpreter does not support foreign function `{}`",
            f.name
        )));
    }
    if f.params.len() != args.len() {
        return Err(Error(format!("wrong number of arguments to `{}`", f.name)));
    }
    let mut env = Env::new();
    for (p, value) in f.params.iter().zip(args) {
        env.insert(p.id, value);
    }
    match block(program, globals, &mut env, &f.body)? {
        Flow::Return(v) => Ok(v),
        Flow::Normal => Ok(Value::Unit),
        Flow::Break | Flow::Continue => Err(Error(format!("loop control escaped `{}`", f.name))),
    }
}
fn block(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    b: &TypedBlock,
) -> Result<Flow, Error> {
    let mut defers: Vec<(FunctionId, Vec<Value>)> = Vec::new();
    let mut flow = Flow::Normal;
    for stmt in &b.statements {
        if !matches!(flow, Flow::Normal) {
            break;
        }
        if let TypedStmt::Defer {
            function,
            arguments,
            ..
        } = stmt
        {
            let mut values = Vec::new();
            for arg in arguments {
                values.push(eval_expr(program, globals, env, arg)?);
            }
            defers.push((*function, values));
        } else {
            flow = statement(program, globals, env, stmt)?;
        }
    }
    for (id, args) in defers.into_iter().rev() {
        let _ = call(program, globals, id, args)?;
    }
    Ok(flow)
}
fn consume_step() -> Result<(), Error> {
    if STEPS.fetch_add(1, Ordering::Relaxed) >= MAX_STEPS {
        return Err(Error("interpreter execution step limit exceeded".into()));
    }
    Ok(())
}

fn statement(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    s: &TypedStmt,
) -> Result<Flow, Error> {
    consume_step()?;
    match s {
        TypedStmt::Declare { id, value, .. } => {
            let v = eval_expr(program, globals, env, value)?;
            if matches!(value, TypedExpr::Propagate { .. }) {
                if let Value::Result { error: true, .. } = v {
                    return Ok(Flow::Return(v));
                }
                let v = match v {
                    Value::Result { value, .. } => *value,
                    other => other,
                };
                env.insert(*id, v);
            } else {
                env.insert(*id, v);
            }
            Ok(Flow::Normal)
        }
        TypedStmt::Store { target, value, .. } => {
            let v = eval_expr(program, globals, env, value)?;
            store(program, globals, env, target, v)?;
            Ok(Flow::Normal)
        }
        TypedStmt::Expr { expression, .. } => {
            eval_expr(program, globals, env, expression)?;
            Ok(Flow::Normal)
        }
        TypedStmt::Return { value, .. } => {
            let v = match value {
                Some(v) => eval_expr(program, globals, env, v)?,
                None => Value::Unit,
            };
            Ok(Flow::Return(v))
        }
        TypedStmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            if truth(&eval_expr(program, globals, env, condition)?)? {
                block(program, globals, env, then_branch)
            } else if let Some(b) = else_branch {
                block(program, globals, env, b)
            } else {
                Ok(Flow::Normal)
            }
        }
        TypedStmt::While {
            condition, body, ..
        } => {
            loop {
                consume_step()?;
                if !truth(&eval_expr(program, globals, env, condition)?)? {
                    break;
                }
                match block(program, globals, env, body)? {
                    Flow::Normal | Flow::Continue => {}
                    Flow::Break => break,
                    Flow::Return(v) => return Ok(Flow::Return(v)),
                }
            }
            Ok(Flow::Normal)
        }
        TypedStmt::Break { .. } => Ok(Flow::Break),
        TypedStmt::Continue { .. } => Ok(Flow::Continue),
        TypedStmt::Defer { .. } => unreachable!(),
    }
}
fn load(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    p: &TypedPlace,
) -> Result<Value, Error> {
    match p {
        TypedPlace::Local { id, .. } => env
            .get(id)
            .cloned()
            .ok_or_else(|| Error("unknown local".into())),
        TypedPlace::Global { id, name, .. } => globals
            .get(id)
            .cloned()
            .ok_or_else(|| Error(format!("unknown global `{name}`"))),
        TypedPlace::Field { base, index, .. } => match load(program, globals, env, base)? {
            Value::Struct(fields) => fields
                .get(*index as usize)
                .cloned()
                .ok_or_else(|| Error("field index out of range".into())),
            _ => Err(Error("field base is not a struct".into())),
        },
        TypedPlace::Index { base, index, .. } => {
            let i = as_usize(&eval_expr(program, globals, env, index)?)?;
            match load(program, globals, env, base)? {
                Value::Array(values) => values
                    .get(i)
                    .cloned()
                    .ok_or_else(|| Error("index out of bounds".into())),
                _ => Err(Error("interpreter only supports array indexing".into())),
            }
        }
        TypedPlace::Temporary { .. } => Err(Error(
            "cannot assign through a temporary in the interpreter".into(),
        )),
        TypedPlace::Dereference { .. } => Err(Error(
            "raw pointer dereference is unsupported by the interpreter".into(),
        )),
    }
}
fn store(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    p: &TypedPlace,
    value: Value,
) -> Result<(), Error> {
    match p {
        TypedPlace::Local { id, .. } => {
            env.insert(*id, value);
            Ok(())
        }
        TypedPlace::Global { id, .. } => {
            globals.insert(*id, value);
            Ok(())
        }
        TypedPlace::Field { base, index, .. } => {
            let mut v = load(program, globals, env, base)?;
            if let Value::Struct(ref mut fs) = v {
                let slot = fs
                    .get_mut(*index as usize)
                    .ok_or_else(|| Error("field index out of range".into()))?;
                *slot = value;
                store(program, globals, env, base, v)
            } else {
                Err(Error("field base is not a struct".into()))
            }
        }
        TypedPlace::Index { base, index, .. } => {
            let mut v = load(program, globals, env, base)?;
            let i = as_usize(&eval_expr(program, globals, env, index)?)?;
            if let Value::Array(ref mut xs) = v {
                let slot = xs
                    .get_mut(i)
                    .ok_or_else(|| Error("index out of bounds".into()))?;
                *slot = value;
                store(program, globals, env, base, v)
            } else {
                Err(Error("interpreter only supports array indexing".into()))
            }
        }
        _ => Err(Error(
            "raw pointer assignment is unsupported by the interpreter".into(),
        )),
    }
}
fn eval_expr(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    e: &TypedExpr,
) -> Result<Value, Error> {
    match e {
        TypedExpr::Integer { value, ty, .. } => Ok(Value::Int(normalize_integer(*value, ty))),
        TypedExpr::Float { value, .. } => Ok(Value::Float(*value)),
        TypedExpr::Bool { value, .. } => Ok(Value::Bool(*value)),
        TypedExpr::Null { .. } => Err(Error(
            "null/pointers are unsupported by the interpreter".into(),
        )),
        TypedExpr::Load { id, .. } => env
            .get(id)
            .cloned()
            .ok_or_else(|| Error("unknown local".into())),
        TypedExpr::GlobalLoad { id, name, .. } => globals
            .get(id)
            .cloned()
            .ok_or_else(|| Error(format!("unknown global `{name}`"))),
        TypedExpr::Field { place, .. } | TypedExpr::Index { place, .. } => {
            load(program, globals, env, place)
        }
        TypedExpr::StructLiteral { fields, .. } => Ok(Value::Struct(
            fields
                .iter()
                .map(|x| eval_expr(program, globals, env, x))
                .collect::<Result<_, _>>()?,
        )),
        TypedExpr::ArrayLiteral { elements, .. } => Ok(Value::Array(
            elements
                .iter()
                .map(|x| eval_expr(program, globals, env, x))
                .collect::<Result<_, _>>()?,
        )),
        TypedExpr::Unary {
            operator,
            operand,
            ty,
            ..
        } => unary(*operator, eval_expr(program, globals, env, operand)?, ty),
        TypedExpr::Binary {
            left,
            operator,
            right,
            operand_type,
            ..
        } => {
            let left_value = eval_expr(program, globals, env, left)?;
            if *operator == BinaryOp::LogicalAnd && !truth(&left_value)? {
                return Ok(Value::Bool(false));
            }
            if *operator == BinaryOp::LogicalOr && truth(&left_value)? {
                return Ok(Value::Bool(true));
            }
            binary(
                *operator,
                left_value,
                eval_expr(program, globals, env, right)?,
                operand_type,
            )
        }
        TypedExpr::Call {
            function,
            name,
            arguments,
            ..
        } => {
            if name == "make_slice" {
                return Err(Error(
                    "interpreter does not support raw-pointer slice construction".into(),
                ));
            }
            let args = arguments
                .iter()
                .map(|x| eval_expr(program, globals, env, x))
                .collect::<Result<Vec<_>, _>>()?;
            call(program, globals, *function, args)
        }
        TypedExpr::Layout {
            kind,
            target,
            field,
            ..
        } => {
            let (size, align, offsets) =
                layout_of(program, target, POINTER_BITS.load(Ordering::Relaxed));
            let value = match kind {
                crate::typed::LayoutKind::Size => size,
                crate::typed::LayoutKind::Align => align,
                crate::typed::LayoutKind::Offset => field
                    .as_ref()
                    .and_then(|name| offsets.get(name).copied())
                    .ok_or_else(|| Error("unknown field in layout query".into()))?,
            };
            Ok(Value::Int(value as i128))
        }
        TypedExpr::ResultOk { value, .. } => Ok(Value::Result {
            error: false,
            value: Box::new(eval_expr(program, globals, env, value)?),
        }),
        TypedExpr::ResultErr { value, .. } => Ok(Value::Result {
            error: true,
            value: Box::new(eval_expr(program, globals, env, value)?),
        }),
        TypedExpr::IsErr { value, .. } => match eval_expr(program, globals, env, value)? {
            Value::Result { error, .. } => Ok(Value::Bool(error)),
            _ => Err(Error("is_err requires a result".into())),
        },
        TypedExpr::Unwrap { value, .. } => match eval_expr(program, globals, env, value)? {
            Value::Result {
                error: false,
                value,
            } => Ok(*value),
            Value::Result { error: true, .. } => Err(Error("unwrap of an error result".into())),
            _ => Err(Error("unwrap requires a result".into())),
        },
        TypedExpr::Propagate { value, .. } => match eval_expr(program, globals, env, value)? {
            Value::Result {
                error: false,
                value,
            } => Ok(*value),
            Value::Result { error: true, value } => Ok(Value::Result { error: true, value }),
            _ => Err(Error("propagation requires a result".into())),
        },
        TypedExpr::AddressOf { .. } | TypedExpr::Dereference { .. } => Err(Error(
            "raw pointers are unsupported by the interpreter".into(),
        )),
    }
}

fn align_up(value: u64, align: u64) -> u64 {
    if align <= 1 {
        value
    } else {
        value.div_ceil(align) * align
    }
}

fn layout_of(
    program: &TypedProgram,
    ty: &ResolvedType,
    pointer_bits: u32,
) -> (u64, u64, HashMap<String, u64>) {
    match ty {
        ResolvedType::Unit => (0, 1, HashMap::new()),
        ResolvedType::Bool => (1, 1, HashMap::new()),
        ResolvedType::Integer {
            width: IntegerWidth::Bits(bits),
            ..
        } => {
            let bytes = ((*bits as u64) / 8).max(1);
            (bytes, bytes.min(8), HashMap::new())
        }
        ResolvedType::Integer {
            width: IntegerWidth::Pointer,
            ..
        } => {
            let bytes = (pointer_bits as u64 / 8).max(1);
            (bytes, bytes, HashMap::new())
        }
        ResolvedType::Float { bits } => {
            let bytes = ((*bits as u64) / 8).max(1);
            (bytes, bytes.min(8), HashMap::new())
        }
        ResolvedType::Pointer(_) => {
            let bytes = (pointer_bits as u64 / 8).max(1);
            (bytes, bytes, HashMap::new())
        }
        ResolvedType::Slice(_) => {
            let bytes = (pointer_bits as u64 / 8).max(1);
            (bytes * 2, bytes, HashMap::new())
        }
        ResolvedType::Array { length, element } => {
            let (size, align, _) = layout_of(program, element, pointer_bits);
            (size * *length, align, HashMap::new())
        }
        ResolvedType::Result { success, error } => {
            let (ss, sa, _) = layout_of(program, success, pointer_bits);
            let (es, ea, _) = layout_of(program, error, pointer_bits);
            let align = sa.max(ea).max(1);
            let offset = align_up(1, sa);
            let end = align_up(offset + ss, ea);
            (align_up(end + es, align), align, HashMap::new())
        }
        ResolvedType::Struct(id) => {
            let Some(structure) = program.structs.iter().find(|item| item.id == *id) else {
                return (0, 1, HashMap::new());
            };
            let mut offset = 0;
            let mut align = 1;
            let mut offsets = HashMap::new();
            for field in &structure.fields {
                let (size, field_align, _) = layout_of(program, &field.ty, pointer_bits);
                offset = align_up(offset, field_align);
                offsets.insert(field.name.clone(), offset);
                offset += size;
                align = align.max(field_align);
            }
            (align_up(offset, align), align, offsets)
        }
    }
}

fn integer_info(ty: &ResolvedType) -> (u32, bool) {
    match ty {
        ResolvedType::Integer {
            width: IntegerWidth::Bits(bits),
            signed,
        } => (*bits as u32, *signed),
        ResolvedType::Integer {
            width: IntegerWidth::Pointer,
            signed,
        } => (POINTER_BITS.load(Ordering::Relaxed), *signed),
        _ => (128, true),
    }
}

fn normalize_integer(value: i128, ty: &ResolvedType) -> i128 {
    let (bits, signed) = integer_info(ty);
    let raw = (value as u128)
        & if bits == 128 {
            u128::MAX
        } else {
            (1u128 << bits) - 1
        };
    if signed && bits < 128 && (raw & (1u128 << (bits - 1))) != 0 {
        (raw | (!0u128 << bits)) as i128
    } else {
        raw as i128
    }
}

fn integer_binary(op: BinaryOp, a: i128, b: i128, ty: &ResolvedType) -> Result<Value, Error> {
    let (bits, signed) = integer_info(ty);
    let mask = if bits == 128 {
        u128::MAX
    } else {
        (1u128 << bits) - 1
    };
    let raw_a = (a as u128) & mask;
    let raw_b = (b as u128) & mask;
    let signed_a = normalize_integer(a, ty);
    let signed_b = normalize_integer(b, ty);
    let signed_min = if bits == 128 {
        i128::MIN
    } else {
        -(1i128 << (bits - 1))
    };
    let out = match op {
        BinaryOp::Add => Value::Int((raw_a.wrapping_add(raw_b) & mask) as i128),
        BinaryOp::Subtract => Value::Int((raw_a.wrapping_sub(raw_b) & mask) as i128),
        BinaryOp::Multiply => Value::Int((raw_a.wrapping_mul(raw_b) & mask) as i128),
        BinaryOp::Divide => {
            if raw_b == 0 || (signed && signed_a == signed_min && signed_b == -1) {
                return Err(Error("integer division failed".into()));
            }
            if signed {
                Value::Int(normalize_integer(signed_a / signed_b, ty))
            } else {
                Value::Int((raw_a / raw_b) as i128)
            }
        }
        BinaryOp::Modulo => {
            if raw_b == 0 || (signed && signed_a == signed_min && signed_b == -1) {
                return Err(Error("integer remainder failed".into()));
            }
            if signed {
                Value::Int(normalize_integer(signed_a % signed_b, ty))
            } else {
                Value::Int((raw_a % raw_b) as i128)
            }
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
        BinaryOp::BitwiseAnd => Value::Int(normalize_integer((raw_a & raw_b) as i128, ty)),
        BinaryOp::BitwiseOr => Value::Int(normalize_integer((raw_a | raw_b) as i128, ty)),
        BinaryOp::BitwiseXor => Value::Int(normalize_integer((raw_a ^ raw_b) as i128, ty)),
        BinaryOp::ShiftLeft | BinaryOp::ShiftRight => {
            if raw_b >= bits as u128 {
                return Err(Error("invalid shift".into()));
            }
            let shift = raw_b as u32;
            if op == BinaryOp::ShiftLeft {
                Value::Int(normalize_integer((raw_a << shift) as i128, ty))
            } else if signed {
                Value::Int(normalize_integer(signed_a >> shift, ty))
            } else {
                Value::Int((raw_a >> shift) as i128)
            }
        }
        _ => return Err(Error("unsupported integer operation".into())),
    };
    Ok(out)
}

fn truth(v: &Value) -> Result<bool, Error> {
    match v {
        Value::Bool(x) => Ok(*x),
        Value::Int(x) => Ok(*x != 0),
        _ => Err(Error("condition is not boolean".into())),
    }
}
fn as_usize(v: &Value) -> Result<usize, Error> {
    match v {
        Value::Int(x) if *x >= 0 => Ok(*x as usize),
        _ => Err(Error("index is not a non-negative integer".into())),
    }
}
fn unary(op: UnaryOp, v: Value, ty: &ResolvedType) -> Result<Value, Error> {
    match (op, v) {
        (UnaryOp::Negate, Value::Int(x)) => Ok(Value::Int(normalize_integer(-x, ty))),
        (UnaryOp::Negate, Value::Float(x)) => Ok(Value::Float(-x)),
        (UnaryOp::Not, Value::Bool(x)) => Ok(Value::Bool(!x)),
        (UnaryOp::BitwiseNot, Value::Int(x)) => Ok(Value::Int(normalize_integer(!x, ty))),
        _ => Err(Error("unsupported unary operation in interpreter".into())),
    }
}
fn binary(op: BinaryOp, a: Value, b: Value, operand_type: &ResolvedType) -> Result<Value, Error> {
    if matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
        return Ok(Value::Bool(if op == BinaryOp::LogicalAnd {
            truth(&a)? && truth(&b)?
        } else {
            truth(&a)? || truth(&b)?
        }));
    }
    match (a, b) {
        (Value::Int(a), Value::Int(b)) => integer_binary(op, a, b, operand_type),
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
            _ => return Err(Error("unsupported floating operation".into())),
        }),
        (Value::Bool(a), Value::Bool(b)) => Ok(match op {
            BinaryOp::Equal => Value::Bool(a == b),
            BinaryOp::NotEqual => Value::Bool(a != b),
            _ => return Err(Error("unsupported boolean operation".into())),
        }),
        _ => Err(Error("operands have incompatible values".into())),
    }
}

// Keep this import used in documentation and make type changes fail loudly at compile time.
#[allow(dead_code)]
fn _type_name(_: &ResolvedType) {}
