//! A small, deliberately explicit interpreter for the typed IR.
//!
//! The interpreter is independent of LLVM and is intended for semantic tests and
//! quick experiments. Operations which need a native address or an FFI symbol
//! are rejected instead of being silently emulated.
use crate::ast::{BinaryOp, UnaryOp};
use crate::mir::{MirInstruction, MirProgram, MirTerminator};
use crate::ops;
use crate::typed::{
    DefId, FunctionId, IntegerWidth, LocalId, ResolvedType, TypedExpr, TypedPlace, TypedProgram,
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

type Env = HashMap<LocalId, Value>;
type Globals = HashMap<DefId, Value>;

pub fn run(program: &TypedProgram) -> Result<i32, Error> {
    run_with_pointer_width(program, usize::BITS)
}

pub fn run_with_pointer_width(program: &TypedProgram, pointer_bits: u32) -> Result<i32, Error> {
    POINTER_BITS.store(pointer_bits, Ordering::Relaxed);
    STEPS.store(0, Ordering::Relaxed);
    CALL_DEPTH.store(0, Ordering::Relaxed);
    let mir = MirProgram::lower(program)
        .map_err(|error| Error(format!("MIR lowering failed: {error}")))?;
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
    let value = call_mir(program, &mir, &mut globals, main.id, Vec::new())?;
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

fn call_mir(
    program: &TypedProgram,
    mir: &MirProgram,
    globals: &mut Globals,
    id: FunctionId,
    args: Vec<Value>,
) -> Result<Value, Error> {
    if CALL_DEPTH.fetch_add(1, Ordering::Relaxed) >= MAX_CALL_DEPTH {
        CALL_DEPTH.fetch_sub(1, Ordering::Relaxed);
        return Err(Error("interpreter call-depth limit exceeded".into()));
    }
    let _guard = CallGuard;
    let typed = function(program, id)?;
    if typed.is_extern {
        return Err(Error(format!(
            "interpreter does not support foreign function `{}`",
            typed.name
        )));
    }
    let mir_function = mir
        .function(id)
        .ok_or_else(|| Error(format!("unknown MIR function id {id}")))?;
    if mir_function.params.len() != args.len() {
        return Err(Error(format!(
            "wrong number of arguments to `{}`",
            typed.name
        )));
    }
    let mut env = Env::new();
    for (parameter, value) in mir_function.params.iter().zip(args) {
        env.insert(parameter.id, value);
    }
    let mut block_id = mir_function.entry;
    loop {
        consume_step()?;
        let block = mir_function
            .blocks
            .iter()
            .find(|block| block.id == block_id)
            .ok_or_else(|| Error(format!("unknown MIR block {}", block_id.0)))?;
        for instruction in &block.instructions {
            match instruction {
                MirInstruction::Declare { id, value, .. } => {
                    let evaluated = eval_expr(program, globals, &mut env, value)?;
                    if matches!(value, TypedExpr::Propagate { .. }) {
                        match evaluated {
                            Value::Result { error: true, .. } => {
                                let actions = mir_function
                                    .cleanup
                                    .get(&block.id)
                                    .cloned()
                                    .unwrap_or_default();
                                run_cleanup(program, globals, &mut env, &actions)?;
                                return Ok(evaluated);
                            }
                            Value::Result { value, .. } => {
                                env.insert(*id, *value);
                            }
                            other => {
                                env.insert(*id, other);
                            }
                        }
                    } else {
                        env.insert(*id, evaluated);
                    }
                }
                MirInstruction::Store { target, value, .. } => {
                    let evaluated = eval_expr(program, globals, &mut env, value)?;
                    if contains_propagation(value)
                        && matches!(evaluated, Value::Result { error: true, .. })
                    {
                        let actions = mir_function
                            .cleanup
                            .get(&block.id)
                            .cloned()
                            .unwrap_or_default();
                        run_cleanup(program, globals, &mut env, &actions)?;
                        return Ok(evaluated);
                    }
                    store(program, globals, &mut env, target, evaluated)?;
                }
                MirInstruction::Expr { expression, .. } => {
                    let evaluated = eval_expr(program, globals, &mut env, expression)?;
                    if contains_propagation(expression)
                        && matches!(evaluated, Value::Result { error: true, .. })
                    {
                        let actions = mir_function
                            .cleanup
                            .get(&block.id)
                            .cloned()
                            .unwrap_or_default();
                        run_cleanup(program, globals, &mut env, &actions)?;
                        return Ok(evaluated);
                    }
                }
                MirInstruction::RunCleanup { actions, .. } => {
                    for action in actions {
                        let values = action
                            .arguments
                            .iter()
                            .map(|argument| eval_expr(program, globals, &mut env, argument))
                            .collect::<Result<Vec<_>, _>>()?;
                        let _ = call(program, globals, action.function, values)?;
                    }
                }
            }
        }
        match &block.terminator {
            MirTerminator::Jump(target) => block_id = *target,
            MirTerminator::Branch {
                condition,
                then_block,
                else_block,
                ..
            } => {
                block_id = if truth(&eval_expr(program, globals, &mut env, condition)?)? {
                    *then_block
                } else {
                    *else_block
                };
            }
            MirTerminator::Return(value) => {
                return match value {
                    Some(value) => Ok(eval_expr(program, globals, &mut env, value)?),
                    None => Ok(Value::Unit),
                };
            }
            MirTerminator::Unreachable => {
                return Err(Error("reached MIR unreachable terminator".into()));
            }
        }
    }
}

fn run_cleanup(
    program: &TypedProgram,
    globals: &mut Globals,
    env: &mut Env,
    actions: &[crate::mir::MirCleanup],
) -> Result<(), Error> {
    for action in actions {
        let values = action
            .arguments
            .iter()
            .map(|argument| eval_expr(program, globals, env, argument))
            .collect::<Result<Vec<_>, _>>()?;
        let _ = call(program, globals, action.function, values)?;
    }
    Ok(())
}

/// Calls from expression evaluation are also lowered and executed as MIR.
/// This keeps the evaluator from growing a second structured-control-flow
/// implementation just for nested calls.
fn call(
    program: &TypedProgram,
    globals: &mut Globals,
    id: FunctionId,
    args: Vec<Value>,
) -> Result<Value, Error> {
    let mir = MirProgram::lower(program)
        .map_err(|error| Error(format!("MIR lowering failed: {error}")))?;
    call_mir(program, &mir, globals, id, args)
}

fn consume_step() -> Result<(), Error> {
    if STEPS.fetch_add(1, Ordering::Relaxed) >= MAX_STEPS {
        return Err(Error("interpreter execution step limit exceeded".into()));
    }
    Ok(())
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
        } => {
            let value = eval_expr(program, globals, env, operand)?;
            if propagated(&value) && contains_propagation(operand) {
                return Ok(value);
            }
            unary(*operator, value, ty)
        }
        TypedExpr::Binary {
            left,
            operator,
            right,
            operand_type,
            ..
        } => {
            let left_value = eval_expr(program, globals, env, left)?;
            if propagated(&left_value) && contains_propagation(left) {
                return Ok(left_value);
            }
            if *operator == BinaryOp::LogicalAnd && !truth(&left_value)? {
                return Ok(Value::Bool(false));
            }
            if *operator == BinaryOp::LogicalOr && truth(&left_value)? {
                return Ok(Value::Bool(true));
            }
            let right_value = eval_expr(program, globals, env, right)?;
            if propagated(&right_value) && contains_propagation(right) {
                return Ok(right_value);
            }
            binary(*operator, left_value, right_value, operand_type)
        }
        TypedExpr::Call {
            function,
            arguments,
            ..
        } => {
            let mut args = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let value = eval_expr(program, globals, env, argument)?;
                if propagated(&value) && contains_propagation(argument) {
                    return Ok(value);
                }
                args.push(value);
            }
            call(program, globals, *function, args)
        }
        TypedExpr::MakeSlice { .. } => Err(Error(
            "interpreter does not support raw-pointer slice construction".into(),
        )),
        TypedExpr::LowLevel { operation, .. } => Err(Error(format!(
            "interpreter does not support low-level operation {operation:?}"
        ))),
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
        TypedExpr::ResultOk { value, .. } => {
            let evaluated = eval_expr(program, globals, env, value)?;
            if propagated(&evaluated) && contains_propagation(value) {
                return Ok(evaluated);
            }
            Ok(Value::Result {
                error: false,
                value: Box::new(evaluated),
            })
        }
        TypedExpr::ResultErr { value, .. } => {
            let evaluated = eval_expr(program, globals, env, value)?;
            if propagated(&evaluated) && contains_propagation(value) {
                return Ok(evaluated);
            }
            Ok(Value::Result {
                error: true,
                value: Box::new(evaluated),
            })
        }
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

fn primitive(value: Value) -> Result<ops::Value, Error> {
    Ok(match value {
        Value::Bool(value) => ops::Value::Bool(value),
        Value::Int(value) => ops::Value::Integer(value),
        Value::Float(value) => ops::Value::Float(value),
        _ => return Err(Error("aggregate is not a primitive value".into())),
    })
}

fn from_primitive(value: ops::Value) -> Value {
    match value {
        ops::Value::Bool(value) => Value::Bool(value),
        ops::Value::Integer(value) => Value::Int(value),
        ops::Value::Float(value) => Value::Float(value),
    }
}

fn normalize_integer(value: i128, ty: &ResolvedType) -> i128 {
    ops::normalize_with_pointer_width(value, ty, POINTER_BITS.load(Ordering::Relaxed))
}

fn ops_error(error: ops::Error) -> Error {
    Error(
        match error {
            ops::Error::DivisionByZero => "integer division failed",
            ops::Error::InvalidOperation => "unsupported primitive operation",
        }
        .into(),
    )
}

/// Whether an expression contains the explicit result-propagation operator.
/// MIR treats an error result from such an expression as a function exit;
/// keeping this fact here prevents the interpreter from inventing a second
/// propagation convention for declarations, stores, and expression statements.
fn propagated(value: &Value) -> bool {
    matches!(value, Value::Result { error: true, .. })
}

fn contains_propagation(expression: &TypedExpr) -> bool {
    match expression {
        TypedExpr::Propagate { .. } => true,
        TypedExpr::Unary { operand, .. } => contains_propagation(operand),
        TypedExpr::Binary { left, right, .. } => {
            contains_propagation(left) || contains_propagation(right)
        }
        TypedExpr::StructLiteral { fields, .. } => fields.iter().any(contains_propagation),
        TypedExpr::ArrayLiteral { elements, .. } => elements.iter().any(contains_propagation),
        TypedExpr::Field { place, .. } | TypedExpr::Index { place, .. } => {
            place_contains_propagation(place)
        }
        TypedExpr::Call { arguments, .. } => arguments.iter().any(contains_propagation),
        TypedExpr::MakeSlice {
            pointer, length, ..
        } => contains_propagation(pointer) || contains_propagation(length),
        TypedExpr::ResultOk { value, .. }
        | TypedExpr::ResultErr { value, .. }
        | TypedExpr::IsErr { value, .. }
        | TypedExpr::Unwrap { value, .. } => contains_propagation(value),
        TypedExpr::AddressOf { place, .. } | TypedExpr::Dereference { place, .. } => {
            place_contains_propagation(place)
        }
        TypedExpr::Integer { .. }
        | TypedExpr::Float { .. }
        | TypedExpr::Bool { .. }
        | TypedExpr::Load { .. }
        | TypedExpr::GlobalLoad { .. }
        | TypedExpr::Null { .. }
        | TypedExpr::Layout { .. }
        | TypedExpr::LowLevel { .. } => false,
    }
}

fn place_contains_propagation(place: &TypedPlace) -> bool {
    match place {
        TypedPlace::Temporary { value, .. } => contains_propagation(value),
        TypedPlace::Field { base, .. } => place_contains_propagation(base),
        TypedPlace::Index { base, index, .. } => {
            place_contains_propagation(base) || contains_propagation(index)
        }
        TypedPlace::Dereference { pointer, .. } => contains_propagation(pointer),
        TypedPlace::Local { .. } | TypedPlace::Global { .. } => false,
    }
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
fn unary(op: UnaryOp, value: Value, ty: &ResolvedType) -> Result<Value, Error> {
    let value = primitive(value)?;
    ops::unary_with_pointer_width(op, value, ty, POINTER_BITS.load(Ordering::Relaxed))
        .map(from_primitive)
        .map_err(ops_error)
}

fn binary(op: BinaryOp, left: Value, right: Value, ty: &ResolvedType) -> Result<Value, Error> {
    let left = primitive(left)?;
    let right = primitive(right)?;
    ops::binary_with_pointer_width(op, left, right, ty, POINTER_BITS.load(Ordering::Relaxed))
        .map(from_primitive)
        .map_err(ops_error)
}

// Keep this import used in documentation and make type changes fail loudly at compile time.
#[allow(dead_code)]
fn _type_name(_: &ResolvedType) {}
