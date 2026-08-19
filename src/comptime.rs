//! Explicit compile-time evaluation and the small, deterministic compiler API.
//!
//! This module intentionally starts with an interpreter rather than executing
//! native code.  That gives compile-time calls the same source semantics while
//! keeping host file, process, network, allocator, and FFI access impossible.

use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, Type, UnaryOp};
use crate::lexer::Span;

const DEFAULT_STEP_LIMIT: u64 = 100_000;
const DEFAULT_RECURSION_LIMIT: usize = 128;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Integer(i128),
    Float(f64),
    Bool(bool),
    Array(Vec<Value>),
    Struct {
        name: String,
        fields: Vec<(String, Value)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    Unsupported { message: String, span: Span },
    Undefined { name: String, span: Span },
    NotConstant { span: Span },
    DivisionByZero { span: Span },
    StepLimit { span: Span },
    RecursionLimit { span: Span },
    InvalidOperation { message: String, span: Span },
}

impl Error {
    pub fn span(&self) -> Span {
        match self {
            Self::Unsupported { span, .. }
            | Self::Undefined { span, .. }
            | Self::NotConstant { span }
            | Self::DivisionByZero { span }
            | Self::StepLimit { span }
            | Self::RecursionLimit { span }
            | Self::InvalidOperation { span, .. } => *span,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { message, .. } | Self::InvalidOperation { message, .. } => {
                write!(f, "{message}")
            }
            Self::Undefined { name, .. } => write!(f, "undefined compile-time name `{name}`"),
            Self::NotConstant { .. } => write!(f, "expression is not compile-time evaluable"),
            Self::DivisionByZero { .. } => {
                write!(f, "division by zero during compile-time evaluation")
            }
            Self::StepLimit { .. } => write!(f, "compile-time evaluation exceeded the step limit"),
            Self::RecursionLimit { .. } => {
                write!(f, "compile-time evaluation exceeded the recursion limit")
            }
        }
    }
}
impl std::error::Error for Error {}

/// Options make resource limits explicit and make evaluation reproducible.
#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub steps: u64,
    pub recursion: usize,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEP_LIMIT,
            recursion: DEFAULT_RECURSION_LIMIT,
        }
    }
}

/// Evaluate and expand all explicit `#` expressions in a program. Ordinary
/// expressions are never executed by this pass.
pub fn expand(program: &Program, pointer_width: u32) -> Result<Program, Error> {
    expand_with_limits(program, pointer_width, Limits::default())
}

pub fn expand_with_limits(
    program: &Program,
    pointer_width: u32,
    limits: Limits,
) -> Result<Program, Error> {
    let specialized = specialize_program(program)?;
    Expander::new(&specialized, pointer_width, limits).expand()
}

/// Monomorphize the explicit `T: type` subset before semantic checking. The
/// resulting declarations have ordinary names and ABI, so the rest of the
/// compiler does not need a generic runtime representation.
fn specialize_program(program: &Program) -> Result<Program, Error> {
    let mut specializer = Specializer::new(program);
    let mut declarations = Vec::new();
    for declaration in &program.declarations {
        match declaration {
            Decl::Function(function) if function.generic_params.is_empty() => {
                declarations.push(Decl::Function(specializer.specialize_function(function)?));
            }
            Decl::Function(_) => {}
            Decl::Variable(variable) => {
                declarations.push(Decl::Variable(crate::ast::VariableDecl {
                    value: specializer.specialize_expr(&variable.value, &HashMap::new())?,
                    ..variable.clone()
                }))
            }
            Decl::Comptime { expression, span } => declarations.push(Decl::Comptime {
                expression: specializer.specialize_expr(expression, &HashMap::new())?,
                span: *span,
            }),
            other => declarations.push(other.clone()),
        }
    }
    let mut generated = specializer.generated.into_values().collect::<Vec<_>>();
    generated.sort_by(|a, b| a.name.cmp(&b.name));
    let mut generated = generated
        .into_iter()
        .map(Decl::Function)
        .collect::<Vec<_>>();
    generated.extend(declarations);
    Ok(Program {
        imports: program.imports.clone(),
        declarations: generated,
    })
}

struct Specializer<'a> {
    generic: HashMap<String, &'a FunctionDecl>,
    generated: HashMap<String, FunctionDecl>,
}
impl<'a> Specializer<'a> {
    fn new(program: &'a Program) -> Self {
        let generic = program
            .declarations
            .iter()
            .filter_map(|d| match d {
                Decl::Function(f) if !f.generic_params.is_empty() => Some((f.name.clone(), f)),
                _ => None,
            })
            .collect();
        Self {
            generic,
            generated: HashMap::new(),
        }
    }
    fn specialize_function(&mut self, function: &FunctionDecl) -> Result<FunctionDecl, Error> {
        let types = function
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        Ok(FunctionDecl {
            body: self.specialize_block(&function.body, &types)?,
            ..function.clone()
        })
    }
    fn ensure(
        &mut self,
        name: &str,
        arguments: &[Expr],
        span: Span,
        env: &HashMap<String, Type>,
    ) -> Result<String, Error> {
        let Some(function) = self.generic.get(name).copied() else {
            return Err(Error::Undefined {
                name: name.into(),
                span,
            });
        };
        let mut substitutions = HashMap::new();
        for (parameter, argument) in function.params.iter().zip(arguments) {
            let Some(argument_type) = infer_expr_type(argument, env) else {
                return Err(Error::InvalidOperation {
                    message: format!("cannot infer type argument for `{name}`"),
                    span,
                });
            };
            infer_generic(
                &parameter.ty,
                &argument_type,
                &mut substitutions,
                &function.generic_params,
                span,
            )?;
        }
        if substitutions.len() != function.generic_params.len() {
            return Err(Error::InvalidOperation {
                message: format!("could not infer all type arguments for `{name}`"),
                span,
            });
        }
        let suffix = function
            .generic_params
            .iter()
            .map(|p| {
                substitutions
                    .get(&p.name)
                    .map(mangle_type)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join("__");
        let specialized_name = format!("{name}__{suffix}");
        if self.generated.contains_key(&specialized_name) {
            return Ok(specialized_name);
        }
        let mut specialized = function.clone();
        specialized.name = specialized_name.clone();
        specialized.generic_params.clear();
        for parameter in &mut specialized.params {
            parameter.ty = substitute_type(&parameter.ty, &substitutions);
        }
        specialized.return_type = substitute_type(&specialized.return_type, &substitutions);
        // Insert a skeleton first: recursive generic calls hit the recursion
        // limit in evaluation rather than endlessly creating declarations.
        self.generated
            .insert(specialized_name.clone(), specialized.clone());
        let env = specialized
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        let source_body = substitute_block_types(&function.body, &substitutions);
        specialized.body = self.specialize_block(&source_body, &env)?;
        self.generated.insert(specialized_name.clone(), specialized);
        Ok(specialized_name)
    }
    fn specialize_block(
        &mut self,
        block: &Block,
        env: &HashMap<String, Type>,
    ) -> Result<Block, Error> {
        let mut block = block.clone();
        let mut locals = env.clone();
        for statement in &mut block.statements {
            match statement {
                Stmt::Variable(v) => {
                    v.value = self.specialize_expr(&v.value, &locals)?;
                    let ty = v.ty.clone().unwrap_or_else(|| {
                        infer_expr_type(&v.value, &locals).unwrap_or(Type::Named("i32".into()))
                    });
                    locals.insert(v.name.clone(), ty);
                }
                Stmt::Expr { expression, .. } => {
                    *expression = self.specialize_expr(expression, &locals)?
                }
                Stmt::Return {
                    value: Some(value), ..
                } => *value = self.specialize_expr(value, &locals)?,
                Stmt::Assignment { target, value, .. } => {
                    *target = self.specialize_expr(target, &locals)?;
                    *value = self.specialize_expr(value, &locals)?;
                }
                Stmt::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    *condition = self.specialize_expr(condition, &locals)?;
                    *then_branch = self.specialize_block(then_branch, &locals)?;
                    if let Some(branch) = else_branch {
                        *branch = self.specialize_block(branch, &locals)?;
                    }
                }
                Stmt::While {
                    condition, body, ..
                } => {
                    *condition = self.specialize_expr(condition, &locals)?;
                    *body = self.specialize_block(body, &locals)?;
                }
                Stmt::Defer { call, .. } => *call = self.specialize_expr(call, &locals)?,
                _ => {}
            }
        }
        Ok(block)
    }
    fn specialize_expr(
        &mut self,
        expression: &Expr,
        env: &HashMap<String, Type>,
    ) -> Result<Expr, Error> {
        let span = expression.span();
        match expression {
            Expr::Call {
                callee, arguments, ..
            } => {
                let arguments = arguments
                    .iter()
                    .map(|x| self.specialize_expr(x, env))
                    .collect::<Result<Vec<_>, _>>()?;
                let mut callee = self.specialize_expr(callee, env)?;
                if let Expr::Identifier {
                    name,
                    span: callee_span,
                } = &callee
                {
                    if self.generic.contains_key(name) {
                        let specialized = self.ensure(name, &arguments, *callee_span, env)?;
                        callee = Expr::Identifier {
                            name: specialized,
                            span: *callee_span,
                        };
                    }
                }
                Ok(Expr::Call {
                    callee: Box::new(callee),
                    arguments,
                    span,
                })
            }
            Expr::Binary {
                left,
                operator,
                right,
                ..
            } => Ok(Expr::Binary {
                left: Box::new(self.specialize_expr(left, env)?),
                operator: *operator,
                right: Box::new(self.specialize_expr(right, env)?),
                span,
            }),
            Expr::Unary {
                operator, operand, ..
            } => Ok(Expr::Unary {
                operator: *operator,
                operand: Box::new(self.specialize_expr(operand, env)?),
                span,
            }),
            Expr::Field { base, name, .. } => Ok(Expr::Field {
                base: Box::new(self.specialize_expr(base, env)?),
                name: name.clone(),
                span,
            }),
            Expr::Index { base, index, .. } => Ok(Expr::Index {
                base: Box::new(self.specialize_expr(base, env)?),
                index: Box::new(self.specialize_expr(index, env)?),
                span,
            }),
            Expr::UncheckedIndex { base, index, .. } => Ok(Expr::UncheckedIndex {
                base: Box::new(self.specialize_expr(base, env)?),
                index: Box::new(self.specialize_expr(index, env)?),
                span,
            }),
            Expr::ArrayLiteral { ty, elements, .. } => Ok(Expr::ArrayLiteral {
                ty: substitute_type(ty, &HashMap::new()),
                elements: elements
                    .iter()
                    .map(|x| self.specialize_expr(x, env))
                    .collect::<Result<_, _>>()?,
                span,
            }),
            Expr::StructLiteral { name, fields, .. } => Ok(Expr::StructLiteral {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| {
                        Ok(crate::ast::StructInit {
                            name: f.name.clone(),
                            value: self.specialize_expr(&f.value, env)?,
                            span: f.span,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
                span,
            }),
            Expr::Propagate { expression, .. } => Ok(Expr::Propagate {
                expression: Box::new(self.specialize_expr(expression, env)?),
                span,
            }),
            Expr::Comptime { expression, .. } => Ok(Expr::Comptime {
                expression: Box::new(self.specialize_expr(expression, env)?),
                span,
            }),
            _ => Ok(expression.clone()),
        }
    }
}
fn infer_expr_type(expression: &Expr, env: &HashMap<String, Type>) -> Option<Type> {
    match expression {
        Expr::Integer { .. } => Some(Type::Named("i32".into())),
        Expr::Float { .. } => Some(Type::Named("f64".into())),
        Expr::Bool { .. } => Some(Type::Named("bool".into())),
        Expr::Identifier { name, .. } => env.get(name).cloned(),
        Expr::StructLiteral { name, .. } => Some(Type::Named(name.clone())),
        Expr::ArrayLiteral { ty, .. } => Some(ty.clone()),
        _ => None,
    }
}
fn infer_generic(
    pattern: &Type,
    actual: &Type,
    substitutions: &mut HashMap<String, Type>,
    params: &[crate::ast::GenericParam],
    span: Span,
) -> Result<(), Error> {
    if let Type::Named(name) = pattern {
        if params.iter().any(|p| p.name == *name) {
            if let Some(old) = substitutions.get(name) {
                if old != actual {
                    return Err(Error::InvalidOperation {
                        message: "conflicting inferred generic type arguments".into(),
                        span,
                    });
                }
            } else {
                substitutions.insert(name.clone(), actual.clone());
            }
            return Ok(());
        }
    }
    match (pattern, actual) {
        (Type::Pointer(p), Type::Pointer(a)) | (Type::Slice(p), Type::Slice(a)) => {
            infer_generic(p, a, substitutions, params, span)
        }
        (
            Type::Array {
                length: pl,
                element: p,
            },
            Type::Array {
                length: al,
                element: a,
            },
        ) if pl == al => infer_generic(p, a, substitutions, params, span),
        (
            Type::Result {
                success: ps,
                error: pe,
            },
            Type::Result {
                success: as_,
                error: ae,
            },
        ) => {
            infer_generic(ps, as_, substitutions, params, span)?;
            infer_generic(pe, ae, substitutions, params, span)
        }
        _ => Ok(()),
    }
}
fn substitute_type(ty: &Type, substitutions: &HashMap<String, Type>) -> Type {
    match ty {
        Type::Named(name) => substitutions
            .get(name)
            .cloned()
            .unwrap_or_else(|| ty.clone()),
        Type::Pointer(x) => Type::Pointer(Box::new(substitute_type(x, substitutions))),
        Type::Slice(x) => Type::Slice(Box::new(substitute_type(x, substitutions))),
        Type::Array { length, element } => Type::Array {
            length: *length,
            element: Box::new(substitute_type(element, substitutions)),
        },
        Type::Result { success, error } => Type::Result {
            success: Box::new(substitute_type(success, substitutions)),
            error: Box::new(substitute_type(error, substitutions)),
        },
        _ => ty.clone(),
    }
}
fn substitute_block_types(block: &Block, substitutions: &HashMap<String, Type>) -> Block {
    let mut block = block.clone();
    for statement in &mut block.statements {
        match statement {
            Stmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                *condition = substitute_expr_types(condition, substitutions);
                *then_branch = substitute_block_types(then_branch, substitutions);
                if let Some(branch) = else_branch {
                    *branch = substitute_block_types(branch, substitutions);
                }
            }
            Stmt::While {
                condition, body, ..
            } => {
                *condition = substitute_expr_types(condition, substitutions);
                *body = substitute_block_types(body, substitutions);
            }
            Stmt::Return {
                value: Some(value), ..
            } => *value = substitute_expr_types(value, substitutions),
            Stmt::Variable(v) => {
                v.ty = v.ty.as_ref().map(|ty| substitute_type(ty, substitutions));
                v.value = substitute_expr_types(&v.value, substitutions);
            }
            Stmt::Assignment { target, value, .. } => {
                *target = substitute_expr_types(target, substitutions);
                *value = substitute_expr_types(value, substitutions);
            }
            Stmt::Expr { expression, .. }
            | Stmt::Defer {
                call: expression, ..
            } => *expression = substitute_expr_types(expression, substitutions),
            _ => {}
        }
    }
    block
}
fn substitute_expr_types(expression: &Expr, substitutions: &HashMap<String, Type>) -> Expr {
    let span = expression.span();
    match expression {
        Expr::ArrayLiteral { ty, elements, .. } => Expr::ArrayLiteral {
            ty: substitute_type(ty, substitutions),
            elements: elements
                .iter()
                .map(|x| substitute_expr_types(x, substitutions))
                .collect(),
            span,
        },
        Expr::SizeOf { ty, .. } => Expr::SizeOf {
            ty: substitute_type(ty, substitutions),
            span,
        },
        Expr::AlignOf { ty, .. } => Expr::AlignOf {
            ty: substitute_type(ty, substitutions),
            span,
        },
        Expr::OffsetOf { ty, field, .. } => Expr::OffsetOf {
            ty: substitute_type(ty, substitutions),
            field: field.clone(),
            span,
        },
        Expr::Binary {
            left,
            operator,
            right,
            ..
        } => Expr::Binary {
            left: Box::new(substitute_expr_types(left, substitutions)),
            operator: *operator,
            right: Box::new(substitute_expr_types(right, substitutions)),
            span,
        },
        Expr::Unary {
            operator, operand, ..
        } => Expr::Unary {
            operator: *operator,
            operand: Box::new(substitute_expr_types(operand, substitutions)),
            span,
        },
        Expr::Call {
            callee, arguments, ..
        } => Expr::Call {
            callee: Box::new(substitute_expr_types(callee, substitutions)),
            arguments: arguments
                .iter()
                .map(|x| substitute_expr_types(x, substitutions))
                .collect(),
            span,
        },
        Expr::Field { base, name, .. } => Expr::Field {
            base: Box::new(substitute_expr_types(base, substitutions)),
            name: name.clone(),
            span,
        },
        Expr::Index { base, index, .. } => Expr::Index {
            base: Box::new(substitute_expr_types(base, substitutions)),
            index: Box::new(substitute_expr_types(index, substitutions)),
            span,
        },
        Expr::UncheckedIndex { base, index, .. } => Expr::UncheckedIndex {
            base: Box::new(substitute_expr_types(base, substitutions)),
            index: Box::new(substitute_expr_types(index, substitutions)),
            span,
        },
        Expr::StructLiteral { name, fields, .. } => Expr::StructLiteral {
            name: name.clone(),
            fields: fields
                .iter()
                .map(|f| crate::ast::StructInit {
                    name: f.name.clone(),
                    value: substitute_expr_types(&f.value, substitutions),
                    span: f.span,
                })
                .collect(),
            span,
        },
        Expr::Propagate { expression, .. } => Expr::Propagate {
            expression: Box::new(substitute_expr_types(expression, substitutions)),
            span,
        },
        Expr::Comptime { expression, .. } => Expr::Comptime {
            expression: Box::new(substitute_expr_types(expression, substitutions)),
            span,
        },
        _ => expression.clone(),
    }
}
fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Named(n) => n.replace('.', "_"),
        Type::Array { length, element } => format!("a{length}_{}", mangle_type(element)),
        Type::Pointer(x) => format!("p_{}", mangle_type(x)),
        Type::Slice(x) => format!("s_{}", mangle_type(x)),
        Type::Unit => "unit".into(),
        Type::Result { .. } => "result".into(),
    }
}

struct Expander<'a> {
    program: &'a Program,
    functions: HashMap<String, &'a FunctionDecl>,
    structs: HashMap<String, &'a crate::ast::StructDecl>,
    globals: HashMap<String, Value>,
    pointer_width: u32,
    limits: Limits,
    steps: u64,
    recursion: usize,
}
impl<'a> Expander<'a> {
    fn new(program: &'a Program, pointer_width: u32, limits: Limits) -> Self {
        let mut functions = HashMap::new();
        let mut structs = HashMap::new();
        for declaration in &program.declarations {
            match declaration {
                Decl::Function(f) => {
                    functions.insert(f.name.clone(), f);
                }
                Decl::Struct(s) => {
                    structs.insert(s.name.clone(), s);
                }
                _ => {}
            }
        }
        Self {
            program,
            functions,
            structs,
            globals: HashMap::new(),
            pointer_width,
            limits,
            steps: 0,
            recursion: 0,
        }
    }
    fn expand(mut self) -> Result<Program, Error> {
        let mut declarations = Vec::new();
        for declaration in &self.program.declarations {
            match declaration {
                Decl::Variable(v) => {
                    let value = self.expand_expr(&v.value)?;
                    if let Expr::Comptime { expression, .. } = &v.value {
                        let value = self.eval(expression, &HashMap::new())?;
                        self.globals.insert(v.name.clone(), value.clone());
                        let hint = self.expression_type_hint(expression);
                        let value =
                            self.value_expr_with_hint(value, v.value.span(), hint.as_ref())?;
                        let mut variable = v.clone();
                        variable.value = value;
                        if variable.ty.is_none() {
                            variable.ty = hint;
                        }
                        declarations.push(Decl::Variable(variable));
                    } else {
                        // Constants are available as inputs to later explicit
                        // calls, but are never evaluated merely because they
                        // are present in a program.
                        if let Ok(value) = self.eval(&value, &HashMap::new()) {
                            self.globals.insert(v.name.clone(), value);
                        }
                        declarations.push(Decl::Variable(crate::ast::VariableDecl {
                            value,
                            ..v.clone()
                        }));
                    }
                }
                Decl::Comptime { expression, span } => {
                    // A directive is intentionally not retained in runtime IR.
                    let value = self.eval(expression, &HashMap::new()).map_err(|e| e)?;
                    if !matches!(value, Value::Unit) {
                        return Err(Error::InvalidOperation {
                            message: "top-level compile-time directive must return unit".into(),
                            span: *span,
                        });
                    }
                }
                Decl::Function(f) => {
                    let mut copy = f.clone();
                    copy.body = self.expand_block(&f.body)?;
                    declarations.push(Decl::Function(copy));
                }
                other => declarations.push(other.clone()),
            }
        }
        Ok(Program {
            imports: self.program.imports.clone(),
            declarations,
        })
    }
    fn expand_block(&mut self, block: &Block) -> Result<Block, Error> {
        let mut copy = block.clone();
        for statement in &mut copy.statements {
            match statement {
                Stmt::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    *condition = self.expand_expr(condition)?;
                    *then_branch = self.expand_block(then_branch)?;
                    if let Some(branch) = else_branch {
                        *branch = self.expand_block(branch)?;
                    }
                }
                Stmt::While {
                    condition, body, ..
                } => {
                    *condition = self.expand_expr(condition)?;
                    *body = self.expand_block(body)?;
                }
                Stmt::Return {
                    value: Some(value), ..
                } => *value = self.expand_expr(value)?,
                Stmt::Variable(v) => {
                    if let Expr::Comptime { expression, .. } = &v.value {
                        let hint = self.expression_type_hint(expression);
                        let value = self.eval(expression, &HashMap::new())?;
                        v.value =
                            self.value_expr_with_hint(value, v.value.span(), hint.as_ref())?;
                        if v.ty.is_none() {
                            v.ty = hint;
                        }
                    } else {
                        v.value = self.expand_expr(&v.value)?;
                    }
                }
                Stmt::Assignment { target, value, .. } => {
                    *target = self.expand_expr(target)?;
                    *value = self.expand_expr(value)?;
                }
                Stmt::Expr { expression, .. }
                | Stmt::Defer {
                    call: expression, ..
                } => *expression = self.expand_expr(expression)?,
                _ => {}
            }
        }
        Ok(copy)
    }
    fn expand_expr(&mut self, expression: &Expr) -> Result<Expr, Error> {
        if let Expr::Comptime {
            expression: inner,
            span,
        } = expression
        {
            let value = self.eval(inner, &HashMap::new())?;
            return self.value_expr(value, *span);
        }
        // Rebuild recursively so a marker can occur below a runtime expression.
        match expression {
            Expr::Binary {
                left,
                right,
                operator,
                span,
            } => Ok(Expr::Binary {
                left: Box::new(self.expand_expr(left)?),
                right: Box::new(self.expand_expr(right)?),
                operator: *operator,
                span: *span,
            }),
            Expr::Unary {
                operand,
                operator,
                span,
            } => Ok(Expr::Unary {
                operand: Box::new(self.expand_expr(operand)?),
                operator: *operator,
                span: *span,
            }),
            Expr::Call {
                callee,
                arguments,
                span,
            } => Ok(Expr::Call {
                callee: Box::new(self.expand_expr(callee)?),
                arguments: arguments
                    .iter()
                    .map(|x| self.expand_expr(x))
                    .collect::<Result<_, _>>()?,
                span: *span,
            }),
            Expr::Field { base, name, span } => Ok(Expr::Field {
                base: Box::new(self.expand_expr(base)?),
                name: name.clone(),
                span: *span,
            }),
            Expr::Index { base, index, span } => Ok(Expr::Index {
                base: Box::new(self.expand_expr(base)?),
                index: Box::new(self.expand_expr(index)?),
                span: *span,
            }),
            Expr::UncheckedIndex { base, index, span } => Ok(Expr::UncheckedIndex {
                base: Box::new(self.expand_expr(base)?),
                index: Box::new(self.expand_expr(index)?),
                span: *span,
            }),
            Expr::StructLiteral { name, fields, span } => Ok(Expr::StructLiteral {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| {
                        Ok(crate::ast::StructInit {
                            name: f.name.clone(),
                            value: self.expand_expr(&f.value)?,
                            span: f.span,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
                span: *span,
            }),
            Expr::ArrayLiteral { ty, elements, span } => Ok(Expr::ArrayLiteral {
                ty: ty.clone(),
                elements: elements
                    .iter()
                    .map(|x| self.expand_expr(x))
                    .collect::<Result<_, _>>()?,
                span: *span,
            }),
            Expr::Propagate { expression, span } => Ok(Expr::Propagate {
                expression: Box::new(self.expand_expr(expression)?),
                span: *span,
            }),
            _ => Ok(expression.clone()),
        }
    }
    fn tick(&mut self, span: Span) -> Result<(), Error> {
        self.steps += 1;
        if self.steps > self.limits.steps {
            Err(Error::StepLimit { span })
        } else {
            Ok(())
        }
    }
    fn eval(&mut self, expression: &Expr, env: &HashMap<String, Value>) -> Result<Value, Error> {
        self.tick(expression.span())?;
        match expression {
            Expr::Integer { value, .. } => Ok(Value::Integer(*value)),
            Expr::Float { value, .. } => Ok(Value::Float(*value)),
            Expr::Bool { value, .. } => Ok(Value::Bool(*value)),
            Expr::ArrayLiteral { elements, .. } => Ok(Value::Array(
                elements
                    .iter()
                    .map(|e| self.eval(e, env))
                    .collect::<Result<_, _>>()?,
            )),
            Expr::StructLiteral { name, fields, .. } => Ok(Value::Struct {
                name: name.clone(),
                fields: fields
                    .iter()
                    .map(|f| Ok((f.name.clone(), self.eval(&f.value, env)?)))
                    .collect::<Result<_, Error>>()?,
            }),
            Expr::Identifier { name, span } => env
                .get(name)
                .cloned()
                .or_else(|| self.globals.get(name).cloned())
                .ok_or(Error::Undefined {
                    name: name.clone(),
                    span: *span,
                }),
            Expr::Comptime { expression, .. } => self.eval(expression, env),
            Expr::SizeOf { ty, span } => Ok(Value::Integer(self.layout(ty, *span)?.0 as i128)),
            Expr::AlignOf { ty, span } => Ok(Value::Integer(self.layout(ty, *span)?.1 as i128)),
            Expr::OffsetOf { ty, field, span } => {
                Ok(Value::Integer(self.offset(ty, field, *span)? as i128))
            }
            Expr::Unary {
                operator,
                operand,
                span,
            } => {
                let x = self.eval(operand, env)?;
                match (operator, x) {
                    (UnaryOp::Negate, Value::Integer(x)) => Ok(Value::Integer(-x)),
                    (UnaryOp::Negate, Value::Float(x)) => Ok(Value::Float(-x)),
                    (UnaryOp::Not, Value::Bool(x)) | (UnaryOp::BitwiseNot, Value::Bool(x)) => {
                        Ok(Value::Bool(!x))
                    }
                    (UnaryOp::BitwiseNot, Value::Integer(x)) => Ok(Value::Integer(!x)),
                    _ => Err(Error::Unsupported {
                        message:
                            "pointers and runtime-only operations are unavailable at compile time"
                                .into(),
                        span: *span,
                    }),
                }
            }
            Expr::Binary {
                left,
                operator,
                right,
                span,
            } => {
                if *operator == BinaryOp::LogicalAnd {
                    let a = self.eval(left, env)?;
                    return match a {
                        Value::Bool(false) => Ok(Value::Bool(false)),
                        Value::Bool(true) => self.eval(right, env),
                        _ => Err(Error::InvalidOperation {
                            message: "logical operators require booleans".into(),
                            span: *span,
                        }),
                    };
                }
                if *operator == BinaryOp::LogicalOr {
                    let a = self.eval(left, env)?;
                    return match a {
                        Value::Bool(true) => Ok(Value::Bool(true)),
                        Value::Bool(false) => self.eval(right, env),
                        _ => Err(Error::InvalidOperation {
                            message: "logical operators require booleans".into(),
                            span: *span,
                        }),
                    };
                }
                let a = self.eval(left, env)?;
                let b = self.eval(right, env)?;
                self.binary(a, *operator, b, *span)
            }
            Expr::Call {
                callee,
                arguments,
                span,
            } => {
                let Expr::Identifier { name, .. } = callee.as_ref() else {
                    return Err(Error::Unsupported {
                        message: "indirect compile-time calls are unavailable".into(),
                        span: *span,
                    });
                };
                if name == "validate_type" {
                    if arguments.len() != 1 {
                        return Err(Error::InvalidOperation {
                            message: "validate_type expects one type argument".into(),
                            span: *span,
                        });
                    }
                    self.validate_type_expr(&arguments[0])?;
                    return Ok(Value::Unit);
                }
                let args = arguments
                    .iter()
                    .map(|x| self.eval(x, env))
                    .collect::<Result<Vec<_>, _>>()?;
                self.call(name, args, *span)
            }
            Expr::Field { base, name, span } => match self.eval(base, env)? {
                Value::Struct { fields, .. } => fields
                    .into_iter()
                    .find(|(n, _)| n == name)
                    .map(|(_, v)| v)
                    .ok_or(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                _ => Err(Error::InvalidOperation {
                    message: "field access requires a compile-time struct".into(),
                    span: *span,
                }),
            },
            Expr::Index { base, index, span } | Expr::UncheckedIndex { base, index, span } => {
                let Value::Array(values) = self.eval(base, env)? else {
                    return Err(Error::InvalidOperation {
                        message: "indexing requires a compile-time array".into(),
                        span: *span,
                    });
                };
                let Value::Integer(i) = self.eval(index, env)? else {
                    return Err(Error::InvalidOperation {
                        message: "array index must be an integer".into(),
                        span: index.span(),
                    });
                };
                values
                    .get(i as usize)
                    .cloned()
                    .ok_or(Error::InvalidOperation {
                        message: "compile-time array index is out of bounds".into(),
                        span: *span,
                    })
            }
            Expr::Null { span } | Expr::Propagate { span, .. } => Err(Error::Unsupported {
                message: "null/result propagation is not available in compile-time evaluation"
                    .into(),
                span: *span,
            }),
        }
    }
    fn binary(&self, a: Value, op: BinaryOp, b: Value, span: Span) -> Result<Value, Error> {
        match op {
            BinaryOp::Add => integer_op(a, b, |x: i128, y: i128| x.wrapping_add(y), span),
            BinaryOp::Subtract => integer_op(a, b, |x: i128, y: i128| x.wrapping_sub(y), span),
            BinaryOp::Multiply => integer_op(a, b, |x: i128, y: i128| x.wrapping_mul(y), span),
            BinaryOp::BitwiseAnd => integer_op(a, b, |x: i128, y: i128| x & y, span),
            BinaryOp::BitwiseOr => integer_op(a, b, |x: i128, y: i128| x | y, span),
            BinaryOp::BitwiseXor => integer_op(a, b, |x: i128, y: i128| x ^ y, span),

            BinaryOp::Divide | BinaryOp::Modulo => match (a, b) {
                (Value::Integer(x), Value::Integer(y)) => {
                    if y == 0 {
                        return Err(Error::DivisionByZero { span });
                    }
                    Ok(Value::Integer(if op == BinaryOp::Divide {
                        x / y
                    } else {
                        x % y
                    }))
                }
                _ => Err(Error::InvalidOperation {
                    message: "integer operands required".into(),
                    span,
                }),
            },
            BinaryOp::Equal => Ok(Value::Bool(a == b)),
            BinaryOp::NotEqual => Ok(Value::Bool(a != b)),
            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => {
                match (a, b) {
                    (Value::Integer(x), Value::Integer(y)) => Ok(Value::Bool(match op {
                        BinaryOp::Less => x < y,
                        BinaryOp::LessEqual => x <= y,
                        BinaryOp::Greater => x > y,
                        _ => x >= y,
                    })),
                    _ => Err(Error::InvalidOperation {
                        message: "ordered integer operands required".into(),
                        span,
                    }),
                }
            }
            BinaryOp::ShiftLeft => {
                integer_op(a, b, |x: i128, y: i128| x.wrapping_shl(y as u32), span)
            }
            BinaryOp::ShiftRight => {
                integer_op(a, b, |x: i128, y: i128| x.wrapping_shr(y as u32), span)
            }
            BinaryOp::LogicalAnd | BinaryOp::LogicalOr => unreachable!(),
        }
    }
    fn call(&mut self, name: &str, args: Vec<Value>, span: Span) -> Result<Value, Error> {
        let Some(function) = self.functions.get(name).copied().cloned() else {
            return Err(Error::Undefined {
                name: name.into(),
                span,
            });
        };
        if function.is_extern {
            return Err(Error::Unsupported {
                message: "FFI calls are runtime-only in compile-time context".into(),
                span,
            });
        }
        if args.len() != function.params.len() {
            return Err(Error::InvalidOperation {
                message: format!("compile-time call `{name}` has the wrong number of arguments"),
                span,
            });
        }
        if self.recursion >= self.limits.recursion {
            return Err(Error::RecursionLimit { span });
        }
        self.recursion += 1;
        let mut env = HashMap::new();
        for (p, value) in function.params.iter().zip(args) {
            env.insert(p.name.clone(), value);
        }
        let result = self.exec_block(&function.body, &mut env);
        self.recursion -= 1;
        result
    }
    fn exec_block(
        &mut self,
        block: &Block,
        env: &mut HashMap<String, Value>,
    ) -> Result<Value, Error> {
        for statement in &block.statements {
            match statement {
                Stmt::Return { value, .. } => {
                    return value
                        .as_ref()
                        .map(|x| self.eval(x, env))
                        .unwrap_or(Ok(Value::Unit));
                }
                Stmt::Variable(v) => {
                    let value = self.eval(&v.value, env)?;
                    env.insert(v.name.clone(), value);
                }
                Stmt::Expr { expression, .. } => {
                    self.eval(expression, env)?;
                }
                Stmt::If {
                    condition,
                    then_branch,
                    else_branch,
                    ..
                } => {
                    if matches!(self.eval(condition, env)?, Value::Bool(true)) {
                        let result = self.exec_block(then_branch, env)?;
                        if !matches!(result, Value::Unit) {
                            return Ok(result);
                        }
                    } else if let Some(branch) = else_branch {
                        let result = self.exec_block(branch, env)?;
                        if !matches!(result, Value::Unit) {
                            return Ok(result);
                        }
                    }
                }
                Stmt::While {
                    condition,
                    body,
                    span,
                } => {
                    let mut guard = 0;
                    while matches!(self.eval(condition, env)?, Value::Bool(true)) {
                        guard += 1;
                        if guard > self.limits.steps {
                            return Err(Error::StepLimit { span: *span });
                        }
                        let result = self.exec_block(body, env)?;
                        if !matches!(result, Value::Unit) {
                            return Ok(result);
                        }
                    }
                }
                Stmt::Assignment {
                    target: Expr::Identifier { name, .. },
                    value,
                    ..
                } => {
                    let value = self.eval(value, env)?;
                    if !env.contains_key(name) {
                        return Err(Error::Undefined {
                            name: name.clone(),
                            span: value_span(value),
                        });
                    }
                    env.insert(name.clone(), value);
                }
                Stmt::Assignment { span, .. } => {
                    return Err(Error::Unsupported {
                        message: "complex assignment is unavailable at compile time".into(),
                        span: *span,
                    });
                }
                Stmt::Break { .. } | Stmt::Continue { .. } => {
                    return Err(Error::Unsupported {
                        message:
                            "break and continue are not yet available in compile-time functions"
                                .into(),
                        span: statement_span(statement),
                    });
                }
                Stmt::Defer { span, .. } => {
                    return Err(Error::Unsupported {
                        message: "defer is runtime-only in compile-time functions".into(),
                        span: *span,
                    });
                }
            }
        }
        Ok(Value::Unit)
    }
    fn expression_type_hint(&self, expression: &Expr) -> Option<Type> {
        match expression {
            Expr::Integer { .. } => Some(Type::Named("i32".into())),
            Expr::Float { .. } => Some(Type::Named("f64".into())),
            Expr::Bool { .. } => Some(Type::Named("bool".into())),
            Expr::SizeOf { .. } | Expr::AlignOf { .. } | Expr::OffsetOf { .. } => {
                Some(Type::Named("usize".into()))
            }
            Expr::StructLiteral { name, .. } => Some(Type::Named(name.clone())),
            Expr::ArrayLiteral { ty, .. } => Some(ty.clone()),
            Expr::Call { callee, .. } => match callee.as_ref() {
                Expr::Identifier { name, .. } => {
                    self.functions.get(name).map(|f| f.return_type.clone())
                }
                _ => None,
            },
            _ => None,
        }
    }
    fn value_expr(&self, value: Value, span: Span) -> Result<Expr, Error> {
        self.value_expr_with_hint(value, span, None)
    }
    fn value_expr_with_hint(
        &self,
        value: Value,
        span: Span,
        hint: Option<&Type>,
    ) -> Result<Expr, Error> {
        Ok(match value {
            Value::Unit => Expr::Integer { value: 0, span },
            Value::Integer(value) => Expr::Integer { value, span },
            Value::Float(value) => Expr::Float { value, span },
            Value::Bool(value) => Expr::Bool { value, span },
            Value::Array(values) => {
                let ty = hint.cloned().unwrap_or(Type::Array {
                    length: values.len() as u64,
                    element: Box::new(Type::Named("i32".into())),
                });
                let element_hint = match &ty {
                    Type::Array { element, .. } => Some((**element).clone()),
                    _ => None,
                };
                Expr::ArrayLiteral {
                    ty,
                    elements: values
                        .into_iter()
                        .map(|v| self.value_expr_with_hint(v, span, element_hint.as_ref()))
                        .collect::<Result<_, _>>()?,
                    span,
                }
            }
            Value::Struct { name, fields } => Expr::StructLiteral {
                name,
                fields: fields
                    .into_iter()
                    .map(|(name, value)| {
                        Ok(crate::ast::StructInit {
                            name,
                            value: self.value_expr(value, span)?,
                            span,
                        })
                    })
                    .collect::<Result<_, Error>>()?,
                span,
            },
        })
    }
    fn validate_type_expr(&self, expression: &Expr) -> Result<(), Error> {
        let Expr::Identifier { name, span } = expression else {
            return Err(Error::InvalidOperation {
                message: "validate_type expects a type name".into(),
                span: expression.span(),
            });
        };
        let builtin = matches!(
            name.as_str(),
            "unit" | "bool" | "f32" | "f64" | "usize" | "isize"
        ) || (name.len() > 1
            && matches!(name.as_bytes()[0], b'i' | b'u')
            && name[1..].parse::<u16>().is_ok());
        if !builtin && !self.structs.contains_key(name) {
            return Err(Error::Undefined {
                name: name.clone(),
                span: *span,
            });
        }
        Ok(())
    }
    fn layout(&self, ty: &Type, span: Span) -> Result<(u64, u64), Error> {
        match ty {
            Type::Unit => Ok((0, 1)),
            Type::Named(n) if n == "bool" => Ok((1, 1)),
            Type::Named(n) if n == "usize" || n == "isize" => Ok((
                (self.pointer_width / 8) as u64,
                (self.pointer_width / 8) as u64,
            )),
            Type::Named(n) if n.starts_with('i') || n.starts_with('u') => {
                let bits: u64 = n[1..].parse().unwrap_or(32);
                Ok(((bits / 8).max(1), (bits / 8).max(1)))
            }
            Type::Named(n) if n == "f32" => Ok((4, 4)),
            Type::Named(n) if n == "f64" => Ok((8, 8)),
            Type::Named(n) => {
                let s = self.structs.get(n).ok_or(Error::Undefined {
                    name: n.clone(),
                    span,
                })?;
                let mut size = 0;
                let mut align = 1;
                for field in &s.fields {
                    let (fs, fa) = self.layout(&field.ty, field.span)?;
                    size = align_to(size, fa);
                    size += fs;
                    align = align.max(fa);
                }
                Ok((align_to(size, align), align))
            }
            Type::Pointer(_) => Ok((
                (self.pointer_width / 8) as u64,
                (self.pointer_width / 8) as u64,
            )),
            Type::Array { length, element } => {
                let (s, a) = self.layout(element, span)?;
                Ok((s * length, a))
            }
            Type::Slice(_) => Ok((
                (self.pointer_width / 8 * 2) as u64,
                (self.pointer_width / 8) as u64,
            )),
            Type::Result { .. } => Err(Error::Unsupported {
                message: "result layout reflection is not yet available".into(),
                span,
            }),
        }
    }
    fn offset(&self, ty: &Type, field: &str, span: Span) -> Result<u64, Error> {
        let Type::Named(name) = ty else {
            return Err(Error::InvalidOperation {
                message: "offset_of requires a struct type".into(),
                span,
            });
        };
        let s = self.structs.get(name).ok_or(Error::Undefined {
            name: name.clone(),
            span,
        })?;
        let mut offset = 0;
        for item in &s.fields {
            let (_, a) = self.layout(&item.ty, span)?;
            offset = align_to(offset, a);
            if item.name == field {
                return Ok(offset);
            }
            offset += self.layout(&item.ty, span)?.0;
        }
        Err(Error::Undefined {
            name: field.into(),
            span,
        })
    }
}
fn integer_op(a: Value, b: Value, f: fn(i128, i128) -> i128, span: Span) -> Result<Value, Error> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(f(x, y))),
        _ => Err(Error::InvalidOperation {
            message: "integer operands required".into(),
            span,
        }),
    }
}
fn align_to(value: u64, alignment: u64) -> u64 {
    (value + alignment - 1) / alignment * alignment
}
fn value_span(_: Value) -> Span {
    Span::new(0, 0)
}
fn statement_span(statement: &Stmt) -> Span {
    match statement {
        Stmt::If { span, .. }
        | Stmt::While { span, .. }
        | Stmt::Break { span }
        | Stmt::Continue { span }
        | Stmt::Defer { span, .. }
        | Stmt::Return { span, .. }
        | Stmt::Variable(crate::ast::VariableDecl { span, .. })
        | Stmt::Assignment { span, .. }
        | Stmt::Expr { span, .. } => *span,
    }
}

/// Structured metadata exposed to compiler extensions without string parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub name: String,
    pub size: u64,
    pub alignment: u64,
    pub fields: Vec<FieldInfo>,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldInfo {
    pub name: String,
    pub ty: Type,
    pub offset: u64,
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionInfo {
    pub name: String,
    pub parameters: Vec<(String, Type)>,
    pub return_type: Type,
}

/// Reflect resolved source declarations. This is deliberately a data API: a
/// caller never has to parse formatted compiler output.
pub fn evaluate(program: &Program, expression: &Expr, pointer_width: u32) -> Result<Value, Error> {
    evaluate_with_limits(program, expression, pointer_width, Limits::default())
}

pub fn evaluate_with_limits(
    program: &Program,
    expression: &Expr,
    pointer_width: u32,
    limits: Limits,
) -> Result<Value, Error> {
    let specialized = specialize_program(program)?;
    let mut expander = Expander::new(&specialized, pointer_width, limits);
    expander.eval(expression, &HashMap::new())
}

pub fn reflect_type(program: &Program, name: &str, pointer_width: u32) -> Result<TypeInfo, Error> {
    let expander = Expander::new(program, pointer_width, Limits::default());
    let s = expander.structs.get(name).ok_or(Error::Undefined {
        name: name.into(),
        span: Span::new(0, 0),
    })?;
    let mut offset = 0;
    let mut fields = Vec::new();
    let mut alignment = 1;
    for field in &s.fields {
        let (size, align) = expander.layout(&field.ty, field.span)?;
        offset = align_to(offset, align);
        fields.push(FieldInfo {
            name: field.name.clone(),
            ty: field.ty.clone(),
            offset,
        });
        offset += size;
        alignment = alignment.max(align);
    }
    Ok(TypeInfo {
        name: name.into(),
        size: align_to(offset, alignment),
        alignment,
        fields,
    })
}

pub fn reflect_function(program: &Program, name: &str) -> Result<FunctionInfo, Error> {
    let function = program
        .declarations
        .iter()
        .find_map(|d| match d {
            Decl::Function(f) if f.name == name => Some(f),
            _ => None,
        })
        .ok_or(Error::Undefined {
            name: name.into(),
            span: Span::new(0, 0),
        })?;
    Ok(FunctionInfo {
        name: function.name.clone(),
        parameters: function
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect(),
        return_type: function.return_type.clone(),
    })
}

/// A structured declaration builder for compiler extensions. Declarations
/// produced here are ordinary AST declarations; callers must pass the result
/// through `pipeline::analyze_program` before lowering.
#[derive(Debug, Default, Clone)]
pub struct DeclarationGenerator {
    declarations: Vec<Decl>,
}
impl DeclarationGenerator {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn declarations(&self) -> &[Decl] {
        &self.declarations
    }
    pub fn add(&mut self, declaration: Decl) -> Result<(), Error> {
        let name = match &declaration {
            Decl::Function(f) => Some((&f.name, f.span)),
            Decl::Variable(v) => Some((&v.name, v.span)),
            Decl::Struct(s) => Some((&s.name, s.span)),
            Decl::Comptime { span, .. } => {
                return Err(Error::InvalidOperation {
                    message: "compile-time directives cannot be generated as runtime declarations"
                        .into(),
                    span: *span,
                });
            }
        };
        if let Some((name, span)) = name {
            if self.declarations.iter().any(|d| match d {
                Decl::Function(f) => f.name == *name,
                Decl::Variable(v) => v.name == *name,
                Decl::Struct(s) => s.name == *name,
                Decl::Comptime { .. } => false,
            }) {
                return Err(Error::InvalidOperation {
                    message: format!(
                        "generated declaration `{name}` conflicts with an existing declaration"
                    ),
                    span,
                });
            }
        }
        self.declarations.push(declaration);
        Ok(())
    }
    pub fn into_program(self) -> Program {
        Program {
            imports: Vec::new(),
            declarations: self.declarations,
        }
    }
    /// Insert generated declarations into a designated module. Duplicate
    /// names are rejected before the normal semantic pass.
    pub fn append_to(self, program: &Program) -> Result<Program, Error> {
        let mut result = program.clone();
        for declaration in self.declarations {
            let name =
                match &declaration {
                    Decl::Function(f) => (&f.name, f.span),
                    Decl::Variable(v) => (&v.name, v.span),
                    Decl::Struct(s) => (&s.name, s.span),
                    Decl::Comptime { span, .. } => return Err(Error::InvalidOperation {
                        message:
                            "compile-time directives cannot be generated as runtime declarations"
                                .into(),
                        span: *span,
                    }),
                };
            if result.declarations.iter().any(|existing| match existing {
                Decl::Function(f) => f.name == *name.0,
                Decl::Variable(v) => v.name == *name.0,
                Decl::Struct(s) => s.name == *name.0,
                Decl::Comptime { .. } => false,
            }) {
                return Err(Error::InvalidOperation {
                    message: format!(
                        "generated declaration `{}` conflicts with the destination module",
                        name.0
                    ),
                    span: name.1,
                });
            }
            result.declarations.push(declaration);
        }
        Ok(result)
    }
}
