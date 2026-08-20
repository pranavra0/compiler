//! Explicit compile-time evaluation and the small, deterministic compiler API.
//!
//! This module intentionally starts with an interpreter rather than executing
//! native code.  That gives compile-time calls the same source semantics while
//! keeping host file, process, network, allocator, and FFI access impossible.

use std::collections::HashMap;
use std::fmt;

use crate::ast::{BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, Type, UnaryOp};
use crate::lexer::Span;
use crate::typed::{DefId, InstantiationKey, InstantiationTable, TypeId, TypedExpr};

const DEFAULT_STEP_LIMIT: u64 = 100_000;
const DEFAULT_RECURSION_LIMIT: usize = 128;
const MAX_SPECIALIZATIONS: usize = 10_000;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Integer(i128),
    Float(f64),
    Bool(bool),
    String(String),
    TypeInfo(TypeMetadata),
    FunctionInfo(FunctionMetadata),
    ModuleInfo(ModuleMetadata),
    DeclarationInfo(DeclarationMetadata),
    FieldInfo(FieldMetadata),
    TypeRef(String),
    Array(Vec<Value>),
    Struct {
        name: String,
        fields: Vec<(String, Value)>,
    },
}

/// Structured values available to compile-time reflection code.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeMetadata {
    pub name: String,
    pub identity: String,
    pub size: u64,
    pub alignment: u64,
    pub fields: Vec<Value>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct FieldMetadata {
    pub name: String,
    pub ty: String,
    pub offset: u64,
}
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionMetadata {
    pub name: String,
    pub parameters: Vec<Value>,
    pub return_type: String,
}
#[derive(Debug, Clone, PartialEq)]
pub struct ModuleMetadata {
    pub declarations: Vec<Value>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct DeclarationMetadata {
    pub name: String,
    pub kind: String,
    pub exported: bool,
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
    /// Approximate interpreter memory units consumed by aggregate values.
    pub memory: u64,
    /// Maximum number of generated declarations.
    pub output: u64,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            steps: DEFAULT_STEP_LIMIT,
            recursion: DEFAULT_RECURSION_LIMIT,
            memory: 1_000_000,
            output: 10_000,
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
            Decl::Struct(structure) if structure.generic_params.is_empty() => {
                declarations.push(Decl::Struct(specializer.specialize_struct(structure)?));
            }
            Decl::Struct(_) => {}
            Decl::Variable(variable) => {
                let mut copy = variable.clone();
                if let Some(ty) = &copy.ty {
                    copy.ty = Some(specializer.specialize_type(ty, copy.span)?);
                }
                copy.value = specializer.specialize_expr(&copy.value, &HashMap::new())?;
                declarations.push(Decl::Variable(copy));
            }
            Decl::Comptime { expression, span } => declarations.push(Decl::Comptime {
                expression: specializer.specialize_expr(expression, &HashMap::new())?,
                span: *span,
            }),
        }
    }
    // Drain structural work only after the source walk has registered all
    // dependencies. The queue is the expansion boundary; linker spellings
    // are still derived below solely for the compatibility AST.
    let _completed_instantiations = specializer
        .instantiations
        .drain_pending()
        .collect::<Vec<_>>();
    let mut generated = specializer.generated.into_values().collect::<Vec<_>>();
    generated.sort_by(|a, b| a.name.cmp(&b.name));
    let mut generated_declarations = generated
        .into_iter()
        .map(Decl::Function)
        .collect::<Vec<_>>();
    let mut generated_structs = specializer
        .generated_structs
        .into_values()
        .collect::<Vec<_>>();
    generated_structs.sort_by(|a, b| a.name.cmp(&b.name));
    generated_declarations.extend(generated_structs.into_iter().map(Decl::Struct));
    // Preserve graph declaration IDs: source declarations stay in their
    // canonical order and generated declarations are appended as a new
    // work-queue result. They still re-enter ordinary name resolution.
    declarations.extend(generated_declarations);
    Ok(Program {
        imports: program.imports.clone(),
        declarations,
    })
}

struct Specializer<'a> {
    generic: HashMap<String, &'a FunctionDecl>,
    generic_structs: HashMap<String, &'a crate::ast::StructDecl>,
    generic_ids: HashMap<String, DefId>,
    generic_struct_ids: HashMap<String, DefId>,
    generated: HashMap<String, FunctionDecl>,
    generated_structs: HashMap<String, crate::ast::StructDecl>,
    /// Structural instantiation identities drive the work queue. Mangled
    /// names remain only as the source-level compatibility spelling.
    instantiations: InstantiationTable,
    type_ids: HashMap<Type, TypeId>,
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
            .collect::<HashMap<_, _>>();
        let generic_structs = program
            .declarations
            .iter()
            .filter_map(|d| match d {
                Decl::Struct(s) if !s.generic_params.is_empty() => Some((s.name.clone(), s)),
                _ => None,
            })
            .collect::<HashMap<_, _>>();
        let generic_ids = program
            .declarations
            .iter()
            .enumerate()
            .filter_map(|(index, declaration)| match declaration {
                Decl::Function(function) if !function.generic_params.is_empty() => {
                    Some((function.name.clone(), DefId(index as u32)))
                }
                _ => None,
            })
            .collect();
        let generic_struct_ids = program
            .declarations
            .iter()
            .enumerate()
            .filter_map(|(index, declaration)| match declaration {
                Decl::Struct(structure) if !structure.generic_params.is_empty() => {
                    Some((structure.name.clone(), DefId(index as u32)))
                }
                _ => None,
            })
            .collect();
        Self {
            generic,
            generic_structs,
            generic_ids,
            generic_struct_ids,
            generated: HashMap::new(),
            generated_structs: HashMap::new(),
            instantiations: InstantiationTable::default(),
            type_ids: HashMap::new(),
        }
    }

    fn type_id(&mut self, ty: &Type) -> TypeId {
        if let Some(id) = self.type_ids.get(ty) {
            return *id;
        }
        // Type IDs are local to this specialization session and are keyed by
        // the structural AST type itself. A linker spelling can never merge
        // distinct types (or split equivalent nested types).
        let id = TypeId(self.type_ids.len() as u32);
        self.type_ids.insert(ty.clone(), id);
        id
    }

    fn register_instantiation(
        &mut self,
        name: &str,
        substitutions: &HashMap<String, Type>,
        parameters: &[crate::ast::GenericParam],
        span: Span,
    ) -> Result<bool, Error> {
        let Some(definition) = self
            .generic_ids
            .get(name)
            .copied()
            .or_else(|| self.generic_struct_ids.get(name).copied())
        else {
            return Ok(true);
        };
        let type_arguments = parameters
            .iter()
            .filter_map(|parameter| substitutions.get(&parameter.name))
            .map(|ty| self.type_id(ty))
            .collect();
        let (_, fresh) = self.instantiations.intern(InstantiationKey {
            definition,
            type_arguments,
            value_arguments: Vec::new(),
        });
        if fresh && self.instantiations.len() > MAX_SPECIALIZATIONS {
            return Err(Error::InvalidOperation {
                message: "generic specialization limit exceeded".into(),
                span,
            });
        }
        Ok(fresh)
    }
    fn specialize_type(&mut self, ty: &Type, span: Span) -> Result<Type, Error> {
        match ty {
            Type::Named(name) => {
                self.ensure_struct_name(name, span)?;
                Ok(ty.clone())
            }
            Type::Pointer(inner) => Ok(Type::Pointer(Box::new(self.specialize_type(inner, span)?))),
            Type::Slice(inner) => Ok(Type::Slice(Box::new(self.specialize_type(inner, span)?))),
            Type::Array { length, element } => Ok(Type::Array {
                length: *length,
                element: Box::new(self.specialize_type(element, span)?),
            }),
            Type::Result { success, error } => Ok(Type::Result {
                success: Box::new(self.specialize_type(success, span)?),
                error: Box::new(self.specialize_type(error, span)?),
            }),
            Type::Unit => Ok(Type::Unit),
        }
    }
    fn specialize_struct(
        &mut self,
        structure: &crate::ast::StructDecl,
    ) -> Result<crate::ast::StructDecl, Error> {
        let mut copy = structure.clone();
        for field in &mut copy.fields {
            field.ty = self.specialize_type(&field.ty, field.span)?;
        }
        Ok(copy)
    }
    fn ensure_struct_name(&mut self, name: &str, span: Span) -> Result<(), Error> {
        let Some((base, suffix)) = name.split_once("__") else {
            return Ok(());
        };
        let Some(structure) = self.generic_structs.get(base).copied() else {
            return Ok(());
        };
        if self.generated_structs.contains_key(name) {
            return Ok(());
        }
        if self.generated_structs.len() >= MAX_SPECIALIZATIONS {
            return Err(Error::InvalidOperation {
                message: "generic specialization limit exceeded".into(),
                span,
            });
        }
        let parts = suffix.split("__").collect::<Vec<_>>();
        if parts.len() != structure.generic_params.len() {
            return Err(Error::InvalidOperation {
                message: format!(
                    "generic type `{base}` expects {} type arguments",
                    structure.generic_params.len()
                ),
                span,
            });
        }
        let mut substitutions = HashMap::new();
        for (parameter, part) in structure.generic_params.iter().zip(parts) {
            substitutions.insert(parameter.name.clone(), demangle_type(part));
        }
        if !self.register_instantiation(base, &substitutions, &structure.generic_params, span)? {
            return Ok(());
        }
        let mut specialized = structure.clone();
        specialized.name = name.into();
        specialized.generic_params.clear();
        for field in &mut specialized.fields {
            field.ty = substitute_type(&field.ty, &substitutions);
        }
        self.generated_structs.insert(name.into(), specialized);
        Ok(())
    }
    fn specialize_function(&mut self, function: &FunctionDecl) -> Result<FunctionDecl, Error> {
        let mut copy = function.clone();
        for parameter in &mut copy.params {
            parameter.ty = self.specialize_type(&parameter.ty, parameter.span)?;
        }
        copy.return_type = self.specialize_type(&copy.return_type, copy.span)?;
        let types = copy
            .params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        copy.body = self.specialize_block(&copy.body, &types)?;
        Ok(copy)
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
        if self.generated.len() >= MAX_SPECIALIZATIONS {
            return Err(Error::InvalidOperation {
                message: "generic specialization limit exceeded".into(),
                span,
            });
        }
        if !self.register_instantiation(name, &substitutions, &function.generic_params, span)? {
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
                    if let Some(ty) = &v.ty {
                        v.ty = Some(self.specialize_type(ty, v.span)?);
                    }
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
                    && self.generic.contains_key(name)
                {
                    let specialized = self.ensure(name, &arguments, *callee_span, env)?;
                    callee = Expr::Identifier {
                        name: specialized,
                        span: *callee_span,
                    };
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
            Expr::StructLiteral { name, fields, span } => {
                self.ensure_struct_name(name, *span)?;
                Ok(Expr::StructLiteral {
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
                    span: *span,
                })
            }
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
    if let Type::Named(name) = pattern
        && params.iter().any(|p| p.name == *name)
    {
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
fn demangle_type(name: &str) -> Type {
    match name {
        "unit" => Type::Unit,
        "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "f32"
        | "f64" | "bool" | "usize" | "isize" => Type::Named(name.into()),
        _ => Type::Named(name.into()),
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

enum ExecResult {
    Normal,
    Return(Value),
    Break,
    Continue,
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
    memory_used: u64,
    generated: Vec<Decl>,
    eval_cache: HashMap<String, Value>,
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
            memory_used: 0,
            generated: Vec::new(),
            eval_cache: HashMap::new(),
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
                    let value = self.eval(expression, &HashMap::new())?;
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
        for generated in self.generated {
            let generated_name =
                declaration_name(&generated).expect("generated runtime declaration");
            if self
                .program
                .declarations
                .iter()
                .any(|d| declaration_name(d) == Some(generated_name))
                || declarations
                    .iter()
                    .any(|d| declaration_name(d) == Some(generated_name))
            {
                return Err(Error::InvalidOperation {
                    message: format!(
                        "generated declaration `{generated_name}` conflicts with an existing declaration"
                    ),
                    span: declaration_span(&generated),
                });
            }
            declarations.push(generated);
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
    fn consume(&mut self, amount: u64, span: Span) -> Result<(), Error> {
        self.memory_used = self.memory_used.saturating_add(amount);
        if self.memory_used > self.limits.memory {
            Err(Error::InvalidOperation {
                message: "compile-time evaluation exceeded the memory limit".into(),
                span,
            })
        } else {
            Ok(())
        }
    }
    fn eval(&mut self, expression: &Expr, env: &HashMap<String, Value>) -> Result<Value, Error> {
        self.tick(expression.span())?;
        match expression {
            Expr::Integer { value, .. } => Ok(Value::Integer(*value)),
            Expr::Float { value, .. } => Ok(Value::Float(*value)),
            Expr::String { value, .. } => Ok(Value::String(value.clone())),
            Expr::Bool { value, .. } => Ok(Value::Bool(*value)),
            Expr::ArrayLiteral { elements, span, .. } => {
                self.consume(elements.len() as u64, *span)?;
                Ok(Value::Array(
                    elements
                        .iter()
                        .map(|e| self.eval(e, env))
                        .collect::<Result<_, _>>()?,
                ))
            }
            Expr::StructLiteral { name, fields, span } => {
                self.consume(fields.len() as u64, *span)?;
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: fields
                        .iter()
                        .map(|f| Ok((f.name.clone(), self.eval(&f.value, env)?)))
                        .collect::<Result<_, Error>>()?,
                })
            }
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
                if name == "reflect_type" {
                    if arguments.len() != 1 {
                        return Err(Error::InvalidOperation {
                            message: "reflect_type expects one type argument".into(),
                            span: *span,
                        });
                    }
                    return self.reflect_type_expr(&arguments[0], *span);
                }
                if name == "validate" || name == "compile_error" {
                    let args = arguments
                        .iter()
                        .map(|x| self.eval(x, env))
                        .collect::<Result<Vec<_>, _>>()?;
                    let valid = if name == "compile_error" {
                        false
                    } else {
                        matches!(args.first(), Some(Value::Bool(true)))
                    };
                    if !valid {
                        let message = args
                            .iter()
                            .find_map(|value| match value {
                                Value::String(message) => Some(message.clone()),
                                _ => None,
                            })
                            .unwrap_or_else(|| "compile-time validation failed".into());
                        return Err(Error::InvalidOperation {
                            message,
                            span: *span,
                        });
                    }
                    return Ok(Value::Unit);
                }
                if name == "length" {
                    if arguments.len() != 1 {
                        return Err(Error::InvalidOperation {
                            message: "length expects one argument".into(),
                            span: *span,
                        });
                    }
                    return match self.eval(&arguments[0], env)? {
                        Value::Array(values) => Ok(Value::Integer(values.len() as i128)),
                        Value::String(value) => Ok(Value::Integer(value.len() as i128)),
                        _ => Err(Error::InvalidOperation {
                            message: "length expects an array or string".into(),
                            span: *span,
                        }),
                    };
                }
                if name == "reflect_module" {
                    if !arguments.is_empty() {
                        return Err(Error::InvalidOperation {
                            message: "reflect_module expects no arguments".into(),
                            span: *span,
                        });
                    }
                    return Ok(Value::ModuleInfo(ModuleMetadata {
                        declarations: self.reflect_module(),
                    }));
                }
                if name == "reflect_function" {
                    if arguments.len() != 1 {
                        return Err(Error::InvalidOperation {
                            message: "reflect_function expects one function name".into(),
                            span: *span,
                        });
                    }
                    let Value::String(function_name) = self.eval(&arguments[0], env)? else {
                        return Err(Error::InvalidOperation {
                            message: "reflect_function expects a string name".into(),
                            span: *span,
                        });
                    };
                    return self.reflect_function_name(&function_name, *span);
                }
                if name == "generate_function" || name == "generate_constant" {
                    if arguments.len() != 2 {
                        return Err(Error::InvalidOperation {
                            message: format!("{name} expects a name and a value"),
                            span: *span,
                        });
                    }
                    let Value::String(generated_name) = self.eval(&arguments[0], env)? else {
                        return Err(Error::InvalidOperation {
                            message: "generated declaration name must be a string".into(),
                            span: *span,
                        });
                    };
                    let value = self.eval(&arguments[1], env)?;
                    self.generate_declaration(name, generated_name, value, *span)?;
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
                Value::TypeInfo(info) => match name.as_str() {
                    "name" => Ok(Value::String(info.name)),
                    "identity" => Ok(Value::String(info.identity)),
                    "size" => Ok(Value::Integer(info.size as i128)),
                    "alignment" => Ok(Value::Integer(info.alignment as i128)),
                    "fields" => Ok(Value::Array(info.fields)),
                    _ => Err(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                },
                Value::FieldInfo(info) => match name.as_str() {
                    "name" => Ok(Value::String(info.name)),
                    "type" => Ok(Value::TypeRef(info.ty)),
                    "offset" => Ok(Value::Integer(info.offset as i128)),
                    _ => Err(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                },
                Value::FunctionInfo(info) => match name.as_str() {
                    "name" => Ok(Value::String(info.name)),
                    "parameters" => Ok(Value::Array(info.parameters)),
                    "return_type" => Ok(Value::TypeRef(info.return_type)),
                    _ => Err(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                },
                Value::ModuleInfo(info) => match name.as_str() {
                    "declarations" => Ok(Value::Array(info.declarations)),
                    _ => Err(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                },
                Value::DeclarationInfo(info) => match name.as_str() {
                    "name" => Ok(Value::String(info.name)),
                    "kind" => Ok(Value::String(info.kind)),
                    "exported" => Ok(Value::Bool(info.exported)),
                    _ => Err(Error::Undefined {
                        name: name.clone(),
                        span: *span,
                    }),
                },
                _ => Err(Error::InvalidOperation {
                    message: "field access requires a compile-time struct or metadata value".into(),
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
            BinaryOp::Add => match (a, b) {
                (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{x}{y}"))),
                (a, b) => integer_op(a, b, |x: i128, y: i128| x.wrapping_add(y), span),
            },
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
    fn reflect_type_expr(&self, expression: &Expr, span: Span) -> Result<Value, Error> {
        let ty = self.type_from_expr(expression, span)?;
        let (size, alignment) = self.layout(&ty, span)?;
        let mut fields = Vec::new();
        if let Type::Named(name) = &ty
            && let Some(structure) = self.structs.get(name)
        {
            let mut offset = 0;
            for field in &structure.fields {
                let (field_size, field_alignment) = self.layout(&field.ty, field.span)?;
                offset = align_to(offset, field_alignment);
                fields.push(Value::FieldInfo(FieldMetadata {
                    name: field.name.clone(),
                    ty: type_display(&field.ty),
                    offset,
                }));
                offset += field_size;
            }
        }
        Ok(Value::TypeInfo(TypeMetadata {
            name: type_display(&ty),
            identity: format!("type:{}", type_display(&ty)),
            size,
            alignment,
            fields,
        }))
    }
    fn reflect_module(&self) -> Vec<Value> {
        self.program
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                Decl::Function(f) => Some(Value::DeclarationInfo(DeclarationMetadata {
                    name: f.name.clone(),
                    kind: "function".into(),
                    exported: f.exported,
                })),
                Decl::Variable(v) => Some(Value::DeclarationInfo(DeclarationMetadata {
                    name: v.name.clone(),
                    kind: "variable".into(),
                    exported: v.exported,
                })),
                Decl::Struct(s) => Some(Value::DeclarationInfo(DeclarationMetadata {
                    name: s.name.clone(),
                    kind: "struct".into(),
                    exported: s.exported,
                })),
                Decl::Comptime { .. } => None,
            })
            .collect()
    }
    fn reflect_function_name(&self, name: &str, span: Span) -> Result<Value, Error> {
        let Some(function) = self.functions.get(name) else {
            return Err(Error::Undefined {
                name: name.into(),
                span,
            });
        };
        let parameters = function
            .params
            .iter()
            .map(|parameter| {
                Value::FieldInfo(FieldMetadata {
                    name: parameter.name.clone(),
                    ty: type_display(&parameter.ty),
                    offset: 0,
                })
            })
            .collect();
        Ok(Value::FunctionInfo(FunctionMetadata {
            name: function.name.clone(),
            parameters,
            return_type: type_display(&function.return_type),
        }))
    }
    fn type_from_expr(&self, expression: &Expr, span: Span) -> Result<Type, Error> {
        match expression {
            Expr::Identifier { name, .. } => {
                if self.structs.contains_key(name) || is_builtin_type(name) {
                    Ok(Type::Named(name.clone()))
                } else {
                    Err(Error::Undefined {
                        name: name.clone(),
                        span,
                    })
                }
            }
            Expr::SizeOf { ty, .. } | Expr::AlignOf { ty, .. } => Ok(ty.clone()),
            _ => Err(Error::InvalidOperation {
                message: "expected a type name for compile-time reflection".into(),
                span,
            }),
        }
    }
    fn generate_declaration(
        &mut self,
        kind: &str,
        name: String,
        value: Value,
        span: Span,
    ) -> Result<(), Error> {
        if name.is_empty() || !is_identifier(&name) {
            return Err(Error::InvalidOperation {
                message: "generated declaration name is not a valid identifier".into(),
                span,
            });
        }
        if self.generated.len() as u64 >= self.limits.output {
            return Err(Error::InvalidOperation {
                message: "compile-time generated output exceeded the output limit".into(),
                span,
            });
        }
        let value_type = value_type(&value).ok_or(Error::InvalidOperation {
            message: "only ordinary scalar values can be generated as declarations".into(),
            span,
        })?;
        let value_expr = self.value_expr_with_hint(value, span, Some(&value_type))?;
        let candidate = if kind == "generate_constant" {
            Decl::Variable(crate::ast::VariableDecl {
                name,
                kind: crate::ast::VariableKind::Immutable,
                ty: Some(value_type),
                value: value_expr,
                span,
                exported: false,
            })
        } else {
            Decl::Function(FunctionDecl {
                name,
                generic_params: Vec::new(),
                params: Vec::new(),
                return_type: value_type,
                body: Block {
                    statements: vec![Stmt::Return {
                        value: Some(value_expr),
                        span,
                    }],
                    span,
                },
                span,
                is_extern: false,
                abi: None,
                link_name: None,
                exported: false,
            })
        };
        if let Some(existing) = self
            .generated
            .iter()
            .find(|d| declaration_name(d) == declaration_name(&candidate))
        {
            if existing == &candidate {
                return Ok(());
            }
            return Err(Error::InvalidOperation {
                message: format!(
                    "generated declaration `{}` conflicts with an existing declaration",
                    declaration_name(&candidate).unwrap_or("<unnamed>")
                ),
                span,
            });
        }
        self.generated.push(candidate);
        Ok(())
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
        let cache_key = format!("{name}:{args:?}");
        if let Some(value) = self.eval_cache.get(&cache_key) {
            return Ok(value.clone());
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
        match result? {
            ExecResult::Return(value) => {
                self.eval_cache.insert(cache_key, value.clone());
                Ok(value)
            }
            ExecResult::Normal => {
                self.eval_cache.insert(cache_key, Value::Unit);
                Ok(Value::Unit)
            }
            ExecResult::Break | ExecResult::Continue => Err(Error::InvalidOperation {
                message: "break or continue escaped a compile-time function".into(),
                span,
            }),
        }
    }
    fn exec_block(
        &mut self,
        block: &Block,
        env: &mut HashMap<String, Value>,
    ) -> Result<ExecResult, Error> {
        for statement in &block.statements {
            match statement {
                Stmt::Return { value, .. } => {
                    return Ok(ExecResult::Return(
                        value
                            .as_ref()
                            .map(|x| self.eval(x, env))
                            .transpose()?
                            .unwrap_or(Value::Unit),
                    ));
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
                    let result = if matches!(self.eval(condition, env)?, Value::Bool(true)) {
                        self.exec_block(then_branch, env)?
                    } else if let Some(branch) = else_branch {
                        self.exec_block(branch, env)?
                    } else {
                        ExecResult::Normal
                    };
                    if !matches!(result, ExecResult::Normal) {
                        return Ok(result);
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
                        match self.exec_block(body, env)? {
                            ExecResult::Normal | ExecResult::Continue => {}
                            ExecResult::Break => break,
                            result @ ExecResult::Return(_) => return Ok(result),
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
                            span: statement_span(statement),
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
                Stmt::Break { .. } => return Ok(ExecResult::Break),
                Stmt::Continue { .. } => return Ok(ExecResult::Continue),
                Stmt::Defer { span, .. } => {
                    return Err(Error::Unsupported {
                        message: "defer is runtime-only in compile-time functions".into(),
                        span: *span,
                    });
                }
            }
        }
        Ok(ExecResult::Normal)
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
            Value::String(_)
            | Value::TypeInfo(_)
            | Value::FunctionInfo(_)
            | Value::ModuleInfo(_)
            | Value::DeclarationInfo(_)
            | Value::FieldInfo(_)
            | Value::TypeRef(_) => {
                return Err(Error::InvalidOperation {
                    message: "metadata and strings cannot be emitted as runtime declarations"
                        .into(),
                    span,
                });
            }
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
fn is_builtin_type(name: &str) -> bool {
    matches!(name, "unit" | "bool" | "f32" | "f64" | "usize" | "isize")
        || (name.len() > 1
            && matches!(name.as_bytes()[0], b'i' | b'u')
            && name[1..].parse::<u16>().is_ok())
}
fn is_identifier(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'a'..=b'z' | b'A'..=b'Z' | b'_'))
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}
fn type_display(ty: &Type) -> String {
    match ty {
        Type::Unit => "unit".into(),
        Type::Named(name) => name.clone(),
        Type::Pointer(inner) => format!("*{}", type_display(inner)),
        Type::Slice(inner) => format!("[]{}", type_display(inner)),
        Type::Array { length, element } => format!("[{length}]{}", type_display(element)),
        Type::Result { success, error } => {
            format!("result({}, {})", type_display(success), type_display(error))
        }
    }
}
fn value_type(value: &Value) -> Option<Type> {
    match value {
        Value::Integer(_) => Some(Type::Named("i32".into())),
        Value::Float(_) => Some(Type::Named("f64".into())),
        Value::Bool(_) => Some(Type::Named("bool".into())),
        Value::Array(values) => Some(Type::Array {
            length: values.len() as u64,
            element: Box::new(
                values
                    .first()
                    .and_then(value_type)
                    .unwrap_or(Type::Named("i32".into())),
            ),
        }),
        Value::Struct { name, .. } => Some(Type::Named(name.clone())),
        _ => None,
    }
}
fn declaration_name(declaration: &Decl) -> Option<&str> {
    match declaration {
        Decl::Function(f) => Some(&f.name),
        Decl::Variable(v) => Some(&v.name),
        Decl::Struct(s) => Some(&s.name),
        Decl::Comptime { .. } => None,
    }
}
fn declaration_span(declaration: &Decl) -> Span {
    match declaration {
        Decl::Function(f) => f.span,
        Decl::Variable(v) => v.span,
        Decl::Struct(s) => s.span,
        Decl::Comptime { span, .. } => *span,
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
    value.div_ceil(alignment) * alignment
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
    pub identity: String,
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclarationInfo {
    pub name: String,
    pub kind: String,
    pub exported: bool,
}

pub fn reflect_module(program: &Program) -> Vec<DeclarationInfo> {
    program
        .declarations
        .iter()
        .filter_map(|declaration| match declaration {
            Decl::Function(f) => Some(DeclarationInfo {
                name: f.name.clone(),
                kind: "function".into(),
                exported: f.exported,
            }),
            Decl::Variable(v) => Some(DeclarationInfo {
                name: v.name.clone(),
                kind: "variable".into(),
                exported: v.exported,
            }),
            Decl::Struct(s) => Some(DeclarationInfo {
                name: s.name.clone(),
                kind: "struct".into(),
                exported: s.exported,
            }),
            Decl::Comptime { .. } => None,
        })
        .collect()
}

/// Reflect resolved source declarations. This is deliberately a data API: a
/// caller never has to parse formatted compiler output.
/// Evaluate an already-resolved pure expression through the same typed
/// evaluator used by constant-folding clients. Syntax evaluation remains
/// available below for declaration generation, where an AST value is needed.
pub fn evaluate_typed(expression: &TypedExpr) -> Result<Value, Error> {
    typed_value(crate::typed_eval::evaluate(expression), expression.span())
}

fn typed_value(
    value: Result<crate::typed_eval::Value, crate::typed_eval::Error>,
    span: Span,
) -> Result<Value, Error> {
    let value = value.map_err(|error| Error::InvalidOperation {
        message: error.to_string(),
        span,
    })?;
    Ok(match value {
        crate::typed_eval::Value::Unit => Value::Unit,
        crate::typed_eval::Value::Bool(value) => Value::Bool(value),
        crate::typed_eval::Value::Integer(value) => Value::Integer(value),
        crate::typed_eval::Value::Float(value) => Value::Float(value),
        crate::typed_eval::Value::Struct(fields) => Value::Array(
            fields
                .into_iter()
                .map(|field| typed_value(Ok(field), span))
                .collect::<Result<_, _>>()?,
        ),
        crate::typed_eval::Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(|value| typed_value(Ok(value), span))
                .collect::<Result<_, _>>()?,
        ),
        crate::typed_eval::Value::Result { error, value } => Value::Struct {
            name: if error { "result_err" } else { "result_ok" }.into(),
            fields: vec![("value".into(), typed_value(Ok(*value), span)?)],
        },
    })
}

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
    let specialized = specialize_program(program)?;
    let expander = Expander::new(&specialized, pointer_width, Limits::default());
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
        identity: format!("type:{name}"),
        size: align_to(offset, alignment),
        alignment,
        fields,
    })
}

pub fn reflect_function(program: &Program, name: &str) -> Result<FunctionInfo, Error> {
    let specialized = specialize_program(program)?;
    let function = specialized
        .declarations
        .iter()
        .find_map(|d| match d {
            Decl::Function(f) if f.name == name => Some(f),
            _ => None,
        })
        .or_else(|| {
            program.declarations.iter().find_map(|d| match d {
                Decl::Function(f) if f.name == name => Some(f),
                _ => None,
            })
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
        if let Some((name, span)) = name
            && self.declarations.iter().any(|d| match d {
                Decl::Function(f) => f.name == *name,
                Decl::Variable(v) => v.name == *name,
                Decl::Struct(s) => s.name == *name,
                Decl::Comptime { .. } => false,
            })
        {
            return Err(Error::InvalidOperation {
                message: format!(
                    "generated declaration `{name}` conflicts with an existing declaration"
                ),
                span,
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline;

    fn parse(source: &str) -> Program {
        pipeline::parse_source(source).expect("test source should parse")
    }

    #[test]
    fn evaluates_control_flow_and_break_continue() {
        let source = r#"
            count :: (limit: i32) -> i32 {
                value := 0;
                while (value < limit) {
                    value = value + 1;
                }
                return value;
            }
            answer :: #count(4);
            main :: () {}
        "#;
        let program = parse(source);
        let expanded = expand(&program, usize::BITS).unwrap();
        let variable = expanded
            .declarations
            .iter()
            .find_map(|d| match d {
                Decl::Variable(v) if v.name == "answer" => Some(v),
                _ => None,
            })
            .unwrap();
        assert!(matches!(variable.value, Expr::Integer { value: 4, .. }));
    }

    #[test]
    fn reflection_is_structured_and_generation_is_checked() {
        let source = r#"
            Pair :: struct { tag: u8; value: i32; }
            add :: (a: i32, b: i32) -> i32 { return a + b; }
            #generate_function("generated", reflect_type(Pair).size);
            main :: () -> i32 { return generated(); }
        "#;
        let program = parse(source);
        let info = reflect_type(&program, "Pair", usize::BITS).unwrap();
        assert_eq!(info.fields[1].name, "value");
        assert_eq!(info.fields[1].offset, 4);
        let function = reflect_function(&program, "add").unwrap();
        assert_eq!(function.parameters.len(), 2);
        let analyzed = pipeline::analyze_program(&program).unwrap();
        assert!(analyzed.functions.iter().any(|f| f.name == "generated"));
    }

    #[test]
    fn runtime_only_calls_and_limits_have_invocation_spans() {
        let source = r#"
            extern "c" puts(value: *u8) -> i32;
            answer :: #puts(null);
            main :: () {}
        "#;
        let program = parse(source);
        let error = expand(&program, usize::BITS).unwrap_err();
        assert!(matches!(error, Error::Unsupported { .. }));
        assert!(error.span().start > 0);

        let invalid = parse("#validate(false, \"bad declaration\");\nmain :: () {}");
        let error = expand(&invalid, usize::BITS).unwrap_err();
        assert!(matches!(error, Error::InvalidOperation { .. }));
        assert!(error.to_string().contains("bad declaration"));

        let recursive = parse("loop :: () { #loop(); }\n#loop();\nmain :: () {}");
        let error = expand_with_limits(
            &recursive,
            usize::BITS,
            Limits {
                recursion: 2,
                ..Limits::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, Error::RecursionLimit { .. }));
    }

    #[test]
    fn expansion_is_reproducible_and_generic_types_specialize() {
        let source = r#"
            Box :: struct(T: type) { value: T; }
            get :: (box: Box(i32)) -> i32 { return box.value; }
            choose :: (T: type, a: T, b: T) -> T { if a > b { return a; } return b; }
            first :: #choose(1, 2);
            second :: #choose(3, 4);
            main :: () -> i32 { x := Box__i32{value = 7}; return get(x) + choose(5, 6); }
        "#;
        let program = parse(source);
        assert_eq!(
            expand(&program, usize::BITS).unwrap(),
            expand(&program, usize::BITS).unwrap()
        );
        let expanded = expand(&program, usize::BITS).unwrap();
        assert!(
            expanded
                .declarations
                .iter()
                .any(|d| matches!(d, Decl::Struct(s) if s.name == "Box__i32"))
        );
        assert_eq!(
            expanded
                .declarations
                .iter()
                .filter(|d| matches!(d, Decl::Function(f) if f.name == "choose__i32"))
                .count(),
            1
        );
    }
}
