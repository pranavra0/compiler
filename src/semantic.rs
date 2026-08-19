use std::collections::HashMap;
use std::fmt;

use crate::ast::{
    BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, StructDecl, Type, UnaryOp,
    VariableDecl, VariableKind,
};
use crate::lexer::Span;
use crate::typed::{
    FunctionId, IntegerWidth, LayoutKind, LocalId, ResolvedType, TypedBlock, TypedConstant,
    TypedExpr, TypedField, TypedFunction, TypedGlobal, TypedParameter, TypedPlace, TypedProgram,
    TypedStmt, TypedStruct,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticError {
    UndefinedName {
        name: String,
        span: Span,
    },
    DuplicateName {
        name: String,
        span: Span,
    },
    UnknownType {
        name: String,
        span: Span,
    },
    TypeMismatch {
        expected: Type,
        found: Type,
        span: Span,
    },
    InvalidLiteral {
        message: String,
        span: Span,
    },
    InvalidOperand {
        message: String,
        span: Span,
    },
    WrongArgumentCount {
        name: String,
        expected: usize,
        found: usize,
        span: Span,
    },
    NotCallable {
        name: String,
        span: Span,
    },
    ImmutableAssignment {
        name: String,
        span: Span,
    },
    InvalidAssignmentTarget {
        span: Span,
    },
    BreakOutsideLoop {
        span: Span,
    },
    ContinueOutsideLoop {
        span: Span,
    },
    InvalidPropagation {
        message: String,
        span: Span,
    },
    InvalidDefer {
        message: String,
        span: Span,
    },
    MissingReturn {
        function: String,
        span: Span,
    },
    TopLevelVariableUnsupported {
        name: String,
        span: Span,
    },
    InvalidEntryPoint {
        message: String,
        span: Span,
    },
    InvalidAbi {
        abi: String,
        span: Span,
    },
    InvalidFfiType {
        message: String,
        span: Span,
    },
}
impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UndefinedName { name, .. } => format!("undefined name `{name}`"),
            Self::DuplicateName { name, .. } => format!("duplicate name `{name}`"),
            Self::UnknownType { name, .. } => format!("unknown type `{name}`"),
            Self::TypeMismatch {
                expected, found, ..
            } => format!(
                "type mismatch: expected {}, found {}",
                type_name(expected),
                type_name(found)
            ),
            Self::InvalidLiteral { message, .. } | Self::InvalidOperand { message, .. } => {
                message.clone()
            }
            Self::WrongArgumentCount {
                name,
                expected,
                found,
                ..
            } => format!("function `{name}` expects {expected} arguments, got {found}"),
            Self::NotCallable { name, .. } => format!("`{name}` is not a function"),
            Self::ImmutableAssignment { name, .. } => {
                format!("cannot assign to immutable variable `{name}`")
            }
            Self::InvalidAssignmentTarget { .. } => "invalid assignment target".into(),
            Self::BreakOutsideLoop { .. } => "break is only valid inside a loop".into(),
            Self::ContinueOutsideLoop { .. } => "continue is only valid inside a loop".into(),
            Self::InvalidPropagation { message, .. } | Self::InvalidDefer { message, .. } => {
                message.clone()
            }
            Self::MissingReturn { function, .. } => {
                format!("function `{function}` does not return a value on every path")
            }
            Self::TopLevelVariableUnsupported { name, .. } => {
                format!("top-level variable `{name}` is not supported yet")
            }
            Self::InvalidEntryPoint { message, .. } => message.clone(),
            Self::InvalidAbi { abi, .. } => {
                format!("unsupported ABI `{abi}` (only `c` is supported)")
            }
            Self::InvalidFfiType { message, .. } => message.clone(),
        };
        let span = self.span();
        write!(f, "{text} at {}..{}", span.start, span.end)
    }
}
impl std::error::Error for SemanticError {}
impl SemanticError {
    pub fn span(&self) -> Span {
        match self {
            Self::UndefinedName { span, .. }
            | Self::DuplicateName { span, .. }
            | Self::UnknownType { span, .. }
            | Self::TypeMismatch { span, .. }
            | Self::InvalidLiteral { span, .. }
            | Self::InvalidOperand { span, .. }
            | Self::WrongArgumentCount { span, .. }
            | Self::NotCallable { span, .. }
            | Self::ImmutableAssignment { span, .. }
            | Self::InvalidAssignmentTarget { span }
            | Self::BreakOutsideLoop { span }
            | Self::ContinueOutsideLoop { span }
            | Self::InvalidPropagation { span, .. }
            | Self::InvalidDefer { span, .. }
            | Self::MissingReturn { span, .. }
            | Self::TopLevelVariableUnsupported { span, .. }
            | Self::InvalidEntryPoint { span, .. }
            | Self::InvalidAbi { span, .. }
            | Self::InvalidFfiType { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone)]
struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
}
#[derive(Debug, Clone)]
struct Variable {
    ty: Type,
    mutable: bool,
    constant_value: Option<Expr>,
}
#[derive(Debug, Clone)]
struct StructInfo {
    fields: Vec<(String, Type)>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Flow(u8);
impl Flow {
    const NORMAL: Self = Self(1);
    const RETURN: Self = Self(2);
    const BREAK: Self = Self(4);
    const CONTINUE: Self = Self(8);
    fn contains(self, x: Self) -> bool {
        self.0 & x.0 != 0
    }
    fn union(self, x: Self) -> Self {
        Self(self.0 | x.0)
    }
    fn without(self, x: Self) -> Self {
        Self(self.0 & !x.0)
    }
}

pub fn analyze(program: &Program) -> Result<(), SemanticError> {
    Analyzer::new().analyze(program)
}
pub fn analyze_typed(program: &Program) -> Result<TypedProgram, SemanticError> {
    Analyzer::new().analyze_typed(program)
}
pub fn analyze_typed_with_pointer_width(
    program: &Program,
    pointer_width: u32,
) -> Result<TypedProgram, SemanticError> {
    Analyzer::with_pointer_width(pointer_width).analyze_typed(program)
}

pub fn validate_entry_point(program: &Program) -> Result<(), SemanticError> {
    let mut main = None;
    for declaration in &program.declarations {
        let Some((name, span)) = (match declaration {
            Decl::Function(x) => Some((&x.name, x.span)),
            Decl::Variable(x) => Some((&x.name, x.span)),
            Decl::Struct(x) => Some((&x.name, x.span)),
            Decl::Comptime { .. } => None,
        }) else {
            continue;
        };
        if name != "main" {
            continue;
        }
        if main.is_some() {
            return Err(SemanticError::InvalidEntryPoint {
                message: "duplicate `main` declarations".into(),
                span,
            });
        }
        main = Some((declaration, span));
    }
    let Some((declaration, span)) = main else {
        return Err(SemanticError::InvalidEntryPoint {
            message: "native build requires exactly one `main` function".into(),
            span: Span::new(0, 0),
        });
    };
    let Decl::Function(function) = declaration else {
        return Err(SemanticError::InvalidEntryPoint {
            message: "`main` must be a function".into(),
            span,
        });
    };
    if !function.params.is_empty() {
        return Err(SemanticError::InvalidEntryPoint {
            message: "`main` must not have parameters".into(),
            span: function.span,
        });
    }
    if function.return_type != Type::Named("i32".into()) {
        return Err(SemanticError::InvalidEntryPoint {
            message: "`main` must return i32".into(),
            span: function.span,
        });
    }
    Ok(())
}

pub struct Analyzer {
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, StructInfo>,
    globals: HashMap<String, Variable>,
    constants: HashMap<String, Variable>,
    scopes: Vec<HashMap<String, Variable>>,
    current_return_type: Option<Type>,
    current_function: Option<String>,
    loop_depth: usize,
    pointer_width: u32,
}
impl Analyzer {
    pub fn new() -> Self {
        Self::with_pointer_width(usize::BITS)
    }
    pub fn with_pointer_width(pointer_width: u32) -> Self {
        Self {
            functions: HashMap::new(),
            structs: HashMap::new(),
            globals: HashMap::new(),
            constants: HashMap::new(),
            scopes: Vec::new(),
            current_return_type: None,
            current_function: None,
            loop_depth: 0,
            pointer_width,
        }
    }
    pub fn analyze(mut self, program: &Program) -> Result<(), SemanticError> {
        self.collect_types(program)?;
        self.collect_values(program)?;
        for d in &program.declarations {
            if let Decl::Function(f) = d {
                if !f.is_extern {
                    self.analyze_function(f)?
                }
            }
        }
        Ok(())
    }
    pub fn analyze_typed(self, program: &Program) -> Result<TypedProgram, SemanticError> {
        let pointer_width = self.pointer_width;
        self.analyze(program)?;
        TypedLowerer::new_with_pointer_width(program, pointer_width).lower()
    }

    fn collect_types(&mut self, p: &Program) -> Result<(), SemanticError> {
        for d in &p.declarations {
            if let Decl::Struct(s) = d {
                if self
                    .structs
                    .insert(s.name.clone(), StructInfo { fields: Vec::new() })
                    .is_some()
                {
                    return Err(SemanticError::DuplicateName {
                        name: s.name.clone(),
                        span: s.span,
                    });
                }
            }
        }
        for d in &p.declarations {
            if let Decl::Struct(s) = d {
                let mut names = HashMap::new();
                let mut fields = Vec::new();
                for field in &s.fields {
                    if names.insert(field.name.clone(), field.span).is_some() {
                        return Err(SemanticError::DuplicateName {
                            name: field.name.clone(),
                            span: field.span,
                        });
                    }
                    self.validate_value_type(&field.ty, field.span)?;
                    fields.push((field.name.clone(), field.ty.clone()));
                }
                self.structs.insert(s.name.clone(), StructInfo { fields });
            }
        }
        Ok(())
    }
    fn collect_values(&mut self, p: &Program) -> Result<(), SemanticError> {
        let mut names = HashMap::<String, Span>::new();
        for d in &p.declarations {
            match d {
                Decl::Function(f) => {
                    if f.exported && f.abi.as_deref() != Some("c") {
                        return Err(SemanticError::InvalidAbi {
                            abi: "missing `c`".into(),
                            span: f.span,
                        });
                    }
                    if let Some(abi) = &f.abi {
                        if abi != "c" {
                            return Err(SemanticError::InvalidAbi {
                                abi: abi.clone(),
                                span: f.span,
                            });
                        }
                        for parameter in &f.params {
                            self.validate_ffi_type(&parameter.ty, parameter.span)?;
                        }
                        self.validate_ffi_type(&f.return_type, f.span)?;
                    }
                    if names.insert(f.name.clone(), f.span).is_some() {
                        return Err(SemanticError::DuplicateName {
                            name: f.name.clone(),
                            span: f.span,
                        });
                    }
                    for x in &f.params {
                        self.validate_value_type(&x.ty, x.span)?
                    }
                    self.validate_return_type(&f.return_type, f.span)?;
                    self.functions.insert(
                        f.name.clone(),
                        FunctionSignature {
                            parameters: f.params.iter().map(|x| x.ty.clone()).collect(),
                            return_type: f.return_type.clone(),
                        },
                    );
                }
                Decl::Variable(v) => {
                    if names.insert(v.name.clone(), v.span).is_some() {
                        return Err(SemanticError::DuplicateName {
                            name: v.name.clone(),
                            span: v.span,
                        });
                    }
                    if matches!(v.kind, VariableKind::MutableInferred) {
                        return Err(SemanticError::TopLevelVariableUnsupported {
                            name: v.name.clone(),
                            span: v.span,
                        });
                    }
                    let ty = if let Some(t) = &v.ty {
                        self.validate_value_type(t, v.span)?;
                        self.check_expression(&v.value, Some(t))?
                    } else {
                        self.check_expression(&v.value, None)?
                    };
                    if let Some(t) = &v.ty {
                        self.expect_type(t, &ty, v.value.span())?
                    }
                    if !self.is_compile_time(&v.value) {
                        return Err(SemanticError::InvalidOperand {
                            message: "global initializer must be a compile-time constant".into(),
                            span: v.value.span(),
                        });
                    }
                    let mutable = matches!(v.kind, VariableKind::MutableTyped);
                    let var = Variable {
                        ty: ty.clone(),
                        mutable,
                        constant_value: (!mutable).then(|| v.value.clone()),
                    };
                    self.globals.insert(v.name.clone(), var.clone());
                    if !var.mutable {
                        self.constants.insert(v.name.clone(), var);
                    }
                }
                Decl::Struct(_) | Decl::Comptime { .. } => {}
            }
        }
        Ok(())
    }
    fn analyze_function(&mut self, f: &FunctionDecl) -> Result<(), SemanticError> {
        self.current_function = Some(f.name.clone());
        self.current_return_type = Some(f.return_type.clone());
        self.scopes.push(HashMap::new());
        for p in &f.params {
            self.declare_local(
                &p.name,
                Variable {
                    ty: p.ty.clone(),
                    mutable: true,
                    constant_value: None,
                },
                p.span,
            )?
        }
        self.loop_depth = 0;
        let flow = self.analyze_block_contents(&f.body)?;
        self.scopes.pop();
        self.current_function = None;
        self.current_return_type = None;
        if !is_unit(&f.return_type) && flow.contains(Flow::NORMAL) {
            return Err(SemanticError::MissingReturn {
                function: f.name.clone(),
                span: f.body.span,
            });
        }
        Ok(())
    }
    fn analyze_block(&mut self, b: &Block) -> Result<Flow, SemanticError> {
        self.scopes.push(HashMap::new());
        let x = self.analyze_block_contents(b);
        self.scopes.pop();
        x
    }
    fn analyze_block_contents(&mut self, b: &Block) -> Result<Flow, SemanticError> {
        let mut flow = Flow::NORMAL;
        for s in &b.statements {
            let sf = self.analyze_statement(s)?;
            if flow.contains(Flow::NORMAL) {
                flow = flow.without(Flow::NORMAL).union(sf)
            }
        }
        Ok(flow)
    }
    fn analyze_statement(&mut self, s: &Stmt) -> Result<Flow, SemanticError> {
        match s {
            Stmt::Variable(v) => {
                let ty = self.variable_type(v)?;
                self.declare_local(
                    &v.name,
                    Variable {
                        ty,
                        mutable: !matches!(v.kind, VariableKind::Immutable),
                        constant_value: if matches!(v.kind, VariableKind::Immutable) {
                            Some(v.value.clone())
                        } else {
                            None
                        },
                    },
                    v.span,
                )?;
                Ok(if expression_propagates(&v.value) {
                    Flow::RETURN
                } else {
                    Flow::NORMAL
                })
            }
            Stmt::Assignment { target, value, .. } => {
                let (ty, mutable) = self.check_place(target)?;
                if !mutable {
                    return Err(SemanticError::ImmutableAssignment {
                        name: place_name(target),
                        span: target.span(),
                    });
                }
                let actual = self.check_expression(value, Some(&ty))?;
                self.expect_type(&ty, &actual, value.span())?;
                Ok(if expression_propagates(value) {
                    Flow::RETURN
                } else {
                    Flow::NORMAL
                })
            }
            Stmt::Expr { expression, .. } => {
                self.check_expression(expression, None)?;
                Ok(if expression_propagates(expression) {
                    Flow::RETURN
                } else {
                    Flow::NORMAL
                })
            }
            Stmt::Defer { call, span } => {
                if !matches!(call, Expr::Call { .. }) {
                    return Err(SemanticError::InvalidDefer {
                        message: "defer requires a function call".into(),
                        span: *span,
                    });
                }
                self.check_expression(call, None)?;
                Ok(Flow::NORMAL)
            }
            Stmt::Break { span } => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakOutsideLoop { span: *span });
                }
                Ok(Flow::BREAK)
            }
            Stmt::Continue { span } => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::ContinueOutsideLoop { span: *span });
                }
                Ok(Flow::CONTINUE)
            }
            Stmt::Return { value, span } => {
                let expected = self.current_return_type.clone().unwrap();
                match (is_unit(&expected), value) {
                    (true, None) => Ok(Flow::RETURN),
                    (true, Some(e)) => Err(SemanticError::TypeMismatch {
                        expected: Type::Unit,
                        found: self.check_expression(e, None)?,
                        span: *span,
                    }),
                    (false, None) => Err(SemanticError::TypeMismatch {
                        expected: expected.clone(),
                        found: Type::Unit,
                        span: *span,
                    }),
                    (false, Some(e)) => {
                        let a = self.check_expression(e, Some(&expected))?;
                        self.expect_type(&expected, &a, e.span())?;
                        Ok(Flow::RETURN)
                    }
                }
            }
            Stmt::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let a = self.check_expression(condition, Some(&named("bool")))?;
                self.expect_type(&named("bool"), &a, condition.span())?;
                let t = self.analyze_block(then_branch)?;
                let e = else_branch
                    .as_ref()
                    .map(|b| self.analyze_block(b))
                    .transpose()?
                    .unwrap_or(Flow::NORMAL);
                Ok(t.union(e))
            }
            Stmt::While {
                condition, body, ..
            } => {
                let a = self.check_expression(condition, Some(&named("bool")))?;
                self.expect_type(&named("bool"), &a, condition.span())?;
                let always = matches!(condition, Expr::Bool { value: true, .. });
                self.loop_depth += 1;
                let bf = self.analyze_block(body)?;
                self.loop_depth -= 1;
                let mut x = if bf.contains(Flow::RETURN) {
                    Flow::RETURN
                } else {
                    Flow(0)
                };
                if !always || bf.contains(Flow::BREAK) {
                    x = x.union(Flow::NORMAL)
                }
                Ok(x)
            }
        }
    }
    fn variable_type(&mut self, v: &VariableDecl) -> Result<Type, SemanticError> {
        if let Some(t) = &v.ty {
            self.validate_value_type(t, v.span)?
        }
        let ty = self.check_expression(&v.value, v.ty.as_ref())?;
        if let Some(t) = &v.ty {
            self.expect_type(t, &ty, v.value.span())?
        }
        Ok(ty)
    }

    fn check_expression(
        &mut self,
        e: &Expr,
        expected: Option<&Type>,
    ) -> Result<Type, SemanticError> {
        let ty = match e {
            Expr::Comptime { span, .. } => {
                return Err(SemanticError::InvalidOperand {
                    message: "compile-time marker was not expanded before semantic analysis".into(),
                    span: *span,
                });
            }
            Expr::Integer { value, span } => {
                let t = expected
                    .filter(|x| is_integer(x))
                    .cloned()
                    .unwrap_or_else(|| named("i32"));
                if !integer_fits_with_width(*value, &t, self.pointer_width) {
                    return Err(SemanticError::InvalidLiteral {
                        message: format!("integer literal does not fit in {}", type_name(&t)),
                        span: *span,
                    });
                }
                t
            }
            Expr::Float { value, span } => {
                let t = expected
                    .filter(|x| is_float(x))
                    .cloned()
                    .unwrap_or_else(|| named("f64"));
                if type_name(&t) == "f32" && !(*value as f32).is_finite() {
                    return Err(SemanticError::InvalidLiteral {
                        message: "floating-point literal does not fit in f32".into(),
                        span: *span,
                    });
                }
                t
            }
            Expr::Bool { .. } => named("bool"),
            Expr::String { span, .. } => {
                return Err(SemanticError::InvalidOperand {
                    message: "strings are only available in compile-time context".into(),
                    span: *span,
                });
            }
            Expr::Propagate { expression, span } => {
                let actual = self.check_expression(expression, None)?;
                let Type::Result { success, error } = actual else {
                    return Err(SemanticError::InvalidPropagation {
                        message: "`?` requires a result value".into(),
                        span: *span,
                    });
                };
                let Some(Type::Result {
                    error: current_error,
                    ..
                }) = self.current_return_type.as_ref()
                else {
                    return Err(SemanticError::InvalidPropagation {
                        message: "`?` requires a result-returning function".into(),
                        span: *span,
                    });
                };
                if **current_error != *error {
                    return Err(SemanticError::InvalidPropagation {
                        message:
                            "propagated error type is incompatible with the current return type"
                                .into(),
                        span: *span,
                    });
                }
                *success
            }
            Expr::Null { span } => match expected {
                Some(Type::Pointer(_)) => expected.cloned().unwrap(),
                _ => {
                    return Err(SemanticError::InvalidOperand {
                        message: "null requires a pointer type context".into(),
                        span: *span,
                    });
                }
            },
            Expr::SizeOf { ty, span } => {
                self.validate_value_type(ty, *span)?;
                named("usize")
            }
            Expr::AlignOf { ty, span } => {
                self.validate_value_type(ty, *span)?;
                named("usize")
            }
            Expr::OffsetOf { ty, field, span } => {
                self.validate_value_type(ty, *span)?;
                let Type::Named(name) = ty else {
                    return Err(SemanticError::InvalidOperand {
                        message: "offset_of requires a struct type".into(),
                        span: *span,
                    });
                };
                let Some(info) = self.structs.get(name) else {
                    return Err(SemanticError::UnknownType {
                        name: name.clone(),
                        span: *span,
                    });
                };
                if !info.fields.iter().any(|(n, _)| n == field) {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("unknown field `{field}`"),
                        span: *span,
                    });
                }
                named("usize")
            }
            Expr::Identifier { name, span } => {
                if let Some(v) = self.lookup_variable(name) {
                    v.ty
                } else if let Some(v) = self.globals.get(name) {
                    v.ty.clone()
                } else if self.functions.contains_key(name) {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("function `{name}` cannot be used as a value"),
                        span: *span,
                    });
                } else {
                    return Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *span,
                    });
                }
            }
            Expr::StructLiteral { name, fields, span } => {
                let info =
                    self.structs
                        .get(name)
                        .cloned()
                        .ok_or_else(|| SemanticError::UnknownType {
                            name: name.clone(),
                            span: *span,
                        })?;
                let t = named(name);
                if let Some(exp) = expected {
                    self.expect_type(exp, &t, *span)?
                }
                let mut seen = HashMap::new();
                for f in fields {
                    if seen.insert(f.name.clone(), ()).is_some() {
                        return Err(SemanticError::DuplicateName {
                            name: f.name.clone(),
                            span: f.span,
                        });
                    }
                    let Some((_, ft)) = info.fields.iter().find(|(n, _)| n == &f.name) else {
                        return Err(SemanticError::InvalidOperand {
                            message: format!("unknown field `{}`", f.name),
                            span: f.span,
                        });
                    };
                    let a = self.check_expression(&f.value, Some(ft))?;
                    self.expect_type(ft, &a, f.value.span())?
                }
                if fields.len() != info.fields.len() {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("struct `{name}` literal is missing a field"),
                        span: *span,
                    });
                }
                t
            }
            Expr::ArrayLiteral {
                ty: at,
                elements,
                span,
            } => {
                self.validate_value_type(at, *span)?;
                let t = at.clone();
                if let Some(exp) = expected {
                    self.expect_type(exp, &t, *span)?
                }
                let Type::Array { length, element } = at else {
                    unreachable!()
                };
                if elements.len() as u64 != *length {
                    return Err(SemanticError::InvalidOperand {
                        message: format!(
                            "array literal expects {length} elements, got {}",
                            elements.len()
                        ),
                        span: *span,
                    });
                }
                for x in elements {
                    let a = self.check_expression(x, Some(element))?;
                    self.expect_type(element, &a, x.span())?
                }
                t
            }
            Expr::Field { base, name, span } => {
                let bt = self.check_expression(base, None)?;
                let Type::Named(s) = &bt else {
                    return Err(SemanticError::InvalidOperand {
                        message: "field access requires a struct".into(),
                        span: *span,
                    });
                };
                let Some(info) = self.structs.get(s) else {
                    unreachable!()
                };
                let Some((_, t)) = info.fields.iter().find(|(n, _)| n == name) else {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("unknown field `{name}`"),
                        span: *span,
                    });
                };
                t.clone()
            }
            Expr::Index { base, index, span } | Expr::UncheckedIndex { base, index, span } => {
                let bt = self.check_expression(base, None)?;
                let it = self.check_expression(index, Some(&named("usize")))?;
                self.expect_type(&named("usize"), &it, index.span())?;
                match bt {
                    Type::Array { length, element } => {
                        if matches!(e, Expr::Index { .. }) {
                            if let Some(i) = self.constant_integer(index) {
                                if i < 0 || i as u64 >= length {
                                    return Err(SemanticError::InvalidLiteral {
                                        message: "array index is out of bounds".into(),
                                        span: index.span(),
                                    });
                                }
                            }
                        }
                        *element
                    }
                    Type::Slice(element) => *element,
                    _ => {
                        return Err(SemanticError::InvalidOperand {
                            message: "indexing requires an array or slice".into(),
                            span: *span,
                        });
                    }
                }
            }
            Expr::Unary {
                operator,
                operand,
                span,
            } => {
                if *operator == UnaryOp::AddressOf {
                    let (ty, mutable) = self.check_place(operand)?;
                    if !mutable {
                        return Err(SemanticError::InvalidOperand {
                            message: "address-of requires a mutable lvalue".into(),
                            span: operand.span(),
                        });
                    }
                    let result = Type::Pointer(Box::new(ty));
                    if let Some(expected) = expected {
                        self.expect_type(expected, &result, *span)?;
                    }
                    return Ok(result);
                }
                if *operator == UnaryOp::Dereference {
                    let a = self.check_expression(operand, None)?;
                    if let Type::Pointer(element) = a {
                        let result = *element;
                        if let Some(expected) = expected {
                            self.expect_type(expected, &result, *span)?;
                        }
                        return Ok(result);
                    }
                    return Err(SemanticError::InvalidOperand {
                        message: "dereference requires a pointer".into(),
                        span: *span,
                    });
                }
                let oe = match operator {
                    UnaryOp::Not => Some(named("bool")),
                    _ => expected.filter(|t| is_numeric(t)).cloned(),
                };
                let a = self.check_expression(operand, oe.as_ref())?;
                match operator {
                    UnaryOp::Negate if !is_signed_integer(&a) && !is_float(&a) => {
                        return Err(SemanticError::InvalidOperand {
                            message: "negation requires a signed integer or floating-point operand"
                                .into(),
                            span: *span,
                        });
                    }
                    UnaryOp::Not if !is_bool(&a) => {
                        return Err(SemanticError::InvalidOperand {
                            message: "logical not requires a boolean operand".into(),
                            span: *span,
                        });
                    }
                    UnaryOp::BitwiseNot if !is_integer(&a) => {
                        return Err(SemanticError::InvalidOperand {
                            message: "bitwise not requires an integer operand".into(),
                            span: *span,
                        });
                    }
                    _ => {}
                }
                a
            }
            Expr::Binary {
                left,
                operator,
                right,
                span,
            } => self.check_binary(left, *operator, right, *span, expected)?,
            Expr::Call {
                callee,
                arguments,
                span,
            } => {
                let Expr::Identifier { name, span: cs } = callee.as_ref() else {
                    return Err(SemanticError::InvalidOperand {
                        message: "only named functions can be called currently".into(),
                        span: callee.span(),
                    });
                };
                if name == "return_ok"
                    || name == "return_err"
                    || name == "is_err"
                    || name == "unwrap"
                {
                    let void_ok = name == "return_ok"
                        && matches!(expected, Some(Type::Result { success, .. }) if is_unit(success));
                    let expected_args = usize::from(!void_ok);
                    if arguments.len() != expected_args {
                        return Err(SemanticError::WrongArgumentCount {
                            name: name.clone(),
                            expected: expected_args,
                            found: arguments.len(),
                            span: *span,
                        });
                    }
                    let arg = if void_ok {
                        None
                    } else {
                        Some(self.check_expression(&arguments[0], None)?)
                    };
                    match name.as_str() {
                        "is_err" => {
                            if !matches!(arg.as_ref(), Some(Type::Result { .. })) {
                                return Err(SemanticError::InvalidOperand {
                                    message: "is_err requires a result value".into(),
                                    span: *span,
                                });
                            }
                            return Ok(named("bool"));
                        }
                        "unwrap" => {
                            let Some(Type::Result { success, .. }) = arg else {
                                return Err(SemanticError::InvalidOperand {
                                    message: "unwrap requires a result value".into(),
                                    span: *span,
                                });
                            };
                            return Ok(*success);
                        }
                        "return_ok" | "return_err" => {
                            let Some(Type::Result { success, error }) = expected else {
                                return Err(SemanticError::InvalidOperand {
                                    message: format!("{name} requires a result return context"),
                                    span: *span,
                                });
                            };
                            let wanted = if name == "return_ok" { success } else { error };
                            if let Some(argument) = arguments.first() {
                                let actual = self.check_expression(argument, Some(wanted))?;
                                self.expect_type(wanted, &actual, argument.span())?;
                            }
                            return Ok(Type::Result {
                                success: Box::new(*success.clone()),
                                error: Box::new(*error.clone()),
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                if name == "make_slice" {
                    if arguments.len() != 2 {
                        return Err(SemanticError::WrongArgumentCount {
                            name: name.clone(),
                            expected: 2,
                            found: arguments.len(),
                            span: *span,
                        });
                    }
                    let default_pointer = Type::Pointer(Box::new(named("u8")));
                    let pointer = if matches!(&arguments[0], Expr::Null { .. }) {
                        self.check_expression(&arguments[0], Some(&default_pointer))?
                    } else {
                        self.check_expression(&arguments[0], None)?
                    };
                    let Type::Pointer(element) = pointer else {
                        return Err(SemanticError::InvalidOperand {
                            message: "make_slice requires a typed pointer".into(),
                            span: arguments[0].span(),
                        });
                    };
                    let length = self.check_expression(&arguments[1], Some(&named("usize")))?;
                    self.expect_type(&named("usize"), &length, arguments[1].span())?;
                    return Ok(Type::Slice(element));
                }
                if self.lookup_variable(name).is_some() {
                    return Err(SemanticError::NotCallable {
                        name: name.clone(),
                        span: *cs,
                    });
                }
                let Some(sig) = self.functions.get(name).cloned() else {
                    return Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *cs,
                    });
                };
                if sig.parameters.len() != arguments.len() {
                    return Err(SemanticError::WrongArgumentCount {
                        name: name.clone(),
                        expected: sig.parameters.len(),
                        found: arguments.len(),
                        span: *span,
                    });
                }
                for (a, t) in arguments.iter().zip(&sig.parameters) {
                    let x = self.check_expression(a, Some(t))?;
                    self.expect_type(t, &x, a.span())?
                }
                sig.return_type
            }
        };
        if let Some(x) = expected {
            self.expect_type(x, &ty, e.span())?
        }
        Ok(ty)
    }
    fn check_binary(
        &mut self,
        l: &Expr,
        op: BinaryOp,
        r: &Expr,
        span: Span,
        expected: Option<&Type>,
    ) -> Result<Type, SemanticError> {
        let left_type = self.check_expression(l, None)?;
        if matches!(left_type, Type::Pointer(_)) {
            match op {
                BinaryOp::Add | BinaryOp::Subtract => {
                    let integer_context = named("usize");
                    let right = self.check_expression(
                        r,
                        if matches!(r, Expr::Integer { .. }) {
                            Some(&integer_context)
                        } else {
                            None
                        },
                    )?;
                    if op == BinaryOp::Subtract && right == left_type {
                        return Ok(named("isize"));
                    }
                    if !matches!(right, Type::Named(ref n) if n == "usize" || n == "isize") {
                        return Err(SemanticError::InvalidOperand {
                            message: "pointer arithmetic requires usize or isize".into(),
                            span,
                        });
                    }
                    return Ok(left_type);
                }
                BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::Less
                | BinaryOp::LessEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterEqual => {
                    let right = self.check_expression(r, Some(&left_type))?;
                    self.expect_type(&left_type, &right, r.span())?;
                    return Ok(named("bool"));
                }
                _ => {}
            }
        }
        if matches!(op, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            let b = named("bool");
            let a = self.check_expression(l, Some(&b))?;
            let c = self.check_expression(r, Some(&b))?;
            self.expect_type(&b, &a, l.span())?;
            self.expect_type(&b, &c, r.span())?;
            return Ok(b);
        }
        if matches!(
            op,
            BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::Less
                | BinaryOp::LessEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterEqual
        ) {
            let a = self.check_expression(l, None)?;
            let b = self.check_expression(r, Some(&a))?;
            self.expect_type(&a, &b, r.span())?;
            let valid = if matches!(op, BinaryOp::Equal | BinaryOp::NotEqual) {
                (is_numeric(&a) && is_numeric(&b)) || (is_bool(&a) && is_bool(&b))
            } else {
                is_numeric(&a) && is_numeric(&b)
            };
            if !valid {
                return Err(SemanticError::InvalidOperand{message:"comparison requires operands of the same numeric type (or bool for == and !=)".into(),span});
            }
            return Ok(named("bool"));
        }
        let oe = expected.filter(|t| is_numeric(t));
        let a = if matches!(left_type, Type::Pointer(_)) {
            left_type
        } else {
            self.check_expression(l, oe)?
        };
        let b = self.check_expression(r, Some(&a))?;
        self.expect_type(&a, &b, r.span())?;
        let valid = match op {
            BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo => is_numeric(&a),
            BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::ShiftLeft
            | BinaryOp::ShiftRight => is_integer(&a),
            _ => false,
        };
        if !valid {
            return Err(SemanticError::InvalidOperand {
                message: "operator requires operands of the same explicit numeric type".into(),
                span,
            });
        }
        Ok(a)
    }
    fn check_place(&mut self, e: &Expr) -> Result<(Type, bool), SemanticError> {
        match e {
            Expr::Identifier { name, span } => {
                if let Some(v) = self.lookup_variable(name) {
                    Ok((v.ty, v.mutable))
                } else if let Some(v) = self.globals.get(name) {
                    Ok((v.ty.clone(), v.mutable))
                } else {
                    Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *span,
                    })
                }
            }
            Expr::Field { base, name, span } => {
                let (bt, m) = self.check_place(base)?;
                let Type::Named(s) = bt else {
                    return Err(SemanticError::InvalidAssignmentTarget { span: *span });
                };
                let info = self.structs.get(&s).unwrap();
                let Some((_, t)) = info.fields.iter().find(|(n, _)| n == name) else {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("unknown field `{name}`"),
                        span: *span,
                    });
                };
                Ok((t.clone(), m))
            }
            Expr::Index { base, index, span } | Expr::UncheckedIndex { base, index, span } => {
                let (bt, m) = self.check_place(base)?;
                let it = self.check_expression(index, Some(&named("usize")))?;
                self.expect_type(&named("usize"), &it, index.span())?;
                match bt {
                    Type::Array { length, element } => {
                        if matches!(e, Expr::Index { .. }) {
                            if let Some(i) = self.constant_integer(index) {
                                if i < 0 || i as u64 >= length {
                                    return Err(SemanticError::InvalidLiteral {
                                        message: "array index is out of bounds".into(),
                                        span: index.span(),
                                    });
                                }
                            }
                        }
                        Ok((*element, m))
                    }
                    Type::Slice(element) => Ok((*element, m)),
                    _ => Err(SemanticError::InvalidAssignmentTarget { span: *span }),
                }
            }
            Expr::Unary {
                operator: UnaryOp::Dereference,
                operand,
                span,
            } => {
                let pointer = self.check_expression(operand, None)?;
                match pointer {
                    Type::Pointer(element) => Ok((*element, true)),
                    _ => Err(SemanticError::InvalidAssignmentTarget { span: *span }),
                }
            }
            _ => Err(SemanticError::InvalidAssignmentTarget { span: e.span() }),
        }
    }
    fn declare_local(&mut self, name: &str, v: Variable, span: Span) -> Result<(), SemanticError> {
        let s = self.scopes.last_mut().unwrap();
        if s.contains_key(name) {
            return Err(SemanticError::DuplicateName {
                name: name.into(),
                span,
            });
        }
        s.insert(name.into(), v);
        Ok(())
    }
    fn lookup_variable(&self, name: &str) -> Option<Variable> {
        self.scopes.iter().rev().find_map(|s| s.get(name).cloned())
    }
    fn expect_type(&self, e: &Type, a: &Type, span: Span) -> Result<(), SemanticError> {
        if e == a {
            Ok(())
        } else {
            Err(SemanticError::TypeMismatch {
                expected: e.clone(),
                found: a.clone(),
                span,
            })
        }
    }
    fn validate_value_type(&self, t: &Type, span: Span) -> Result<(), SemanticError> {
        if is_unit(t) {
            return Err(SemanticError::UnknownType {
                name: type_name(t),
                span,
            });
        }
        self.validate_type(t, span)
    }
    fn validate_return_type(&self, t: &Type, span: Span) -> Result<(), SemanticError> {
        self.validate_type(t, span)
    }
    fn validate_ffi_type(&self, t: &Type, span: Span) -> Result<(), SemanticError> {
        fn compatible(analyzer: &Analyzer, t: &Type, seen: &mut Vec<String>) -> bool {
            match t {
                Type::Unit => true,
                Type::Named(name)
                    if matches!(
                        name.as_str(),
                        "i8" | "i16"
                            | "i32"
                            | "i64"
                            | "i128"
                            | "u8"
                            | "u16"
                            | "u32"
                            | "u64"
                            | "u128"
                            | "usize"
                            | "isize"
                            | "f32"
                            | "f64"
                    ) =>
                {
                    true
                }
                Type::Named(name) => {
                    if seen.contains(name) {
                        return true;
                    }
                    let Some(info) = analyzer.structs.get(name) else {
                        return false;
                    };
                    seen.push(name.clone());
                    let result = info
                        .fields
                        .iter()
                        .all(|(_, field)| compatible(analyzer, field, seen));
                    seen.pop();
                    result
                }
                Type::Pointer(element) => {
                    **element == Type::Unit || compatible(analyzer, element, seen)
                }
                Type::Array { element, .. } => compatible(analyzer, element, seen),
                Type::Slice(_) | Type::Result { .. } => false,
            }
        }
        if compatible(self, t, &mut Vec::new()) {
            return Ok(());
        }
        Err(SemanticError::InvalidFfiType {
            message: format!("type `{}` is not representable in the C ABI", type_name(t)),
            span,
        })
    }
    fn validate_type(&self, t: &Type, span: Span) -> Result<(), SemanticError> {
        match t {
            Type::Unit => Ok(()),
            Type::Named(n) if is_known_type_name(n) || self.structs.contains_key(n) => Ok(()),
            Type::Named(n) => Err(SemanticError::UnknownType {
                name: n.clone(),
                span,
            }),
            Type::Array { element, .. } | Type::Slice(element) => {
                self.validate_value_type(element, span)
            }
            Type::Pointer(element) => {
                if is_unit(element) {
                    Ok(())
                } else {
                    self.validate_value_type(element, span)
                }
            }
            Type::Result { success, error } => {
                self.validate_type(success, span)?;
                self.validate_value_type(error, span)
            }
        }
    }
    fn constant_integer(&self, e: &Expr) -> Option<i128> {
        match e {
            Expr::Integer { value, .. } => Some(*value),
            Expr::Identifier { name, .. } => self
                .lookup_variable(name)
                .or_else(|| self.globals.get(name).cloned())
                .and_then(|v| {
                    v.constant_value
                        .as_ref()
                        .and_then(|x| self.constant_integer(x))
                }),
            Expr::Unary {
                operator, operand, ..
            } => {
                let value = self.constant_integer(operand)?;
                match operator {
                    UnaryOp::Negate => Some(value.wrapping_neg()),
                    UnaryOp::BitwiseNot => Some(!value),
                    UnaryOp::Not | UnaryOp::AddressOf | UnaryOp::Dereference => None,
                }
            }
            Expr::Binary {
                left,
                operator,
                right,
                ..
            } => {
                let a = self.constant_integer(left)?;
                let b = self.constant_integer(right)?;
                match operator {
                    BinaryOp::Add => Some(a.wrapping_add(b)),
                    BinaryOp::Subtract => Some(a.wrapping_sub(b)),
                    BinaryOp::Multiply => Some(a.wrapping_mul(b)),
                    BinaryOp::Divide if b != 0 => Some(a.wrapping_div(b)),
                    BinaryOp::Modulo if b != 0 => Some(a.wrapping_rem(b)),
                    BinaryOp::BitwiseAnd => Some(a & b),
                    BinaryOp::BitwiseOr => Some(a | b),
                    BinaryOp::BitwiseXor => Some(a ^ b),
                    BinaryOp::ShiftLeft if b >= 0 && b < 128 => Some(a.wrapping_shl(b as u32)),
                    BinaryOp::ShiftRight if b >= 0 && b < 128 => Some(a.wrapping_shr(b as u32)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn is_compile_time(&self, e: &Expr) -> bool {
        match e {
            Expr::Integer { .. } | Expr::Float { .. } | Expr::Bool { .. } => true,
            Expr::Identifier { name, .. } => self.constants.contains_key(name),
            Expr::StructLiteral { fields, .. } => {
                fields.iter().all(|f| self.is_compile_time(&f.value))
            }
            Expr::ArrayLiteral { elements, .. } => elements.iter().all(|x| self.is_compile_time(x)),
            Expr::SizeOf { .. } | Expr::AlignOf { .. } | Expr::OffsetOf { .. } => true,
            Expr::Unary { operand, .. } => self.is_compile_time(operand),
            Expr::Binary { left, right, .. } => {
                self.is_compile_time(left) && self.is_compile_time(right)
            }
            _ => false,
        }
    }
}

struct TypedLowerer<'a> {
    program: &'a Program,
    functions: HashMap<String, (FunctionId, &'a FunctionDecl)>,
    structs: HashMap<String, StructDecl>,
    globals: HashMap<String, (Type, bool)>,
    constants: HashMap<String, &'a VariableDecl>,
    scopes: Vec<HashMap<String, (LocalId, ResolvedType)>>,
    next_local: LocalId,
    current_return_type: ResolvedType,
}
impl<'a> TypedLowerer<'a> {
    fn new_with_pointer_width(p: &'a Program, _pointer_width: u32) -> Self {
        let mut functions = HashMap::new();
        let mut structs = HashMap::new();
        let mut globals = HashMap::new();
        let mut constants = HashMap::new();
        for (i, d) in p.declarations.iter().enumerate() {
            match d {
                Decl::Function(f) => {
                    functions.insert(f.name.clone(), (i, f));
                }
                Decl::Struct(s) => {
                    structs.insert(s.name.clone(), s.clone());
                }
                Decl::Variable(v) => {
                    let t =
                        v.ty.clone()
                            .unwrap_or_else(|| infer_ast_type_with_globals(&v.value, &globals));
                    let m = !matches!(v.kind, VariableKind::Immutable);
                    globals.insert(v.name.clone(), (t, m));
                    if !m {
                        constants.insert(v.name.clone(), v);
                    }
                }
                Decl::Comptime { .. } => {}
            }
        }
        Self {
            program: p,
            functions,
            structs,
            globals,
            constants,
            scopes: Vec::new(),
            next_local: 0,
            current_return_type: ResolvedType::Unit,
        }
    }
    fn lower(mut self) -> Result<TypedProgram, SemanticError> {
        let structs = self
            .structs
            .values()
            .map(|s| TypedStruct {
                name: s.name.clone(),
                fields: s
                    .fields
                    .iter()
                    .map(|f| TypedField {
                        name: f.name.clone(),
                        ty: self.resolve(&f.ty),
                    })
                    .collect(),
                span: s.span,
            })
            .collect();
        let mut globals_out = Vec::new();
        let mut constants_out = Vec::new();
        for d in &self.program.declarations {
            if let Decl::Variable(v) = d {
                let ty =
                    self.resolve(&v.ty.clone().unwrap_or_else(|| self.inferred_type(&v.value)));
                let value = self.lower_const_expr(&v.value, Some(ty.clone()))?;
                if matches!(v.kind, VariableKind::Immutable) {
                    constants_out.push(TypedConstant {
                        name: v.name.clone(),
                        ty,
                        value,
                        span: v.span,
                    })
                } else {
                    globals_out.push(TypedGlobal {
                        name: v.name.clone(),
                        ty,
                        value,
                        span: v.span,
                    })
                }
            }
        }
        let mut functions = Vec::new();
        for d in &self.program.declarations {
            if let Decl::Function(f) = d {
                functions.push(self.lower_function(f)?);
            }
        }
        Ok(TypedProgram {
            structs,
            globals: globals_out,
            constants: constants_out,
            functions,
        })
    }
    fn lower_const_expr(
        &mut self,
        e: &Expr,
        expected: Option<ResolvedType>,
    ) -> Result<TypedExpr, SemanticError> {
        let lowered = self.lower_expr(e, expected)?;
        let value = self.fold_constant(lowered)?;
        if !is_folded_constant(&value) {
            return Err(SemanticError::InvalidOperand {
                message: "global initializer must be a compile-time evaluable constant".into(),
                span: e.span(),
            });
        }
        Ok(value)
    }
    fn fold_constant(&self, value: TypedExpr) -> Result<TypedExpr, SemanticError> {
        match value {
            TypedExpr::Unary {
                operator,
                operand,
                ty,
                span,
            } => {
                let operand = self.fold_constant(*operand)?;
                match (&operator, &operand) {
                    (UnaryOp::Negate, TypedExpr::Integer { value, .. }) => Ok(TypedExpr::Integer {
                        value: wrap_integer(value.wrapping_neg(), &ty),
                        ty,
                        span,
                    }),
                    (UnaryOp::Negate, TypedExpr::Float { value, .. }) => Ok(TypedExpr::Float {
                        value: -*value,
                        ty,
                        span,
                    }),
                    (UnaryOp::Not | UnaryOp::BitwiseNot, TypedExpr::Bool { value, .. }) => {
                        Ok(TypedExpr::Bool {
                            value: !value,
                            ty,
                            span,
                        })
                    }
                    (UnaryOp::BitwiseNot, TypedExpr::Integer { value, .. }) => {
                        Ok(TypedExpr::Integer {
                            value: wrap_integer(!value, &ty),
                            ty,
                            span,
                        })
                    }
                    _ => Ok(TypedExpr::Unary {
                        operator,
                        operand: Box::new(operand),
                        ty,
                        span,
                    }),
                }
            }
            TypedExpr::StructLiteral { ty, fields, span } => Ok(TypedExpr::StructLiteral {
                ty,
                fields: fields
                    .into_iter()
                    .map(|field| self.fold_constant(field))
                    .collect::<Result<_, _>>()?,
                span,
            }),
            TypedExpr::ArrayLiteral { ty, elements, span } => Ok(TypedExpr::ArrayLiteral {
                ty,
                elements: elements
                    .into_iter()
                    .map(|element| self.fold_constant(element))
                    .collect::<Result<_, _>>()?,
                span,
            }),
            TypedExpr::Binary {
                left,
                operator,
                right,
                ty,
                operand_type,
                span,
            } => {
                let left = self.fold_constant(*left)?;
                let right = self.fold_constant(*right)?;
                if let (TypedExpr::Integer { value: a, .. }, TypedExpr::Integer { value: b, .. }) =
                    (&left, &right)
                {
                    let a = wrap_integer(*a, &operand_type);
                    let b = wrap_integer(*b, &operand_type);
                    if matches!(
                        operator,
                        BinaryOp::Equal
                            | BinaryOp::NotEqual
                            | BinaryOp::Less
                            | BinaryOp::LessEqual
                            | BinaryOp::Greater
                            | BinaryOp::GreaterEqual
                    ) {
                        let result = integer_compare(a, b, &operand_type, operator);
                        return Ok(TypedExpr::Bool {
                            value: result,
                            ty,
                            span,
                        });
                    }
                    let signed = operand_type.is_signed_integer();
                    let bits = integer_bit_width(&operand_type);
                    let result = match operator {
                        BinaryOp::Add => Some(a.wrapping_add(b)),
                        BinaryOp::Subtract => Some(a.wrapping_sub(b)),
                        BinaryOp::Multiply => Some(a.wrapping_mul(b)),
                        BinaryOp::BitwiseAnd => Some(a & b),
                        BinaryOp::BitwiseOr => Some(a | b),
                        BinaryOp::BitwiseXor => Some(a ^ b),
                        BinaryOp::ShiftLeft if b >= 0 && (b as u32) < bits => {
                            Some(a.wrapping_shl(b as u32))
                        }
                        BinaryOp::ShiftRight if b >= 0 && (b as u32) < bits => {
                            if signed {
                                Some(a.wrapping_shr(b as u32))
                            } else {
                                Some(((a as u128) >> (b as u32)) as i128)
                            }
                        }
                        BinaryOp::Divide
                            if b != 0
                                && !(signed && a == signed_minimum(&operand_type) && b == -1) =>
                        {
                            if signed {
                                Some(a.wrapping_div(b))
                            } else {
                                Some(((a as u128) / (b as u128)) as i128)
                            }
                        }
                        BinaryOp::Modulo
                            if b != 0
                                && !(signed && a == signed_minimum(&operand_type) && b == -1) =>
                        {
                            if signed {
                                Some(a.wrapping_rem(b))
                            } else {
                                Some(((a as u128) % (b as u128)) as i128)
                            }
                        }
                        _ => None,
                    };
                    if let Some(result) = result {
                        return Ok(TypedExpr::Integer {
                            value: wrap_integer(result, &operand_type),
                            ty,
                            span,
                        });
                    }
                }
                if let (TypedExpr::Float { value: a, .. }, TypedExpr::Float { value: b, .. }) =
                    (&left, &right)
                {
                    if matches!(
                        operator,
                        BinaryOp::Equal
                            | BinaryOp::NotEqual
                            | BinaryOp::Less
                            | BinaryOp::LessEqual
                            | BinaryOp::Greater
                            | BinaryOp::GreaterEqual
                    ) {
                        let result = match operator {
                            BinaryOp::Equal => *a == *b,
                            BinaryOp::NotEqual => *a != *b,
                            BinaryOp::Less => *a < *b,
                            BinaryOp::LessEqual => *a <= *b,
                            BinaryOp::Greater => *a > *b,
                            BinaryOp::GreaterEqual => *a >= *b,
                            _ => unreachable!(),
                        };
                        return Ok(TypedExpr::Bool {
                            value: result,
                            ty,
                            span,
                        });
                    }
                    let result = match operator {
                        BinaryOp::Add => Some(*a + *b),
                        BinaryOp::Subtract => Some(*a - *b),
                        BinaryOp::Multiply => Some(*a * *b),
                        BinaryOp::Divide => Some(*a / *b),
                        BinaryOp::Modulo => Some(*a % *b),
                        _ => None,
                    };
                    if let Some(result) = result {
                        return Ok(TypedExpr::Float {
                            value: result,
                            ty,
                            span,
                        });
                    }
                }
                if let (TypedExpr::Bool { value: a, .. }, TypedExpr::Bool { value: b, .. }) =
                    (&left, &right)
                {
                    if operator == BinaryOp::LogicalAnd || operator == BinaryOp::LogicalOr {
                        return Ok(TypedExpr::Bool {
                            value: if operator == BinaryOp::LogicalAnd {
                                *a && *b
                            } else {
                                *a || *b
                            },
                            ty,
                            span,
                        });
                    }
                    if operator == BinaryOp::Equal || operator == BinaryOp::NotEqual {
                        return Ok(TypedExpr::Bool {
                            value: if operator == BinaryOp::Equal {
                                *a == *b
                            } else {
                                *a != *b
                            },
                            ty,
                            span,
                        });
                    }
                }
                Ok(TypedExpr::Binary {
                    left: Box::new(left),
                    operator,
                    right: Box::new(right),
                    ty,
                    operand_type,
                    span,
                })
            }
            other => Ok(other),
        }
    }
    fn inferred_type(&self, e: &Expr) -> Type {
        infer_ast_type_with_globals(e, &self.globals)
    }
    fn lower_function(&mut self, f: &FunctionDecl) -> Result<TypedFunction, SemanticError> {
        self.scopes.push(HashMap::new());
        self.next_local = 0;
        self.current_return_type = self.resolve(&f.return_type);
        let mut params = Vec::new();
        for p in &f.params {
            let ty = self.resolve(&p.ty);
            let id = self.new_local(&p.name, ty.clone(), p.span)?;
            params.push(TypedParameter {
                id,
                name: p.name.clone(),
                ty,
                span: p.span,
            })
        }
        let body = self.lower_block_contents(&f.body)?;
        self.scopes.pop();
        Ok(TypedFunction {
            id: self.functions[&f.name].0,
            name: f.name.clone(),
            params,
            return_type: self.resolve(&f.return_type),
            body,
            span: f.span,
            is_extern: f.is_extern,
            abi: f.abi.clone(),
            link_name: f.link_name.clone(),
            exported: f.exported,
        })
    }
    fn lower_block(&mut self, b: &Block) -> Result<TypedBlock, SemanticError> {
        self.scopes.push(HashMap::new());
        let x = self.lower_block_contents(b);
        self.scopes.pop();
        x
    }
    fn lower_block_contents(&mut self, b: &Block) -> Result<TypedBlock, SemanticError> {
        Ok(TypedBlock {
            statements: b
                .statements
                .iter()
                .map(|s| self.lower_statement(s))
                .collect::<Result<_, _>>()?,
            span: b.span,
        })
    }
    fn lower_statement(&mut self, s: &Stmt) -> Result<TypedStmt, SemanticError> {
        Ok(match s {
            Stmt::If {
                condition,
                then_branch,
                else_branch,
                span,
            } => TypedStmt::If {
                condition: self.lower_expr(condition, Some(ResolvedType::Bool))?,
                then_branch: self.lower_block(then_branch)?,
                else_branch: else_branch
                    .as_ref()
                    .map(|b| self.lower_block(b))
                    .transpose()?,
                span: *span,
            },
            Stmt::While {
                condition,
                body,
                span,
            } => TypedStmt::While {
                condition: self.lower_expr(condition, Some(ResolvedType::Bool))?,
                body: self.lower_block(body)?,
                span: *span,
            },
            Stmt::Break { span } => TypedStmt::Break { span: *span },
            Stmt::Continue { span } => TypedStmt::Continue { span: *span },
            Stmt::Defer { call, span } => {
                let lowered = self.lower_expr(call, None)?;
                let TypedExpr::Call {
                    function,
                    name,
                    arguments,
                    ..
                } = lowered
                else {
                    return Err(SemanticError::InvalidDefer {
                        message: "defer requires a named function call".into(),
                        span: *span,
                    });
                };
                TypedStmt::Defer {
                    function,
                    name,
                    arguments,
                    span: *span,
                }
            }
            Stmt::Return { value, span } => TypedStmt::Return {
                value: value
                    .as_ref()
                    .map(|x| self.lower_expr(x, Some(self.current_return_type.clone())))
                    .transpose()?,
                span: *span,
            },
            Stmt::Variable(v) => {
                let declared = v.ty.as_ref().map(|x| self.resolve(x));
                let value = self.lower_expr(&v.value, declared.clone())?;
                let ty = declared.unwrap_or_else(|| value.ty());
                let id = self.new_local(&v.name, ty.clone(), v.span)?;
                TypedStmt::Declare {
                    id,
                    name: v.name.clone(),
                    ty,
                    mutable: !matches!(v.kind, VariableKind::Immutable),
                    value,
                    span: v.span,
                }
            }
            Stmt::Assignment {
                target,
                value,
                span,
            } => {
                let place = self.lower_place(target)?;
                let ty = place.ty();
                TypedStmt::Store {
                    target: place,
                    value: self.lower_expr(value, Some(ty.clone()))?,
                    ty,
                    span: *span,
                }
            }
            Stmt::Expr { expression, span } => TypedStmt::Expr {
                expression: self.lower_expr(expression, None)?,
                span: *span,
            },
        })
    }
    fn lower_place(&mut self, e: &Expr) -> Result<TypedPlace, SemanticError> {
        match e {
            Expr::Identifier { name, span } => {
                if let Some((id, t)) = self.lookup(name) {
                    Ok(TypedPlace::Local { id, ty: t })
                } else if let Some((t, _)) = self.globals.get(name) {
                    Ok(TypedPlace::Global {
                        name: name.clone(),
                        ty: self.resolve(t),
                    })
                } else {
                    Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *span,
                    })
                }
            }
            Expr::Field { base, name, span } => {
                let p = self.lower_place(base)?;
                let Type::Named(s) = self.ast_type_of_place(&p) else {
                    return Err(SemanticError::InvalidAssignmentTarget { span: *span });
                };
                let st = self.structs.get(&s).unwrap();
                let Some((i, f)) = st.fields.iter().enumerate().find(|(_, f)| f.name == *name)
                else {
                    return Err(SemanticError::InvalidOperand {
                        message: format!("unknown field `{name}`"),
                        span: *span,
                    });
                };
                Ok(TypedPlace::Field {
                    base: Box::new(p),
                    index: i as u32,
                    ty: self.resolve(&f.ty),
                })
            }
            Expr::Unary {
                operator: UnaryOp::Dereference,
                operand,
                span,
            } => {
                let pointer = self.lower_expr(operand, None)?;
                let Type::Pointer(element) = self.ast_from_resolved(&pointer.ty()) else {
                    return Err(SemanticError::InvalidAssignmentTarget { span: *span });
                };
                Ok(TypedPlace::Dereference {
                    pointer: Box::new(pointer),
                    ty: self.resolve(&element),
                })
            }
            Expr::Index { base, index, span } | Expr::UncheckedIndex { base, index, span } => {
                let p = self.lower_place(base)?;
                let t = self.ast_type_of_place(&p);
                let (element, length) = match t {
                    Type::Array { length, element } => (element, Some(length)),
                    Type::Slice(element) => (element, None),
                    _ => return Err(SemanticError::InvalidAssignmentTarget { span: *span }),
                };
                let ix = self.lower_expr(
                    index,
                    Some(ResolvedType::Integer {
                        width: IntegerWidth::Pointer,
                        signed: false,
                    }),
                )?;
                Ok(TypedPlace::Index {
                    base: Box::new(p),
                    index: Box::new(ix),
                    ty: self.resolve(&element),
                    length,
                    checked: matches!(e, Expr::Index { .. }),
                })
            }
            _ => {
                // A field or index expression may be based on an rvalue (for
                // example, `make().field`). Materialize that value so nested
                // aggregate access still has an address in the typed IR.
                let value = self.lower_expr(e, None)?;
                let ty = value.ty();
                Ok(TypedPlace::Temporary {
                    value: Box::new(value),
                    ty,
                })
            }
        }
    }
    fn ast_type_of_place(&self, p: &TypedPlace) -> Type {
        match p {
            TypedPlace::Local { ty, .. }
            | TypedPlace::Global { ty, .. }
            | TypedPlace::Temporary { ty, .. }
            | TypedPlace::Field { ty, .. }
            | TypedPlace::Index { ty, .. }
            | TypedPlace::Dereference { ty, .. } => self.ast_from_resolved(ty),
        }
    }
    fn ast_from_resolved(&self, t: &ResolvedType) -> Type {
        match t {
            ResolvedType::Unit => Type::Unit,
            ResolvedType::Bool => named("bool"),
            ResolvedType::Integer { width, signed } => {
                let n = match (width, signed) {
                    (IntegerWidth::Bits(8), true) => "i8",
                    (IntegerWidth::Bits(16), true) => "i16",
                    (IntegerWidth::Bits(32), true) => "i32",
                    (IntegerWidth::Bits(64), true) => "i64",
                    (IntegerWidth::Bits(128), true) => "i128",
                    (IntegerWidth::Bits(8), false) => "u8",
                    (IntegerWidth::Bits(16), false) => "u16",
                    (IntegerWidth::Bits(32), false) => "u32",
                    (IntegerWidth::Bits(64), false) => "u64",
                    (IntegerWidth::Bits(128), false) => "u128",
                    (IntegerWidth::Pointer, true) => "isize",
                    (IntegerWidth::Pointer, false) => "usize",
                    _ => "i32",
                };
                named(n)
            }
            ResolvedType::Float { bits } => named(if *bits == 32 { "f32" } else { "f64" }),
            ResolvedType::Struct(n) => named(n),
            ResolvedType::Array { length, element } => Type::Array {
                length: *length,
                element: Box::new(self.ast_from_resolved(element)),
            },
            ResolvedType::Pointer(element) => {
                Type::Pointer(Box::new(self.ast_from_resolved(element)))
            }
            ResolvedType::Slice(element) => Type::Slice(Box::new(self.ast_from_resolved(element))),
            ResolvedType::Result { success, error } => Type::Result {
                success: Box::new(self.ast_from_resolved(success)),
                error: Box::new(self.ast_from_resolved(error)),
            },
        }
    }
    fn lower_expr(
        &mut self,
        e: &Expr,
        expected: Option<ResolvedType>,
    ) -> Result<TypedExpr, SemanticError> {
        let span = e.span();
        match e {
            Expr::Comptime { span, .. } => Err(SemanticError::InvalidOperand {
                message: "compile-time marker was not expanded before lowering".into(),
                span: *span,
            }),
            Expr::Integer { value, .. } => Ok(TypedExpr::Integer {
                value: *value,
                ty: expected
                    .filter(|t| t.is_integer())
                    .unwrap_or(ResolvedType::Integer {
                        width: IntegerWidth::Bits(32),
                        signed: true,
                    }),
                span,
            }),
            Expr::Float { value, .. } => Ok(TypedExpr::Float {
                value: *value,
                ty: expected
                    .filter(|t| matches!(t, ResolvedType::Float { .. }))
                    .unwrap_or(ResolvedType::Float { bits: 64 }),
                span,
            }),
            Expr::Bool { value, .. } => Ok(TypedExpr::Bool {
                value: *value,
                ty: ResolvedType::Bool,
                span,
            }),
            Expr::String { .. } => Err(SemanticError::InvalidOperand {
                message: "strings are only available in compile-time context".into(),
                span,
            }),
            Expr::Propagate { expression, .. } => {
                let value = self.lower_expr(expression, None)?;
                let ResolvedType::Result { success, error } = value.ty() else {
                    return Err(SemanticError::InvalidPropagation {
                        message: "`?` requires a result value".into(),
                        span,
                    });
                };
                let ResolvedType::Result {
                    error: current_error,
                    ..
                } = &self.current_return_type
                else {
                    return Err(SemanticError::InvalidPropagation {
                        message: "`?` requires a result-returning function".into(),
                        span,
                    });
                };
                if **current_error != *error {
                    return Err(SemanticError::InvalidPropagation {
                        message:
                            "propagated error type is incompatible with the current return type"
                                .into(),
                        span,
                    });
                }
                Ok(TypedExpr::Propagate {
                    value: Box::new(value),
                    ty: *success,
                    span,
                })
            }
            Expr::Null { .. } => Ok(TypedExpr::Null {
                ty: expected.ok_or_else(|| SemanticError::InvalidOperand {
                    message: "null requires a pointer context".into(),
                    span,
                })?,
                span,
            }),
            Expr::SizeOf { ty, .. } => Ok(TypedExpr::Layout {
                kind: LayoutKind::Size,
                ty: intp(false),
                target: self.resolve(ty),
                field: None,
                span,
            }),
            Expr::AlignOf { ty, .. } => Ok(TypedExpr::Layout {
                kind: LayoutKind::Align,
                ty: intp(false),
                target: self.resolve(ty),
                field: None,
                span,
            }),
            Expr::OffsetOf { ty, field, .. } => Ok(TypedExpr::Layout {
                kind: LayoutKind::Offset,
                ty: intp(false),
                target: self.resolve(ty),
                field: Some(field.clone()),
                span,
            }),
            Expr::Identifier { name, .. } => {
                if let Some((id, t)) = self.lookup(name) {
                    return Ok(TypedExpr::Load {
                        id,
                        name: name.clone(),
                        ty: t,
                        span,
                    });
                }
                if let Some((t, m)) = self.globals.get(name) {
                    let ty = self.resolve(t);
                    if *m {
                        return Ok(TypedExpr::GlobalLoad {
                            name: name.clone(),
                            ty,
                            span,
                        });
                    }
                }
                if let Some(v) = self.constants.get(name).copied() {
                    return self.lower_expr(
                        &v.value,
                        Some(self.resolve(
                            &v.ty.clone().unwrap_or_else(|| self.inferred_type(&v.value)),
                        )),
                    );
                }
                Err(SemanticError::UndefinedName {
                    name: name.clone(),
                    span,
                })
            }
            Expr::StructLiteral { name, fields, .. } => {
                let st = self.structs.get(name).unwrap().clone();
                let mut out = Vec::new();
                for f in &st.fields {
                    let x = fields.iter().find(|x| x.name == f.name).unwrap();
                    out.push(self.lower_expr(&x.value, Some(self.resolve(&f.ty)))?)
                }
                Ok(TypedExpr::StructLiteral {
                    ty: ResolvedType::Struct(name.clone()),
                    fields: out,
                    span,
                })
            }
            Expr::ArrayLiteral { ty, elements, .. } => {
                let rt = self.resolve(ty);
                let Type::Array { element, .. } = ty else {
                    unreachable!()
                };
                let out = elements
                    .iter()
                    .map(|x| self.lower_expr(x, Some(self.resolve(element))))
                    .collect::<Result<_, _>>()?;
                Ok(TypedExpr::ArrayLiteral {
                    ty: rt,
                    elements: out,
                    span,
                })
            }
            Expr::Field { .. } => {
                let p = self.lower_place(e)?;
                let ty = p.ty();
                Ok(TypedExpr::Field { place: p, ty, span })
            }
            Expr::Index { .. } | Expr::UncheckedIndex { .. } => {
                let p = self.lower_place(e)?;
                let ty = p.ty();
                Ok(TypedExpr::Index { place: p, ty, span })
            }
            Expr::Unary {
                operator: UnaryOp::AddressOf,
                operand,
                ..
            } => {
                let place = self.lower_place(operand)?;
                let ty = ResolvedType::Pointer(Box::new(place.ty()));
                Ok(TypedExpr::AddressOf { place, ty, span })
            }
            Expr::Unary {
                operator: UnaryOp::Dereference,
                ..
            } => {
                let place = self.lower_place(e)?;
                let ty = place.ty();
                Ok(TypedExpr::Dereference { place, ty, span })
            }
            Expr::Unary {
                operator, operand, ..
            } => {
                let oe = if *operator == UnaryOp::Not {
                    Some(ResolvedType::Bool)
                } else {
                    expected.clone()
                };
                let x = self.lower_expr(operand, oe)?;
                Ok(TypedExpr::Unary {
                    operator: *operator,
                    ty: x.ty(),
                    operand: Box::new(x),
                    span,
                })
            }
            Expr::Binary {
                left,
                operator,
                right,
                ..
            } => {
                let logical = matches!(operator, BinaryOp::LogicalAnd | BinaryOp::LogicalOr);
                let comparison = matches!(
                    operator,
                    BinaryOp::Equal
                        | BinaryOp::NotEqual
                        | BinaryOp::Less
                        | BinaryOp::LessEqual
                        | BinaryOp::Greater
                        | BinaryOp::GreaterEqual
                );
                let le = self.lower_expr(
                    left,
                    if logical {
                        Some(ResolvedType::Bool)
                    } else {
                        expected
                            .filter(|t| t.is_integer() || matches!(t, ResolvedType::Float { .. }))
                    },
                )?;
                let re = self.lower_expr(
                    right,
                    if logical {
                        Some(ResolvedType::Bool)
                    } else if matches!(le.ty(), ResolvedType::Pointer(_))
                        && matches!(operator, BinaryOp::Add | BinaryOp::Subtract)
                        && matches!(&**right, Expr::Integer { .. })
                    {
                        Some(intp(false))
                    } else {
                        Some(le.ty())
                    },
                )?;
                let ot = le.ty();
                Ok(TypedExpr::Binary {
                    left: Box::new(le),
                    operator: *operator,
                    right: Box::new(re),
                    ty: if logical || comparison {
                        ResolvedType::Bool
                    } else {
                        ot.clone()
                    },
                    operand_type: ot,
                    span,
                })
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                let Expr::Identifier { name, .. } = callee.as_ref() else {
                    unreachable!()
                };
                if name == "return_ok"
                    || name == "return_err"
                    || name == "is_err"
                    || name == "unwrap"
                {
                    let arg = arguments
                        .first()
                        .map(|a| self.lower_expr(a, None))
                        .transpose()?;
                    match name.as_str() {
                        "is_err" => {
                            return Ok(TypedExpr::IsErr {
                                value: Box::new(arg.unwrap()),
                                ty: ResolvedType::Bool,
                                span,
                            });
                        }
                        "unwrap" => {
                            let arg = arg.unwrap();
                            let ResolvedType::Result { success, .. } = arg.ty() else {
                                unreachable!()
                            };
                            return Ok(TypedExpr::Unwrap {
                                ty: *success,
                                value: Box::new(arg),
                                span,
                            });
                        }
                        "return_ok" | "return_err" => {
                            let ResolvedType::Result { success, error } =
                                self.current_return_type.clone()
                            else {
                                unreachable!()
                            };
                            let wanted = if name == "return_ok" {
                                *success.clone()
                            } else {
                                *error.clone()
                            };
                            let value = if let Some(argument) = arguments.first() {
                                self.lower_expr(argument, Some(wanted))?
                            } else {
                                TypedExpr::Integer {
                                    value: 0,
                                    ty: int(8, false),
                                    span,
                                }
                            };
                            return Ok(if name == "return_ok" {
                                TypedExpr::ResultOk {
                                    value: Box::new(value),
                                    ty: ResolvedType::Result { success, error },
                                    span,
                                }
                            } else {
                                TypedExpr::ResultErr {
                                    value: Box::new(value),
                                    ty: ResolvedType::Result { success, error },
                                    span,
                                }
                            });
                        }
                        _ => unreachable!(),
                    }
                }
                if name == "make_slice" {
                    let pointer = if matches!(&arguments[0], Expr::Null { .. }) {
                        self.lower_expr(
                            &arguments[0],
                            Some(ResolvedType::Pointer(Box::new(int(8, false)))),
                        )?
                    } else {
                        self.lower_expr(&arguments[0], None)?
                    };
                    let length = self.lower_expr(&arguments[1], Some(intp(false)))?;
                    let element = match pointer.ty() {
                        ResolvedType::Pointer(x) => *x,
                        _ => unreachable!(),
                    };
                    return Ok(TypedExpr::Call {
                        function: usize::MAX,
                        name: name.clone(),
                        arguments: vec![pointer, length],
                        parameter_types: vec![],
                        ty: ResolvedType::Slice(Box::new(element)),
                        span,
                    });
                }
                let (id, f) = self.functions.get(name).copied().unwrap();
                let pts = f
                    .params
                    .iter()
                    .map(|p| self.resolve(&p.ty))
                    .collect::<Vec<_>>();
                let args = arguments
                    .iter()
                    .zip(&pts)
                    .map(|(a, t)| self.lower_expr(a, Some(t.clone())))
                    .collect::<Result<_, _>>()?;
                Ok(TypedExpr::Call {
                    function: id,
                    name: name.clone(),
                    arguments: args,
                    parameter_types: pts,
                    ty: self.resolve(&f.return_type),
                    span,
                })
            }
        }
    }
    fn new_local(
        &mut self,
        n: &str,
        t: ResolvedType,
        span: Span,
    ) -> Result<LocalId, SemanticError> {
        let id = self.next_local;
        self.next_local += 1;
        let s = self.scopes.last_mut().unwrap();
        if s.contains_key(n) {
            return Err(SemanticError::DuplicateName {
                name: n.into(),
                span,
            });
        }
        s.insert(n.into(), (id, t));
        Ok(id)
    }
    fn lookup(&self, n: &str) -> Option<(LocalId, ResolvedType)> {
        self.scopes.iter().rev().find_map(|s| s.get(n).cloned())
    }
    fn resolve(&self, t: &Type) -> ResolvedType {
        match t {
            Type::Unit => ResolvedType::Unit,
            Type::Named(n) => match n.as_str() {
                "bool" => ResolvedType::Bool,
                "i8" => int(8, true),
                "i16" => int(16, true),
                "i32" => int(32, true),
                "i64" => int(64, true),
                "i128" => int(128, true),
                "u8" => int(8, false),
                "u16" => int(16, false),
                "u32" => int(32, false),
                "u64" => int(64, false),
                "u128" => int(128, false),
                "usize" => intp(false),
                "isize" => intp(true),
                "f32" => ResolvedType::Float { bits: 32 },
                "f64" => ResolvedType::Float { bits: 64 },
                "void" => ResolvedType::Unit,
                _ => ResolvedType::Struct(n.clone()),
            },
            Type::Array { length, element } => ResolvedType::Array {
                length: *length,
                element: Box::new(self.resolve(element)),
            },
            Type::Pointer(element) => ResolvedType::Pointer(Box::new(self.resolve(element))),
            Type::Slice(element) => ResolvedType::Slice(Box::new(self.resolve(element))),
            Type::Result { success, error } => ResolvedType::Result {
                success: Box::new(self.resolve(success)),
                error: Box::new(self.resolve(error)),
            },
        }
    }
}
fn infer_ast_type_with_globals(e: &Expr, globals: &HashMap<String, (Type, bool)>) -> Type {
    match e {
        Expr::Integer { .. } => named("i32"),
        Expr::Float { .. } => named("f64"),
        Expr::Bool { .. } => named("bool"),
        Expr::Identifier { name, .. } => globals
            .get(name)
            .map(|(ty, _)| ty.clone())
            .unwrap_or_else(|| named("i32")),
        Expr::StructLiteral { name, .. } => named(name),
        Expr::ArrayLiteral { ty, .. } => ty.clone(),
        Expr::Null { .. } => Type::Pointer(Box::new(named("u8"))),
        Expr::SizeOf { .. } | Expr::AlignOf { .. } | Expr::OffsetOf { .. } => named("usize"),
        Expr::UncheckedIndex { base, .. } | Expr::Index { base, .. } => {
            infer_ast_type_with_globals(base, globals)
        }
        Expr::Propagate { expression, .. } => infer_ast_type_with_globals(expression, globals),
        Expr::Unary {
            operator, operand, ..
        } => match operator {
            UnaryOp::AddressOf => {
                Type::Pointer(Box::new(infer_ast_type_with_globals(operand, globals)))
            }
            UnaryOp::Dereference => match infer_ast_type_with_globals(operand, globals) {
                Type::Pointer(x) => *x,
                _ => named("i32"),
            },
            _ => infer_ast_type_with_globals(operand, globals),
        },
        Expr::Binary { left, .. } => infer_ast_type_with_globals(left, globals),
        Expr::Call { arguments, .. } if !arguments.is_empty() => {
            infer_ast_type_with_globals(&arguments[0], globals)
        }
        _ => named("i32"),
    }
}
fn expression_propagates(e: &Expr) -> bool {
    match e {
        Expr::Propagate { .. } => true,
        Expr::Call { arguments, .. } => arguments.iter().any(expression_propagates),
        Expr::Binary { left, right, .. } => {
            expression_propagates(left) || expression_propagates(right)
        }
        Expr::Unary { operand, .. } => expression_propagates(operand),
        Expr::Field { base, .. } | Expr::Index { base, .. } | Expr::UncheckedIndex { base, .. } => {
            expression_propagates(base)
        }
        _ => false,
    }
}
fn place_name(e: &Expr) -> String {
    match e {
        Expr::Identifier { name, .. } => name.clone(),
        Expr::Field { name, .. } => name.clone(),
        _ => "value".into(),
    }
}
fn named(n: &str) -> Type {
    Type::Named(n.into())
}
fn is_folded_constant(e: &TypedExpr) -> bool {
    match e {
        TypedExpr::Integer { .. }
        | TypedExpr::Float { .. }
        | TypedExpr::Bool { .. }
        | TypedExpr::Layout { .. } => true,
        TypedExpr::StructLiteral { fields, .. } => fields.iter().all(is_folded_constant),
        TypedExpr::ArrayLiteral { elements, .. } => elements.iter().all(is_folded_constant),
        _ => false,
    }
}

fn integer_bit_width(t: &ResolvedType) -> u32 {
    match t {
        ResolvedType::Integer {
            width: IntegerWidth::Bits(bits),
            ..
        } => *bits as u32,
        ResolvedType::Integer {
            width: IntegerWidth::Pointer,
            ..
        } => usize::BITS,
        _ => 128,
    }
}

/// Keep the i128 representation in the same two's-complement bit pattern as
/// the declared integer width. This makes compile-time evaluation agree with
/// LLVM's wrapping integer operations.
fn wrap_integer(value: i128, t: &ResolvedType) -> i128 {
    let bits = integer_bit_width(t);
    if bits >= 128 {
        return value;
    }
    let modulus = 1_i128 << bits;
    let raw = value.rem_euclid(modulus);
    if t.is_signed_integer() && raw >= (modulus / 2) {
        raw - modulus
    } else {
        raw
    }
}

fn signed_minimum(t: &ResolvedType) -> i128 {
    let bits = integer_bit_width(t);
    if bits >= 128 {
        i128::MIN
    } else {
        -(1_i128 << (bits - 1))
    }
}

fn integer_compare(a: i128, b: i128, t: &ResolvedType, op: BinaryOp) -> bool {
    if t.is_signed_integer() {
        match op {
            BinaryOp::Equal => a == b,
            BinaryOp::NotEqual => a != b,
            BinaryOp::Less => a < b,
            BinaryOp::LessEqual => a <= b,
            BinaryOp::Greater => a > b,
            BinaryOp::GreaterEqual => a >= b,
            _ => unreachable!(),
        }
    } else {
        let (a, b) = (a as u128, b as u128);
        match op {
            BinaryOp::Equal => a == b,
            BinaryOp::NotEqual => a != b,
            BinaryOp::Less => a < b,
            BinaryOp::LessEqual => a <= b,
            BinaryOp::Greater => a > b,
            BinaryOp::GreaterEqual => a >= b,
            _ => unreachable!(),
        }
    }
}

fn int(bits: u16, signed: bool) -> ResolvedType {
    ResolvedType::Integer {
        width: IntegerWidth::Bits(bits),
        signed,
    }
}
fn intp(signed: bool) -> ResolvedType {
    ResolvedType::Integer {
        width: IntegerWidth::Pointer,
        signed,
    }
}
fn is_unit(t: &Type) -> bool {
    matches!(t, Type::Unit) || matches!(t,Type::Named(n)if n=="void")
}
fn is_known_type_name(n: &str) -> bool {
    matches!(
        n,
        "bool"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "isize"
            | "f32"
            | "f64"
            | "void"
    )
}
fn type_name(t: &Type) -> String {
    match t {
        Type::Unit => "void".into(),
        Type::Named(n) => n.clone(),
        Type::Array { length, element } => format!("[{length}]{}", type_name(element)),
        Type::Pointer(element) => format!("*{}", type_name(element)),
        Type::Slice(element) => format!("[]{}", type_name(element)),
        Type::Result { success, error } => format!("{} | {}", type_name(success), type_name(error)),
    }
}
fn is_bool(t: &Type) -> bool {
    matches!(t,Type::Named(n)if n=="bool")
}
fn is_integer(t: &Type) -> bool {
    matches!(t,Type::Named(n)if matches!(n.as_str(),"i8"|"i16"|"i32"|"i64"|"i128"|"u8"|"u16"|"u32"|"u64"|"u128"|"usize"|"isize"))
}
fn is_signed_integer(t: &Type) -> bool {
    matches!(t,Type::Named(n)if matches!(n.as_str(),"i8"|"i16"|"i32"|"i64"|"i128"|"isize"))
}
fn is_float(t: &Type) -> bool {
    matches!(t,Type::Named(n)if n=="f32"||n=="f64")
}
fn is_numeric(t: &Type) -> bool {
    is_integer(t) || is_float(t)
}
fn integer_fits_with_width(v: i128, t: &Type, pointer_width: u32) -> bool {
    match type_name(t).as_str() {
        "i8" => i8::try_from(v).is_ok(),
        "i16" => i16::try_from(v).is_ok(),
        "i32" => i32::try_from(v).is_ok(),
        "i64" => i64::try_from(v).is_ok(),
        "i128" => true,
        "u8" => u8::try_from(v).is_ok(),
        "u16" => u16::try_from(v).is_ok(),
        "u32" => u32::try_from(v).is_ok(),
        "u64" => u64::try_from(v).is_ok(),
        "u128" => v >= 0,
        "usize" => v >= 0 && (pointer_width >= 128 || (v as u128) < (1u128 << pointer_width)),
        "isize" => {
            pointer_width >= 128
                || (v >= -(1i128 << (pointer_width - 1)) && v < (1i128 << (pointer_width - 1)))
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::parse_source;

    fn check(source: &str) -> Result<TypedProgram, SemanticError> {
        let program = parse_source(source).expect("source should parse");
        Analyzer::new().analyze_typed(&program)
    }

    #[test]
    fn accepts_structs_arrays_globals_and_rvalue_access() {
        assert!(check(
            "Pair :: struct { x: i32; y: i32; }\n\
             make :: () -> Pair { return Pair{x = 1, y = 2}; }\n\
             limit :: i32 = 4; counter : i32 = 5;\n\
             main :: () -> i32 { xs := [2]i32{3, 4}; xs[0] = 8; return make().x + xs[0] + limit + counter; }"
        )
        .is_ok());
    }

    #[test]
    fn rejects_invalid_struct_and_array_operations() {
        assert!(check(
            "Pair :: struct { x: i32; } main :: () -> i32 { p := Pair{x = 1, z = 2}; return 0; }"
        )
        .is_err());
        assert!(check(
            "Pair :: struct { x: i32; } main :: () -> i32 { p := Pair{x = 1, x = 2}; return 0; }"
        )
        .is_err());
        assert!(check("main :: () -> i32 { xs := [2]i32{1}; return 0; }").is_err());
        assert!(check("main :: () -> i32 { xs := [2]i32{1, 2}; return xs[2]; }").is_err());
        assert!(
            check("idx :: usize = 2; main :: () -> i32 { xs := [2]i32{1, 2}; return xs[idx]; }")
                .is_err()
        );
    }

    #[test]
    fn rejects_nonconstant_global_initializers_and_constant_traps() {
        assert!(
            check("f :: () -> i32 { return 1; } bad : i32 = f(); main :: () -> i32 { return 0; }")
                .is_err()
        );
        assert!(
            check("counter : i32 = 1; bad :: i32 = counter; main :: () -> i32 { return 0; }")
                .is_err()
        );
        assert!(check("bad :: i32 = 1 / 0; main :: () -> i32 { return 0; }").is_err());
    }
}
