use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ast::{
    BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, StructDecl, Type, UnaryOp,
    VariableDecl, VariableKind,
};
use crate::lexer::Span;
use crate::modules::ModuleGraph;
use crate::typed::{
    DefId, FunctionId, IntegerWidth, Intrinsic, LayoutKind, LocalId, LowLevelOperation,
    ResolvedType, TypedBlock, TypedConstant, TypedExpr, TypedField, TypedFunction, TypedGlobal,
    TypedParameter, TypedPlace, TypedProgram, TypedStmt, TypedStruct,
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

pub fn analyze(program: &Program) -> Result<(), SemanticError> {
    analyze_typed(program).map(|_| ())
}

pub fn analyze_typed(program: &Program) -> Result<TypedProgram, SemanticError> {
    TypedLowerer::new_with_pointer_width(program, usize::BITS).lower()
}

pub fn analyze_typed_with_pointer_width(
    program: &Program,
    pointer_width: u32,
) -> Result<TypedProgram, SemanticError> {
    TypedLowerer::new_with_pointer_width(program, pointer_width).lower()
}

/// Lower a graph-resolved compatibility view while taking declaration IDs
/// from the canonical module graph.
pub fn analyze_typed_with_graph(
    program: &Program,
    graph: &ModuleGraph,
    pointer_width: u32,
) -> Result<TypedProgram, SemanticError> {
    TypedLowerer::new_with_graph(program, graph, pointer_width).lower()
}

/// Validate the native entry point after typed lowering. Keeping this check on
/// the typed representation means clients never need a second AST pass.
pub fn validate_typed_entry_point(program: &TypedProgram) -> Result<(), SemanticError> {
    let mains: Vec<_> = program
        .functions
        .iter()
        .filter(|function| function.name == "main" && !function.is_extern)
        .collect();
    let Some(main) = mains.first() else {
        return Err(SemanticError::InvalidEntryPoint {
            message: "native build requires exactly one `main` function".into(),
            span: Span::new(0, 0),
        });
    };
    if mains.len() != 1 {
        return Err(SemanticError::InvalidEntryPoint {
            message: "duplicate `main` declarations".into(),
            span: main.span,
        });
    }
    if !main.params.is_empty() {
        return Err(SemanticError::InvalidEntryPoint {
            message: "`main` must not have parameters".into(),
            span: main.span,
        });
    }
    if !matches!(
        main.return_type,
        ResolvedType::Integer {
            width: IntegerWidth::Bits(32),
            signed: true
        }
    ) {
        return Err(SemanticError::InvalidEntryPoint {
            message: "`main` must return i32".into(),
            span: main.span,
        });
    }
    Ok(())
}

struct TypedLowerer<'a> {
    program: &'a Program,
    functions: HashMap<String, (FunctionId, &'a FunctionDecl)>,
    structs: HashMap<String, (DefId, StructDecl)>,
    globals: HashMap<String, (Type, bool)>,
    global_ids: HashMap<String, DefId>,
    definition_ids: HashMap<String, DefId>,
    constants: HashMap<String, &'a VariableDecl>,
    scopes: Vec<HashMap<String, (LocalId, ResolvedType)>>,
    immutable_locals: HashSet<LocalId>,
    next_local: u32,
    current_return_type: ResolvedType,
    loop_depth: usize,
}
impl<'a> TypedLowerer<'a> {
    fn new_with_pointer_width(p: &'a Program, _pointer_width: u32) -> Self {
        Self::new_with_ids(p, None)
    }

    fn new_with_graph(p: &'a Program, graph: &ModuleGraph, _pointer_width: u32) -> Self {
        let mut ids = HashMap::new();
        for definition in &graph.definitions {
            ids.insert(definition.source_name.clone(), definition.id);
            ids.insert(definition.linker_name.clone(), definition.id);
        }
        Self::new_with_ids(p, Some(ids))
    }

    fn new_with_ids(p: &'a Program, graph_ids: Option<HashMap<String, DefId>>) -> Self {
        let mut functions = HashMap::new();
        let mut structs = HashMap::new();
        let mut globals = HashMap::new();
        let mut constants = HashMap::new();
        let mut global_ids = HashMap::new();
        let mut definition_ids = HashMap::new();
        let mut next_def = graph_ids
            .as_ref()
            .and_then(|ids| ids.values().map(|id| id.0).max())
            .map_or(0, |id| id.saturating_add(1));
        for d in &p.declarations {
            let name = match d {
                Decl::Function(f) => &f.name,
                Decl::Struct(s) => &s.name,
                Decl::Variable(v) => &v.name,
                Decl::Comptime { .. } => continue,
            };
            let id = graph_ids
                .as_ref()
                .and_then(|ids| ids.get(name).copied())
                .unwrap_or_else(|| {
                    let id = DefId(next_def);
                    next_def = next_def.saturating_add(1);
                    id
                });
            definition_ids.insert(name.clone(), id);
            match d {
                Decl::Function(f) => {
                    functions.insert(f.name.clone(), (id, f));
                }
                Decl::Struct(s) => {
                    structs.insert(s.name.clone(), (id, s.clone()));
                }
                Decl::Variable(v) => {
                    let t =
                        v.ty.clone()
                            .unwrap_or_else(|| infer_ast_type_with_globals(&v.value, &globals));
                    let m = !matches!(v.kind, VariableKind::Immutable);
                    globals.insert(v.name.clone(), (t, m));
                    global_ids.insert(v.name.clone(), id);
                    if !m {
                        constants.insert(v.name.clone(), v);
                    }
                }
                Decl::Comptime { .. } => unreachable!(),
            }
        }
        Self {
            program: p,
            functions,
            structs,
            globals,
            global_ids,
            definition_ids,
            constants,
            scopes: Vec::new(),
            immutable_locals: HashSet::new(),
            next_local: 0,
            current_return_type: ResolvedType::Unit,
            loop_depth: 0,
        }
    }
    fn lower(mut self) -> Result<TypedProgram, SemanticError> {
        self.validate_program()?;
        let structs: Vec<TypedStruct> = self
            .program
            .declarations
            .iter()
            .filter_map(|declaration| match declaration {
                Decl::Struct(s) => Some((DefId(self.declaration_index(&s.name)), s)),
                _ => None,
            })
            .map(|(id, s)| TypedStruct {
                id,
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
                        id: DefId(self.declaration_index(&v.name)),
                        name: v.name.clone(),
                        ty,
                        value,
                        span: v.span,
                    })
                } else {
                    globals_out.push(TypedGlobal {
                        id: DefId(self.declaration_index(&v.name)),
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
        let symbols = self.symbol_table();
        Ok(TypedProgram {
            symbols,
            structs,
            globals: globals_out,
            constants: constants_out,
            functions,
        })
    }
    fn validate_program(&self) -> Result<(), SemanticError> {
        let mut names = HashSet::new();
        for declaration in &self.program.declarations {
            let (name, span) = match declaration {
                Decl::Function(function) => (&function.name, function.span),
                Decl::Struct(structure) => (&structure.name, structure.span),
                Decl::Variable(variable) => (&variable.name, variable.span),
                Decl::Comptime { .. } => continue,
            };
            if !names.insert(name.clone()) {
                return Err(SemanticError::DuplicateName {
                    name: name.clone(),
                    span,
                });
            }
            match declaration {
                Decl::Function(function) => {
                    if function.exported && function.abi.as_deref() != Some("c") {
                        return Err(SemanticError::InvalidAbi {
                            abi: "missing `c`".into(),
                            span: function.span,
                        });
                    }
                    if let Some(abi) = &function.abi
                        && abi != "c"
                    {
                        return Err(SemanticError::InvalidAbi {
                            abi: abi.clone(),
                            span: function.span,
                        });
                    }
                    for parameter in &function.params {
                        self.validate_value_type(&parameter.ty, parameter.span)?;
                    }
                    self.validate_type(&function.return_type, function.span)?;
                }
                Decl::Struct(structure) => {
                    let mut fields = HashSet::new();
                    for field in &structure.fields {
                        if !fields.insert(field.name.clone()) {
                            return Err(SemanticError::DuplicateName {
                                name: field.name.clone(),
                                span: field.span,
                            });
                        }
                        self.validate_value_type(&field.ty, field.span)?;
                    }
                }
                Decl::Variable(variable) => {
                    if let Some(ty) = &variable.ty {
                        self.validate_value_type(ty, variable.span)?;
                    }
                }
                Decl::Comptime { .. } => unreachable!(),
            }
        }
        Ok(())
    }

    fn validate_type(&self, ty: &Type, span: Span) -> Result<(), SemanticError> {
        match ty {
            Type::Unit => Ok(()),
            Type::Named(name)
                if matches!(
                    name.as_str(),
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
                ) || self.structs.contains_key(name) =>
            {
                Ok(())
            }
            Type::Named(name) => Err(SemanticError::UnknownType {
                name: name.clone(),
                span,
            }),
            Type::Array { element, .. } | Type::Slice(element) => {
                self.validate_value_type(element, span)
            }
            Type::Pointer(element) => {
                if matches!(element.as_ref(), Type::Unit)
                    || matches!(element.as_ref(), Type::Named(name) if name == "void")
                {
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

    fn validate_value_type(&self, ty: &Type, span: Span) -> Result<(), SemanticError> {
        if matches!(ty, Type::Unit) || matches!(ty, Type::Named(name) if name == "void") {
            return Err(SemanticError::UnknownType {
                name: "void".into(),
                span,
            });
        }
        self.validate_type(ty, span)
    }

    fn declaration_index(&self, name: &str) -> u32 {
        self.definition_ids
            .get(name)
            .copied()
            .unwrap_or(DefId(u32::MAX))
            .0
    }

    fn symbol_table(&self) -> crate::typed::SymbolTable {
        let definitions = self
            .program
            .declarations
            .iter()
            .filter_map(|declaration| {
                let (name, kind) = match declaration {
                    Decl::Function(f) => (&f.name, crate::typed::DefinitionKind::Function),
                    Decl::Struct(s) => (&s.name, crate::typed::DefinitionKind::Struct),
                    Decl::Variable(v) => (
                        &v.name,
                        if matches!(v.kind, VariableKind::Immutable) {
                            crate::typed::DefinitionKind::Constant
                        } else {
                            crate::typed::DefinitionKind::Global
                        },
                    ),
                    Decl::Comptime { .. } => return None,
                };
                Some(crate::typed::Definition {
                    id: self
                        .definition_ids
                        .get(name)
                        .copied()
                        .unwrap_or(DefId(u32::MAX)),
                    name: name.clone(),
                    kind,
                })
            })
            .collect();
        crate::typed::SymbolTable { definitions }
    }

    fn lower_const_expr(
        &mut self,
        e: &Expr,
        expected: Option<ResolvedType>,
    ) -> Result<TypedExpr, SemanticError> {
        let lowered = self.lower_expr(e, expected)?;
        let value = self.fold_constant(lowered)?;
        // Pure typed constants use the shared evaluator. Layout queries are
        // kept as typed operations because their target data belongs to the
        // backend-aware frontend, not to this target-neutral helper.
        if !contains_layout_query(&value) && !crate::typed_eval::is_constant(&value) {
            return Err(SemanticError::InvalidOperand {
                message: "global initializer is not supported by the typed evaluator".into(),
                span: e.span(),
            });
        }
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
        self.immutable_locals.clear();
        self.next_local = 0;
        self.loop_depth = 0;
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
        if !f.is_extern
            && self.current_return_type != ResolvedType::Unit
            && block_may_fall_through(&f.body)
        {
            return Err(SemanticError::MissingReturn {
                function: f.name.clone(),
                span: f.body.span,
            });
        }
        Ok(TypedFunction {
            id: self
                .functions
                .get(&f.name)
                .map(|(id, _)| *id)
                .ok_or_else(|| SemanticError::UndefinedName {
                    name: f.name.clone(),
                    span: f.span,
                })?,
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
            } => {
                let condition = self.lower_expr(condition, Some(ResolvedType::Bool))?;
                self.loop_depth += 1;
                let body = self.lower_block(body)?;
                self.loop_depth -= 1;
                TypedStmt::While {
                    condition,
                    body,
                    span: *span,
                }
            }
            Stmt::Break { span } => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakOutsideLoop { span: *span });
                }
                TypedStmt::Break { span: *span }
            }
            Stmt::Continue { span } => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::ContinueOutsideLoop { span: *span });
                }
                TypedStmt::Continue { span: *span }
            }
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
                if matches!(v.kind, VariableKind::Immutable) {
                    self.immutable_locals.insert(id);
                }
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
                if !self.place_is_mutable(&place) {
                    return Err(SemanticError::ImmutableAssignment {
                        name: place_name(target),
                        span: *span,
                    });
                }
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
                        id: self.global_ids[name],
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
                let Some(st) = self.structs.get(&s).map(|(_, structure)| structure) else {
                    return Err(SemanticError::UnknownType {
                        name: s,
                        span: *span,
                    });
                };
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
                if matches!(e, Expr::Index { .. })
                    && let Some(value) = self.constant_integer(index)
                    && (value.is_negative() || length.is_some_and(|bound| value as u64 >= bound))
                {
                    return Err(SemanticError::InvalidLiteral {
                        message: "array index is out of bounds".into(),
                        span: index.span(),
                    });
                }
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
    fn place_is_mutable(&self, place: &TypedPlace) -> bool {
        match place {
            TypedPlace::Local { id, .. } => !self.immutable_locals.contains(id),
            TypedPlace::Global { name, .. } => {
                self.globals.get(name).is_some_and(|(_, mutable)| *mutable)
            }
            TypedPlace::Temporary { .. } => false,
            TypedPlace::Field { base, .. } | TypedPlace::Index { base, .. } => {
                self.place_is_mutable(base)
            }
            TypedPlace::Dereference { .. } => true,
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
            ResolvedType::Struct(n) => named(self.struct_name(*n)),
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
            Expr::Integer { value, .. } => {
                let ty = expected
                    .as_ref()
                    .filter(|t| t.is_integer())
                    .cloned()
                    .unwrap_or(ResolvedType::Integer {
                        width: IntegerWidth::Bits(32),
                        signed: true,
                    });
                self.ensure_expected(expected.as_ref(), &ty, span)?;
                Ok(TypedExpr::Integer {
                    value: *value,
                    ty,
                    span,
                })
            }
            Expr::Float { value, .. } => {
                let ty = expected
                    .as_ref()
                    .filter(|t| matches!(t, ResolvedType::Float { .. }))
                    .cloned()
                    .unwrap_or(ResolvedType::Float { bits: 64 });
                self.ensure_expected(expected.as_ref(), &ty, span)?;
                Ok(TypedExpr::Float {
                    value: *value,
                    ty,
                    span,
                })
            }
            Expr::Bool { value, .. } => {
                self.ensure_expected(expected.as_ref(), &ResolvedType::Bool, span)?;
                Ok(TypedExpr::Bool {
                    value: *value,
                    ty: ResolvedType::Bool,
                    span,
                })
            }
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
                    self.ensure_expected(expected.as_ref(), &t, span)?;
                    return Ok(TypedExpr::Load {
                        id,
                        name: name.clone(),
                        ty: t,
                        span,
                    });
                }
                if let Some((t, m)) = self.globals.get(name) {
                    let ty = self.resolve(t);
                    self.ensure_expected(expected.as_ref(), &ty, span)?;
                    if *m {
                        return Ok(TypedExpr::GlobalLoad {
                            id: self.global_ids[name],
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
                let Some((struct_id, structure)) = self.structs.get(name).cloned() else {
                    return Err(SemanticError::UnknownType {
                        name: name.clone(),
                        span,
                    });
                };
                let mut out = Vec::new();
                let mut seen = std::collections::HashSet::new();
                for initializer in fields {
                    if !seen.insert(initializer.name.as_str()) {
                        return Err(SemanticError::DuplicateName {
                            name: initializer.name.clone(),
                            span: initializer.span,
                        });
                    }
                    if !structure
                        .fields
                        .iter()
                        .any(|field| field.name == initializer.name)
                    {
                        return Err(SemanticError::InvalidOperand {
                            message: format!("unknown field `{}`", initializer.name),
                            span: initializer.span,
                        });
                    }
                }
                for field in &structure.fields {
                    let Some(initializer) = fields.iter().find(|x| x.name == field.name) else {
                        return Err(SemanticError::InvalidOperand {
                            message: format!("missing field `{}` in `{name}`", field.name),
                            span,
                        });
                    };
                    out.push(self.lower_expr(&initializer.value, Some(self.resolve(&field.ty)))?)
                }
                Ok(TypedExpr::StructLiteral {
                    ty: ResolvedType::Struct(struct_id),
                    fields: out,
                    span,
                })
            }
            Expr::ArrayLiteral { ty, elements, .. } => {
                let rt = self.resolve(ty);
                let Type::Array { length, element } = ty else {
                    return Err(SemanticError::InvalidOperand {
                        message: "array literal requires an array type".into(),
                        span,
                    });
                };
                if elements.len() as u64 != *length {
                    return Err(SemanticError::InvalidOperand {
                        message: format!(
                            "array literal expects {length} elements, got {}",
                            elements.len()
                        ),
                        span,
                    });
                }
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
                    } else if matches!(left.as_ref(), Expr::Integer { .. } | Expr::Float { .. }) {
                        expected.as_ref().and_then(|t| {
                            (t.is_integer() || matches!(t, ResolvedType::Float { .. }))
                                .then(|| t.clone())
                        })
                    } else {
                        None
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
                let pointer_arithmetic = matches!(ot, ResolvedType::Pointer(_))
                    && matches!(operator, BinaryOp::Add | BinaryOp::Subtract)
                    && re.ty().is_integer();
                if !logical && !comparison && !pointer_arithmetic && re.ty() != ot {
                    return Err(SemanticError::TypeMismatch {
                        expected: self.ast_from_resolved(&ot),
                        found: self.ast_from_resolved(&re.ty()),
                        span: right.span(),
                    });
                }
                let result_ty = if logical || comparison {
                    ResolvedType::Bool
                } else if matches!(
                    (&ot, &re.ty()),
                    (ResolvedType::Pointer(_), ResolvedType::Pointer(_))
                ) && *operator == BinaryOp::Subtract
                {
                    intp(true)
                } else {
                    ot.clone()
                };
                self.ensure_expected(expected.as_ref(), &result_ty, span)?;
                Ok(TypedExpr::Binary {
                    left: Box::new(le),
                    operator: *operator,
                    right: Box::new(re),
                    ty: result_ty,
                    operand_type: ot,
                    span,
                })
            }
            Expr::Call {
                callee, arguments, ..
            } => {
                let Expr::Identifier { name, .. } = callee.as_ref() else {
                    return Err(SemanticError::NotCallable {
                        name: "<expression>".into(),
                        span,
                    });
                };
                if let Some(intrinsic) = Intrinsic::from_name(name).filter(|x| x.is_result()) {
                    let arg = arguments
                        .first()
                        .map(|a| self.lower_expr(a, None))
                        .transpose()?;
                    match intrinsic {
                        Intrinsic::IsErr => {
                            if arguments.len() != 1 {
                                return Err(SemanticError::WrongArgumentCount {
                                    name: name.clone(),
                                    expected: 1,
                                    found: arguments.len(),
                                    span,
                                });
                            }
                            return Ok(TypedExpr::IsErr {
                                value: Box::new(arg.ok_or_else(|| {
                                    SemanticError::WrongArgumentCount {
                                        name: name.clone(),
                                        expected: 1,
                                        found: 0,
                                        span,
                                    }
                                })?),
                                ty: ResolvedType::Bool,
                                span,
                            });
                        }
                        Intrinsic::Unwrap => {
                            if arguments.len() != 1 {
                                return Err(SemanticError::WrongArgumentCount {
                                    name: name.clone(),
                                    expected: 1,
                                    found: arguments.len(),
                                    span,
                                });
                            }
                            let arg = arg.ok_or_else(|| SemanticError::WrongArgumentCount {
                                name: name.clone(),
                                expected: 1,
                                found: 0,
                                span,
                            })?;
                            let ResolvedType::Result { success, .. } = arg.ty() else {
                                return Err(SemanticError::InvalidOperand {
                                    message: "unwrap requires a result value".into(),
                                    span,
                                });
                            };
                            return Ok(TypedExpr::Unwrap {
                                ty: *success,
                                value: Box::new(arg),
                                span,
                            });
                        }
                        Intrinsic::ReturnOk | Intrinsic::ReturnErr => {
                            let ResolvedType::Result { success, error } =
                                self.current_return_type.clone()
                            else {
                                return Err(SemanticError::InvalidPropagation {
                                    message: "return_ok/return_err requires a result return type"
                                        .into(),
                                    span,
                                });
                            };
                            let wanted = if intrinsic == Intrinsic::ReturnOk {
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
                            return Ok(if intrinsic == Intrinsic::ReturnOk {
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
                        _ => {
                            return Err(SemanticError::InvalidOperand {
                                message: "invalid result builtin".into(),
                                span,
                            });
                        }
                    }
                }
                if matches!(Intrinsic::from_name(name), Some(Intrinsic::MakeSlice)) {
                    if arguments.len() != 2 {
                        return Err(SemanticError::WrongArgumentCount {
                            name: name.clone(),
                            expected: 2,
                            found: arguments.len(),
                            span,
                        });
                    }
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
                        _ => {
                            return Err(SemanticError::InvalidOperand {
                                message: "make_slice requires a pointer argument".into(),
                                span,
                            });
                        }
                    };
                    return Ok(TypedExpr::MakeSlice {
                        pointer: Box::new(pointer),
                        length: Box::new(length),
                        element: Box::new(element.clone()),
                        ty: ResolvedType::Slice(Box::new(element)),
                        span,
                    });
                }
                if let Some(intrinsic) = Intrinsic::from_name(name).filter(|x| x.is_low_level()) {
                    let operation = match intrinsic {
                        Intrinsic::VolatileLoad => LowLevelOperation::VolatileLoad,
                        Intrinsic::VolatileStore => LowLevelOperation::VolatileStore,
                        Intrinsic::AtomicLoad => LowLevelOperation::AtomicLoad,
                        Intrinsic::AtomicStore => LowLevelOperation::AtomicStore,
                        Intrinsic::Fence => LowLevelOperation::Fence,
                        _ => unreachable!("low-level intrinsic registry is exhaustive"),
                    };
                    let expected_arguments = if matches!(intrinsic, Intrinsic::Fence) {
                        0
                    } else if matches!(intrinsic, Intrinsic::VolatileStore | Intrinsic::AtomicStore)
                    {
                        2
                    } else {
                        1
                    };
                    if arguments.len() != expected_arguments {
                        return Err(SemanticError::WrongArgumentCount {
                            name: name.clone(),
                            expected: expected_arguments,
                            found: arguments.len(),
                            span,
                        });
                    }
                    if matches!(intrinsic, Intrinsic::Fence) {
                        return Ok(TypedExpr::LowLevel {
                            operation,
                            arguments: Vec::new(),
                            ty: ResolvedType::Unit,
                            span,
                        });
                    }
                    let pointer = self.lower_expr(&arguments[0], None)?;
                    let ResolvedType::Pointer(element) = pointer.ty() else {
                        return Err(SemanticError::InvalidOperand {
                            message: format!("{name} requires a pointer argument"),
                            span,
                        });
                    };
                    let mut lowered = vec![pointer];
                    if expected_arguments == 2 {
                        lowered.push(self.lower_expr(&arguments[1], Some(*element.clone()))?);
                    }
                    let ty = if expected_arguments == 1 {
                        *element
                    } else {
                        ResolvedType::Unit
                    };
                    return Ok(TypedExpr::LowLevel {
                        operation,
                        arguments: lowered,
                        ty,
                        span,
                    });
                }
                let Some((id, f)) = self.functions.get(name).copied() else {
                    return Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span,
                    });
                };
                let pts = f
                    .params
                    .iter()
                    .map(|p| self.resolve(&p.ty))
                    .collect::<Vec<_>>();
                if arguments.len() != pts.len() {
                    return Err(SemanticError::WrongArgumentCount {
                        name: name.clone(),
                        expected: pts.len(),
                        found: arguments.len(),
                        span,
                    });
                }
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
    fn struct_name(&self, id: DefId) -> &str {
        self.structs
            .values()
            .find(|(struct_id, _)| *struct_id == id)
            .map(|(_, structure)| structure.name.as_str())
            .unwrap_or("<unknown-struct>")
    }

    fn new_local(
        &mut self,
        n: &str,
        t: ResolvedType,
        span: Span,
    ) -> Result<LocalId, SemanticError> {
        let id = LocalId(self.next_local);
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
    fn ensure_expected(
        &self,
        expected: Option<&ResolvedType>,
        actual: &ResolvedType,
        span: Span,
    ) -> Result<(), SemanticError> {
        if let Some(expected) = expected
            && expected != actual
        {
            eprintln!("lower mismatch expected={expected:?} actual={actual:?} at {span:?}");
            return Err(SemanticError::TypeMismatch {
                expected: self.ast_from_resolved(expected),
                found: self.ast_from_resolved(actual),
                span,
            });
        }
        Ok(())
    }

    fn constant_integer(&self, expression: &Expr) -> Option<i128> {
        match expression {
            Expr::Integer { value, .. } => Some(*value),
            Expr::Identifier { name, .. } => {
                self.constants
                    .get(name)
                    .and_then(|constant| match &constant.value {
                        Expr::Integer { value, .. } => Some(*value),
                        _ => None,
                    })
            }
            _ => None,
        }
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
                _ => ResolvedType::Struct(
                    self.structs
                        .get(n)
                        .map(|(id, _)| *id)
                        .unwrap_or(DefId(u32::MAX)),
                ),
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

fn block_may_fall_through(block: &Block) -> bool {
    let Some(last) = block.statements.last() else {
        return true;
    };
    match last {
        Stmt::Return { .. } | Stmt::Break { .. } | Stmt::Continue { .. } => false,
        Stmt::If {
            then_branch,
            else_branch: Some(else_branch),
            ..
        } => block_may_fall_through(then_branch) || block_may_fall_through(else_branch),
        Stmt::While {
            condition, body, ..
        } if matches!(condition, Expr::Bool { value: true, .. }) && !block_contains_break(body) => {
            false
        }
        _ => true,
    }
}

fn block_contains_break(block: &Block) -> bool {
    block.statements.iter().any(|statement| match statement {
        Stmt::Break { .. } => true,
        Stmt::If {
            then_branch,
            else_branch,
            ..
        } => {
            block_contains_break(then_branch)
                || else_branch.as_ref().is_some_and(block_contains_break)
        }
        Stmt::While { body, .. } => block_contains_break(body),
        _ => false,
    })
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
fn contains_layout_query(e: &TypedExpr) -> bool {
    match e {
        TypedExpr::Layout { .. } => true,
        TypedExpr::Unary { operand, .. } => contains_layout_query(operand),
        TypedExpr::Binary { left, right, .. } => {
            contains_layout_query(left) || contains_layout_query(right)
        }
        TypedExpr::StructLiteral { fields, .. } => fields.iter().any(contains_layout_query),
        TypedExpr::ArrayLiteral { elements, .. } => elements.iter().any(contains_layout_query),
        _ => false,
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::parse_source;

    fn check(source: &str) -> Result<TypedProgram, SemanticError> {
        let program = parse_source(source).expect("source should parse");
        analyze_typed(&program)
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
    fn typed_program_uses_stable_definition_ids() {
        let typed = check(
            "Pair :: struct { x: i32; }\nmain :: () -> i32 { p := Pair{x = 1}; return p.x; }",
        )
        .expect("program should type check");
        let pair = typed.symbols.find("Pair").expect("struct symbol");
        let main = typed.symbols.find("main").expect("function symbol");
        assert_eq!(typed.structs[0].id, pair.id);
        assert_eq!(typed.functions[0].id, main.id);
        assert_ne!(pair.id, main.id);
        assert_eq!(typed.symbols.get(pair.id).unwrap().name, "Pair");
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
