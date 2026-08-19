use crate::ast::{BinaryOp, UnaryOp};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutKind {
    Size,
    Align,
    Offset,
}
use crate::lexer::Span;

/// Stable identity for a declaration. IDs are compiler-owned; source names are
/// retained separately for diagnostics and linker metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DefId(pub u32);

/// Stable identity for a local binding within a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalId(pub u32);

/// Reserved for interned types and generic instantiations as those phases are
/// migrated to structured identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeId(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstantiationId(pub u32);

impl DefId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}
impl std::fmt::Display for DefId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "def#{}", self.0)
    }
}
impl LocalId {
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegerWidth {
    Bits(u16),
    Pointer,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedType {
    Unit,
    Bool,
    Integer {
        width: IntegerWidth,
        signed: bool,
    },
    Float {
        bits: u16,
    },
    Struct(DefId),
    Array {
        length: u64,
        element: Box<ResolvedType>,
    },
    Pointer(Box<ResolvedType>),
    Slice(Box<ResolvedType>),
    /// `{ i1, success, error }` in the LLVM backend. The tag is true for error.
    Result {
        success: Box<ResolvedType>,
        error: Box<ResolvedType>,
    },
}
impl ResolvedType {
    pub fn is_integer(&self) -> bool {
        matches!(self, Self::Integer { .. })
    }
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, Self::Integer { signed: true, .. })
    }
    pub fn is_aggregate(&self) -> bool {
        matches!(
            self,
            Self::Struct(_) | Self::Array { .. } | Self::Slice(_) | Self::Result { .. }
        )
    }
}

pub type FunctionId = DefId;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DefinitionKind {
    Function,
    Struct,
    Global,
    Constant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Definition {
    pub id: DefId,
    pub name: String,
    pub kind: DefinitionKind,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SymbolTable {
    pub definitions: Vec<Definition>,
}

impl SymbolTable {
    pub fn get(&self, id: DefId) -> Option<&Definition> {
        self.definitions
            .iter()
            .find(|definition| definition.id == id)
    }

    pub fn find(&self, name: &str) -> Option<&Definition> {
        self.definitions
            .iter()
            .find(|definition| definition.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedProgram {
    pub symbols: SymbolTable,
    pub structs: Vec<TypedStruct>,
    pub globals: Vec<TypedGlobal>,
    pub constants: Vec<TypedConstant>,
    pub functions: Vec<TypedFunction>,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedStruct {
    pub id: DefId,
    pub name: String,
    pub fields: Vec<TypedField>,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedField {
    pub name: String,
    pub ty: ResolvedType,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedGlobal {
    pub id: DefId,
    pub name: String,
    pub ty: ResolvedType,
    pub value: TypedExpr,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedConstant {
    pub id: DefId,
    pub name: String,
    pub ty: ResolvedType,
    pub value: TypedExpr,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFunction {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<TypedParameter>,
    pub return_type: ResolvedType,
    pub body: TypedBlock,
    pub span: Span,
    pub is_extern: bool,
    pub abi: Option<String>,
    pub link_name: Option<String>,
    pub exported: bool,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedParameter {
    pub id: LocalId,
    pub name: String,
    pub ty: ResolvedType,
    pub span: Span,
}
#[derive(Debug, Clone, PartialEq)]
pub struct TypedBlock {
    pub statements: Vec<TypedStmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedPlace {
    Local {
        id: LocalId,
        ty: ResolvedType,
    },
    Global {
        id: DefId,
        name: String,
        ty: ResolvedType,
    },
    /// A value expression materialized into storage so a nested field or
    /// index can use the same address-based lowering as an lvalue.
    Temporary {
        value: Box<TypedExpr>,
        ty: ResolvedType,
    },
    Field {
        base: Box<TypedPlace>,
        index: u32,
        ty: ResolvedType,
    },
    Index {
        base: Box<TypedPlace>,
        index: Box<TypedExpr>,
        ty: ResolvedType,
        length: Option<u64>,
        checked: bool,
    },
    Dereference {
        pointer: Box<TypedExpr>,
        ty: ResolvedType,
    },
}
impl TypedPlace {
    pub fn ty(&self) -> ResolvedType {
        match self {
            Self::Local { ty, .. }
            | Self::Global { ty, .. }
            | Self::Temporary { ty, .. }
            | Self::Field { ty, .. }
            | Self::Index { ty, .. }
            | Self::Dereference { ty, .. } => ty.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedStmt {
    If {
        condition: TypedExpr,
        then_branch: TypedBlock,
        else_branch: Option<TypedBlock>,
        span: Span,
    },
    While {
        condition: TypedExpr,
        body: TypedBlock,
        span: Span,
    },
    Break {
        span: Span,
    },
    Continue {
        span: Span,
    },
    Defer {
        function: FunctionId,
        name: String,
        arguments: Vec<TypedExpr>,
        span: Span,
    },
    Return {
        value: Option<TypedExpr>,
        span: Span,
    },
    Declare {
        id: LocalId,
        name: String,
        ty: ResolvedType,
        mutable: bool,
        value: TypedExpr,
        span: Span,
    },
    Store {
        target: TypedPlace,
        value: TypedExpr,
        ty: ResolvedType,
        span: Span,
    },
    Expr {
        expression: TypedExpr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedExpr {
    Integer {
        value: i128,
        ty: ResolvedType,
        span: Span,
    },
    Float {
        value: f64,
        ty: ResolvedType,
        span: Span,
    },
    Bool {
        value: bool,
        ty: ResolvedType,
        span: Span,
    },
    StructLiteral {
        ty: ResolvedType,
        fields: Vec<TypedExpr>,
        span: Span,
    },
    ArrayLiteral {
        ty: ResolvedType,
        elements: Vec<TypedExpr>,
        span: Span,
    },
    Load {
        id: LocalId,
        name: String,
        ty: ResolvedType,
        span: Span,
    },
    GlobalLoad {
        id: DefId,
        name: String,
        ty: ResolvedType,
        span: Span,
    },
    Field {
        place: TypedPlace,
        ty: ResolvedType,
        span: Span,
    },
    Index {
        place: TypedPlace,
        ty: ResolvedType,
        span: Span,
    },
    Unary {
        operator: UnaryOp,
        operand: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
    Binary {
        left: Box<TypedExpr>,
        operator: BinaryOp,
        right: Box<TypedExpr>,
        ty: ResolvedType,
        operand_type: ResolvedType,
        span: Span,
    },
    Call {
        function: FunctionId,
        name: String,
        arguments: Vec<TypedExpr>,
        parameter_types: Vec<ResolvedType>,
        ty: ResolvedType,
        span: Span,
    },
    Null {
        ty: ResolvedType,
        span: Span,
    },
    AddressOf {
        place: TypedPlace,
        ty: ResolvedType,
        span: Span,
    },
    Dereference {
        place: TypedPlace,
        ty: ResolvedType,
        span: Span,
    },
    Layout {
        kind: LayoutKind,
        ty: ResolvedType,
        target: ResolvedType,
        field: Option<String>,
        span: Span,
    },
    ResultOk {
        value: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
    ResultErr {
        value: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
    IsErr {
        value: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
    Unwrap {
        value: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
    Propagate {
        value: Box<TypedExpr>,
        ty: ResolvedType,
        span: Span,
    },
}
impl TypedExpr {
    pub fn ty(&self) -> ResolvedType {
        match self {
            Self::Integer { ty, .. }
            | Self::Float { ty, .. }
            | Self::Bool { ty, .. }
            | Self::StructLiteral { ty, .. }
            | Self::ArrayLiteral { ty, .. }
            | Self::Load { ty, .. }
            | Self::GlobalLoad { ty, .. }
            | Self::Field { ty, .. }
            | Self::Index { ty, .. }
            | Self::Unary { ty, .. }
            | Self::Binary { ty, .. }
            | Self::Call { ty, .. }
            | Self::Null { ty, .. }
            | Self::AddressOf { ty, .. }
            | Self::Dereference { ty, .. }
            | Self::Layout { ty, .. }
            | Self::ResultOk { ty, .. }
            | Self::ResultErr { ty, .. }
            | Self::IsErr { ty, .. }
            | Self::Unwrap { ty, .. }
            | Self::Propagate { ty, .. } => ty.clone(),
        }
    }
    pub fn span(&self) -> Span {
        match self {
            Self::Integer { span, .. }
            | Self::Float { span, .. }
            | Self::Bool { span, .. }
            | Self::StructLiteral { span, .. }
            | Self::ArrayLiteral { span, .. }
            | Self::Load { span, .. }
            | Self::GlobalLoad { span, .. }
            | Self::Field { span, .. }
            | Self::Index { span, .. }
            | Self::Unary { span, .. }
            | Self::Binary { span, .. }
            | Self::Call { span, .. }
            | Self::Null { span, .. }
            | Self::AddressOf { span, .. }
            | Self::Dereference { span, .. }
            | Self::Layout { span, .. }
            | Self::ResultOk { span, .. }
            | Self::ResultErr { span, .. }
            | Self::IsErr { span, .. }
            | Self::Unwrap { span, .. }
            | Self::Propagate { span, .. } => *span,
        }
    }
}
