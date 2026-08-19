use crate::ast::{BinaryOp, UnaryOp};
use crate::lexer::Span;

/// A primitive type after semantic resolution.  `Pointer` widths are resolved
/// by the backend for the selected target, while signedness is retained here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerWidth {
    Bits(u16),
    Pointer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedType {
    Unit,
    Bool,
    Integer { width: IntegerWidth, signed: bool },
    Float { bits: u16 },
}

impl ResolvedType {
    pub fn is_integer(self) -> bool {
        matches!(self, Self::Integer { .. })
    }

    pub fn is_signed_integer(self) -> bool {
        matches!(self, Self::Integer { signed: true, .. })
    }
}

pub type FunctionId = usize;
pub type LocalId = usize;

#[derive(Debug, Clone, PartialEq)]
pub struct TypedProgram {
    pub functions: Vec<TypedFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedFunction {
    pub id: FunctionId,
    pub name: String,
    pub params: Vec<TypedParameter>,
    pub return_type: ResolvedType,
    pub body: TypedBlock,
    pub span: Span,
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
        id: LocalId,
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
    /// A resolved local load.  The source name is retained for diagnostics,
    /// but backends use `id`, never the spelling, to identify the binding.
    Load {
        id: LocalId,
        name: String,
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
}

impl TypedExpr {
    pub fn ty(&self) -> ResolvedType {
        match self {
            Self::Integer { ty, .. }
            | Self::Float { ty, .. }
            | Self::Bool { ty, .. }
            | Self::Load { ty, .. }
            | Self::Unary { ty, .. }
            | Self::Binary { ty, .. }
            | Self::Call { ty, .. } => *ty,
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Self::Integer { span, .. }
            | Self::Float { span, .. }
            | Self::Bool { span, .. }
            | Self::Load { span, .. }
            | Self::Unary { span, .. }
            | Self::Binary { span, .. }
            | Self::Call { span, .. } => *span,
        }
    }
}
