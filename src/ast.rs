use crate::lexer::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub declarations: Vec<Decl>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function(FunctionDecl),
    Variable(VariableDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub kind: VariableKind,
    pub ty: Option<Type>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    MutableInferred, // :=
    MutableTyped,    // : T =
    Immutable,       // :: expr
}

#[derive(Debug, Clone, PartialEq)]
pub struct Parameter {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub statements: Vec<Stmt>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    If {
        condition: Expr,
        then_branch: Block,
        else_branch: Option<Block>,
        span: Span,
    },

    While {
        condition: Expr,
        body: Block,
        span: Span,
    },

    Break {
        span: Span,
    },

    Continue {
        span: Span,
    },

    Return {
        value: Option<Expr>,
        span: Span,
    },

    Variable(VariableDecl),

    Assignment {
        target: Expr,
        value: Expr,
        span: Span,
    },

    Expr {
        expression: Expr,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Integer {
        value: i128,
        span: Span,
    },

    Float {
        value: f64,
        span: Span,
    },

    Bool {
        value: bool,
        span: Span,
    },

    Identifier {
        name: String,
        span: Span,
    },

    Unary {
        operator: UnaryOp,
        operand: Box<Expr>,
        span: Span,
    },

    Binary {
        left: Box<Expr>,
        operator: BinaryOp,
        right: Box<Expr>,
        span: Span,
    },

    Call {
        callee: Box<Expr>,
        arguments: Vec<Expr>,
        span: Span,
    },
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Integer { span, .. } => *span,
            Expr::Float { span, .. } => *span,
            Expr::Bool { span, .. } => *span,
            Expr::Identifier { span, .. } => *span,
            Expr::Unary { span, .. } => *span,
            Expr::Binary { span, .. } => *span,
            Expr::Call { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
    BitwiseNot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,

    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,

    LogicalAnd,
    LogicalOr,

    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Unit,
    Named(String),
}
