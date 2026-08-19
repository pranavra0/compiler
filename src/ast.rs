use crate::lexer::Span;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub imports: Vec<ImportDecl>,
    pub declarations: Vec<Decl>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportDecl {
    /// A path such as `os` or `os.compy`, relative to the importing file.
    pub path: String,
    /// The namespace used at call sites.  The first implementation defaults
    /// this to the final path component.
    pub alias: String,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Function(FunctionDecl),
    Variable(VariableDecl),
    Struct(StructDecl),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructField>,
    pub span: Span,
    pub exported: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Parameter>,
    pub return_type: Type,
    pub body: Block,
    pub span: Span,
    /// `extern` declarations have no source body but use the same signature
    /// representation as ordinary functions.
    pub is_extern: bool,
    pub abi: Option<String>,
    /// External symbol name, when it differs from the source/module name.
    pub link_name: Option<String>,
    pub exported: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VariableDecl {
    pub name: String,
    pub kind: VariableKind,
    pub ty: Option<Type>,
    pub value: Expr,
    pub span: Span,
    pub exported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    MutableInferred,
    MutableTyped,
    Immutable,
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
    Defer {
        call: Expr,
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
    Null {
        span: Span,
    },
    SizeOf {
        ty: Type,
        span: Span,
    },
    AlignOf {
        ty: Type,
        span: Span,
    },
    OffsetOf {
        ty: Type,
        field: String,
        span: Span,
    },
    UncheckedIndex {
        base: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    Identifier {
        name: String,
        span: Span,
    },
    StructLiteral {
        name: String,
        fields: Vec<StructInit>,
        span: Span,
    },
    ArrayLiteral {
        ty: Type,
        elements: Vec<Expr>,
        span: Span,
    },
    Field {
        base: Box<Expr>,
        name: String,
        span: Span,
    },
    Index {
        base: Box<Expr>,
        index: Box<Expr>,
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
    Propagate {
        expression: Box<Expr>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct StructInit {
    pub name: String,
    pub value: Expr,
    pub span: Span,
}

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Integer { span, .. }
            | Expr::Float { span, .. }
            | Expr::Bool { span, .. }
            | Expr::Null { span }
            | Expr::SizeOf { span, .. }
            | Expr::AlignOf { span, .. }
            | Expr::OffsetOf { span, .. }
            | Expr::UncheckedIndex { span, .. }
            | Expr::Identifier { span, .. }
            | Expr::StructLiteral { span, .. }
            | Expr::ArrayLiteral { span, .. }
            | Expr::Field { span, .. }
            | Expr::Index { span, .. }
            | Expr::Unary { span, .. }
            | Expr::Binary { span, .. }
            | Expr::Call { span, .. }
            | Expr::Propagate { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
    BitwiseNot,
    AddressOf,
    Dereference,
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
    Array {
        length: u64,
        element: Box<Type>,
    },
    Pointer(Box<Type>),
    Slice(Box<Type>),
    /// A tagged result value. Exactly one of success or error is active.
    Result {
        success: Box<Type>,
        error: Box<Type>,
    },
}
