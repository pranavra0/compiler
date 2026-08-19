use std::collections::HashMap;
use std::fmt;

use crate::ast::{
    BinaryOp, Block, Decl, Expr, FunctionDecl, Program, Stmt, Type, UnaryOp, VariableDecl,
    VariableKind,
};
use crate::lexer::Span;

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
    MissingReturn {
        function: String,
        span: Span,
    },
    TopLevelVariableUnsupported {
        name: String,
        span: Span,
    },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UndefinedName { name, span } => {
                write!(f, "undefined name `{name}` at {}..{}", span.start, span.end)
            }
            Self::DuplicateName { name, span } => {
                write!(f, "duplicate name `{name}` at {}..{}", span.start, span.end)
            }
            Self::UnknownType { name, span } => {
                write!(f, "unknown type `{name}` at {}..{}", span.start, span.end)
            }
            Self::TypeMismatch {
                expected,
                found,
                span,
            } => write!(
                f,
                "type mismatch: expected {}, found {} at {}..{}",
                type_name(expected),
                type_name(found),
                span.start,
                span.end
            ),
            Self::InvalidLiteral { message, span } => {
                write!(f, "{message} at {}..{}", span.start, span.end)
            }
            Self::InvalidOperand { message, span } => {
                write!(f, "{message} at {}..{}", span.start, span.end)
            }
            Self::WrongArgumentCount {
                name,
                expected,
                found,
                span,
            } => write!(
                f,
                "function `{name}` expects {expected} arguments, got {found} at {}..{}",
                span.start, span.end
            ),
            Self::NotCallable { name, span } => {
                write!(
                    f,
                    "`{name}` is not a function at {}..{}",
                    span.start, span.end
                )
            }
            Self::ImmutableAssignment { name, span } => write!(
                f,
                "cannot assign to immutable variable `{name}` at {}..{}",
                span.start, span.end
            ),
            Self::InvalidAssignmentTarget { span } => {
                write!(
                    f,
                    "invalid assignment target at {}..{}",
                    span.start, span.end
                )
            }
            Self::BreakOutsideLoop { span } => write!(
                f,
                "break is only valid inside a loop at {}..{}",
                span.start, span.end
            ),
            Self::ContinueOutsideLoop { span } => write!(
                f,
                "continue is only valid inside a loop at {}..{}",
                span.start, span.end
            ),
            Self::MissingReturn { function, span } => write!(
                f,
                "function `{function}` does not return a value on every path at {}..{}",
                span.start, span.end
            ),
            Self::TopLevelVariableUnsupported { name, span } => write!(
                f,
                "top-level variable `{name}` is not supported yet at {}..{}",
                span.start, span.end
            ),
        }
    }
}

impl std::error::Error for SemanticError {}

#[derive(Debug, Clone)]
struct FunctionSignature {
    parameters: Vec<Type>,
    return_type: Type,
}

#[derive(Debug, Clone)]
struct Variable {
    ty: Type,
    mutable: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Flow(u8);

impl Flow {
    const NORMAL: Self = Self(1);
    const RETURN: Self = Self(2);
    const BREAK: Self = Self(4);
    const CONTINUE: Self = Self(8);

    fn contains(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    fn without(self, other: Self) -> Self {
        Self(self.0 & !other.0)
    }
}

/// Performs name resolution and the type rules which are independent of a
/// backend.  The analyzer deliberately does not rewrite the AST yet; its
/// result is a validated program ready for lowering.
pub struct Analyzer {
    functions: HashMap<String, FunctionSignature>,
    scopes: Vec<HashMap<String, Variable>>,
    current_return_type: Option<Type>,
    current_function: Option<String>,
    loop_depth: usize,
}

/// Analyze a complete program using the language's semantic rules.
pub fn analyze(program: &Program) -> Result<(), SemanticError> {
    Analyzer::new().analyze(program)
}

impl Analyzer {
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            scopes: Vec::new(),
            current_return_type: None,
            current_function: None,
            loop_depth: 0,
        }
    }

    pub fn analyze(mut self, program: &Program) -> Result<(), SemanticError> {
        self.collect_signatures(program)?;

        for declaration in &program.declarations {
            match declaration {
                Decl::Function(function) => self.analyze_function(function)?,
                Decl::Variable(variable) => {
                    return Err(SemanticError::TopLevelVariableUnsupported {
                        name: variable.name.clone(),
                        span: variable.span,
                    });
                }
            }
        }

        Ok(())
    }

    /// Collect every function before checking any body. This permits forward
    /// calls and makes calls use the declared signature, not codegen guesses.
    fn collect_signatures(&mut self, program: &Program) -> Result<(), SemanticError> {
        let mut names = HashMap::<String, Span>::new();

        for declaration in &program.declarations {
            let (name, span) = match declaration {
                Decl::Function(function) => (&function.name, function.span),
                Decl::Variable(variable) => (&variable.name, variable.span),
            };

            if names.insert(name.clone(), span).is_some() {
                return Err(SemanticError::DuplicateName {
                    name: name.clone(),
                    span,
                });
            }

            if let Decl::Function(function) = declaration {
                for parameter in &function.params {
                    self.validate_value_type(&parameter.ty, parameter.span)?;
                }
                self.validate_return_type(&function.return_type, function.span)?;

                self.functions.insert(
                    function.name.clone(),
                    FunctionSignature {
                        parameters: function
                            .params
                            .iter()
                            .map(|parameter| parameter.ty.clone())
                            .collect(),
                        return_type: function.return_type.clone(),
                    },
                );
            }
        }

        Ok(())
    }

    fn analyze_function(&mut self, function: &FunctionDecl) -> Result<(), SemanticError> {
        self.current_function = Some(function.name.clone());
        self.current_return_type = Some(function.return_type.clone());
        self.scopes.push(HashMap::new());

        for parameter in &function.params {
            self.declare_local(
                &parameter.name,
                Variable {
                    ty: parameter.ty.clone(),
                    mutable: true,
                },
                parameter.span,
            )?;
        }

        // Parameters and declarations directly in the function body share one
        // lexical scope, so a local cannot silently redeclare a parameter.
        self.loop_depth = 0;
        let flow = self.analyze_block_contents(&function.body)?;
        self.scopes.pop();

        self.current_return_type = None;
        self.current_function = None;
        self.loop_depth = 0;

        if !is_unit(&function.return_type) && flow.contains(Flow::NORMAL) {
            return Err(SemanticError::MissingReturn {
                function: function.name.clone(),
                span: function.body.span,
            });
        }

        Ok(())
    }

    fn analyze_block(&mut self, block: &Block) -> Result<Flow, SemanticError> {
        self.scopes.push(HashMap::new());
        let result = self.analyze_block_contents(block);
        self.scopes.pop();
        result
    }

    fn analyze_block_contents(&mut self, block: &Block) -> Result<Flow, SemanticError> {
        let mut flow = Flow::NORMAL;

        for statement in &block.statements {
            let statement_flow = self.analyze_statement(statement)?;
            if flow.contains(Flow::NORMAL) {
                flow = flow.without(Flow::NORMAL).union(statement_flow);
            }
        }

        Ok(flow)
    }

    fn analyze_statement(&mut self, statement: &Stmt) -> Result<Flow, SemanticError> {
        match statement {
            Stmt::Variable(variable) => {
                let ty = self.variable_type(variable)?;
                self.declare_local(
                    &variable.name,
                    Variable {
                        ty,
                        mutable: !matches!(variable.kind, VariableKind::Immutable),
                    },
                    variable.span,
                )?;
                Ok(Flow::NORMAL)
            }

            Stmt::Assignment { target, value, .. } => {
                let Expr::Identifier { name, span } = target else {
                    return Err(SemanticError::InvalidAssignmentTarget {
                        span: target.span(),
                    });
                };

                let variable = self.lookup_variable(name).ok_or_else(|| {
                    if self.functions.contains_key(name) {
                        SemanticError::InvalidAssignmentTarget { span: *span }
                    } else {
                        SemanticError::UndefinedName {
                            name: name.clone(),
                            span: *span,
                        }
                    }
                })?;

                if !variable.mutable {
                    return Err(SemanticError::ImmutableAssignment {
                        name: name.clone(),
                        span: *span,
                    });
                }

                let value_type = self.check_expression(value, Some(&variable.ty))?;
                self.expect_type(&variable.ty, &value_type, value.span())?;
                Ok(Flow::NORMAL)
            }

            Stmt::Expr { expression, .. } => {
                self.check_expression(expression, None)?;
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
                let expected = self
                    .current_return_type
                    .as_ref()
                    .expect("return outside function")
                    .clone();
                match (is_unit(&expected), value) {
                    (true, None) => Ok(Flow::RETURN),
                    (true, Some(expression)) => Err(SemanticError::TypeMismatch {
                        expected: Type::Unit,
                        found: self.check_expression(expression, None)?,
                        span: *span,
                    }),
                    (false, None) => Err(SemanticError::TypeMismatch {
                        expected: expected.clone(),
                        found: Type::Unit,
                        span: *span,
                    }),
                    (false, Some(expression)) => {
                        let actual = self.check_expression(expression, Some(&expected))?;
                        self.expect_type(&expected, &actual, expression.span())?;
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
                let condition_type = self.check_expression(condition, Some(&named("bool")))?;
                self.expect_type(&named("bool"), &condition_type, condition.span())?;
                let then_flow = self.analyze_block(then_branch)?;
                let else_flow = match else_branch {
                    Some(block) => self.analyze_block(block)?,
                    None => Flow::NORMAL,
                };
                Ok(then_flow.union(else_flow))
            }

            Stmt::While {
                condition, body, ..
            } => {
                let condition_type = self.check_expression(condition, Some(&named("bool")))?;
                self.expect_type(&named("bool"), &condition_type, condition.span())?;

                let statically_true = matches!(condition, Expr::Bool { value: true, .. });
                self.loop_depth += 1;
                let body_result = self.analyze_block(body);
                self.loop_depth -= 1;
                let body_flow = body_result?;

                let mut flow = if body_flow.contains(Flow::RETURN) {
                    Flow::RETURN
                } else {
                    Flow(0)
                };
                if !statically_true || body_flow.contains(Flow::BREAK) {
                    flow = flow.union(Flow::NORMAL);
                }
                Ok(flow)
            }
        }
    }

    fn variable_type(&mut self, variable: &VariableDecl) -> Result<Type, SemanticError> {
        if let Some(ty) = &variable.ty {
            self.validate_value_type(ty, variable.span)?;
        }

        let ty = match &variable.ty {
            Some(ty) => self.check_expression(&variable.value, Some(ty))?,
            None => self.check_expression(&variable.value, None)?,
        };

        if let Some(expected) = &variable.ty {
            self.expect_type(expected, &ty, variable.value.span())?;
        }

        Ok(ty)
    }

    fn check_expression(
        &mut self,
        expression: &Expr,
        expected: Option<&Type>,
    ) -> Result<Type, SemanticError> {
        let ty = match expression {
            Expr::Integer { value, span } => {
                let ty = expected
                    .filter(|ty| is_integer(ty))
                    .cloned()
                    .unwrap_or_else(|| named("i32"));
                if !integer_fits(*value, &ty) {
                    return Err(SemanticError::InvalidLiteral {
                        message: format!("integer literal does not fit in {}", type_name(&ty)),
                        span: *span,
                    });
                }
                ty
            }

            Expr::Float { value, span } => {
                let ty = expected
                    .filter(|ty| is_float(ty))
                    .cloned()
                    .unwrap_or_else(|| named("f64"));
                if matches!(type_name(&ty), "f32") && !(*value as f32).is_finite() {
                    return Err(SemanticError::InvalidLiteral {
                        message: "floating-point literal does not fit in f32".to_string(),
                        span: *span,
                    });
                }
                ty
            }

            Expr::Bool { .. } => named("bool"),

            Expr::Identifier { name, span } => {
                let Some(variable) = self.lookup_variable(name) else {
                    if self.functions.contains_key(name) {
                        return Err(SemanticError::InvalidOperand {
                            message: format!("function `{name}` cannot be used as a value"),
                            span: *span,
                        });
                    }
                    return Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *span,
                    });
                };
                variable.ty
            }

            Expr::Unary {
                operator,
                operand,
                span,
            } => {
                let operand_expected = match operator {
                    UnaryOp::Not => Some(named("bool")),
                    _ => expected.filter(|ty| is_numeric(ty)).cloned(),
                };
                let operand_type = self.check_expression(operand, operand_expected.as_ref())?;
                match operator {
                    UnaryOp::Negate
                        if !is_signed_integer(&operand_type) && !is_float(&operand_type) =>
                    {
                        return Err(SemanticError::InvalidOperand {
                            message: "negation requires a signed integer or floating-point operand"
                                .to_string(),
                            span: *span,
                        });
                    }
                    UnaryOp::Not if !is_bool(&operand_type) => {
                        return Err(SemanticError::InvalidOperand {
                            message: "logical not requires a boolean operand".to_string(),
                            span: *span,
                        });
                    }
                    UnaryOp::BitwiseNot if !is_integer(&operand_type) => {
                        return Err(SemanticError::InvalidOperand {
                            message: "bitwise not requires an integer operand".to_string(),
                            span: *span,
                        });
                    }
                    _ => {}
                }
                operand_type
            }

            Expr::Binary {
                left,
                operator,
                right,
                span,
            } => self.check_binary_expression(left, *operator, right, *span, expected)?,

            Expr::Call {
                callee,
                arguments,
                span,
            } => {
                let Expr::Identifier {
                    name,
                    span: callee_span,
                } = callee.as_ref()
                else {
                    return Err(SemanticError::InvalidOperand {
                        message: "only named functions can be called currently".to_string(),
                        span: callee.span(),
                    });
                };

                // A local binding shadows a function with the same spelling.
                // Resolve the lexical variable namespaces before the global
                // function namespace.
                if self.lookup_variable(name).is_some() {
                    return Err(SemanticError::NotCallable {
                        name: name.clone(),
                        span: *callee_span,
                    });
                }

                let Some(signature) = self.functions.get(name).cloned() else {
                    return Err(SemanticError::UndefinedName {
                        name: name.clone(),
                        span: *callee_span,
                    });
                };

                if signature.parameters.len() != arguments.len() {
                    return Err(SemanticError::WrongArgumentCount {
                        name: name.clone(),
                        expected: signature.parameters.len(),
                        found: arguments.len(),
                        span: *span,
                    });
                }

                for (argument, parameter_type) in arguments.iter().zip(&signature.parameters) {
                    let actual = self.check_expression(argument, Some(parameter_type))?;
                    self.expect_type(parameter_type, &actual, argument.span())?;
                }

                signature.return_type
            }
        };

        if let Some(expected) = expected {
            self.expect_type(expected, &ty, expression.span())?;
        }
        Ok(ty)
    }

    fn check_binary_expression(
        &mut self,
        left: &Expr,
        operator: BinaryOp,
        right: &Expr,
        span: Span,
        expected: Option<&Type>,
    ) -> Result<Type, SemanticError> {
        if matches!(operator, BinaryOp::LogicalAnd | BinaryOp::LogicalOr) {
            let bool_type = named("bool");
            let left_type = self.check_expression(left, Some(&bool_type))?;
            let right_type = self.check_expression(right, Some(&bool_type))?;
            self.expect_type(&bool_type, &left_type, left.span())?;
            self.expect_type(&bool_type, &right_type, right.span())?;
            return Ok(bool_type);
        }

        if matches!(
            operator,
            BinaryOp::Equal
                | BinaryOp::NotEqual
                | BinaryOp::Less
                | BinaryOp::LessEqual
                | BinaryOp::Greater
                | BinaryOp::GreaterEqual
        ) {
            let left_type = self.check_expression(left, None)?;
            let right_type = self.check_expression(right, Some(&left_type))?;
            self.expect_type(&left_type, &right_type, right.span())?;
            let valid = if matches!(operator, BinaryOp::Equal | BinaryOp::NotEqual) {
                (is_numeric(&left_type) && is_numeric(&right_type))
                    || (is_bool(&left_type) && is_bool(&right_type))
            } else {
                is_numeric(&left_type) && is_numeric(&right_type)
            };
            if !valid {
                return Err(SemanticError::InvalidOperand {
                    message: "comparison requires operands of the same numeric type (or bool for == and !=)".to_string(),
                    span,
                });
            }
            return Ok(named("bool"));
        }

        let operand_expected = expected.filter(|ty| is_numeric(ty));
        let left_type = self.check_expression(left, operand_expected)?;
        let right_type = self.check_expression(right, Some(&left_type))?;
        self.expect_type(&left_type, &right_type, right.span())?;

        let valid = match operator {
            BinaryOp::Add
            | BinaryOp::Subtract
            | BinaryOp::Multiply
            | BinaryOp::Divide
            | BinaryOp::Modulo => is_numeric(&left_type),
            BinaryOp::BitwiseAnd
            | BinaryOp::BitwiseOr
            | BinaryOp::BitwiseXor
            | BinaryOp::ShiftLeft
            | BinaryOp::ShiftRight => is_integer(&left_type),
            _ => false,
        };

        if !valid {
            return Err(SemanticError::InvalidOperand {
                message: "operator requires operands of the same explicit numeric type".to_string(),
                span,
            });
        }

        Ok(left_type)
    }

    fn declare_local(
        &mut self,
        name: &str,
        variable: Variable,
        span: Span,
    ) -> Result<(), SemanticError> {
        let scope = self
            .scopes
            .last_mut()
            .expect("a local scope is always active");
        if scope.contains_key(name) {
            return Err(SemanticError::DuplicateName {
                name: name.to_string(),
                span,
            });
        }
        scope.insert(name.to_string(), variable);
        Ok(())
    }

    fn lookup_variable(&self, name: &str) -> Option<Variable> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).cloned())
    }

    fn expect_type(&self, expected: &Type, found: &Type, span: Span) -> Result<(), SemanticError> {
        if expected == found {
            Ok(())
        } else {
            Err(SemanticError::TypeMismatch {
                expected: expected.clone(),
                found: found.clone(),
                span,
            })
        }
    }

    fn validate_value_type(&self, ty: &Type, span: Span) -> Result<(), SemanticError> {
        if is_unit(ty) {
            return Err(SemanticError::UnknownType {
                name: type_name(ty).to_string(),
                span,
            });
        }
        self.validate_type(ty, span)
    }

    fn validate_return_type(&self, ty: &Type, span: Span) -> Result<(), SemanticError> {
        self.validate_type(ty, span)
    }

    fn validate_type(&self, ty: &Type, span: Span) -> Result<(), SemanticError> {
        if matches!(ty, Type::Unit) || is_known_type(ty) {
            Ok(())
        } else {
            Err(SemanticError::UnknownType {
                name: type_name(ty).to_string(),
                span,
            })
        }
    }
}

fn named(name: &str) -> Type {
    Type::Named(name.to_string())
}

fn type_name(ty: &Type) -> &str {
    match ty {
        Type::Unit => "void",
        Type::Named(name) => name,
    }
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Unit) || matches!(ty, Type::Named(name) if name == "void")
}

fn is_known_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Named(name)
            if matches!(name.as_str(), "bool" | "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "f32" | "f64" | "usize" | "isize" | "void")
    )
}

fn is_bool(ty: &Type) -> bool {
    matches!(ty, Type::Named(name) if name == "bool")
}

fn is_integer(ty: &Type) -> bool {
    matches!(ty, Type::Named(name) if matches!(name.as_str(), "i8" | "i16" | "i32" | "i64" | "i128" | "u8" | "u16" | "u32" | "u64" | "u128" | "usize" | "isize"))
}

fn is_signed_integer(ty: &Type) -> bool {
    matches!(ty, Type::Named(name) if matches!(name.as_str(), "i8" | "i16" | "i32" | "i64" | "i128" | "isize"))
}

fn is_float(ty: &Type) -> bool {
    matches!(ty, Type::Named(name) if name == "f32" || name == "f64")
}

fn is_numeric(ty: &Type) -> bool {
    is_integer(ty) || is_float(ty)
}

fn integer_fits(value: i128, ty: &Type) -> bool {
    match type_name(ty) {
        "i8" => i8::try_from(value).is_ok(),
        "i16" => i16::try_from(value).is_ok(),
        "i32" => i32::try_from(value).is_ok(),
        "i64" => i64::try_from(value).is_ok(),
        "i128" => true,
        "u8" => u8::try_from(value).is_ok(),
        "u16" => u16::try_from(value).is_ok(),
        "u32" => u32::try_from(value).is_ok(),
        "u64" => u64::try_from(value).is_ok(),
        "u128" => value >= 0,
        "usize" => usize::try_from(value).is_ok(),
        "isize" => isize::try_from(value).is_ok(),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::{Lexer, TokenKind};
    use crate::parser::Parser;

    fn analyze_source(source: &str) -> Result<(), SemanticError> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            let eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if eof {
                break;
            }
        }
        let program = Parser::new(tokens).parse().unwrap();
        analyze(&program)
    }

    #[test]
    fn checks_names_and_mutability() {
        assert!(matches!(
            analyze_source("answer :: 42; main :: () -> i32 { return 0; }"),
            Err(SemanticError::TopLevelVariableUnsupported { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { x := 1; x = 2; return x; }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { x :: 1; x = 2; return x; }"),
            Err(SemanticError::ImmutableAssignment { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { return missing; }"),
            Err(SemanticError::UndefinedName { .. })
        ));
    }

    #[test]
    fn checks_signatures_and_returns() {
        assert!(matches!(
            analyze_source(
                "add :: (a: i64, b: i64) -> i64 { return a + b; } main :: () -> i64 { return add(1, 2); }"
            ),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { return add(1); }"),
            Err(SemanticError::UndefinedName { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i64 { x: i32 = 1; return x; }"),
            Err(SemanticError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn rejects_implicit_numeric_conversions() {
        assert!(matches!(
            analyze_source("main :: () -> i64 { x: i64 = 1; y: i32 = 2; return x + y; }"),
            Err(SemanticError::TypeMismatch { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { x: f64 = 1.0; return x; }"),
            Err(SemanticError::TypeMismatch { .. })
        ));
    }

    #[test]
    fn nested_scopes_can_shadow_but_not_duplicate() {
        assert!(matches!(
            analyze_source("main :: () -> i32 { x := 1; if true { x := 2; } return x; }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { x := 1; x := 2; return x; }"),
            Err(SemanticError::DuplicateName { .. })
        ));
        assert!(matches!(
            analyze_source("main :: (x: i32) -> i32 { x := 2; return x; }"),
            Err(SemanticError::DuplicateName { .. })
        ));
        assert!(matches!(
            analyze_source(
                "f :: () -> i32 { return 1; } main :: () -> i32 { f := 2; return f(); }"
            ),
            Err(SemanticError::NotCallable { .. })
        ));
    }

    #[test]
    fn checks_loop_context_and_conditions() {
        assert!(matches!(
            analyze_source("main :: () -> i32 { while true { break; } return 0; }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { while 1 { break; } return 0; }"),
            Err(SemanticError::TypeMismatch { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { break; return 0; }"),
            Err(SemanticError::BreakOutsideLoop { .. })
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { continue; return 0; }"),
            Err(SemanticError::ContinueOutsideLoop { .. })
        ));
        assert!(matches!(
            analyze_source(
                "main :: () -> i32 { while true { if true { break; } if false { continue; } } return 0; }"
            ),
            Ok(())
        ));
    }

    #[test]
    fn loop_flow_handles_nested_loops_and_returns() {
        assert!(matches!(
            analyze_source("main :: () -> i32 { while true { while true { break; } continue; } }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("missing :: (flag: bool) -> i32 { while flag { return 1; } }"),
            Err(SemanticError::MissingReturn { .. })
        ));
        assert!(matches!(
            analyze_source("missing :: () -> i32 { while true { break; } }"),
            Err(SemanticError::MissingReturn { .. })
        ));
        assert!(matches!(
            analyze_source("spin :: () -> i32 { while true { continue; } }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("f :: () -> i32 { while true { return 7; } }"),
            Ok(())
        ));
    }

    #[test]
    fn checks_mutability_inside_loop_bodies() {
        assert!(matches!(
            analyze_source("main :: () -> i32 { x := 0; while x < 1 { x = x + 1; } return x; }"),
            Ok(())
        ));
        assert!(matches!(
            analyze_source("main :: () -> i32 { x :: 0; while true { x = 1; } return x; }"),
            Err(SemanticError::ImmutableAssignment { .. })
        ));
    }
}
