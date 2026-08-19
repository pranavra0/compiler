use std::fmt;

use crate::ast::{
    BinaryOp, Block, Decl, Expr, FunctionDecl, GenericParam, ImportDecl, Parameter, Program, Stmt,
    StructDecl, StructField, StructInit, Type, UnaryOp, VariableDecl, VariableKind,
};
use crate::lexer::{Span, Token, TokenKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    UnexpectedToken {
        expected: String,
        found: TokenKind,
        span: Span,
    },

    UnexpectedEof {
        expected: String,
        span: Span,
    },

    InvalidInteger {
        lexeme: String,
        span: Span,
    },

    InvalidFloat {
        lexeme: String,
        span: Span,
    },

    InvalidDeclaration {
        span: Span,
    },

    InvalidAssignmentTarget {
        span: Span,
    },

    Unsupported {
        message: String,
        span: Span,
    },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::UnexpectedToken {
                expected,
                found,
                span,
            } => {
                write!(
                    f,
                    "expected {expected}, found {found} at {}..{}",
                    span.start, span.end
                )
            }

            ParseError::UnexpectedEof { expected, span } => {
                write!(
                    f,
                    "expected {expected}, found end of file at {}..{}",
                    span.start, span.end
                )
            }

            ParseError::InvalidInteger { lexeme, span } => {
                write!(
                    f,
                    "invalid integer literal `{lexeme}` at {}..{}",
                    span.start, span.end
                )
            }

            ParseError::InvalidFloat { lexeme, span } => {
                write!(
                    f,
                    "invalid floating-point literal `{lexeme}` at {}..{}",
                    span.start, span.end
                )
            }

            ParseError::InvalidDeclaration { span } => {
                write!(f, "invalid declaration at {}..{}", span.start, span.end)
            }

            ParseError::InvalidAssignmentTarget { span } => {
                write!(
                    f,
                    "invalid assignment target at {}..{}",
                    span.start, span.end
                )
            }
            ParseError::Unsupported { message, span } => {
                write!(f, "{message} at {}..{}", span.start, span.end)
            }
        }
    }
}

impl std::error::Error for ParseError {}

pub struct Parser {
    tokens: Vec<Token>,
    position: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    pub fn parse(&mut self) -> Result<Program, ParseError> {
        let mut imports = Vec::new();
        let mut declarations = Vec::new();
        while !self.at(TokenKind::Eof) {
            if self.at(TokenKind::Import) {
                imports.push(self.parse_import()?);
            } else if self.at(TokenKind::Hash) {
                let start = self.current().span.start;
                let expression = self.parse_expression()?;
                let end = self.expect(TokenKind::Semicolon)?.span.end;
                declarations.push(Decl::Comptime {
                    expression,
                    span: Span::new(start, end),
                });
            } else {
                declarations.push(self.parse_declaration()?);
            }
        }
        Ok(Program {
            imports,
            declarations,
        })
    }

    fn parse_import(&mut self) -> Result<ImportDecl, ParseError> {
        let start = self.expect(TokenKind::Import)?.span.start;
        let token = if self.at(TokenKind::String) || self.at(TokenKind::Identifier) {
            self.advance().clone()
        } else {
            return Err(ParseError::UnexpectedToken {
                expected: "module name or string path".into(),
                found: self.current().kind,
                span: self.current().span,
            });
        };
        let path = token.lexeme.trim_matches('"').to_string();
        let alias = path
            .rsplit('/')
            .next()
            .unwrap_or(&path)
            .trim_end_matches(".compy")
            .to_string();
        let end = self.expect(TokenKind::Semicolon)?.span.end;
        Ok(ImportDecl {
            path,
            alias,
            span: Span::new(start, end),
        })
    }

    fn parse_declaration(&mut self) -> Result<Decl, ParseError> {
        let start_token = self.current().clone();
        let mut is_extern = false;
        let mut is_export = false;
        loop {
            if self.match_token(TokenKind::Extern) {
                is_extern = true;
                continue;
            }
            if self.match_token(TokenKind::Export) {
                is_export = true;
                continue;
            }
            break;
        }
        let abi = if is_extern || is_export {
            if self.at(TokenKind::String) {
                Some(self.advance().lexeme.trim_matches('"').to_string())
            } else {
                None
            }
        } else {
            None
        };
        let fn_start = start_token.span.start;
        let has_fn_keyword = self.match_token(TokenKind::Fn);
        let name = self.expect_identifier()?;
        let start = if has_fn_keyword || is_extern || is_export {
            fn_start
        } else {
            name.span.start
        };

        if has_fn_keyword || is_extern {
            return Ok(Decl::Function(self.parse_function_after_name(
                name, start, is_extern, abi, is_export,
            )?));
        }
        if is_export {
            if abi.is_some() && !self.at(TokenKind::DoubleColon) {
                return Err(ParseError::Unsupported {
                    message: "an ABI string is only valid on exported functions".into(),
                    span: start_token.span,
                });
            }
            self.expect(TokenKind::DoubleColon)?;
            if abi.is_some() && !self.at(TokenKind::LParen) {
                return Err(ParseError::Unsupported {
                    message: "an ABI string is only valid on exported functions".into(),
                    span: start_token.span,
                });
            }
            if self.at(TokenKind::LParen) {
                return Ok(Decl::Function(
                    self.parse_function_after_name(name, start, false, abi, true)?,
                ));
            }
            if self.at(TokenKind::Struct) {
                let mut structure = self.parse_struct_after_name(name, start)?;
                structure.exported = true;
                return Ok(Decl::Struct(structure));
            }
            return self.parse_variable_after_operator(name, start, TokenKind::DoubleColon, true);
        }

        let operator = if self.at(TokenKind::DoubleColon) {
            self.advance();
            TokenKind::DoubleColon
        } else {
            self.expect(TokenKind::Colon)?;
            TokenKind::Colon
        };

        if operator == TokenKind::DoubleColon && self.at(TokenKind::LParen) {
            return Ok(Decl::Function(
                self.parse_function_after_name(name, start, false, None, false)?,
            ));
        }
        if operator == TokenKind::DoubleColon && self.at(TokenKind::Struct) {
            return Ok(Decl::Struct(self.parse_struct_after_name(name, start)?));
        }

        self.parse_variable_after_operator(name, start, operator, false)
    }

    fn parse_variable_after_operator(
        &mut self,
        name: Token,
        start: usize,
        operator: TokenKind,
        exported: bool,
    ) -> Result<Decl, ParseError> {
        let typed = self.looks_like_typed_declaration();
        let ty = if typed {
            Some(self.parse_type()?)
        } else {
            None
        };
        if typed {
            self.expect(TokenKind::Equal)?;
        }
        let value = self.parse_expression()?;
        let span = Span::new(start, value.span().end);
        self.expect(TokenKind::Semicolon)?;
        Ok(Decl::Variable(VariableDecl {
            name: name.lexeme,
            kind: if operator == TokenKind::Colon {
                VariableKind::MutableTyped
            } else {
                VariableKind::Immutable
            },
            ty,
            value,
            span,
            exported,
        }))
    }

    fn parse_function_after_name(
        &mut self,
        name: Token,
        start: usize,
        is_extern: bool,
        abi: Option<String>,
        exported: bool,
    ) -> Result<FunctionDecl, ParseError> {
        self.expect(TokenKind::LParen)?;

        let mut params = Vec::new();

        if !self.at(TokenKind::RParen) {
            loop {
                if self.at(TokenKind::Ellipsis) {
                    return Err(ParseError::Unsupported {
                        message: "variadic foreign functions are not implemented".into(),
                        span: self.current().span,
                    });
                }
                params.push(self.parse_parameter()?);

                if !self.match_token(TokenKind::Comma) {
                    break;
                }

                if self.at(TokenKind::RParen) {
                    break;
                }
            }
        }

        self.expect(TokenKind::RParen)?;

        // `T: type` is a declaration of a type parameter, not a runtime
        // argument. Keeping it explicit makes generic intent unambiguous.
        let mut generic_params = Vec::new();
        params.retain(|parameter| {
            if parameter.ty == Type::Named("type".into()) {
                generic_params.push(GenericParam {
                    name: parameter.name.clone(),
                    span: parameter.span,
                });
                false
            } else {
                true
            }
        });

        let return_type = if self.match_token(TokenKind::Arrow) {
            self.parse_type()?
        } else {
            Type::Unit
        };

        let body = if is_extern {
            let end = self.expect(TokenKind::Semicolon)?.span.end;
            Block {
                statements: Vec::new(),
                span: Span::new(start, end),
            }
        } else {
            self.parse_block()?
        };

        Ok(FunctionDecl {
            name: name.lexeme.clone(),
            generic_params,
            params,
            return_type,
            body: body.clone(),
            span: Span::new(start, body.span.end),
            is_extern,
            abi,
            link_name: (is_extern || exported).then(|| name.lexeme.clone()),
            exported,
        })
    }

    fn parse_parameter(&mut self) -> Result<Parameter, ParseError> {
        let name = self.expect_identifier()?;
        let start = name.span.start;

        self.expect(TokenKind::Colon)?;

        let ty = self.parse_type()?;

        Ok(Parameter {
            name: name.lexeme,
            ty,
            span: Span::new(start, self.previous().span.end),
        })
    }

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        let mut ty = if self.match_token(TokenKind::Star) {
            Type::Pointer(Box::new(self.parse_type_atom()?))
        } else if self.match_token(TokenKind::LBracket) {
            if self.match_token(TokenKind::RBracket) {
                Type::Slice(Box::new(self.parse_type_atom()?))
            } else {
                let token = self.expect(TokenKind::Integer)?;
                let length = token.lexeme.replace('_', "").parse::<u64>().map_err(|_| {
                    ParseError::InvalidInteger {
                        lexeme: token.lexeme.clone(),
                        span: token.span,
                    }
                })?;
                self.expect(TokenKind::RBracket)?;
                Type::Array {
                    length,
                    element: Box::new(self.parse_type_atom()?),
                }
            }
        } else {
            self.parse_type_atom()?
        };
        // Result types are right associative: T | E.  The error arm is a
        // type as well, which permits named error structs and void.
        if self.match_token(TokenKind::Pipe) {
            let error = self.parse_type()?;
            ty = Type::Result {
                success: Box::new(ty),
                error: Box::new(error),
            };
        }
        Ok(ty)
    }

    fn parse_type_atom(&mut self) -> Result<Type, ParseError> {
        if self.match_token(TokenKind::Star) {
            return Ok(Type::Pointer(Box::new(self.parse_type_atom()?)));
        }
        if self.match_token(TokenKind::LBracket) {
            if self.match_token(TokenKind::RBracket) {
                return Ok(Type::Slice(Box::new(self.parse_type_atom()?)));
            }
            let token = self.expect(TokenKind::Integer)?;
            let length = token.lexeme.replace('_', "").parse::<u64>().map_err(|_| {
                ParseError::InvalidInteger {
                    lexeme: token.lexeme.clone(),
                    span: token.span,
                }
            })?;
            self.expect(TokenKind::RBracket)?;
            return Ok(Type::Array {
                length,
                element: Box::new(self.parse_type_atom()?),
            });
        }
        let first = self.expect_identifier()?;
        if self.match_token(TokenKind::Dot) {
            let second = self.expect_identifier()?;
            Ok(Type::Named(format!("{}.{}", first.lexeme, second.lexeme)))
        } else {
            Ok(Type::Named(first.lexeme))
        }
    }

    fn parse_struct_after_name(
        &mut self,
        name: Token,
        start: usize,
    ) -> Result<StructDecl, ParseError> {
        self.expect(TokenKind::Struct)?;
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            let field = self.expect_identifier()?;
            let field_start = field.span.start;
            self.expect(TokenKind::Colon)?;
            let ty = self.parse_type()?;
            let end = self.previous().span.end;
            fields.push(StructField {
                name: field.lexeme,
                ty,
                span: Span::new(field_start, end),
            });
            if !self.match_token(TokenKind::Semicolon) {
                self.expect(TokenKind::Comma)?;
            }
        }
        let closing = self.expect(TokenKind::RBrace)?;
        self.match_token(TokenKind::Semicolon);
        Ok(StructDecl {
            name: name.lexeme,
            fields,
            span: Span::new(start, closing.span.end),
            exported: false,
        })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let opening = self.expect(TokenKind::LBrace)?;
        let start = opening.span.start;

        let mut statements = Vec::new();

        while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
            statements.push(self.parse_statement()?);
        }

        let closing = self.expect(TokenKind::RBrace)?;

        Ok(Block {
            statements,
            span: Span::new(start, closing.span.end),
        })
    }

    fn parse_statement(&mut self) -> Result<Stmt, ParseError> {
        if self.match_token(TokenKind::Return) {
            return self.parse_return_statement();
        }

        if self.at(TokenKind::If) {
            return self.parse_if_statement();
        }

        if self.at(TokenKind::While) {
            return self.parse_while_statement();
        }

        if self.at(TokenKind::Break) {
            return self.parse_loop_control_statement(true);
        }

        if self.at(TokenKind::Continue) {
            return self.parse_loop_control_statement(false);
        }

        if self.at(TokenKind::Defer) {
            let start = self.advance().span.start;
            let call = self.parse_expression()?;
            if !matches!(call, Expr::Call { .. }) {
                return Err(ParseError::UnexpectedToken {
                    expected: "deferred function call".into(),
                    found: self.current().kind,
                    span: call.span(),
                });
            }
            let end = self.expect(TokenKind::Semicolon)?.span.end;
            return Ok(Stmt::Defer {
                call,
                span: Span::new(start, end),
            });
        }

        if self.at(TokenKind::Let) || self.at(TokenKind::Var) {
            return self.parse_variable_statement();
        }

        if self.at(TokenKind::Identifier)
            && (self.peek_kind(1) == Some(TokenKind::ColonEqual)
                || self.peek_kind(1) == Some(TokenKind::Colon)
                || self.peek_kind(1) == Some(TokenKind::DoubleColon))
        {
            return self.parse_variable_statement();
        }

        let expression = self.parse_expression()?;

        if self.match_token(TokenKind::Equal) {
            let start = expression.span().start;
            let value = self.parse_expression()?;

            if !is_assignable(&expression) {
                return Err(ParseError::InvalidAssignmentTarget {
                    span: expression.span(),
                });
            }

            let span = Span::new(start, value.span().end);

            self.expect(TokenKind::Semicolon)?;

            return Ok(Stmt::Assignment {
                target: expression,
                value,
                span,
            });
        }

        let span = expression.span();

        self.expect(TokenKind::Semicolon)?;

        Ok(Stmt::Expr { expression, span })
    }

    fn parse_if_statement(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect(TokenKind::If)?.span.start;
        let condition = self.parse_expression()?;
        let then_branch = self.parse_block()?;

        let else_branch = if self.match_token(TokenKind::Else) {
            Some(self.parse_block()?)
        } else {
            None
        };

        let end = else_branch
            .as_ref()
            .map_or(then_branch.span.end, |branch| branch.span.end);

        Ok(Stmt::If {
            condition,
            then_branch,
            else_branch,
            span: Span::new(start, end),
        })
    }

    fn parse_while_statement(&mut self) -> Result<Stmt, ParseError> {
        let start = self.expect(TokenKind::While)?.span.start;
        let condition = self.parse_expression()?;
        let body = self.parse_block()?;

        Ok(Stmt::While {
            condition,
            span: Span::new(start, body.span.end),
            body,
        })
    }

    fn parse_loop_control_statement(&mut self, is_break: bool) -> Result<Stmt, ParseError> {
        let start = self.advance().span.start;
        let semicolon = self.expect(TokenKind::Semicolon)?;
        let span = Span::new(start, semicolon.span.end);

        Ok(if is_break {
            Stmt::Break { span }
        } else {
            Stmt::Continue { span }
        })
    }

    fn parse_return_statement(&mut self) -> Result<Stmt, ParseError> {
        let start = self.previous().span.start;

        if self.match_token(TokenKind::Semicolon) {
            return Ok(Stmt::Return {
                value: None,
                span: Span::new(start, self.previous().span.end),
            });
        }

        let value = self.parse_expression()?;
        let end = value.span().end;

        self.expect(TokenKind::Semicolon)?;

        Ok(Stmt::Return {
            value: Some(value),
            span: Span::new(start, end),
        })
    }

    fn parse_variable_statement(&mut self) -> Result<Stmt, ParseError> {
        let keyword = self.current().kind;
        let (name, start, kind, ty, needs_equal) = match keyword {
            TokenKind::Let => {
                self.advance();
                let name = self.expect_identifier()?;
                let start = name.span.start;
                let ty = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                (name, start, VariableKind::Immutable, ty, true)
            }

            TokenKind::Var => {
                self.advance();
                let name = self.expect_identifier()?;
                let start = name.span.start;
                let ty = if self.match_token(TokenKind::Colon) {
                    Some(self.parse_type()?)
                } else {
                    None
                };
                let kind = if ty.is_some() {
                    VariableKind::MutableTyped
                } else {
                    VariableKind::MutableInferred
                };
                (name, start, kind, ty, true)
            }

            TokenKind::Identifier => {
                let name = self.expect_identifier()?;
                let start = name.span.start;

                if self.match_token(TokenKind::ColonEqual) {
                    (name, start, VariableKind::MutableInferred, None, false)
                } else if self.match_token(TokenKind::Colon) {
                    let ty = Some(self.parse_type()?);
                    (name, start, VariableKind::MutableTyped, ty, true)
                } else if self.match_token(TokenKind::DoubleColon) {
                    let ty = if self.looks_like_typed_declaration() {
                        Some(self.parse_type()?)
                    } else {
                        None
                    };
                    let needs_equal = ty.is_some();
                    (name, start, VariableKind::Immutable, ty, needs_equal)
                } else {
                    return Err(ParseError::InvalidDeclaration { span: name.span });
                }
            }

            _ => {
                return Err(ParseError::InvalidDeclaration {
                    span: self.current().span,
                });
            }
        };

        if needs_equal {
            self.expect(TokenKind::Equal)?;
        }

        let value = self.parse_expression()?;
        let end = value.span().end;

        self.expect(TokenKind::Semicolon)?;

        Ok(Stmt::Variable(VariableDecl {
            name: name.lexeme,
            kind,
            ty,
            value,
            span: Span::new(start, end),
            exported: false,
        }))
    }

    fn parse_expression(&mut self) -> Result<Expr, ParseError> {
        self.parse_binary_expression(0)
    }

    fn parse_binary_expression(&mut self, minimum_precedence: u8) -> Result<Expr, ParseError> {
        let mut left = self.parse_unary_expression()?;

        while let Some(operator) = self.current_binary_operator() {
            let precedence = operator.precedence();

            if precedence < minimum_precedence {
                break;
            }

            self.advance();

            let right = self.parse_binary_expression(precedence + 1)?;

            let span = Span::new(left.span().start, right.span().end);

            left = Expr::Binary {
                left: Box::new(left),
                operator,
                right: Box::new(right),
                span,
            };
        }

        Ok(left)
    }

    fn parse_unary_expression(&mut self) -> Result<Expr, ParseError> {
        let operator = match self.current().kind {
            TokenKind::Minus => Some(UnaryOp::Negate),
            TokenKind::Bang => Some(UnaryOp::Not),
            TokenKind::Tilde => Some(UnaryOp::BitwiseNot),
            TokenKind::Ampersand => Some(UnaryOp::AddressOf),
            TokenKind::Star => Some(UnaryOp::Dereference),
            _ => None,
        };

        if let Some(operator) = operator {
            let token = self.advance().clone();
            let operand = self.parse_unary_expression()?;

            let span = Span::new(token.span.start, operand.span().end);

            return Ok(Expr::Unary {
                operator,
                operand: Box::new(operand),
                span,
            });
        }

        self.parse_postfix_expression()
    }

    fn parse_postfix_expression(&mut self) -> Result<Expr, ParseError> {
        let mut expression = self.parse_primary_expression()?;

        loop {
            if self.match_token(TokenKind::Dot) {
                let field = self.expect_identifier()?;
                let span = Span::new(expression.span().start, field.span.end);
                expression = Expr::Field {
                    base: Box::new(expression),
                    name: field.lexeme,
                    span,
                };
                continue;
            }
            if self.match_token(TokenKind::LBracket) {
                let index = self.parse_expression()?;
                let closing = self.expect(TokenKind::RBracket)?;
                let span = Span::new(expression.span().start, closing.span.end);
                expression = Expr::Index {
                    base: Box::new(expression),
                    index: Box::new(index),
                    span,
                };
                continue;
            }
            if self.match_token(TokenKind::Question) {
                let span = Span::new(expression.span().start, self.previous().span.end);
                expression = Expr::Propagate {
                    expression: Box::new(expression),
                    span,
                };
                continue;
            }
            if !self.match_token(TokenKind::LParen) {
                break;
            }

            // Layout intrinsics take types, not runtime expressions.
            if let Expr::Identifier {
                name,
                span: name_span,
            } = &expression
            {
                if name == "size_of" || name == "align_of" {
                    let ty = self.parse_type()?;
                    let closing = self.expect(TokenKind::RParen)?;
                    let span = Span::new(name_span.start, closing.span.end);
                    expression = if name == "size_of" {
                        Expr::SizeOf { ty, span }
                    } else {
                        Expr::AlignOf { ty, span }
                    };
                    continue;
                }
                if name == "offset_of" {
                    let ty = self.parse_type()?;
                    self.expect(TokenKind::Comma)?;
                    let field = self.expect_identifier()?;
                    let closing = self.expect(TokenKind::RParen)?;
                    let span = Span::new(name_span.start, closing.span.end);
                    expression = Expr::OffsetOf {
                        ty,
                        field: field.lexeme,
                        span,
                    };
                    continue;
                }
            }

            let mut arguments = Vec::new();
            if !self.at(TokenKind::RParen) {
                loop {
                    arguments.push(self.parse_expression()?);
                    if !self.match_token(TokenKind::Comma) {
                        break;
                    }
                    if self.at(TokenKind::RParen) {
                        break;
                    }
                }
            }
            let closing = self.expect(TokenKind::RParen)?;
            let span = Span::new(expression.span().start, closing.span.end);
            if let Expr::Identifier { name, .. } = &expression {
                if name == "unchecked_index" {
                    if arguments.len() != 2 {
                        return Err(ParseError::UnexpectedToken {
                            expected: "two arguments".into(),
                            found: TokenKind::RParen,
                            span: closing.span,
                        });
                    }
                    expression = Expr::UncheckedIndex {
                        base: Box::new(arguments.remove(0)),
                        index: Box::new(arguments.remove(0)),
                        span,
                    };
                    continue;
                }
            }
            expression = Expr::Call {
                callee: Box::new(expression),
                arguments,
                span,
            };
        }

        Ok(expression)
    }

    fn parse_primary_expression(&mut self) -> Result<Expr, ParseError> {
        let token = self.current().clone();

        if token.kind == TokenKind::Hash {
            self.advance();
            let expression = self.parse_unary_expression()?;
            let span = Span::new(token.span.start, expression.span().end);
            return Ok(Expr::Comptime {
                expression: Box::new(expression),
                span,
            });
        }

        match token.kind {
            TokenKind::Integer => {
                self.advance();

                let cleaned = token.lexeme.replace('_', "");

                let value = cleaned
                    .parse::<i128>()
                    .map_err(|_| ParseError::InvalidInteger {
                        lexeme: token.lexeme.clone(),
                        span: token.span,
                    })?;

                Ok(Expr::Integer {
                    value,
                    span: token.span,
                })
            }

            TokenKind::Float => {
                self.advance();

                let cleaned = token.lexeme.replace('_', "");
                let value = cleaned
                    .parse::<f64>()
                    .map_err(|_| ParseError::InvalidFloat {
                        lexeme: token.lexeme.clone(),
                        span: token.span,
                    })?;

                Ok(Expr::Float {
                    value,
                    span: token.span,
                })
            }

            TokenKind::True | TokenKind::False => {
                self.advance();

                Ok(Expr::Bool {
                    value: token.kind == TokenKind::True,
                    span: token.span,
                })
            }

            TokenKind::Null => {
                self.advance();
                Ok(Expr::Null { span: token.span })
            }

            TokenKind::Identifier => {
                self.advance();
                let mut qualified_name = token.lexeme.clone();
                if self.match_token(TokenKind::Dot) {
                    let part = self.expect_identifier()?;
                    qualified_name = format!("{}.{}", qualified_name, part.lexeme);
                }
                if self.at(TokenKind::LBrace)
                    && (self.peek_kind(1) == Some(TokenKind::RBrace)
                        || (self.peek_kind(1) == Some(TokenKind::Identifier)
                            && self.peek_kind(2) == Some(TokenKind::Equal)))
                {
                    self.advance();
                    let mut fields = Vec::new();
                    while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                        let field = self.expect_identifier()?;
                        self.expect(TokenKind::Equal)?;
                        let value = self.parse_expression()?;
                        let span = Span::new(field.span.start, value.span().end);
                        fields.push(StructInit {
                            name: field.lexeme,
                            value,
                            span,
                        });
                        if !self.match_token(TokenKind::Comma)
                            && !self.match_token(TokenKind::Semicolon)
                            && !self.at(TokenKind::RBrace)
                        {
                            return Err(ParseError::UnexpectedToken {
                                expected: "field separator".into(),
                                found: self.current().kind,
                                span: self.current().span,
                            });
                        }
                    }
                    let closing = self.expect(TokenKind::RBrace)?;
                    return Ok(Expr::StructLiteral {
                        name: qualified_name,
                        fields,
                        span: Span::new(token.span.start, closing.span.end),
                    });
                }
                if qualified_name != token.lexeme {
                    let span = Span::new(token.span.start, self.previous().span.end);
                    Ok(Expr::Field {
                        base: Box::new(Expr::Identifier {
                            name: token.lexeme,
                            span: token.span,
                        }),
                        name: qualified_name.split('.').nth(1).unwrap().to_string(),
                        span,
                    })
                } else {
                    Ok(Expr::Identifier {
                        name: token.lexeme,
                        span: token.span,
                    })
                }
            }

            TokenKind::LBracket => {
                self.advance();
                let length_token = self.expect(TokenKind::Integer)?;
                let length = length_token
                    .lexeme
                    .replace('_', "")
                    .parse::<u64>()
                    .map_err(|_| ParseError::InvalidInteger {
                        lexeme: length_token.lexeme.clone(),
                        span: length_token.span,
                    })?;
                self.expect(TokenKind::RBracket)?;
                let element = self.parse_type()?;
                self.expect(TokenKind::LBrace)?;
                let mut elements = Vec::new();
                while !self.at(TokenKind::RBrace) && !self.at(TokenKind::Eof) {
                    elements.push(self.parse_expression()?);
                    if !self.match_token(TokenKind::Comma) {
                        self.expect(TokenKind::RBrace).map(|_| ())?;
                        break;
                    }
                }
                let closing = if self.previous().kind == TokenKind::RBrace {
                    self.previous().clone()
                } else {
                    self.expect(TokenKind::RBrace)?
                };
                Ok(Expr::ArrayLiteral {
                    ty: Type::Array {
                        length,
                        element: Box::new(element),
                    },
                    elements,
                    span: Span::new(length_token.span.start - 1, closing.span.end),
                })
            }

            TokenKind::LParen => {
                self.advance();

                let expression = self.parse_expression()?;

                self.expect(TokenKind::RParen)?;

                Ok(expression)
            }

            _ => Err(ParseError::UnexpectedToken {
                expected: "expression".to_string(),
                found: token.kind,
                span: token.span,
            }),
        }
    }

    fn current_binary_operator(&self) -> Option<BinaryOp> {
        match self.current().kind {
            TokenKind::Plus => Some(BinaryOp::Add),
            TokenKind::Minus => Some(BinaryOp::Subtract),
            TokenKind::Star => Some(BinaryOp::Multiply),
            TokenKind::Slash => Some(BinaryOp::Divide),
            TokenKind::Percent => Some(BinaryOp::Modulo),

            TokenKind::EqualEqual => Some(BinaryOp::Equal),
            TokenKind::BangEqual => Some(BinaryOp::NotEqual),

            TokenKind::Less => Some(BinaryOp::Less),
            TokenKind::LessEqual => Some(BinaryOp::LessEqual),
            TokenKind::Greater => Some(BinaryOp::Greater),
            TokenKind::GreaterEqual => Some(BinaryOp::GreaterEqual),
            TokenKind::ShiftLeft => Some(BinaryOp::ShiftLeft),
            TokenKind::ShiftRight => Some(BinaryOp::ShiftRight),

            TokenKind::AmpersandAmpersand => Some(BinaryOp::LogicalAnd),
            TokenKind::PipePipe => Some(BinaryOp::LogicalOr),

            TokenKind::Ampersand => Some(BinaryOp::BitwiseAnd),
            TokenKind::Pipe => Some(BinaryOp::BitwiseOr),
            TokenKind::Caret => Some(BinaryOp::BitwiseXor),

            _ => None,
        }
    }

    fn looks_like_typed_declaration(&self) -> bool {
        fn type_end(tokens: &[Token], position: usize) -> Option<usize> {
            match tokens.get(position)?.kind {
                TokenKind::Identifier => Some(position + 1),
                TokenKind::Star => type_end(tokens, position + 1),
                TokenKind::Pipe => type_end(tokens, position + 1),
                TokenKind::LBracket => {
                    if tokens.get(position + 1)?.kind == TokenKind::RBracket {
                        return type_end(tokens, position + 2);
                    }
                    if tokens.get(position + 1)?.kind != TokenKind::Integer
                        || tokens.get(position + 2)?.kind != TokenKind::RBracket
                    {
                        return None;
                    }
                    type_end(tokens, position + 3)
                }
                _ => None,
            }
        }
        type_end(&self.tokens, self.position)
            .and_then(|end| self.tokens.get(end).map(|token| token.kind))
            == Some(TokenKind::Equal)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.position]
    }

    fn previous(&self) -> &Token {
        &self.tokens[self.position - 1]
    }

    fn peek_kind(&self, offset: usize) -> Option<TokenKind> {
        self.tokens
            .get(self.position + offset)
            .map(|token| token.kind)
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

    fn advance(&mut self) -> &Token {
        let token = &self.tokens[self.position];

        if token.kind != TokenKind::Eof {
            self.position += 1;
        }

        token
    }

    fn match_token(&mut self, kind: TokenKind) -> bool {
        if self.at(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Token, ParseError> {
        if self.at(kind) {
            return Ok(self.advance().clone());
        }

        let token = self.current();

        if token.kind == TokenKind::Eof {
            return Err(ParseError::UnexpectedEof {
                expected: kind.to_string(),
                span: token.span,
            });
        }

        Err(ParseError::UnexpectedToken {
            expected: kind.to_string(),
            found: token.kind,
            span: token.span,
        })
    }

    fn expect_identifier(&mut self) -> Result<Token, ParseError> {
        if self.at(TokenKind::Identifier) {
            return Ok(self.advance().clone());
        }

        let token = self.current();

        if token.kind == TokenKind::Eof {
            return Err(ParseError::UnexpectedEof {
                expected: "identifier".to_string(),
                span: token.span,
            });
        }

        Err(ParseError::UnexpectedToken {
            expected: "identifier".to_string(),
            found: token.kind,
            span: token.span,
        })
    }
}

impl BinaryOp {
    fn precedence(self) -> u8 {
        match self {
            BinaryOp::LogicalOr => 1,
            BinaryOp::LogicalAnd => 2,

            BinaryOp::BitwiseOr => 3,
            BinaryOp::BitwiseXor => 4,
            BinaryOp::BitwiseAnd => 5,

            BinaryOp::Equal | BinaryOp::NotEqual => 6,

            BinaryOp::Less | BinaryOp::LessEqual | BinaryOp::Greater | BinaryOp::GreaterEqual => 7,

            BinaryOp::ShiftLeft | BinaryOp::ShiftRight => 8,

            BinaryOp::Add | BinaryOp::Subtract => 9,

            BinaryOp::Multiply | BinaryOp::Divide | BinaryOp::Modulo => 10,
        }
    }
}

fn is_assignable(expression: &Expr) -> bool {
    matches!(
        expression,
        Expr::Identifier { .. }
            | Expr::Field { .. }
            | Expr::Index { .. }
            | Expr::UncheckedIndex { .. }
            | Expr::Unary {
                operator: UnaryOp::Dereference,
                ..
            }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;

    fn parse(source: &str) -> Program {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token().expect("lexing failed");
            let eof = token.kind == TokenKind::Eof;

            tokens.push(token);

            if eof {
                break;
            }
        }

        Parser::new(tokens).parse().expect("parsing failed")
    }

    #[test]
    fn parses_hello_program_syntax() {
        let program = parse(
            r#"
            fn main() {
                let answer = 42;
                let other = 10.5;

                if answer >= 40 {
                    return;
                }
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        assert_eq!(function.name, "main");
        assert_eq!(function.return_type, Type::Unit);
        assert_eq!(function.body.statements.len(), 3);
        assert!(matches!(
            function.body.statements[0],
            Stmt::Variable(VariableDecl {
                kind: VariableKind::Immutable,
                ..
            })
        ));
        assert!(matches!(
            function.body.statements[1],
            Stmt::Variable(VariableDecl {
                value: Expr::Float { .. },
                ..
            })
        ));
        assert!(matches!(function.body.statements[2], Stmt::If { .. }));
    }

    #[test]
    fn parses_minimal_main() {
        let program = parse(
            r#"
            main :: () -> i32 {
                return 42;
            }
            "#,
        );

        assert_eq!(program.declarations.len(), 1);

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        assert_eq!(function.name, "main");
        assert!(function.params.is_empty());
        assert_eq!(function.return_type, Type::Named("i32".into()));
        assert_eq!(function.body.statements.len(), 1);

        let Stmt::Return {
            value: Some(Expr::Integer { value, .. }),
            ..
        } = &function.body.statements[0]
        else {
            panic!("expected return statement");
        };

        assert_eq!(*value, 42);
    }

    #[test]
    fn parses_function_parameters() {
        let program = parse(
            r#"
            add :: (a: i64, b: i64) -> i64 {
                return a + b;
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        assert_eq!(function.params.len(), 2);

        assert_eq!(function.params[0].name, "a");
        assert_eq!(function.params[0].ty, Type::Named("i64".into()));

        assert_eq!(function.params[1].name, "b");
        assert_eq!(function.params[1].ty, Type::Named("i64".into()));
    }

    #[test]
    fn parses_operator_precedence() {
        let program = parse(
            r#"
            main :: () -> i32 {
                return 1 + 2 * 3;
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        let Stmt::Return {
            value: Some(expression),
            ..
        } = &function.body.statements[0]
        else {
            panic!("expected return");
        };

        let Expr::Binary {
            operator: BinaryOp::Add,
            left,
            right,
            ..
        } = expression
        else {
            panic!("expected addition");
        };

        assert!(matches!(left.as_ref(), Expr::Integer { value: 1, .. }));

        let Expr::Binary {
            operator: BinaryOp::Multiply,
            left,
            right,
            ..
        } = right.as_ref()
        else {
            panic!("expected multiplication");
        };

        assert!(matches!(left.as_ref(), Expr::Integer { value: 2, .. }));

        assert!(matches!(right.as_ref(), Expr::Integer { value: 3, .. }));
    }

    #[test]
    fn parses_function_calls() {
        let program = parse(
            r#"
            main :: () -> i32 {
                return add(10, 20);
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        let Stmt::Return {
            value: Some(Expr::Call { arguments, .. }),
            ..
        } = &function.body.statements[0]
        else {
            panic!("expected function call");
        };

        assert_eq!(arguments.len(), 2);
    }

    #[test]
    fn parses_local_declarations() {
        let program = parse(
            r#"
            main :: () -> i32 {
                x := 10;
                y : i32 = 20;
                z :: 30;
                return x + y + z;
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        assert_eq!(function.body.statements.len(), 4);

        let Stmt::Variable(variable) = &function.body.statements[0] else {
            panic!("expected variable");
        };

        assert_eq!(variable.name, "x");
        assert_eq!(variable.kind, VariableKind::MutableInferred);
        assert_eq!(variable.ty, None);

        let Stmt::Variable(variable) = &function.body.statements[1] else {
            panic!("expected variable");
        };

        assert_eq!(variable.kind, VariableKind::MutableTyped);
        assert_eq!(variable.ty, Some(Type::Named("i32".into())));

        let Stmt::Variable(variable) = &function.body.statements[2] else {
            panic!("expected variable");
        };

        assert_eq!(variable.kind, VariableKind::Immutable);
    }

    #[test]
    fn parses_structs_arrays_fields_and_indices() {
        let program = parse(
            "Pair :: struct { x: i32; y: i32; } main :: () -> i32 { p := Pair{ x = 1, y = 2, }; xs := [2]i32{3, 4}; p.x = xs[1]; return p.x; }",
        );
        assert!(matches!(program.declarations[0], Decl::Struct(_)));
        let Decl::Function(function) = &program.declarations[1] else {
            panic!("expected function");
        };
        assert!(matches!(
            function.body.statements[0],
            Stmt::Variable(VariableDecl {
                value: Expr::StructLiteral { .. },
                ..
            })
        ));
        assert!(matches!(
            function.body.statements[1],
            Stmt::Variable(VariableDecl {
                value: Expr::ArrayLiteral { .. },
                ..
            })
        ));
        assert!(matches!(
            function.body.statements[2],
            Stmt::Assignment { .. }
        ));
    }

    #[test]
    fn parses_empty_struct_literals() {
        let program = parse("Empty :: struct {} main :: () { x := Empty{}; }");
        let Decl::Function(function) = &program.declarations[1] else {
            panic!("expected function");
        };
        assert!(matches!(
            &function.body.statements[0],
            Stmt::Variable(VariableDecl {
                value: Expr::StructLiteral { fields, .. },
                ..
            }) if fields.is_empty()
        ));
    }

    #[test]
    fn parses_assignment() {
        let program = parse(
            r#"
            main :: () -> i32 {
                x := 10;
                x = 20;
                return x;
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        assert!(matches!(
            function.body.statements[1],
            Stmt::Assignment { .. }
        ));
    }

    #[test]
    fn parses_unary_expression() {
        let program = parse(
            r#"
            main :: () -> i32 {
                return -42;
            }
            "#,
        );

        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        let Stmt::Return {
            value: Some(Expr::Unary { operator, .. }),
            ..
        } = &function.body.statements[0]
        else {
            panic!("expected unary expression");
        };

        assert_eq!(*operator, UnaryOp::Negate);
    }

    #[test]
    fn parses_while_and_loop_controls_with_spans() {
        let source = "main :: () -> i32 { while true { if true { break; } continue; } return 0; }";
        let program = parse(source);
        let Decl::Function(function) = &program.declarations[0] else {
            panic!("expected function");
        };

        let Stmt::While {
            condition,
            body,
            span,
        } = &function.body.statements[0]
        else {
            panic!("expected while statement");
        };
        assert!(matches!(condition, Expr::Bool { value: true, .. }));
        assert_eq!(span.start, source.find("while").unwrap());
        assert_eq!(span.end, source.find("} return").unwrap() + 1);
        assert!(matches!(body.statements[0], Stmt::If { .. }));

        let Stmt::If { then_branch, .. } = &body.statements[0] else {
            panic!("expected nested if");
        };
        let Stmt::Break { span } = then_branch.statements[0] else {
            panic!("expected break");
        };
        assert_eq!(&source[span.start..span.end], "break;");

        let Stmt::Continue { span } = body.statements[1] else {
            panic!("expected continue");
        };
        assert_eq!(&source[span.start..span.end], "continue;");
    }

    #[test]
    fn reports_missing_loop_syntax() {
        let mut lexer = Lexer::new("main :: () -> i32 { while true return 0; }");
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            let eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if eof {
                break;
            }
        }
        let error = Parser::new(tokens).parse().unwrap_err();
        assert!(matches!(error, ParseError::UnexpectedToken { .. }));

        let mut lexer = Lexer::new("main :: () -> i32 { while true { break } return 0; }");
        let mut tokens = Vec::new();
        loop {
            let token = lexer.next_token().unwrap();
            let eof = token.kind == TokenKind::Eof;
            tokens.push(token);
            if eof {
                break;
            }
        }
        let error = Parser::new(tokens).parse().unwrap_err();
        assert!(matches!(error, ParseError::UnexpectedToken { .. }));
    }
}
