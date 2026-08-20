use std::fmt;

use crate::ast::Program;
use crate::comptime;
use crate::lexer::{LexError, Lexer, Span, Token, TokenKind};
use crate::mir::{MirError, MirProgram};
use crate::parser::{ParseError, Parser};
use crate::semantic::{self, SemanticError};
use crate::typed::TypedProgram;

#[derive(Debug)]
pub enum FrontendError {
    Lexer(LexError),
    Parser(ParseError),
    Semantic(SemanticError),
    Comptime(comptime::Error),
    Mir(MirError),
}

impl FrontendError {
    pub fn stage(&self) -> &'static str {
        match self {
            Self::Lexer(_) => "lexer",
            Self::Parser(_) => "parser",
            Self::Semantic(_) => "semantic",
            Self::Comptime(_) => "comptime",
            Self::Mir(_) => "mir",
        }
    }

    pub fn span(&self) -> Span {
        match self {
            Self::Lexer(error) => match error {
                LexError::UnexpectedCharacter { span, .. }
                | LexError::UnterminatedBlockComment { span } => *span,
            },
            Self::Parser(error) => match error {
                ParseError::UnexpectedToken { span, .. }
                | ParseError::UnexpectedEof { span, .. }
                | ParseError::InvalidInteger { span, .. }
                | ParseError::InvalidFloat { span, .. }
                | ParseError::InvalidDeclaration { span }
                | ParseError::InvalidAssignmentTarget { span }
                | ParseError::Unsupported { span, .. } => *span,
            },
            Self::Semantic(error) => error.span(),
            Self::Comptime(error) => error.span(),
            Self::Mir(error) => error.span(),
        }
    }
}

impl fmt::Display for FrontendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} error: {}",
            self.stage(),
            trim_byte_location(&self.to_string_inner(), self.span())
        )
    }
}

impl FrontendError {
    fn to_string_inner(&self) -> String {
        match self {
            Self::Lexer(error) => error.to_string(),
            Self::Parser(error) => error.to_string(),
            Self::Semantic(error) => error.to_string(),
            Self::Comptime(error) => error.to_string(),
            Self::Mir(error) => error.to_string(),
        }
    }
}

impl std::error::Error for FrontendError {}

fn trim_byte_location(message: &str, span: Span) -> String {
    let suffix = format!(" at {}..{}", span.start, span.end);
    message.strip_suffix(&suffix).unwrap_or(message).to_string()
}

pub fn lex_source(source: &str) -> Result<Vec<Token>, FrontendError> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        let token = lexer.next_token().map_err(FrontendError::Lexer)?;
        let eof = token.kind == TokenKind::Eof;
        tokens.push(token);
        if eof {
            return Ok(tokens);
        }
    }
}

pub fn parse_source(source: &str) -> Result<Program, FrontendError> {
    let tokens = lex_source(source)?;
    Parser::new(tokens).parse().map_err(FrontendError::Parser)
}

pub fn analyze_program(program: &Program) -> Result<TypedProgram, FrontendError> {
    analyze_program_with_pointer_width(program, usize::BITS)
}

pub fn analyze_program_with_pointer_width(
    program: &Program,
    pointer_width: u32,
) -> Result<TypedProgram, FrontendError> {
    let expanded = comptime::expand(program, pointer_width).map_err(FrontendError::Comptime)?;
    semantic::analyze_typed_with_pointer_width(&expanded, pointer_width)
        .map_err(FrontendError::Semantic)
}

pub fn analyze_source(source: &str) -> Result<TypedProgram, FrontendError> {
    let program = parse_source(source)?;
    analyze_program(&program)
}

/// Lower a validated typed program to the shared control-flow representation.
/// Keeping this as a pipeline entry point prevents individual backends from
/// inventing their own structured-control-flow lowering.
pub fn lower_mir(program: &Program) -> Result<MirProgram, FrontendError> {
    let typed = analyze_program(program)?;
    lower_typed_mir(&typed)
}

/// Lower an already validated program without re-running expansion or name
/// resolution. Backends and tools should use this entry point when they have
/// already selected a typed frontend result.
pub fn lower_typed_mir(program: &TypedProgram) -> Result<MirProgram, FrontendError> {
    MirProgram::lower(program).map_err(FrontendError::Mir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_stages_share_the_same_entry_points() {
        assert_eq!(
            lex_source("main :: () {}").unwrap().last().unwrap().kind,
            TokenKind::Eof
        );
        assert!(parse_source("main :: () {}").is_ok());
        assert!(analyze_source("main :: () {}").is_ok());
    }
}
