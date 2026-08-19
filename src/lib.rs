pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod pipeline;
pub mod semantic;
pub mod source;
pub mod typed;

pub use pipeline::{FrontendError, analyze_program, analyze_source, lex_source, parse_source};
