pub mod ast;
pub mod codegen;
pub mod comptime;
pub mod lexer;
pub mod modules;
pub mod parser;
pub mod pipeline;
pub mod semantic;
pub mod source;
pub mod typed;

pub use pipeline::{
    FrontendError, analyze_program, analyze_program_with_pointer_width, analyze_source, lex_source,
    parse_source,
};
