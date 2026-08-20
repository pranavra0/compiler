pub mod ast;
pub mod codegen;
pub mod comptime;
pub mod formatter;
pub mod interpreter;
pub mod lexer;
pub mod mir;
pub mod modules;
pub mod parser;
pub mod pipeline;
pub mod semantic;
pub mod source;
pub mod typed;

pub use pipeline::{
    FrontendError, analyze_program, analyze_program_with_pointer_width, analyze_source, lex_source,
    lower_mir, lower_typed_mir, parse_source,
};
