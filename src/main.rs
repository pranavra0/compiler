mod ast;
mod codegen;
mod lexer;
mod parser;

use std::env;
use std::fs;
use std::process;

use inkwell::context::Context;

use lexer::Lexer;

fn main() {
    let arguments: Vec<String> = env::args().collect();

    if arguments.len() != 3 {
        print_usage();
        process::exit(1);
    }

    let command = &arguments[1];
    let filename = &arguments[2];

    let source = match fs::read_to_string(filename) {
        Ok(source) => source,

        Err(error) => {
            eprintln!("error: could not read `{filename}`: {error}");
            process::exit(1);
        }
    };

    match command.as_str() {
        "lex" => lex_command(&source),
        "parse" => parse_command(&source),
        "ir" => ir_command(&source),
        _ => {
            eprintln!("error: unknown command `{command}`");
            print_usage();
            process::exit(1);
        }
    }
}

fn lex_command(source: &str) {
    let mut lexer = Lexer::new(source);

    loop {
        match lexer.next_token() {
            Ok(token) => {
                println!(
                    "{:>4}..{:<4}  {:<24} {}",
                    token.span.start,
                    token.span.end,
                    token.kind,
                    token.lexeme.escape_default()
                );

                if matches!(token.kind, lexer::TokenKind::Eof) {
                    break;
                }
            }

            Err(error) => {
                eprintln!("lexer error: {error}");
                process::exit(1);
            }
        }
    }
}

fn parse_command(source: &str) {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();

    loop {
        match lexer.next_token() {
            Ok(token) => {
                let eof = token.kind == lexer::TokenKind::Eof;
                tokens.push(token);

                if eof {
                    break;
                }
            }

            Err(error) => {
                eprintln!("lexer error: {error}");
                process::exit(1);
            }
        }
    }

    let mut parser = parser::Parser::new(tokens);

    match parser.parse() {
        Ok(program) => {
            println!("{program:#?}");
        }

        Err(error) => {
            eprintln!("parser error: {error}");
            process::exit(1);
        }
    }
}

fn ir_command(source: &str) {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();

    loop {
        match lexer.next_token() {
            Ok(token) => {
                let eof = token.kind == lexer::TokenKind::Eof;
                tokens.push(token);

                if eof {
                    break;
                }
            }

            Err(error) => {
                eprintln!("lexer error: {error}");
                process::exit(1);
            }
        }
    }

    let mut parser = parser::Parser::new(tokens);
    let program = match parser.parse() {
        Ok(program) => program,
        Err(error) => {
            eprintln!("parser error: {error}");
            process::exit(1);
        }
    };

    let context = Context::create();
    match codegen::CodeGenerator::new(&context, "compy").generate(&program) {
        Ok(module) => print!("{}", module.print_to_string().to_string()),
        Err(error) => {
            eprintln!("code generation error: {error}");
            process::exit(1);
        }
    }
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("    compiler lex <file>");
    eprintln!("    compiler parse <file>");
    eprintln!("    compiler ir <file>");
}
