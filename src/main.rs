mod lexer;

use std::env;
use std::fs;
use std::process;

use lexer::Lexer;

fn main() {
    let arguments: Vec<String> = env::args().collect();

    if arguments.len() != 3 {
        print_usage();
        process::exit(1);
    }

    let command = &arguments[1];
    let filename = &arguments[2];

    if command != "lex" {
        eprintln!("error: unknown command `{command}`");
        print_usage();
        process::exit(1);
    }

    let source = match fs::read_to_string(filename) {
        Ok(source) => source,

        Err(error) => {
            eprintln!("error: could not read `{filename}`: {error}");
            process::exit(1);
        }
    };

    let mut lexer = Lexer::new(&source);

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

fn print_usage() {
    eprintln!("usage:");
    eprintln!("    compiler lex <file>");
}