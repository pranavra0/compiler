mod ast;
mod codegen;
mod lexer;
mod parser;
mod semantic;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};

use lexer::Lexer;

fn main() {
    let arguments: Vec<String> = env::args().collect();

    if arguments.len() < 3 {
        print_usage();
        process::exit(1);
    }

    let command = &arguments[1];
    let filename = &arguments[2];

    match command.as_str() {
        "lex" | "parse" | "check" | "ir" => {
            if arguments.len() != 3 {
                print_usage();
                process::exit(1);
            }

            let source = match read_source(filename) {
                Ok(source) => source,
                Err(error) => exit_with_error(error),
            };

            match command.as_str() {
                "lex" => lex_command(&source),
                "parse" => parse_command(&source),
                "check" => check_command(&source),
                "ir" => ir_command(&source),
                _ => unreachable!(),
            }
        }

        // `compile` is kept as an alias so the command reads naturally in
        // scripts, while `build` makes it clear that linking also happens.
        "build" | "compile" => {
            if let Err(error) = build_command(filename, &arguments[3..]) {
                exit_with_error(error);
            }
        }

        _ => {
            eprintln!("error: unknown command `{command}`");
            print_usage();
            process::exit(1);
        }
    }
}

fn read_source(filename: &str) -> Result<String, String> {
    fs::read_to_string(filename).map_err(|error| format!("could not read `{filename}`: {error}"))
}

fn exit_with_error(error: String) -> ! {
    eprintln!("error: {error}");
    process::exit(1);
}

fn build_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let output = output_path(filename, arguments)?;
    // The build produces both requested artifacts: the executable and the
    // object file used to link it.
    let object = output.with_extension("o");
    let source = read_source(filename)?;
    let program = parse_source(&source)?;
    semantic::analyze(&program).map_err(|error| format!("semantic error: {error}"))?;

    // Select the target before lowering so usize/isize use the target's
    // pointer width rather than a host-specific constant.
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|error| format!("could not initialize the native LLVM target: {error}"))?;

    let target_triple = TargetMachine::get_default_triple();
    let target = Target::from_triple(&target_triple)
        .map_err(|error| format!("could not find the native LLVM target: {error}"))?;
    let target_machine = target
        .create_target_machine(
            &target_triple,
            "generic",
            "",
            OptimizationLevel::Default,
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "could not create the native LLVM target machine".to_string())?;

    let target_data = target_machine.get_target_data();
    let pointer_width = target_data.get_pointer_byte_size(None) * 8;
    let context = Context::create();
    let module =
        codegen::CodeGenerator::with_pointer_width(&context, module_name(&output), pointer_width)
            .generate(&program)
            .map_err(|error| format!("code generation error: {error}"))?;

    module.set_triple(&target_triple);
    let data_layout = target_data.get_data_layout();
    module.set_data_layout(&data_layout);

    create_parent_directory(&object)?;
    target_machine
        .write_to_file(&module, FileType::Object, &object)
        .map_err(|error| format!("could not emit object file `{}`: {error}", object.display()))?;

    link_native(&object, &output)?;

    println!("object: {}", object.display());
    println!("executable: {}", output.display());
    Ok(())
}

fn parse_source(source: &str) -> Result<ast::Program, String> {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();

    loop {
        let token = lexer
            .next_token()
            .map_err(|error| format!("lexer error: {error}"))?;
        let eof = token.kind == lexer::TokenKind::Eof;
        tokens.push(token);

        if eof {
            break;
        }
    }

    parser::Parser::new(tokens)
        .parse()
        .map_err(|error| format!("parser error: {error}"))
}

fn output_path(filename: &str, arguments: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut position = 0;

    while position < arguments.len() {
        match arguments[position].as_str() {
            "-o" | "--output" => {
                position += 1;
                let value = arguments
                    .get(position)
                    .ok_or_else(|| "missing output path after `-o`/`--output`".to_string())?;
                output = Some(PathBuf::from(value));
            }
            argument => return Err(format!("unknown build argument `{argument}`")),
        }

        position += 1;
    }

    if let Some(output) = output {
        return Ok(output);
    }

    let input = Path::new(filename);
    let stem = input
        .file_stem()
        .ok_or_else(|| format!("could not determine an output name from `{filename}`"))?;

    Ok(PathBuf::from(stem))
}

fn module_name(output: &Path) -> &str {
    output
        .file_stem()
        .and_then(|name| name.to_str())
        .unwrap_or("compy")
}

fn create_parent_directory(path: &Path) -> Result<(), String> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)
            .map_err(|error| format!("could not create `{}`: {error}", parent.display()))?;
    }

    Ok(())
}

fn link_native(object: &Path, output: &Path) -> Result<(), String> {
    create_parent_directory(output)?;

    let compiler = env::var_os("CC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("clang"));
    let status = Command::new(&compiler)
        .arg(object)
        .arg("-o")
        .arg(output)
        .status()
        .map_err(|error| format!("could not run linker `{}`: {error}", compiler.display()))?;

    if !status.success() {
        return Err(format!(
            "linker `{}` failed with status {status}",
            compiler.display()
        ));
    }

    Ok(())
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

fn check_command(source: &str) {
    match parse_source(source).and_then(|program| {
        semantic::analyze(&program).map_err(|error| format!("semantic error: {error}"))
    }) {
        Ok(()) => println!("semantic analysis succeeded"),
        Err(error) => exit_with_error(error),
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

    if let Err(error) = semantic::analyze(&program) {
        eprintln!("semantic error: {error}");
        process::exit(1);
    }

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
    eprintln!("    compiler check <file>");
    eprintln!("    compiler ir <file>");
    eprintln!("    compiler build <file> [-o <executable>]");
    eprintln!("    compiler compile <file> [-o <executable>]");
}
