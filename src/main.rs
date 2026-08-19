use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use compiler::codegen::CodeGenerator;
use compiler::pipeline::{self, FrontendError};
use compiler::semantic;
use compiler::source::SourceMap;
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() < 3 {
        print_usage();
        process::exit(1);
    }
    let command = &arguments[1];
    let filename = &arguments[2];

    let result = match command.as_str() {
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
                "lex" => lex_command(filename, &source),
                "parse" => parse_command(filename, &source),
                "check" => check_command(filename, &source),
                "ir" => ir_command(filename, &source),
                _ => unreachable!(),
            }
        }
        "build" | "compile" => build_command(filename, &arguments[3..]),
        _ => {
            eprintln!("error: unknown command `{command}`");
            print_usage();
            process::exit(1);
        }
    };
    if let Err(error) = result {
        exit_with_error(error);
    }
}

fn read_source(filename: &str) -> Result<String, String> {
    fs::read_to_string(filename).map_err(|error| format!("could not read `{filename}`: {error}"))
}

fn diagnostic(filename: &str, source: &str, error: &FrontendError) -> String {
    let map = SourceMap::new(source);
    let position = map.position(error.span().start);
    format!(
        "error: {} at {filename}:{}:{}",
        error, position.line, position.column
    )
}

fn frontend_error(filename: &str, source: &str, error: FrontendError) -> String {
    diagnostic(filename, source, &error)
}

fn exit_with_error(error: String) -> ! {
    eprintln!("{error}");
    process::exit(1);
}

fn lex_command(filename: &str, source: &str) -> Result<(), String> {
    let tokens =
        pipeline::lex_source(source).map_err(|error| frontend_error(filename, source, error))?;
    for token in tokens {
        println!(
            "{:>4}..{:<4}  {:<24} {}",
            token.span.start,
            token.span.end,
            token.kind,
            token.lexeme.escape_default()
        );
    }
    Ok(())
}

fn parse_command(filename: &str, source: &str) -> Result<(), String> {
    let program =
        pipeline::parse_source(source).map_err(|error| frontend_error(filename, source, error))?;
    println!("{program:#?}");
    Ok(())
}

fn check_command(filename: &str, source: &str) -> Result<(), String> {
    pipeline::analyze_source(source).map_err(|error| frontend_error(filename, source, error))?;
    println!("semantic analysis succeeded");
    Ok(())
}

fn ir_command(filename: &str, source: &str) -> Result<(), String> {
    let typed = pipeline::analyze_source(source)
        .map_err(|error| frontend_error(filename, source, error))?;
    let context = Context::create();
    let module = CodeGenerator::new(&context, "compy")
        .generate_typed(&typed)
        .map_err(|error| format!("error: codegen error: {error}"))?;
    print!("{}", module.print_to_string().to_string());
    Ok(())
}

fn build_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let output = output_path(filename, arguments)?;
    let object = output.with_extension("o");
    let source = read_source(filename)?;
    let program = pipeline::parse_source(&source)
        .map_err(|error| frontend_error(filename, &source, error))?;
    semantic::validate_entry_point(&program).map_err(|error| {
        let frontend = FrontendError::Semantic(error);
        frontend_error(filename, &source, frontend)
    })?;
    let typed = pipeline::analyze_program(&program)
        .map_err(|error| frontend_error(filename, &source, error))?;

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
    let module = CodeGenerator::with_pointer_width(&context, module_name(&output), pointer_width)
        .generate_typed(&typed)
        .map_err(|error| format!("error: code generation error: {error}"))?;
    module.set_triple(&target_triple);
    module.set_data_layout(&target_data.get_data_layout());
    create_parent_directory(&object)?;
    target_machine
        .write_to_file(&module, FileType::Object, &object)
        .map_err(|error| format!("could not emit object file `{}`: {error}", object.display()))?;
    link_native(&object, &output)?;
    println!("object: {}", object.display());
    println!("executable: {}", output.display());
    Ok(())
}

fn output_path(filename: &str, arguments: &[String]) -> Result<PathBuf, String> {
    let mut output = None;
    let mut position = 0;
    while position < arguments.len() {
        match arguments[position].as_str() {
            "-o" | "--output" => {
                position += 1;
                output = Some(PathBuf::from(arguments.get(position).ok_or_else(|| {
                    "missing output path after `-o`/`--output`".to_string()
                })?));
            }
            argument => return Err(format!("unknown build argument `{argument}`")),
        }
        position += 1;
    }
    if let Some(output) = output {
        return Ok(output);
    }
    let stem = Path::new(filename)
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

fn print_usage() {
    eprintln!("usage:");
    eprintln!("    compiler lex <file>");
    eprintln!("    compiler parse <file>");
    eprintln!("    compiler check <file>");
    eprintln!("    compiler ir <file>");
    eprintln!("    compiler build <file> [-o <executable>]");
    eprintln!("    compiler compile <file> [-o <executable>]");
}
