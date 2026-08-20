use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use compiler::codegen::CodeGenerator;
use compiler::comptime;
use compiler::formatter;
use compiler::interpreter;
use compiler::modules;
use compiler::pipeline::{self, FrontendError};
use compiler::semantic;
use compiler::source::SourceMap;
use inkwell::OptimizationLevel;
use inkwell::context::Context;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};

fn main() {
    let arguments: Vec<String> = env::args().collect();
    if arguments.len() < 2 {
        print_usage();
        process::exit(1);
    }
    let command = &arguments[1];
    if command == "fmt" {
        let result = fmt_command(&arguments[2..]);
        if let Err(error) = result {
            exit_with_error(error);
        }
        return;
    }
    if arguments.len() < 3 {
        print_usage();
        process::exit(1);
    }
    let filename = &arguments[2];

    let result = match command.as_str() {
        "lex" | "parse" => {
            if arguments.len() != 3 {
                print_usage();
                process::exit(1);
            }
            let source = match read_source(filename) {
                Ok(source) => source,
                Err(error) => exit_with_error(error),
            };
            if command == "lex" {
                lex_command(filename, &source)
            } else {
                parse_command(filename, &source)
            }
        }
        "check" | "ir" => {
            if command == "check" {
                check_command(filename, &arguments[3..])
            } else {
                ir_command(filename, &arguments[3..])
            }
        }
        "run" | "interpret" => run_command(filename, &arguments[3..]),
        "reflect" => reflect_command(filename, &arguments[3..]),
        "generated" => generated_command(filename, &arguments[3..]),
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

fn reflect_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    if arguments.len() != 1 && arguments.len() != 2 {
        return Err(
            "usage: compiler reflect <file> <type> | compiler reflect <file> function <name>"
                .into(),
        );
    }
    let source = read_source(filename)?;
    let program = pipeline::parse_source(&source)
        .map_err(|error| frontend_error(filename, &source, error))?;
    if arguments.len() == 2 {
        if arguments[0] != "function" {
            return Err("reflect kind must be `function`".into());
        }
        let info = comptime::reflect_function(&program, &arguments[1]).map_err(|error| {
            format!(
                "comptime error: {error} at {filename}:{}",
                SourceMap::new(&source).position(error.span().start).line
            )
        })?;
        println!("{info:#?}");
    } else {
        let info =
            comptime::reflect_type(&program, &arguments[0], usize::BITS).map_err(|error| {
                format!(
                    "comptime error: {error} at {filename}:{}",
                    SourceMap::new(&source).position(error.span().start).line
                )
            })?;
        println!("{info:#?}");
    }
    Ok(())
}

fn generated_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    if !arguments.is_empty() {
        return Err("usage: compiler generated <file>".into());
    }
    let source = read_source(filename)?;
    let program = pipeline::parse_source(&source)
        .map_err(|error| frontend_error(filename, &source, error))?;
    let expanded = comptime::expand(&program, usize::BITS).map_err(|error| {
        format!(
            "comptime error: {error} at {filename}:{}",
            SourceMap::new(&source).position(error.span().start).line
        )
    })?;
    println!("{expanded:#?}");
    Ok(())
}

fn read_source(filename: &str) -> Result<String, String> {
    fs::read_to_string(filename).map_err(|error| format!("could not read `{filename}`: {error}"))
}

fn fmt_command(arguments: &[String]) -> Result<(), String> {
    let check = arguments.iter().any(|arg| arg == "--check");
    let files: Vec<&String> = arguments
        .iter()
        .filter(|arg| arg.as_str() != "--check")
        .collect();
    if files.is_empty() {
        return Err("usage: compiler fmt [--check] <files...>".into());
    }
    for filename in files {
        let source = read_source(filename)?;
        let formatted = formatter::format_source(&source)?;
        if check {
            if formatted != source {
                return Err(format!("{filename} is not formatted"));
            }
        } else {
            fs::write(filename, formatted)
                .map_err(|error| format!("could not write `{filename}`: {error}"))?;
        }
    }
    Ok(())
}

fn run_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let options = BuildOptions::parse(arguments)?;
    let project = project(filename, arguments)?;
    let pointer_width = pointer_width_for(&options)?;
    let typed = project
        .analyze(pointer_width)
        .map_err(|error| frontend_error(filename, &project.root_source, error))?;
    let status = interpreter::run_with_pointer_width(&typed, pointer_width)
        .map_err(|error| format!("interpreter error: {error}"))?;
    process::exit(status);
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

fn project(filename: &str, arguments: &[String]) -> Result<modules::Project, String> {
    let roots = module_roots(arguments)?;
    modules::resolve(filename, &roots).map_err(|error| {
        let source = read_source(filename).unwrap_or_default();
        format!(
            "error: module error: {} at {filename}:{}:{}",
            error,
            SourceMap::new(&source).position(error.span.start).line,
            SourceMap::new(&source).position(error.span.start).column
        )
    })
}

fn check_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let options = BuildOptions::parse(arguments)?;
    let project = project(filename, arguments)?;
    project
        .analyze(pointer_width_for(&options)?)
        .map_err(|error| frontend_error(filename, &project.root_source, error))?;
    println!("semantic analysis succeeded");
    Ok(())
}

fn ir_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let options = BuildOptions::parse(arguments)?;
    let project = project(filename, arguments)?;
    let typed = project
        .analyze(pointer_width_for(&options)?)
        .map_err(|error| frontend_error(filename, &project.root_source, error))?;
    let context = Context::create();
    let module = if options.target.is_some() {
        Target::initialize_all(&InitializationConfig::default());
        let triple = options.target.as_deref().map(TargetTriple::create).unwrap();
        let target =
            Target::from_triple(&triple).map_err(|e| format!("could not find target: {e}"))?;
        let machine = target
            .create_target_machine(
                &triple,
                "generic",
                "",
                optimization_level(options.opt_level),
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| "could not create target machine".to_string())?;
        let data = machine.get_target_data();
        let generator = CodeGenerator::with_target_data(&context, "compy", &data);
        let generator = if options.debug {
            generator.with_debug_info_source(filename, &project.root_source, options.opt_level > 0)
        } else {
            generator
        };
        let module = generator
            .generate_typed(&typed)
            .map_err(|error| format!("error: codegen error: {error}"))?;
        module.set_triple(&triple);
        module
    } else {
        {
            let generator = CodeGenerator::new(&context, "compy");
            let generator = if options.debug {
                generator.with_debug_info_source(
                    filename,
                    &project.root_source,
                    options.opt_level > 0,
                )
            } else {
                generator
            };
            generator
                .generate_typed(&typed)
                .map_err(|error| format!("error: codegen error: {error}"))?
        }
    };
    let text = module.print_to_string().to_string();
    if let Some(output) = options.output {
        ensure_not_input(filename, &output)?;
        create_parent_directory(&output)?;
        fs::write(&output, text)
            .map_err(|error| format!("could not emit IR file `{}`: {error}", output.display()))?;
        println!("ir: {}", output.display());
    } else {
        print!("{text}");
    }
    Ok(())
}

fn build_command(filename: &str, arguments: &[String]) -> Result<(), String> {
    let options = BuildOptions::parse(arguments)?;
    let output = if options.emit_ir {
        options
            .output
            .clone()
            .unwrap_or_else(|| Path::new(filename).with_extension("ll"))
    } else if options.emit_assembly {
        options
            .output
            .clone()
            .unwrap_or_else(|| Path::new(filename).with_extension("s"))
    } else if options.emit_object {
        options
            .output
            .clone()
            .unwrap_or_else(|| Path::new(filename).with_extension("o"))
    } else {
        options
            .output
            .clone()
            .unwrap_or(output_path(filename, &[])?)
    };
    let object = if options.emit_object {
        output.clone()
    } else {
        output.with_extension("o")
    };
    ensure_not_input(filename, &output)?;
    let project = project(filename, arguments)?;
    if let Some(depfile) = &options.depfile {
        write_dependency_file(depfile, &output, &project.dependencies)?;
    }
    Target::initialize_all(&InitializationConfig::default());
    let target_triple = options
        .target
        .as_deref()
        .map(TargetTriple::create)
        .unwrap_or_else(TargetMachine::get_default_triple);
    let target = Target::from_triple(&target_triple)
        .map_err(|error| format!("could not find the native LLVM target: {error}"))?;
    let target_machine = target
        .create_target_machine(
            &target_triple,
            "generic",
            "",
            optimization_level(options.opt_level),
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "could not create the native LLVM target machine".to_string())?;
    let target_data = target_machine.get_target_data();
    if !options.emit_object && !options.emit_ir && !options.emit_assembly {
        semantic::validate_entry_point(&project.program).map_err(|error| {
            let frontend = FrontendError::Semantic(error);
            frontend_error(filename, &project.root_source, frontend)
        })?;
    }
    let typed = project
        .analyze(target_data.get_pointer_byte_size(None) * 8)
        .map_err(|error| frontend_error(filename, &project.root_source, error))?;
    let context = Context::create();
    let generator = CodeGenerator::with_target_data(&context, module_name(&output), &target_data);
    let generator = if options.debug {
        generator.with_debug_info_source(filename, &project.root_source, options.opt_level > 0)
    } else {
        generator
    };
    let module = generator
        .generate_typed(&typed)
        .map_err(|error| format!("error: code generation error: {error}"))?;
    module.set_triple(&target_triple);
    module.set_data_layout(&target_data.get_data_layout());
    if options.emit_ir {
        create_parent_directory(&output)?;
        fs::write(&output, module.print_to_string().to_string())
            .map_err(|error| format!("could not emit IR file `{}`: {error}", output.display()))?;
        println!("ir: {}", output.display());
        return Ok(());
    }
    if options.emit_assembly {
        create_parent_directory(&output)?;
        target_machine
            .write_to_file(&module, FileType::Assembly, &output)
            .map_err(|error| {
                format!(
                    "could not emit assembly file `{}`: {error}",
                    output.display()
                )
            })?;
        println!("assembly: {}", output.display());
        return Ok(());
    }
    create_parent_directory(&object)?;
    target_machine
        .write_to_file(&module, FileType::Object, &object)
        .map_err(|error| format!("could not emit object file `{}`: {error}", object.display()))?;
    println!("object: {}", object.display());
    if options.emit_object {
        return Ok(());
    }
    link_native(&object, &output, &options)?;
    println!("executable: {}", output.display());
    Ok(())
}

fn ensure_not_input(filename: &str, output: &Path) -> Result<(), String> {
    let input = Path::new(filename);
    let same_canonical = match (fs::canonicalize(output), fs::canonicalize(input)) {
        (Ok(output), Ok(input)) => output == input,
        _ => false,
    };
    if output == input || same_canonical {
        return Err(format!("refusing to overwrite input source `{filename}`"));
    }
    Ok(())
}

fn output_path(filename: &str, _arguments: &[String]) -> Result<PathBuf, String> {
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
#[derive(Debug, Default)]
struct BuildOptions {
    output: Option<PathBuf>,
    library_paths: Vec<PathBuf>,
    libraries: Vec<String>,
    objects: Vec<PathBuf>,
    target: Option<String>,
    linker: Option<PathBuf>,
    link_args: Vec<String>,
    depfile: Option<PathBuf>,
    opt_level: u8,
    emit_object: bool,
    emit_ir: bool,
    emit_assembly: bool,
    emit_executable: bool,
    debug: bool,
}
impl BuildOptions {
    fn parse(arguments: &[String]) -> Result<Self, String> {
        let mut out = Self::default();
        let mut i = 0;
        while i < arguments.len() {
            let flag = arguments[i].as_str();
            let value = |i: &mut usize, name: &str| -> Result<String, String> {
                *i += 1;
                arguments
                    .get(*i)
                    .cloned()
                    .ok_or_else(|| format!("missing value after `{name}`"))
            };
            match flag {
                "-o" | "--output" => out.output = Some(PathBuf::from(value(&mut i, flag)?)),
                "-g" | "--debug" => out.debug = true,
                "-O0" => out.opt_level = 0,
                "-O1" => out.opt_level = 1,
                "-O2" => out.opt_level = 2,
                "-O3" => out.opt_level = 3,
                "-I" | "--module-root" => {
                    let _ = value(&mut i, flag)?;
                }
                "-L" => out.library_paths.push(PathBuf::from(value(&mut i, flag)?)),
                "-l" | "--library" => out.libraries.push(value(&mut i, flag)?),
                "--object" => out.objects.push(PathBuf::from(value(&mut i, flag)?)),
                "--target" => out.target = Some(value(&mut i, flag)?),
                "--linker" => out.linker = Some(PathBuf::from(value(&mut i, flag)?)),
                "--link-arg" => out.link_args.push(value(&mut i, flag)?),
                "--depfile" => out.depfile = Some(PathBuf::from(value(&mut i, flag)?)),
                "-O" | "--opt-level" => {
                    out.opt_level = value(&mut i, flag)?
                        .parse()
                        .map_err(|_| "optimization level must be 0, 1, 2, or 3".to_string())?;
                    if out.opt_level > 3 {
                        return Err("optimization level must be 0, 1, 2, or 3".into());
                    }
                }
                "--emit-object" | "--emit-obj" | "--object-only" => out.emit_object = true,
                "--emit-ir" => out.emit_ir = true,
                "--emit-asm" => out.emit_assembly = true,
                "--emit-exe" => out.emit_executable = true,
                other => return Err(format!("unknown build argument `{other}`")),
            }
            i += 1;
        }
        let artifact_count = [
            out.emit_ir,
            out.emit_assembly,
            out.emit_object,
            out.emit_executable,
        ]
        .into_iter()
        .filter(|selected| *selected)
        .count();
        if artifact_count > 1 {
            return Err("artifact emission options are mutually exclusive".into());
        }
        if out.emit_object && out.emit_executable {
            return Err("artifact options --emit-obj and --emit-exe are mutually exclusive".into());
        }
        Ok(out)
    }
}
fn optimization_level(level: u8) -> OptimizationLevel {
    match level {
        0 => OptimizationLevel::None,
        1 => OptimizationLevel::Less,
        2 => OptimizationLevel::Default,
        _ => OptimizationLevel::Aggressive,
    }
}
fn pointer_width_for(options: &BuildOptions) -> Result<u32, String> {
    let Some(target_name) = options.target.as_deref() else {
        return Ok(usize::BITS);
    };
    Target::initialize_all(&InitializationConfig::default());
    let triple = TargetTriple::create(target_name);
    let target = Target::from_triple(&triple).map_err(|e| format!("could not find target: {e}"))?;
    let machine = target
        .create_target_machine(
            &triple,
            "generic",
            "",
            optimization_level(options.opt_level),
            RelocMode::PIC,
            CodeModel::Default,
        )
        .ok_or_else(|| "could not create target machine".to_string())?;
    Ok(machine.get_target_data().get_pointer_byte_size(None) * 8)
}
fn module_roots(arguments: &[String]) -> Result<Vec<PathBuf>, String> {
    let mut roots = Vec::new();
    let mut i = 0;
    while i < arguments.len() {
        if matches!(arguments[i].as_str(), "-I" | "--module-root") {
            i += 1;
            roots.push(PathBuf::from(
                arguments
                    .get(i)
                    .ok_or_else(|| "missing module root".to_string())?,
            ));
        }
        i += 1;
    }
    Ok(roots)
}

fn write_dependency_file(
    path: &Path,
    output: &Path,
    dependencies: &[PathBuf],
) -> Result<(), String> {
    create_parent_directory(path)?;
    let mut text = format!("{}:", output.display());
    for dependency in dependencies {
        text.push(' ');
        text.push_str(&dependency.display().to_string());
    }
    text.push('\n');
    fs::write(path, text).map_err(|error| {
        format!(
            "could not write dependency file `{}`: {error}",
            path.display()
        )
    })
}

fn link_native(object: &Path, output: &Path, options: &BuildOptions) -> Result<(), String> {
    create_parent_directory(output)?;
    let compiler = options
        .linker
        .clone()
        .or_else(|| env::var_os("CC").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("clang"));
    let mut command = Command::new(&compiler);
    if let Some(target) = &options.target {
        command.arg(format!("--target={target}"));
    }
    command.arg(object);
    for item in &options.objects {
        command.arg(item);
    }
    for path in &options.library_paths {
        command.arg("-L").arg(path);
    }
    for library in &options.libraries {
        command.arg("-l").arg(library);
    }
    command.args(&options.link_args);
    let status = command
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
    eprintln!("    compiler reflect <file> <type>");
    eprintln!("    compiler reflect <file> function <name>");
    eprintln!("    compiler generated <file>");
    eprintln!("    compiler fmt [--check] <files...>");
    eprintln!("    compiler run <file> [--target <triple>]");
    eprintln!("    compiler check <file> [-I <module-root>]");
    eprintln!("    compiler ir <file> [-I <module-root>]");
    eprintln!(
        "    compiler build <file> [-o <path>] [-I <module-root>] [-L <path>] [-l <library>] [--object <file>] [--target <triple>] [--linker <path>] [--link-arg <arg>] [-O0|-O1|-O2|-O3] [-g] [--emit-ir|--emit-asm|--emit-obj|--emit-exe] [--depfile <path>]"
    );
    eprintln!("    compiler compile <file> [-o <executable>]");
}
