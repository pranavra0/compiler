#![no_main]
use compiler::codegen::CodeGenerator;
use compiler::pipeline;
use inkwell::context::Context;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    let Ok(program) = pipeline::parse_source(&source) else { return };
    let Ok(typed) = pipeline::analyze_program(&program) else { return };
    let context = Context::create();
    // A successful frontend result must either verify as typed IR or return a
    // structured code-generation error; this target is primarily a panic check.
    let _ = CodeGenerator::new(&context, "fuzz").generate_typed(&typed);
});
