#![no_main]
use compiler::interpreter;
use compiler::pipeline;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    let Ok(program) = pipeline::parse_source(&source) else { return };
    let Ok(typed) = pipeline::analyze_program(&program) else { return };
    // The interpreter has a step limit, so arbitrary accepted loops cannot
    // make this target run forever.
    let _ = interpreter::run(&typed);
});
