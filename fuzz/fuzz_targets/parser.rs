#![no_main]
use compiler::pipeline;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    // Parsing is fallible by design; arbitrary input must only produce a
    // structured frontend error.
    let _ = pipeline::parse_source(&source);
});
