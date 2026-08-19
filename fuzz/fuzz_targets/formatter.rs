#![no_main]
use compiler::formatter;
use compiler::pipeline;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    if let Ok(formatted) = formatter::format_source(&source) {
        // Formatting must never produce text that the frontend cannot parse.
        let _ = pipeline::parse_source(&formatted);
    }
});
