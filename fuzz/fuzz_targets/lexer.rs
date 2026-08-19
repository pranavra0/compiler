#![no_main]
use compiler::lexer::Lexer;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|bytes: &[u8]| {
    let source = String::from_utf8_lossy(bytes);
    let mut lexer = Lexer::new(&source);
    // Errors are expected. The invariant is that they do not panic or loop.
    loop {
        match lexer.next_token() {
            Ok(token) if token.kind == compiler::lexer::TokenKind::Eof => break,
            Ok(_) => {}
            Err(_) => break,
        }
    }
});
