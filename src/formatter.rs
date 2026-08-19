//! Deterministic source formatter. It formats the token stream rather than the
//! AST, so it remains usable for programs containing experimental constructs.
use crate::lexer::{Lexer, TokenKind};

pub fn format_source(source: &str) -> Result<String, String> {
    let comments = comments(source);
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        let token = lexer.next_token().map_err(|e| e.to_string())?;
        let eof = token.kind == TokenKind::Eof;
        tokens.push(token);
        if eof {
            break;
        }
    }
    let mut out = String::new();
    let mut indent = 0usize;
    let mut line_start = true;
    let mut need_space = false;
    let mut prev: Option<TokenKind> = None;
    let write_indent = |out: &mut String, line_start: &mut bool, indent: usize| {
        if *line_start {
            out.push_str(&"    ".repeat(indent));
            *line_start = false;
        }
    };
    let mut comment_index = 0;
    for token in tokens.into_iter().filter(|t| t.kind != TokenKind::Eof) {
        while comment_index < comments.len() && comments[comment_index].end <= token.span.start {
            if !line_start && !out.ends_with('\n') {
                out.push(' ');
            }
            write_indent(&mut out, &mut line_start, indent);
            out.push_str(comments[comment_index].text.trim());
            out.push('\n');
            line_start = true;
            need_space = false;
            prev = None;
            comment_index += 1;
        }
        let k = token.kind;
        if matches!(prev, Some(TokenKind::RBrace))
            && !matches!(
                k,
                TokenKind::Else
                    | TokenKind::Semicolon
                    | TokenKind::Comma
                    | TokenKind::RParen
                    | TokenKind::RBracket
            )
            && !line_start
        {
            out.push('\n');
            line_start = true;
            need_space = false;
        }
        if k == TokenKind::LBrace {
            if !line_start && !out.ends_with(' ') {
                out.push(' ');
            }
            write_indent(&mut out, &mut line_start, indent);
            out.push('{');
            out.push('\n');
            line_start = true;
            indent += 1;
            need_space = false;
        } else if k == TokenKind::RBrace {
            if !line_start {
                out.push('\n');
            }
            indent = indent.saturating_sub(1);
            line_start = true;
            write_indent(&mut out, &mut line_start, indent);
            out.push('}');
            need_space = true;
        } else if k == TokenKind::Semicolon {
            out = out.trim_end().to_string();
            out.push(';');
            out.push('\n');
            line_start = true;
            need_space = false;
        } else if k == TokenKind::Comma {
            out = out.trim_end().to_string();
            out.push_str(", ");
            line_start = false;
            need_space = false;
        } else if k == TokenKind::LParen {
            write_indent(&mut out, &mut line_start, indent);
            if need_space && !matches!(prev, Some(TokenKind::Fn)) {
                out.push(' ');
            }
            out.push('(');
            need_space = false;
        } else if k == TokenKind::RParen || k == TokenKind::RBracket {
            out = out.trim_end().to_string();
            out.push_str(token.lexeme.as_str());
            need_space = true;
        } else if k == TokenKind::LBracket {
            write_indent(&mut out, &mut line_start, indent);
            out.push('[');
            need_space = false;
        } else if k == TokenKind::Dot {
            out = out.trim_end().to_string();
            out.push('.');
            need_space = false;
        } else if k == TokenKind::Colon {
            out = out.trim_end().to_string();
            out.push_str(": ");
            need_space = false;
        } else if matches!(
            k,
            TokenKind::DoubleColon
                | TokenKind::Arrow
                | TokenKind::Equal
                | TokenKind::EqualEqual
                | TokenKind::BangEqual
                | TokenKind::Less
                | TokenKind::LessEqual
                | TokenKind::Greater
                | TokenKind::GreaterEqual
                | TokenKind::Plus
                | TokenKind::Minus
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::AmpersandAmpersand
                | TokenKind::PipePipe
                | TokenKind::Ampersand
                | TokenKind::Pipe
                | TokenKind::Caret
                | TokenKind::ShiftLeft
                | TokenKind::ShiftRight
        ) {
            out = out.trim_end().to_string();
            out.push(' ');
            out.push_str(&token.lexeme);
            out.push(' ');
            need_space = false;
        } else {
            write_indent(&mut out, &mut line_start, indent);
            if (need_space || matches!(prev, Some(TokenKind::RBrace)))
                && !out.ends_with(' ')
                && !out.ends_with('\n')
            {
                out.push(' ');
            }
            out.push_str(&token.lexeme);
            need_space = true;
        }
        prev = Some(k);
    }
    while comment_index < comments.len() {
        if !line_start && !out.ends_with('\n') {
            out.push(' ');
        }
        write_indent(&mut out, &mut line_start, indent);
        out.push_str(comments[comment_index].text.trim());
        out.push('\n');
        line_start = true;
        comment_index += 1;
    }
    let mut result = out.trim_end().to_string();
    result.push('\n');
    Ok(result)
}

#[derive(Debug, Clone)]
struct Comment {
    end: usize,
    text: String,
}

/// Extract comments without interpreting their contents.
fn comments(source: &str) -> Vec<Comment> {
    let bytes = source.as_bytes();
    let mut i = 0;
    let mut result = Vec::new();
    while i < bytes.len() {
        if bytes[i] == b'"' {
            i += 1;
            while i < bytes.len() {
                let escaped = bytes[i] == b'\\';
                i += 1;
                if !escaped && i <= bytes.len() && bytes[i - 1] == b'"' {
                    break;
                }
            }
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'/' {
            let start = i;
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            result.push(Comment {
                end: i,
                text: source[start..i].to_string(),
            });
            continue;
        }
        if i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            result.push(Comment {
                end: i,
                text: source[start..i].to_string(),
            });
            continue;
        }
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn formatter_is_idempotent_and_keeps_comments() {
        let s = "// keep\nmain::()->i32{return 1;}";
        let once = format_source(s).unwrap();
        assert!(once.contains("// keep"));
        assert!(crate::pipeline::parse_source(&once).is_ok());
        assert_eq!(format_source(&once).unwrap(), once);
    }
}
