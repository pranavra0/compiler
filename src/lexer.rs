use std::fmt;

/// A half-open range into the original source string
/// `start..end` means:
///     source[start..end]
/// The end position is exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenKind {
    Eof,

    Identifier,
    Integer,
    Float,
    String,
    Ellipsis,

    // Keywords
    Fn,
    Import,
    Extern,
    Export,
    Let,
    Var,
    If,
    Else,
    Return,
    Struct,
    Enum,
    Union,
    While,
    Break,
    Continue,
    Defer,
    For,
    True,
    False,
    Null,
    /// Explicit compile-time execution marker (`#`).
    Hash,

    // Arithmetic
    Plus,
    Minus,
    Star,
    Slash,
    Percent,

    // Comparison / assignment
    Equal,
    EqualEqual,
    Bang,
    BangEqual,

    Less,
    LessEqual,
    ShiftLeft,
    Greater,
    GreaterEqual,
    ShiftRight,

    // Bitwise / logical
    Ampersand,
    AmpersandAmpersand,
    Pipe,
    PipePipe,
    Caret,
    Tilde,

    // Other operators
    Arrow,       // ->
    FatArrow,    // =>
    Question,    // ? (explicit result propagation)
    ColonEqual,  // :=
    DoubleColon, // ::

    // Punctuation
    Dot,
    Comma,
    Semicolon,
    Colon,

    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
}

impl fmt::Display for TokenKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            TokenKind::Eof => "Eof",

            TokenKind::Identifier => "Identifier",
            TokenKind::Integer => "Integer",
            TokenKind::Float => "Float",
            TokenKind::String => "String",
            TokenKind::Ellipsis => "Ellipsis",

            TokenKind::Fn => "Fn",
            TokenKind::Import => "Import",
            TokenKind::Extern => "Extern",
            TokenKind::Export => "Export",
            TokenKind::Let => "Let",
            TokenKind::Var => "Var",
            TokenKind::If => "If",
            TokenKind::Else => "Else",
            TokenKind::Return => "Return",
            TokenKind::Struct => "Struct",
            TokenKind::Enum => "Enum",
            TokenKind::Union => "Union",
            TokenKind::While => "While",
            TokenKind::Break => "Break",
            TokenKind::Continue => "Continue",
            TokenKind::Defer => "Defer",
            TokenKind::For => "For",
            TokenKind::True => "True",
            TokenKind::False => "False",
            TokenKind::Null => "Null",
            TokenKind::Hash => "Hash",

            TokenKind::Plus => "Plus",
            TokenKind::Minus => "Minus",
            TokenKind::Star => "Star",
            TokenKind::Slash => "Slash",
            TokenKind::Percent => "Percent",

            TokenKind::Equal => "Equal",
            TokenKind::EqualEqual => "EqualEqual",
            TokenKind::Bang => "Bang",
            TokenKind::BangEqual => "BangEqual",

            TokenKind::Less => "Less",
            TokenKind::LessEqual => "LessEqual",
            TokenKind::ShiftLeft => "ShiftLeft",
            TokenKind::Greater => "Greater",
            TokenKind::GreaterEqual => "GreaterEqual",
            TokenKind::ShiftRight => "ShiftRight",

            TokenKind::Ampersand => "Ampersand",
            TokenKind::AmpersandAmpersand => "AmpersandAmpersand",
            TokenKind::Pipe => "Pipe",
            TokenKind::PipePipe => "PipePipe",
            TokenKind::Caret => "Caret",
            TokenKind::Tilde => "Tilde",

            TokenKind::Arrow => "Arrow",
            TokenKind::FatArrow => "FatArrow",
            TokenKind::Question => "Question",
            TokenKind::DoubleColon => "DoubleColon",

            TokenKind::Dot => "Dot",
            TokenKind::Comma => "Comma",
            TokenKind::Semicolon => "Semicolon",
            TokenKind::Colon => "Colon",
            TokenKind::ColonEqual => "ColonEqual",

            TokenKind::LParen => "LParen",
            TokenKind::RParen => "RParen",
            TokenKind::LBracket => "LBracket",
            TokenKind::RBracket => "RBracket",
            TokenKind::LBrace => "LBrace",
            TokenKind::RBrace => "RBrace",
        };

        write!(f, "{name}")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub span: Span,
}

impl Token {
    fn new(kind: TokenKind, source: &str, span: Span) -> Self {
        Self {
            kind,
            lexeme: source[span.start..span.end].to_string(),
            span,
        }
    }
}

/// produced while lexing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    UnexpectedCharacter { character: char, span: Span },

    UnterminatedBlockComment { span: Span },
}

impl fmt::Display for LexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LexError::UnexpectedCharacter { character, span } => {
                write!(
                    f,
                    "unexpected character {:?} at {}..{}",
                    character, span.start, span.end
                )
            }

            LexError::UnterminatedBlockComment { span } => {
                write!(
                    f,
                    "unterminated block comment at {}..{}",
                    span.start, span.end
                )
            }
        }
    }
}

impl std::error::Error for LexError {}

/// `position` is a byte offset into source
pub struct Lexer<'a> {
    source: &'a str,
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            source,
            bytes: source.as_bytes(),
            position: 0,
        }
    }

    pub fn next_token(&mut self) -> Result<Token, LexError> {
        self.skip_whitespace_and_comments()?;

        let start = self.position;

        let Some(byte) = self.peek() else {
            return Ok(Token::new(
                TokenKind::Eof,
                self.source,
                Span::new(start, start),
            ));
        };

        let token = match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_identifier_or_keyword(),

            b'0'..=b'9' => self.lex_number(),

            b'+' => self.single_character_token(TokenKind::Plus),
            b'-' => self.lex_minus(),
            b'*' => self.single_character_token(TokenKind::Star),
            b'/' => self.single_character_token(TokenKind::Slash),
            b'%' => self.single_character_token(TokenKind::Percent),

            b'=' => self.lex_equal(),
            b'!' => self.lex_bang(),
            b'#' => self.single_character_token(TokenKind::Hash),
            b'?' => self.single_character_token(TokenKind::Question),
            b'"' => self.lex_string(),
            b'.' if self.peek_next() == Some(b'.')
                && self.bytes.get(self.position + 2) == Some(&b'.') =>
            {
                self.lex_ellipsis()
            }

            b'<' => self.lex_less(),
            b'>' => self.lex_greater(),

            b'&' => self.lex_ampersand(),
            b'|' => self.lex_pipe(),
            b'^' => self.single_character_token(TokenKind::Caret),
            b'~' => self.single_character_token(TokenKind::Tilde),

            b'.' => self.single_character_token(TokenKind::Dot),
            b',' => self.single_character_token(TokenKind::Comma),
            b';' => self.single_character_token(TokenKind::Semicolon),
            b':' => self.lex_colon(),

            b'(' => self.single_character_token(TokenKind::LParen),
            b')' => self.single_character_token(TokenKind::RParen),

            b'[' => self.single_character_token(TokenKind::LBracket),
            b']' => self.single_character_token(TokenKind::RBracket),

            b'{' => self.single_character_token(TokenKind::LBrace),
            b'}' => self.single_character_token(TokenKind::RBrace),

            _ => {
                let character = self.bump().expect("peek returned Some");

                return Err(LexError::UnexpectedCharacter {
                    character,
                    span: Span::new(start, self.position),
                });
            }
        };

        Ok(token)
    }

    /// The first character has already been checked by the caller.
    fn lex_identifier_or_keyword(&mut self) -> Token {
        let start = self.position;

        self.bump();

        while let Some(byte) = self.peek() {
            if is_identifier_continue(byte) {
                self.bump();
            } else {
                break;
            }
        }

        let span = Span::new(start, self.position);
        let text = &self.source[span.start..span.end];

        let kind = keyword_kind(text).unwrap_or(TokenKind::Identifier);

        Token::new(kind, self.source, span)
    }

    /// Read either an integer or floating-point litera
    fn lex_string(&mut self) -> Token {
        let start = self.position;
        self.bump();
        while let Some(byte) = self.peek() {
            self.bump();
            if byte == b'"' {
                break;
            }
        }
        Token::new(
            TokenKind::String,
            self.source,
            Span::new(start, self.position),
        )
    }

    fn lex_ellipsis(&mut self) -> Token {
        let start = self.position;
        self.bump();
        self.bump();
        self.bump();
        Token::new(
            TokenKind::Ellipsis,
            self.source,
            Span::new(start, self.position),
        )
    }

    fn lex_number(&mut self) -> Token {
        let start = self.position;

        self.consume_digits();

        let is_float = match (self.peek(), self.peek_next()) {
            (Some(b'.'), Some(next)) if next.is_ascii_digit() => {
                self.bump(); // '.'
                self.consume_digits();
                true
            }

            _ => false,
        };

        let span = Span::new(start, self.position);

        let kind = if is_float {
            TokenKind::Float
        } else {
            TokenKind::Integer
        };

        Token::new(kind, self.source, span)
    }

    fn consume_digits(&mut self) {
        while let Some(byte) = self.peek() {
            if byte.is_ascii_digit() || byte == b'_' {
                self.bump();
            } else {
                break;
            }
        }
    }

    fn lex_minus(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'>') {
            self.bump();
            TokenKind::Arrow
        } else {
            TokenKind::Minus
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_equal(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::EqualEqual
        } else if self.peek() == Some(b'>') {
            self.bump();
            TokenKind::FatArrow
        } else {
            TokenKind::Equal
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_bang(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::BangEqual
        } else {
            TokenKind::Bang
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_less(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::LessEqual
        } else if self.peek() == Some(b'<') {
            self.bump();
            TokenKind::ShiftLeft
        } else {
            TokenKind::Less
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_greater(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'=') {
            self.bump();
            TokenKind::GreaterEqual
        } else if self.peek() == Some(b'>') {
            self.bump();
            TokenKind::ShiftRight
        } else {
            TokenKind::Greater
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_ampersand(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'&') {
            self.bump();
            TokenKind::AmpersandAmpersand
        } else {
            TokenKind::Ampersand
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_pipe(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = if self.peek() == Some(b'|') {
            self.bump();
            TokenKind::PipePipe
        } else {
            TokenKind::Pipe
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn lex_colon(&mut self) -> Token {
        let start = self.position;

        self.bump();

        let kind = match self.peek() {
            Some(b':') => {
                self.bump();
                TokenKind::DoubleColon
            }

            Some(b'=') => {
                self.bump();
                TokenKind::ColonEqual
            }

            _ => TokenKind::Colon,
        };

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn single_character_token(&mut self, kind: TokenKind) -> Token {
        let start = self.position;

        self.bump();

        Token::new(kind, self.source, Span::new(start, self.position))
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<(), LexError> {
        loop {
            while let Some(byte) = self.peek() {
                if byte.is_ascii_whitespace() {
                    self.bump();
                } else {
                    break;
                }
            }

            if self.peek() == Some(b'/') && self.peek_next() == Some(b'/') {
                self.skip_line_comment();
                continue;
            }

            if self.peek() == Some(b'/') && self.peek_next() == Some(b'*') {
                self.skip_block_comment()?;
                continue;
            }

            break;
        }

        Ok(())
    }

    fn skip_line_comment(&mut self) {
        self.bump(); // first '/'
        self.bump(); // second '/'

        while let Some(byte) = self.peek() {
            self.bump();

            if byte == b'\n' {
                break;
            }
        }
    }

    fn skip_block_comment(&mut self) -> Result<(), LexError> {
        let start = self.position;

        self.bump(); // '/'
        self.bump(); // '*'

        loop {
            match self.peek() {
                None => {
                    return Err(LexError::UnterminatedBlockComment {
                        span: Span::new(start, self.position),
                    });
                }

                Some(b'*') if self.peek_next() == Some(b'/') => {
                    self.bump();
                    self.bump();
                    return Ok(());
                }

                Some(_) => {
                    self.bump();
                }
            }
        }
    }

    /// Look at the current byte without consuming it.
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.position).copied()
    }

    /// Look one byte beyond the current byte.
    fn peek_next(&self) -> Option<u8> {
        self.bytes.get(self.position + 1).copied()
    }

    /// Consume one UTF-8 character.
    ///
    /// Chapter 1 intentionally recognizes ASCII syntax.
    /// This method still decodes a UTF-8 character so that errors can
    /// report the actual character rather than just a raw byte.
    fn bump(&mut self) -> Option<char> {
        let remaining = &self.source[self.position..];
        let character = remaining.chars().next()?;

        self.position += character.len_utf8();

        Some(character)
    }
}

fn is_identifier_continue(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn keyword_kind(text: &str) -> Option<TokenKind> {
    Some(match text {
        "fn" => TokenKind::Fn,
        "import" => TokenKind::Import,
        "extern" => TokenKind::Extern,
        "export" => TokenKind::Export,
        "let" => TokenKind::Let,
        "var" => TokenKind::Var,
        "if" => TokenKind::If,
        "else" => TokenKind::Else,
        "return" => TokenKind::Return,
        "struct" => TokenKind::Struct,
        "enum" => TokenKind::Enum,
        "union" => TokenKind::Union,
        "while" => TokenKind::While,
        "break" => TokenKind::Break,
        "continue" => TokenKind::Continue,
        "defer" => TokenKind::Defer,
        "for" => TokenKind::For,
        "true" => TokenKind::True,
        "false" => TokenKind::False,
        "null" => TokenKind::Null,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(source: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(source);
        let mut tokens = Vec::new();

        loop {
            let token = lexer.next_token().expect("lexing failed");

            let done = token.kind == TokenKind::Eof;

            tokens.push(token);

            if done {
                break;
            }
        }

        tokens
    }

    #[test]
    fn lex_keywords_and_identifiers() {
        let tokens = lex_all("fn main let value return while break continue");

        let kinds: Vec<_> = tokens.iter().map(|token| token.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::Fn,
                TokenKind::Identifier,
                TokenKind::Let,
                TokenKind::Identifier,
                TokenKind::Return,
                TokenKind::While,
                TokenKind::Break,
                TokenKind::Continue,
                TokenKind::Eof,
            ]
        );

        assert_eq!(tokens[1].lexeme, "main");
        assert_eq!(tokens[3].lexeme, "value");
    }

    #[test]
    fn lex_numbers() {
        let tokens = lex_all("123 1_000 12.34 1_000.25");

        assert_eq!(tokens[0].kind, TokenKind::Integer);
        assert_eq!(tokens[0].lexeme, "123");

        assert_eq!(tokens[1].kind, TokenKind::Integer);
        assert_eq!(tokens[1].lexeme, "1_000");

        assert_eq!(tokens[2].kind, TokenKind::Float);
        assert_eq!(tokens[2].lexeme, "12.34");

        assert_eq!(tokens[3].kind, TokenKind::Float);
        assert_eq!(tokens[3].lexeme, "1_000.25");

        assert_eq!(tokens[4].kind, TokenKind::Eof);
    }

    #[test]
    fn lex_punctuation() {
        let tokens = lex_all("() [] {} , ; : := :: .");

        let kinds: Vec<_> = tokens.iter().map(|token| token.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBracket,
                TokenKind::RBracket,
                TokenKind::LBrace,
                TokenKind::RBrace,
                TokenKind::Comma,
                TokenKind::Semicolon,
                TokenKind::Colon,
                TokenKind::ColonEqual,
                TokenKind::DoubleColon,
                TokenKind::Dot,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_two_character_operators() {
        let tokens = lex_all("== != <= >= << >> -> => && ||");

        let kinds: Vec<_> = tokens.iter().map(|token| token.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::EqualEqual,
                TokenKind::BangEqual,
                TokenKind::LessEqual,
                TokenKind::GreaterEqual,
                TokenKind::ShiftLeft,
                TokenKind::ShiftRight,
                TokenKind::Arrow,
                TokenKind::FatArrow,
                TokenKind::AmpersandAmpersand,
                TokenKind::PipePipe,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn comments_are_ignored() {
        let tokens = lex_all(
            r#"
            // this is a comment
            fn main() {
                /* another comment */
                let x = 42;
            }
            "#,
        );

        let kinds: Vec<_> = tokens.iter().map(|token| token.kind).collect();

        assert_eq!(
            kinds,
            vec![
                TokenKind::Fn,
                TokenKind::Identifier,
                TokenKind::LParen,
                TokenKind::RParen,
                TokenKind::LBrace,
                TokenKind::Let,
                TokenKind::Identifier,
                TokenKind::Equal,
                TokenKind::Integer,
                TokenKind::Semicolon,
                TokenKind::RBrace,
                TokenKind::Eof,
            ]
        );
    }

    #[test]
    fn spans_point_into_original_source() {
        let source = "fn main()";

        let tokens = lex_all(source);

        assert_eq!(tokens[0].span, Span::new(0, 2));
        assert_eq!(tokens[0].lexeme, "fn");

        assert_eq!(tokens[1].span, Span::new(3, 7));
        assert_eq!(tokens[1].lexeme, "main");

        assert_eq!(tokens[2].span, Span::new(7, 8));
        assert_eq!(tokens[2].lexeme, "(");
    }

    #[test]
    fn unexpected_character_is_an_error() {
        let mut lexer = Lexer::new("@");

        let error = lexer.next_token().unwrap_err();

        assert_eq!(
            error,
            LexError::UnexpectedCharacter {
                character: '@',
                span: Span::new(0, 1),
            }
        );
    }

    #[test]
    fn unterminated_block_comment_is_an_error() {
        let mut lexer = Lexer::new("/* hello");

        let error = lexer.next_token().unwrap_err();

        assert_eq!(
            error,
            LexError::UnterminatedBlockComment {
                span: Span::new(0, 8),
            }
        );
    }
}
