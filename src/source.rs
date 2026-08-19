use crate::lexer::Span;

/// One-based source position. Columns are UTF-8 byte columns for now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePosition {
    pub line: usize,
    pub column: usize,
}

/// Maps byte spans to human-readable locations without changing the spans
/// stored in tokens and AST/IR nodes.
#[derive(Debug, Clone)]
pub struct SourceMap {
    line_starts: Vec<usize>,
}

impl SourceMap {
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0];
        for (offset, byte) in source.as_bytes().iter().enumerate() {
            if *byte == b'\n' {
                line_starts.push(offset + 1);
            }
        }
        Self { line_starts }
    }

    pub fn position(&self, byte_offset: usize) -> SourcePosition {
        let offset = self
            .line_starts
            .partition_point(|start| *start <= byte_offset);
        let line_index = offset.saturating_sub(1);
        SourcePosition {
            line: line_index + 1,
            // Deliberately byte-based: UTF-8 character columns can be added
            // later without changing Span's byte-range contract.
            column: byte_offset.saturating_sub(self.line_starts[line_index]) + 1,
        }
    }

    pub fn span_positions(&self, span: Span) -> (SourcePosition, SourcePosition) {
        (self.position(span.start), self.position(span.end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_beginning_middle_and_end_of_lines() {
        let map = SourceMap::new("abc\ndef\nghi");
        assert_eq!(map.position(0), SourcePosition { line: 1, column: 1 });
        assert_eq!(map.position(5), SourcePosition { line: 2, column: 2 });
        assert_eq!(map.position(10), SourcePosition { line: 3, column: 3 });
        assert_eq!(map.position(11), SourcePosition { line: 3, column: 4 });
    }

    #[test]
    fn columns_are_utf8_bytes() {
        let map = SourceMap::new("éx");
        assert_eq!(map.position(2), SourcePosition { line: 1, column: 3 });
    }
}
