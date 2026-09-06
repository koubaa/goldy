//! Decoder for the `Sdeb` debug chunk of a `.slang-module` container
//! (`source/slang/slang-serialize-source-loc.{h,cpp}`).
//!
//! Instructions carry a *serial source location*: an offset into a virtual address space in
//! which every source file that contributed a location gets a contiguous range. The debug
//! chunk maps ranges back to file paths and records the start offset of every line that has
//! at least one location, so a serial location decodes to `path:line:column`. Lines affected
//! by `#line` directives are stored separately with their adjusted line number and path.

use super::{riff::Chunk, IrError, SourceLocation};

#[derive(Debug, Clone, Copy)]
struct LineInfo {
    start: u32,
    line_index: u32,
}

#[derive(Debug, Clone, Copy)]
struct AdjustedLineInfo {
    line: LineInfo,
    adjusted_line_index: u32,
    path_index: u32,
}

#[derive(Debug, Clone, Copy)]
struct SourceInfo {
    path_index: u32,
    start: u32,
    end: u32,
    lines_start: u32,
    lines_count: u32,
    adjusted_start: u32,
    adjusted_count: u32,
}

#[derive(Debug, Default)]
pub(super) struct DebugInfo {
    /// Index 0 is the null string and 1 the empty string, as in `StringSlicePool`.
    strings: Vec<String>,
    lines: Vec<LineInfo>,
    adjusted: Vec<AdjustedLineInfo>,
    sources: Vec<SourceInfo>,
}

fn u32_at(b: &[u8], off: usize) -> Result<u32, IrError> {
    b.get(off..off + 4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .ok_or(IrError::Malformed("debug chunk truncated"))
}

/// `SerialRiffUtil::writeArrayChunk`: a `u32` count followed by the entries.
fn array_entries(data: &[u8], entry_size: usize) -> Result<impl Iterator<Item = &[u8]>, IrError> {
    let n = u32_at(data, 0)? as usize;
    let body = &data[4..];
    if body.len() < n * entry_size {
        return Err(IrError::Malformed("debug array chunk truncated"));
    }
    Ok(body.chunks_exact(entry_size).take(n))
}

/// `SerialStringTableUtil`: each string is prefixed with its length encoded as a UTF-8 code point.
fn decode_string_table(table: &[u8]) -> Vec<String> {
    let mut out = vec![String::new(), String::new()];
    let mut i = 0;
    while i < table.len() {
        let b0 = table[i];
        let (len, adv) = if b0 < 0x80 {
            (u32::from(b0), 1)
        } else if b0 < 0xE0 {
            (
                (u32::from(b0 & 0x1F) << 6) | u32::from(table.get(i + 1).copied().unwrap_or(0) & 0x3F),
                2,
            )
        } else if b0 < 0xF0 {
            (
                (u32::from(b0 & 0x0F) << 12)
                    | (u32::from(table.get(i + 1).copied().unwrap_or(0) & 0x3F) << 6)
                    | u32::from(table.get(i + 2).copied().unwrap_or(0) & 0x3F),
                3,
            )
        } else {
            (
                (u32::from(b0 & 0x07) << 18)
                    | (u32::from(table.get(i + 1).copied().unwrap_or(0) & 0x3F) << 12)
                    | (u32::from(table.get(i + 2).copied().unwrap_or(0) & 0x3F) << 6)
                    | u32::from(table.get(i + 3).copied().unwrap_or(0) & 0x3F),
                4,
            )
        };
        i += adv;
        let end = (i + len as usize).min(table.len());
        out.push(String::from_utf8_lossy(&table[i..end]).into_owned());
        i = end;
    }
    out
}

impl DebugInfo {
    /// Decode a `LIST/Sdeb` chunk.
    pub fn parse(chunk: &Chunk<'_>) -> Result<DebugInfo, IrError> {
        let mut info = DebugInfo::default();
        if let Some(c) = chunk.data_child(b"Sdst") {
            let n = u32_at(c.data, 0)? as usize;
            let table = c
                .data
                .get(4..4 + n)
                .ok_or(IrError::Malformed("debug string table truncated"))?;
            info.strings = decode_string_table(table);
        } else {
            info.strings = vec![String::new(), String::new()];
        }
        if let Some(c) = chunk.data_child(b"Sdln") {
            for e in array_entries(c.data, 8)? {
                info.lines.push(LineInfo {
                    start: u32_at(e, 0)?,
                    line_index: u32_at(e, 4)?,
                });
            }
        }
        if let Some(c) = chunk.data_child(b"Sdal") {
            for e in array_entries(c.data, 16)? {
                info.adjusted.push(AdjustedLineInfo {
                    line: LineInfo {
                        start: u32_at(e, 0)?,
                        line_index: u32_at(e, 4)?,
                    },
                    adjusted_line_index: u32_at(e, 8)?,
                    path_index: u32_at(e, 12)?,
                });
            }
        }
        if let Some(c) = chunk.data_child(b"Sdso") {
            for e in array_entries(c.data, 32)? {
                info.sources.push(SourceInfo {
                    path_index: u32_at(e, 0)?,
                    start: u32_at(e, 4)?,
                    end: u32_at(e, 8)?,
                    lines_start: u32_at(e, 16)?,
                    lines_count: u32_at(e, 20)?,
                    adjusted_start: u32_at(e, 24)?,
                    adjusted_count: u32_at(e, 28)?,
                });
            }
        }
        Ok(info)
    }

    fn string(&self, index: u32) -> Option<&str> {
        self.strings
            .get(index as usize)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
    }

    /// Decode a serial source location (0 means "no location").
    pub fn lookup(&self, raw: u32) -> Option<SourceLocation> {
        if raw == 0 {
            return None;
        }
        let src = self.sources.iter().find(|s| s.start <= raw && raw <= s.end)?;
        let offset = raw - src.start;
        let plain = self
            .lines
            .get(src.lines_start as usize..(src.lines_start + src.lines_count) as usize)?
            .iter()
            .filter(|l| l.start <= offset)
            .max_by_key(|l| l.start);
        let adjusted = self
            .adjusted
            .get(src.adjusted_start as usize..(src.adjusted_start + src.adjusted_count) as usize)?
            .iter()
            .filter(|l| l.line.start <= offset)
            .max_by_key(|l| l.line.start);
        let file_path = self.string(src.path_index).unwrap_or("<unknown>");
        match (plain, adjusted) {
            (Some(p), Some(a)) if p.start > a.line.start => Some(SourceLocation {
                file: file_path.to_string(),
                line: p.line_index + 1,
                column: offset - p.start + 1,
            }),
            (_, Some(a)) => Some(SourceLocation {
                file: self.string(a.path_index).unwrap_or(file_path).to_string(),
                line: a.adjusted_line_index + 1,
                column: offset - a.line.start + 1,
            }),
            (Some(p), None) => Some(SourceLocation {
                file: file_path.to_string(),
                line: p.line_index + 1,
                column: offset - p.start + 1,
            }),
            (None, None) => None,
        }
    }
}
