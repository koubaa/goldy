//! Minimal reader for the RIFF container Slang uses for `.slang-module` files
//! (`source/core/slang-riff.{h,cpp}`): 8-byte aligned chunks, `RIFF`/`LIST` chunks carry a
//! FourCC sub-type and nest, everything else is a data chunk.

use super::IrError;

/// One parsed chunk. Data chunks have `children.is_empty()` and `data` set to their payload;
/// list chunks carry their children and `data` covers the list body.
#[derive(Debug)]
pub(super) struct Chunk<'a> {
    pub tag: [u8; 4],
    /// Sub-type for `RIFF`/`LIST` chunks, otherwise equal to `tag`.
    pub kind: [u8; 4],
    pub data: &'a [u8],
    pub children: Vec<Chunk<'a>>,
}

impl<'a> Chunk<'a> {
    pub fn is_list(&self) -> bool {
        &self.tag == b"RIFF" || &self.tag == b"LIST"
    }

    /// Direct children that are lists of the given sub-type.
    pub fn lists<'s>(&'s self, kind: &'s [u8; 4]) -> impl Iterator<Item = &'s Chunk<'a>> + 's {
        self.children.iter().filter(move |c| c.is_list() && &c.kind == kind)
    }

    /// First direct data child with the given tag.
    pub fn data_child(&self, tag: &[u8; 4]) -> Option<&Chunk<'a>> {
        self.children.iter().find(|c| !c.is_list() && &c.tag == tag)
    }

    /// Depth-first search for the first list chunk of the given sub-type.
    pub fn find_list(&self, kind: &[u8; 4]) -> Option<&Chunk<'a>> {
        if self.is_list() && &self.kind == kind {
            return Some(self);
        }
        self.children.iter().find_map(|c| c.find_list(kind))
    }
}

fn align8(v: usize) -> usize {
    (v + 7) & !7
}

/// Parse a complete RIFF file. The top-level chunk must be `RIFF`.
pub(super) fn parse(bytes: &[u8]) -> Result<Chunk<'_>, IrError> {
    let mut top = parse_range(bytes, 0, bytes.len())?;
    match top.len() {
        1 if &top[0].tag == b"RIFF" => Ok(top.remove(0)),
        _ => Err(IrError::Malformed("container is not a single RIFF chunk")),
    }
}

fn parse_range(bytes: &[u8], mut off: usize, end: usize) -> Result<Vec<Chunk<'_>>, IrError> {
    let mut out = Vec::new();
    while off + 8 <= end {
        let tag: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
        let size = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap()) as usize;
        let body = off + 8;
        let body_end = body
            .checked_add(size)
            .filter(|e| *e <= end)
            .ok_or(IrError::Malformed("RIFF chunk exceeds its parent"))?;
        if &tag == b"RIFF" || &tag == b"LIST" {
            if size < 4 {
                return Err(IrError::Malformed("RIFF list chunk without sub-type"));
            }
            let kind: [u8; 4] = bytes[body..body + 4].try_into().unwrap();
            let children = parse_range(bytes, align8(body + 4), body_end)?;
            out.push(Chunk {
                tag,
                kind,
                data: &bytes[body + 4..body_end],
                children,
            });
        } else {
            out.push(Chunk {
                tag,
                kind: tag,
                data: &bytes[body..body_end],
                children: Vec::new(),
            });
        }
        off = align8(body_end);
    }
    Ok(out)
}
