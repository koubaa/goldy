//! Reader for Slang's "fossil" serialization format (`source/slang/slang-fossil.h`,
//! `docs/design/serialization.md`): a relative-pointer object graph whose root is a variant
//! carrying its own layout. Values are navigated with the embedded layout, so the reader
//! validates the shape it expects instead of assuming field offsets.

use super::IrError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    StringObj,
    ArrayObj,
    OptionalObj,
    DictionaryObj,
    Tuple,
    Struct,
    Ptr,
    VariantObj,
}

impl Kind {
    fn from_tag(tag: u32) -> Option<Kind> {
        Some(match tag {
            0 => Kind::Bool,
            1 => Kind::Int8,
            2 => Kind::Int16,
            3 => Kind::Int32,
            4 => Kind::Int64,
            5 => Kind::UInt8,
            6 => Kind::UInt16,
            7 => Kind::UInt32,
            8 => Kind::UInt64,
            9 => Kind::Float32,
            10 => Kind::Float64,
            11 => Kind::StringObj,
            12 => Kind::ArrayObj,
            13 => Kind::OptionalObj,
            14 => Kind::DictionaryObj,
            15 => Kind::Tuple,
            16 => Kind::Struct,
            17 => Kind::Ptr,
            18 => Kind::VariantObj,
            _ => return None,
        })
    }

    /// Kinds whose data lives in a separately addressed object (count / layout stored just
    /// before it). In a field position they are encoded as a relative pointer to that object.
    fn is_object(self) -> bool {
        matches!(
            self,
            Kind::StringObj | Kind::ArrayObj | Kind::DictionaryObj | Kind::VariantObj
        )
    }
}

/// Decoded layout tree for one value type.
#[derive(Debug, Clone)]
pub(super) struct Layout {
    pub kind: Kind,
    /// Element layout for pointers, optionals, arrays and dictionaries.
    pub elem: Option<Box<Layout>>,
    /// Element stride for arrays and dictionaries.
    pub stride: u32,
    /// `(layout, offset)` per field for structs and tuples. A missing layout means it was
    /// elided from the blob.
    pub fields: Vec<(Option<Layout>, u32)>,
}

pub(super) struct Fossil<'a> {
    bytes: &'a [u8],
}

/// A value located in the blob together with its layout.
///
/// For object kinds (strings, arrays, ...) `obj` tells whether `off` is the object itself
/// (reached through a pointer) or a field holding a relative pointer to it.
#[derive(Clone, Copy)]
pub(super) struct Val<'a, 'l> {
    f: &'a Fossil<'a>,
    off: usize,
    layout: &'l Layout,
    obj: bool,
}

const MAGIC: &[u8; 5] = b"\xABfoss";
const HEADER_SIZE: usize = 32;
const MAX_LAYOUT_DEPTH: u32 = 32;

fn malformed(what: &'static str) -> IrError {
    IrError::Malformed(what)
}

impl<'a> Fossil<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Fossil<'a>, IrError> {
        if bytes.len() < HEADER_SIZE || &bytes[..5] != MAGIC {
            return Err(malformed("fossil blob header"));
        }
        Ok(Fossil { bytes })
    }

    fn u32(&self, off: usize) -> Result<u32, IrError> {
        self.bytes
            .get(off..off + 4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or(malformed("fossil read past end"))
    }

    fn u64(&self, off: usize) -> Result<u64, IrError> {
        self.bytes
            .get(off..off + 8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ok_or(malformed("fossil read past end"))
    }

    /// Relative pointer at `off`; `None` when null.
    fn rel(&self, off: usize) -> Result<Option<usize>, IrError> {
        let r = self.u32(off)? as i32;
        if r == 0 {
            return Ok(None);
        }
        let t = (off as i64) + i64::from(r);
        if t < 0 || t as usize >= self.bytes.len() {
            return Err(malformed("fossil relative pointer out of range"));
        }
        Ok(Some(t as usize))
    }

    fn layout_at(&self, off: Option<usize>, depth: u32) -> Result<Option<Layout>, IrError> {
        let Some(off) = off else { return Ok(None) };
        if depth > MAX_LAYOUT_DEPTH {
            return Err(malformed("fossil layout nesting too deep"));
        }
        let kind = Kind::from_tag(self.u32(off)?).ok_or(malformed("fossil layout kind"))?;
        let mut layout = Layout {
            kind,
            elem: None,
            stride: 0,
            fields: Vec::new(),
        };
        match kind {
            Kind::Ptr | Kind::OptionalObj => {
                layout.elem = self.layout_at(self.rel(off + 4)?, depth + 1)?.map(Box::new);
            }
            Kind::ArrayObj | Kind::DictionaryObj => {
                layout.elem = self.layout_at(self.rel(off + 4)?, depth + 1)?.map(Box::new);
                layout.stride = self.u32(off + 8)?;
            }
            Kind::Struct | Kind::Tuple => {
                let n = self.u32(off + 4)? as usize;
                if n > 4096 {
                    return Err(malformed("fossil record field count"));
                }
                for i in 0..n {
                    let fo = off + 8 + 8 * i;
                    let field_layout = self.layout_at(self.rel(fo)?, depth + 1)?;
                    layout.fields.push((field_layout, self.u32(fo + 4)?));
                }
            }
            _ => {}
        }
        Ok(Some(layout))
    }

    /// The root value (always a variant) and its layout.
    pub fn root(&'a self) -> Result<(Layout, usize), IrError> {
        let root = self.rel(28)?.ok_or(malformed("fossil root pointer"))?;
        if root < 4 {
            return Err(malformed("fossil root variant"));
        }
        let layout = self
            .layout_at(self.rel(root - 4)?, 0)?
            .ok_or(malformed("fossil root layout missing"))?;
        Ok((layout, root))
    }

    /// The root value's content (object position: the variant's payload is inline at `off`).
    pub fn val<'l>(&'a self, off: usize, layout: &'l Layout) -> Val<'a, 'l> {
        Val {
            f: self,
            off,
            layout,
            obj: true,
        }
    }
}

impl<'a, 'l> Val<'a, 'l> {
    pub fn kind(&self) -> Kind {
        self.layout.kind
    }

    fn expect(&self, kinds: &[Kind], what: &'static str) -> Result<(), IrError> {
        if kinds.contains(&self.layout.kind) {
            Ok(())
        } else {
            Err(malformed(what))
        }
    }

    /// Field `i` of a struct/tuple, located inline.
    pub fn field(&self, i: usize) -> Result<Val<'a, 'l>, IrError> {
        self.expect(&[Kind::Struct, Kind::Tuple], "fossil: expected record")?;
        let (layout, off) = self
            .layout
            .fields
            .get(i)
            .ok_or(malformed("fossil: record field index"))?;
        let layout = layout.as_ref().ok_or(malformed("fossil: record field layout elided"))?;
        Ok(Val {
            f: self.f,
            off: self.off + *off as usize,
            layout,
            obj: false,
        })
    }

    pub fn field_count(&self) -> usize {
        self.layout.fields.len()
    }

    /// Follow a pointer or optional. `None` when null. The target holds the element: object
    /// kinds (strings, arrays, ...) as the object itself, optionals transparently as their
    /// content, everything else inline.
    pub fn deref(&self) -> Result<Option<Val<'a, 'l>>, IrError> {
        self.expect(&[Kind::Ptr, Kind::OptionalObj], "fossil: expected pointer")?;
        let mut elem = self
            .layout
            .elem
            .as_deref()
            .ok_or(malformed("fossil: pointer element layout elided"))?;
        let Some(t) = self.f.rel(self.off)? else {
            return Ok(None);
        };
        // A pointer to an optional is a single indirection: the optional's content lives at
        // the target (Slang writes `Optional<T>` as "pointer to T").
        while elem.kind == Kind::OptionalObj {
            elem = elem
                .elem
                .as_deref()
                .ok_or(malformed("fossil: optional element layout elided"))?;
        }
        Ok(Some(Val {
            f: self.f,
            off: t,
            layout: elem,
            obj: true,
        }))
    }

    /// Address of the object a string/array value refers to, or `None` for an empty one.
    fn object(&self) -> Result<Option<usize>, IrError> {
        if !self.layout.kind.is_object() {
            return Err(malformed("fossil: expected object"));
        }
        if self.obj {
            Ok(Some(self.off))
        } else {
            self.f.rel(self.off)
        }
    }

    /// Bytes of a string.
    pub fn string(&self) -> Result<&'a [u8], IrError> {
        self.expect(&[Kind::StringObj], "fossil: expected string")?;
        let Some(t) = self.object()? else {
            return Ok(&[]);
        };
        if t < 4 {
            return Err(malformed("fossil: string object"));
        }
        let n = self.f.u32(t - 4)? as usize;
        self.f.bytes.get(t..t + n).ok_or(malformed("fossil: string bytes"))
    }

    /// Elements of an array.
    pub fn array(&self) -> Result<Array<'a, 'l>, IrError> {
        self.expect(&[Kind::ArrayObj], "fossil: expected array")?;
        let Some(t) = self.object()? else {
            return Ok(Array {
                f: self.f,
                off: 0,
                len: 0,
                stride: 0,
                elem: self.layout.elem.as_deref(),
            });
        };
        if t < 4 {
            return Err(malformed("fossil: array object"));
        }
        let len = self.f.u32(t - 4)? as usize;
        let stride = self.layout.stride as usize;
        if len
            .checked_mul(stride)
            .and_then(|n| n.checked_add(t))
            .is_none_or(|end| end > self.f.bytes.len())
        {
            return Err(malformed("fossil: array exceeds blob"));
        }
        Ok(Array {
            f: self.f,
            off: t,
            len,
            stride,
            elem: self.layout.elem.as_deref(),
        })
    }

    pub fn u32(&self) -> Result<u32, IrError> {
        self.expect(&[Kind::UInt32, Kind::Int32], "fossil: expected 32-bit integer")?;
        self.f.u32(self.off)
    }

    pub fn i64(&self) -> Result<i64, IrError> {
        self.expect(&[Kind::Int64, Kind::UInt64], "fossil: expected 64-bit integer")?;
        Ok(self.f.u64(self.off)? as i64)
    }

    pub fn u64(&self) -> Result<u64, IrError> {
        self.expect(&[Kind::Int64, Kind::UInt64], "fossil: expected 64-bit integer")?;
        self.f.u64(self.off)
    }
}

pub(super) struct Array<'a, 'l> {
    f: &'a Fossil<'a>,
    off: usize,
    len: usize,
    stride: usize,
    elem: Option<&'l Layout>,
}

impl<'a, 'l> Array<'a, 'l> {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn get(&self, i: usize) -> Result<Val<'a, 'l>, IrError> {
        if i >= self.len {
            return Err(malformed("fossil: array index"));
        }
        let elem = self.elem.ok_or(malformed("fossil: array element layout elided"))?;
        Ok(Val {
            f: self.f,
            off: self.off + i * self.stride,
            layout: elem,
            obj: false,
        })
    }

    /// Contiguous bytes of a byte array.
    pub fn bytes(&self) -> Result<&'a [u8], IrError> {
        if self.stride != 1 && self.len != 0 {
            return Err(malformed("fossil: expected byte array"));
        }
        self.f
            .bytes
            .get(self.off..self.off + self.len)
            .ok_or(malformed("fossil: array bytes"))
    }
}
