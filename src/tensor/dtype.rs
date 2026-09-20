//! Element types for the dense tensor layer.

use crate::error::GoldyError;

/// Dense tensor element type. Operation support is per-op, not universal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorDType {
    /// IEEE-754 binary32.
    F32,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
}

impl TensorDType {
    /// Size of one element in bytes.
    pub const fn size_bytes(self) -> usize {
        4
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
        }
    }

    pub(crate) fn require_f32(self, op: &str) -> Result<(), GoldyError> {
        if self == Self::F32 {
            Ok(())
        } else {
            Err(GoldyError::Validation(format!(
                "tensor {op}: dtype {} is not supported (F32 required)",
                self.name()
            )))
        }
    }
}
