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
    /// bfloat16. Stored packed; matmul promotes each element to f32 in the dot.
    BF16,
    /// Unsigned 8-bit. Used for NVFP4 code bytes (two E2M1 nibbles).
    U8,
    /// OCP float8 E4M3. Used for NVFP4 block scales.
    F8E4M3,
}

impl TensorDType {
    /// Size of one element in bytes.
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 | Self::I32 => 4,
            Self::BF16 => 2,
            Self::U8 | Self::F8E4M3 => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::BF16 => "bf16",
            Self::U8 => "u8",
            Self::F8E4M3 => "f8_e4m3",
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
