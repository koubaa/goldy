//! Frame-table protocol — record/execute index routing (retained-scheme spike).
//!
//! Bindless indices are staged on the CPU and copied into a device-local table
//! before each submission; shader preambles resolve `index = table[selector][base + k]`
//! instead of reading baked push-constant words.

/// Metal argument-buffer slot for the (unused) selector cell.
///
/// On DX12/Vulkan the selector/table slots are **per-context** (allocated from
/// the device descriptor registry) and reach shaders via push constants
/// `_rs1`/`_rs2`; these constants only describe Metal's fixed device-level
/// argument-buffer layout.
#[cfg_attr(
    not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub const FRAME_TABLE_SELECTOR_SLOT: u32 = 0;
/// Metal argument-buffer slot for the device-local index table (`Scattered<u32>`).
#[cfg_attr(
    not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub const FRAME_TABLE_DEVICE_SLOT: u32 = 1;
/// First bindless slot available to program resources (low protocol slots reserved).
#[cfg_attr(
    not(any(
        feature = "vulkan",
        all(feature = "dx12", target_os = "windows"),
        all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    )),
    allow(dead_code)
)]
pub const FRAME_TABLE_USER_SLOT_BASE: u32 = 2;

/// Address stride of one device-table row.
///
/// Staging and uploads copy only the indices a submission wrote. This stride is
/// the reservation those indices are placed in, kept in sync with
/// `GOLDY_FRAME_TABLE_ROW_STRIDE` in `access.slang`. A submission that needs
/// more indices than this is a record-time error.
pub const FRAME_TABLE_ROW_STRIDE: u32 = 32768;
/// In-flight rows for submissions that are not retained. The ring wraps; a
/// later submit waits until the GPU has finished with the row it is about to
/// overwrite.
pub const FRAME_TABLE_TRANSIENT_ROWS: u32 = 8;
/// Rows owned by retained command buffers. A retained buffer's bindings do not
/// change, so its prologue keeps reading the same row and must not share the
/// transient ring (that ring is only 8 deep, and pinning a ring row evicted the
/// buffer on the next wrap).
pub const FRAME_TABLE_PERSISTENT_ROWS: u32 = 512;
/// Transient ring followed by the persistent retained-buffer rows.
pub const FRAME_TABLE_MAX_ROWS: u32 = FRAME_TABLE_TRANSIENT_ROWS + FRAME_TABLE_PERSISTENT_ROWS;

/// Total `u32` elements in staging and device-local table buffers.
pub const FRAME_TABLE_TABLE_U32S: usize = FRAME_TABLE_ROW_STRIDE as usize * FRAME_TABLE_MAX_ROWS as usize;

/// Byte size of one device-local table buffer.
#[cfg_attr(
    not(all(feature = "metal", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
pub const FRAME_TABLE_TABLE_BYTES: u64 = (FRAME_TABLE_TABLE_U32S * 4) as u64;
/// Per-row selector slots at the front of CPU upload staging.
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"),)),
    allow(dead_code)
)]
pub const FRAME_TABLE_STAGING_SELECTOR_U32S: usize = FRAME_TABLE_MAX_ROWS as usize;
/// Total `u32` elements in CPU upload staging (selectors + row payloads).
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"),)),
    allow(dead_code)
)]
pub const FRAME_TABLE_STAGING_U32S: usize = FRAME_TABLE_STAGING_SELECTOR_U32S + FRAME_TABLE_TABLE_U32S;
/// Byte size of CPU upload staging (per-row selectors + row-strided table payloads).
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"),)),
    allow(dead_code)
)]
pub const FRAME_TABLE_STAGING_BYTES: u64 = (FRAME_TABLE_STAGING_U32S * 4) as u64;

/// Byte offset of row `row`'s selector word in CPU staging.
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"),)),
    allow(dead_code)
)]
#[inline]
pub fn staging_selector_byte_offset(row: u32) -> u64 {
    (row as u64) * 4
}

/// Byte offset of row `row`'s payload in CPU staging.
#[cfg_attr(
    not(any(feature = "vulkan", all(feature = "dx12", target_os = "windows"),)),
    allow(dead_code)
)]
#[inline]
pub fn staging_row_payload_byte_offset(row: u32) -> u64 {
    (FRAME_TABLE_STAGING_SELECTOR_U32S as u64 + row as u64 * FRAME_TABLE_ROW_STRIDE as u64) * 4
}

/// Persistent row whose last GPU token is the oldest.
///
/// `in_use` is the sentinel a submitter stores while it still owns the row on
/// the CPU. Those rows lose to any row the GPU has already been given.
pub fn least_recent_persistent_row(tokens: &[std::sync::atomic::AtomicU64], in_use: u64) -> u32 {
    let mut victim = FRAME_TABLE_TRANSIENT_ROWS;
    let mut oldest = u64::MAX;
    for row in FRAME_TABLE_TRANSIENT_ROWS..FRAME_TABLE_MAX_ROWS {
        let tok = tokens[row as usize].load(std::sync::atomic::Ordering::Acquire);
        let age = if tok == in_use { u64::MAX - 1 } else { tok };
        if age < oldest {
            oldest = age;
            victim = row;
        }
    }
    victim
}

/// PushLayout `_reserved[0]` carries the dispatch's base offset within its row.
#[inline]
pub fn dispatch_table_base_word_index() -> usize {
    0
}

/// Accumulates the bindless indices one submission actually uses.
///
/// The device row is [`FRAME_TABLE_ROW_STRIDE`] words apart. This buffer is only
/// the prefix that was written, so a small scheme does not allocate or copy a
/// whole row.
#[derive(Debug, Default, Clone)]
pub struct FrameTableStaging {
    /// Indices at `dispatch_base + k`. Length is [`Self::used_words`].
    pub data: Vec<u32>,
    next_dispatch_base: u32,
    overflowed: bool,
}

impl FrameTableStaging {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            next_dispatch_base: 0,
            overflowed: false,
        }
    }

    /// Words a prologue must copy. Zero when nothing was allocated.
    pub fn used_words(&self) -> u32 {
        self.next_dispatch_base
    }

    /// Reserve a contiguous run of table slots for one dispatch; returns the base offset.
    ///
    /// Past [`FRAME_TABLE_ROW_STRIDE`] the reservation is refused. Call
    /// [`Self::check`] before submitting; the indices are not clamped into a
    /// neighbour's row.
    pub fn alloc_dispatch(&mut self, slot_count: u32) -> u32 {
        let base = self.next_dispatch_base;
        let Some(end) = base.checked_add(slot_count) else {
            self.overflowed = true;
            return base;
        };
        if end > FRAME_TABLE_ROW_STRIDE {
            self.overflowed = true;
            return base;
        }
        self.data.resize(end as usize, 0);
        self.next_dispatch_base = end;
        base
    }

    /// `Err` when a dispatch did not fit in the row.
    pub fn check(&self) -> Result<(), crate::error::GoldyError> {
        if self.overflowed {
            Err(crate::error::GoldyError::Validation(format!(
                "frame table overflow: a submission needs more than {FRAME_TABLE_ROW_STRIDE} bindless indices"
            )))
        } else {
            Ok(())
        }
    }

    /// Returns `true` if any dispatch has written to this staging (i.e. a prologue is needed).
    pub fn has_bindings(&self) -> bool {
        self.next_dispatch_base > 0
    }

    /// Write bindless indices for one dispatch at `dispatch_base`.
    ///
    /// Indices past the reservation set the overflow flag and are not stored.
    pub fn write_dispatch_indices(&mut self, dispatch_base: u32, indices: &[u32]) {
        let base = dispatch_base as usize;
        for (i, &idx) in indices.iter().enumerate() {
            let pos = base + i;
            if pos < self.data.len() {
                self.data[pos] = idx;
            } else {
                self.overflowed = true;
            }
        }
    }

    /// The indices this submission wrote, and nothing after them.
    pub fn as_arc(&self) -> std::sync::Arc<[u32]> {
        let used = (self.used_words() as usize).min(self.data.len());
        std::sync::Arc::from(&self.data[..used])
    }
}

/// Lower render bind commands into frame-table routing (`BindResourcesRaw` + base).
pub fn lower_render_pass_commands(
    staging: &mut FrameTableStaging,
    commands: &[crate::backend::RenderCommand],
) -> Vec<crate::backend::RenderCommand> {
    commands
        .iter()
        .map(|cmd| match cmd {
            crate::backend::RenderCommand::BindResourcesTyped { handles } => {
                let indices: Vec<u32> = handles.iter().map(|h| h.index()).collect();
                let frame_table_base = staging.alloc_dispatch(indices.len() as u32);
                staging.write_dispatch_indices(frame_table_base, &indices);
                crate::backend::RenderCommand::BindResourcesRaw {
                    indices: Vec::new(),
                    user: Vec::new(),
                    frame_table_base,
                }
            }
            crate::backend::RenderCommand::BindResourcesRaw {
                indices,
                user: _,
                frame_table_base: _,
            } if indices.is_empty() => cmd.clone(),
            crate::backend::RenderCommand::BindResourcesRaw {
                indices,
                user,
                frame_table_base: _,
            } => {
                let frame_table_base = staging.alloc_dispatch(indices.len() as u32);
                staging.write_dispatch_indices(frame_table_base, indices);
                crate::backend::RenderCommand::BindResourcesRaw {
                    indices: Vec::new(),
                    user: user.clone(),
                    frame_table_base,
                }
            }
            other => other.clone(),
        })
        .collect()
}

/// Frame-table staging is built directly by the task-graph analyzer; kept for call-site stability.
#[cfg_attr(
    not(any(
        feature = "vulkan",
        all(feature = "dx12", target_os = "windows"),
        all(feature = "metal", any(target_os = "macos", target_os = "ios")),
    )),
    allow(dead_code)
)]
pub fn lower_gpu_commands(_commands: &mut Vec<crate::backend::GpuCommand>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{GpuCommand, RenderCommand};

    #[test]
    fn lower_render_skips_already_lowered_raw() {
        let mut staging = FrameTableStaging::new();
        let cmds = vec![RenderCommand::BindResourcesRaw {
            indices: vec![],
            user: vec![],
            frame_table_base: 3,
        }];
        let lowered = lower_render_pass_commands(&mut staging, &cmds);
        assert_eq!(lowered.len(), 1);
        assert!(matches!(
            lowered[0],
            RenderCommand::BindResourcesRaw {
                frame_table_base: 3,
                ..
            }
        ));
    }

    /// A bind-free stream (copies, uploads, barriers) must NOT receive a
    /// FrameTableStaging prefix; doing so would bump the submission counter and
    /// silently overwrite the selector with zeros, corrupting every in-flight frame.
    #[test]
    fn lower_gpu_no_staging_for_bind_free_stream() {
        let mut cmds = vec![GpuCommand::CopyTexture { src: 1u64, dst: 2u64 }];
        lower_gpu_commands(&mut cmds);
        assert!(
            !cmds.iter().any(|c| matches!(c, GpuCommand::FrameTableStaging { .. })),
            "bind-free stream must not receive a FrameTableStaging prefix"
        );
    }

    /// `FrameTableStaging::has_bindings` must be false on a fresh instance and
    /// true after any `alloc_dispatch` call.
    #[test]
    fn has_bindings_tracks_alloc_dispatch() {
        let mut s = FrameTableStaging::new();
        assert!(!s.has_bindings(), "fresh staging should have no bindings");
        s.alloc_dispatch(3);
        assert!(s.has_bindings(), "after alloc_dispatch staging must report bindings");
    }

    /// A dispatch past the row is an error. It does not clamp into the next row
    /// and it does not grow the staging buffer to the whole table.
    #[test]
    fn alloc_dispatch_reports_overflow_instead_of_clamping() {
        let mut s = FrameTableStaging::new();
        let base = s.alloc_dispatch(FRAME_TABLE_ROW_STRIDE);
        assert_eq!(base, 0);
        assert!(s.check().is_ok());
        assert_eq!(s.used_words(), FRAME_TABLE_ROW_STRIDE);
        assert_eq!(s.as_arc().len(), FRAME_TABLE_ROW_STRIDE as usize);

        let overflow_base = s.alloc_dispatch(10);
        assert_eq!(overflow_base, FRAME_TABLE_ROW_STRIDE);
        assert!(s.check().is_err());
        assert_eq!(
            s.data.len(),
            FRAME_TABLE_ROW_STRIDE as usize,
            "overflow must not allocate another row"
        );
        s.write_dispatch_indices(overflow_base, &[42u32; 10]);
        assert!(s.check().is_err());
        assert_eq!(s.as_arc().len(), FRAME_TABLE_ROW_STRIDE as usize);
    }

    /// Indices live in the used prefix. A small submission's arc is that prefix,
    /// not a row of zeros.
    #[test]
    fn as_arc_is_only_the_words_written() {
        let mut s = FrameTableStaging::new();
        assert!(s.as_arc().is_empty());
        let base = s.alloc_dispatch(4);
        s.write_dispatch_indices(base, &[1, 2, 3, 4]);
        assert_eq!(s.as_arc().as_ref(), &[1, 2, 3, 4]);
        s.write_dispatch_indices(base, &[1, 2, 3, 4, 5]);
        assert!(s.check().is_err(), "a write past the reservation is overflow");
    }

    /// Rust and Slang must agree on `GOLDY_FRAME_TABLE_ROW_STRIDE` in access.slang.
    #[test]
    fn row_stride_matches_slang_constant() {
        let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let slang_path = manifest.join("shaders/goldy_exp/access.slang");
        let content =
            std::fs::read_to_string(&slang_path).unwrap_or_else(|e| panic!("read {}: {e}", slang_path.display()));
        let needle = "GOLDY_FRAME_TABLE_ROW_STRIDE";
        let line = content
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("{needle} not found in {}", slang_path.display()));
        let rhs = line
            .split('=')
            .nth(1)
            .unwrap_or_else(|| panic!("expected '=' in slang line: {line}"))
            .trim()
            .trim_end_matches(';')
            .trim();
        let slang_stride: u32 = rhs
            .parse()
            .unwrap_or_else(|e| panic!("parse slang stride '{rhs}': {e}"));
        assert_eq!(
            slang_stride, FRAME_TABLE_ROW_STRIDE,
            "update goldy/shaders/goldy_exp/access.slang or FRAME_TABLE_ROW_STRIDE in frame_table.rs"
        );
    }

    #[test]
    fn staging_row_payload_offsets_are_non_overlapping() {
        for row in 0..FRAME_TABLE_MAX_ROWS {
            let sel = staging_selector_byte_offset(row);
            let payload = staging_row_payload_byte_offset(row);
            assert_eq!(sel, row as u64 * 4);
            assert!(
                payload + (FRAME_TABLE_ROW_STRIDE as u64 * 4) <= FRAME_TABLE_STAGING_BYTES,
                "row {row} payload overflows staging buffer"
            );
            if row > 0 {
                let prev_payload_end = staging_row_payload_byte_offset(row - 1) + FRAME_TABLE_ROW_STRIDE as u64 * 4;
                assert!(
                    payload >= prev_payload_end,
                    "row {row} payload overlaps row {}",
                    row - 1
                );
            }
        }
    }

    /// Successive `alloc_dispatch` calls within one staging instance produce
    /// non-overlapping base offsets within a row.
    #[test]
    fn dispatch_bases_do_not_overlap_within_row() {
        let mut s = FrameTableStaging::new();
        let base_a = s.alloc_dispatch(5);
        let base_b = s.alloc_dispatch(3);
        let base_c = s.alloc_dispatch(7);
        assert_eq!(base_a, 0);
        assert_eq!(base_b, 5);
        assert_eq!(base_c, 8);
        // Indices written for dispatch A must not overlap with dispatch B's region.
        s.write_dispatch_indices(base_a, &[10, 11, 12, 13, 14]);
        s.write_dispatch_indices(base_b, &[20, 21, 22]);
        assert_eq!(s.data[0], 10);
        assert_eq!(s.data[4], 14);
        assert_eq!(s.data[5], 20); // base_b
        assert_eq!(s.data[7], 22);
        assert_eq!(s.data[8], 0); // base_c region untouched
    }
}
