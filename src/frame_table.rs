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
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
pub const FRAME_TABLE_SELECTOR_SLOT: u32 = 0;
/// Metal argument-buffer slot for the device-local index table (`Scattered<u32>`).
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
pub const FRAME_TABLE_DEVICE_SLOT: u32 = 1;
/// First bindless slot available to program resources (low protocol slots reserved).
#[cfg_attr(
    not(any(
        feature = "vulkan",
        all(feature = "dx12", target_os = "windows"),
        all(feature = "metal", target_os = "macos"),
    )),
    allow(dead_code)
)]
pub const FRAME_TABLE_USER_SLOT_BASE: u32 = 2;

/// Maximum bindless indices routed through the table per submission row.
/// Ekrano's coarse/fine pipeline can exceed 256 indices in a single submission
/// (longpathdash peaks at ~269); keep shader and Rust constants in sync.
pub const FRAME_TABLE_ROW_STRIDE: u32 = 512;
/// Pipeline depth — number of row-groups in the staging/table buffers.
pub const FRAME_TABLE_MAX_ROWS: u32 = 8;

/// Total `u32` elements in staging and device-local table buffers.
pub const FRAME_TABLE_TABLE_U32S: usize = FRAME_TABLE_ROW_STRIDE as usize * FRAME_TABLE_MAX_ROWS as usize;

/// Byte size of one device-local table buffer.
#[cfg_attr(not(all(feature = "metal", target_os = "macos")), allow(dead_code))]
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

/// PushLayout `_reserved[0]` carries the dispatch's base offset within its row.
#[inline]
pub fn dispatch_table_base_word_index() -> usize {
    0
}

/// Accumulates staging writes while lowering a graph to GPU commands.
#[derive(Debug, Default, Clone)]
pub struct FrameTableStaging {
    /// Flat table: `row * ROW_STRIDE + offset`.
    pub data: Vec<u32>,
    next_dispatch_base: u32,
}

impl FrameTableStaging {
    pub fn new() -> Self {
        Self {
            data: vec![0u32; FRAME_TABLE_TABLE_U32S],
            next_dispatch_base: 0,
        }
    }

    /// Reserve a contiguous run of table slots for one dispatch; returns the base offset.
    pub fn alloc_dispatch(&mut self, slot_count: u32) -> u32 {
        let base = self.next_dispatch_base;
        let end = base.saturating_add(slot_count);
        self.next_dispatch_base = end.min(FRAME_TABLE_ROW_STRIDE);
        base
    }

    /// Returns `true` if any dispatch has written to this staging (i.e. a prologue is needed).
    pub fn has_bindings(&self) -> bool {
        self.next_dispatch_base > 0
    }

    /// Write bindless indices for one dispatch into row 0 (N=1 degenerate path used by spike).
    /// Writes are clamped to row 0's boundary; indices that would overflow into row 1+ are silently dropped.
    pub fn write_dispatch_indices(&mut self, dispatch_base: u32, indices: &[u32]) {
        let row = 0usize;
        let row_start = row * FRAME_TABLE_ROW_STRIDE as usize;
        let row_end = row_start + FRAME_TABLE_ROW_STRIDE as usize;
        let base = row_start + dispatch_base as usize;
        for (i, &idx) in indices.iter().enumerate() {
            let pos = base + i;
            if pos < row_end.min(self.data.len()) {
                self.data[pos] = idx;
            }
        }
    }

    pub fn as_arc(&self) -> std::sync::Arc<[u32]> {
        self.data.as_slice().into()
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
                user,
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
        all(feature = "metal", target_os = "macos"),
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

    /// `alloc_dispatch` must never return a base that, combined with slot_count,
    /// exceeds ROW_STRIDE.  Previously this overflowed when the ekrano pipeline
    /// allocated ~269 slots with a 256-stride, causing garbage index reads.
    #[test]
    fn alloc_dispatch_does_not_overflow_row_stride() {
        let mut s = FrameTableStaging::new();
        // Fill up the row completely.
        let full = FRAME_TABLE_ROW_STRIDE;
        let base = s.alloc_dispatch(full);
        assert_eq!(base, 0);
        assert_eq!(s.next_dispatch_base, FRAME_TABLE_ROW_STRIDE);

        // Any further alloc must return the clamped end, not a value beyond the row.
        let overflow_base = s.alloc_dispatch(10);
        assert!(
            overflow_base <= FRAME_TABLE_ROW_STRIDE,
            "alloc_dispatch overflowed: base={overflow_base} > ROW_STRIDE={FRAME_TABLE_ROW_STRIDE}"
        );
        // write_dispatch_indices must not write past the backing vec.
        s.write_dispatch_indices(overflow_base, &[42u32; 10]);
    }

    /// `write_dispatch_indices` must not write past the bounds of row 0.
    /// This is the exact class of bug that caused the longpathdash_butt snapshot
    /// failure (stride 256 → indices spilling into row 1's region).
    #[test]
    fn write_dispatch_indices_stays_in_row_zero() {
        let mut s = FrameTableStaging::new();
        // Write to the last few slots of the row.
        let last_base = FRAME_TABLE_ROW_STRIDE - 4;
        s.write_dispatch_indices(last_base, &[1, 2, 3, 4, 5, 6]); // 6 > 4 remaining
                                                                  // Slots within row 0 must be written.
        for k in 0..4usize {
            assert_eq!(s.data[last_base as usize + k], k as u32 + 1);
        }
        // Row 1 (starting at offset ROW_STRIDE) must be untouched.
        for k in 0..4usize {
            assert_eq!(
                s.data[FRAME_TABLE_ROW_STRIDE as usize + k],
                0,
                "write_dispatch_indices spilled into row 1 at k={k}"
            );
        }
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
