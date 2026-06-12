//! Metal frame-table realization — UMA CPU write, no GPU copy.
//!
//! Selector (arg slot 0) and device table (arg slot 1) are encoded once at device
//! init. Each submission CPU-memwrites the active row payload and selector cell;
//! the N-frame ring guard prevents overwriting a row still read by in-flight GPU work.

use super::types::{LogicalDevice, ARGUMENT_BUFFER_SIZE};
use crate::backend::GpuCommand;
use crate::frame_table::{
    FRAME_TABLE_MAX_ROWS, FRAME_TABLE_ROW_STRIDE, FRAME_TABLE_TABLE_BYTES, FRAME_TABLE_TABLE_U32S,
    FRAME_TABLE_USER_SLOT_BASE, FrameTableStaging,
};
use crate::timeline::TimelineValue;
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::MTLResourceOptions;

/// Ring guard for N-frame table row reuse (pure bookkeeping — unit-tested first).
pub(super) struct FrameTableRing {
    n: u32,
    counter: u64,
    last_token: Vec<Option<TimelineValue>>,
}

impl FrameTableRing {
    pub(super) fn new(n: u32) -> Self {
        assert!(n >= 1, "frame table ring depth must be >= 1");
        Self {
            n,
            counter: 0,
            last_token: vec![None; n as usize],
        }
    }

    #[allow(dead_code)]
    pub(super) fn depth(&self) -> u32 {
        self.n
    }

    /// Advance submission counter and return `counter % N` before the increment is visible
    /// to the next caller (row for *this* submission).
    pub(super) fn next_row(&mut self) -> u32 {
        let row = (self.counter % self.n as u64) as u32;
        self.counter += 1;
        row
    }

    /// If the prior writer of `row` is still in flight relative to `completed`, return its token.
    pub(super) fn wait_required(&self, row: u32, completed: TimelineValue) -> Option<TimelineValue> {
        let tok = self.last_token.get(row as usize).and_then(|t| *t)?;
        if tok > completed {
            Some(tok)
        } else {
            None
        }
    }

    pub(super) fn record(&mut self, row: u32, token: TimelineValue) {
        if let Some(slot) = self.last_token.get_mut(row as usize) {
            *slot = Some(token);
        }
    }
}

/// Byte offset of ring row `row` within the device-local table buffer.
pub(super) fn row_byte_offset(row: u32) -> u64 {
    (row as u64) * FRAME_TABLE_ROW_STRIDE as u64 * 4
}

/// Per-device frame-table GPU resources (stable arg-buffer descriptors + ring).
pub(super) struct MetalFrameTable {
    pub selector: mtl::Buffer,
    pub table: mtl::Buffer,
    ring: FrameTableRing,
}

impl MetalFrameTable {
    pub(super) fn init(
        device: &mtl::DeviceRef,
        argument_buffer: &mtl::BufferRef,
        argument_encoder: &mtl::ArgumentEncoderRef,
    ) -> Self {
        let table = device.new_buffer(FRAME_TABLE_TABLE_BYTES, MTLResourceOptions::StorageModeShared);
        let selector = device.new_buffer(256, MTLResourceOptions::StorageModeShared);

        let encoded_length = argument_encoder.encoded_length();
        for (slot, buf) in [(0u32, &selector), (1, &table)] {
            let offset = (slot as u64) * encoded_length;
            if offset + encoded_length <= ARGUMENT_BUFFER_SIZE {
                argument_encoder.set_argument_buffer(argument_buffer, offset);
                argument_encoder.set_buffer(0, buf, 0);
            }
        }

        Self {
            selector,
            table,
            ring: FrameTableRing::new(FRAME_TABLE_MAX_ROWS),
        }
    }

    pub(super) fn selector_buffer(&self) -> &mtl::BufferRef {
        &self.selector
    }

    pub(super) fn table_buffer(&self) -> &mtl::BufferRef {
        &self.table
    }

    /// CPU prologue: pick ring row, wait if needed, memcpy row payload into the table.
    ///
    /// The row number is returned so the record path can embed the absolute table
    /// offset (`row * ROW_STRIDE + dispatch_base`) directly into `_reserved[0]`.
    /// The selector buffer (slot 0) is intentionally left unwritten on Metal:
    /// `goldy_frame_table_index` on Metal bypasses the selector read entirely and
    /// uses `dispatch_base` as the absolute offset, avoiding the shared-mutable-
    /// selector race when multiple command buffers are in flight simultaneously.
    pub(super) fn run_prologue(
        &mut self,
        staging_data: &[u32],
        completed: TimelineValue,
        wait_fn: impl FnOnce(TimelineValue) -> Result<()>,
    ) -> Result<u32> {
        let row = self.ring.next_row();
        if let Some(tok) = self.ring.wait_required(row, completed) {
            wait_fn(tok)?;
        }

        let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
        let copy_u32s = staging_data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
        let src = &staging_data[0..copy_u32s];

        let table_ptr = self.table.contents() as *mut u32;
        anyhow::ensure!(!table_ptr.is_null(), "frame table buffer has null contents");
        let dest_ptr = unsafe {
            (table_ptr as *mut u8)
                .add(row_byte_offset(row) as usize) as *mut u32
        };
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dest_ptr, copy_u32s);
        }

        Ok(row)
    }

    pub(super) fn record_submission(&mut self, row: u32, token: TimelineValue) {
        self.ring.record(row, token);
    }
}

/// Reserve user bindless storage slots and create frame-table buffers at device init.
pub(super) fn init_device(ld: &LogicalDevice) {
    ld.ledger
        .lock()
        .unwrap()
        .resource_registry
        .ensure_storage_start(FRAME_TABLE_USER_SLOT_BASE);
}

/// Extract staging payload from a compute command slice (first `FrameTableStaging` wins).
pub(super) fn extract_staging_from_commands(commands: &[GpuCommand]) -> Option<std::sync::Arc<[u32]>> {
    commands.iter().find_map(|c| match c {
        GpuCommand::FrameTableStaging { data } => Some(std::sync::Arc::clone(data)),
        _ => None,
    })
}

pub(super) fn extract_staging_from_graph(
    commands: &[crate::backend::GraphCommand],
) -> Option<std::sync::Arc<[u32]>> {
    commands.iter().find_map(|c| match c {
        crate::backend::GraphCommand::Compute(GpuCommand::FrameTableStaging { data }) => {
            Some(std::sync::Arc::clone(data))
        }
        _ => None,
    })
}

/// Lower render commands and build staging for standalone render passes.
pub(super) fn prepare_render_commands(
    buffers: &std::collections::HashMap<super::BufferHandle, super::types::BufferState>,
    commands: &[crate::backend::RenderCommand],
) -> Result<(Vec<u32>, Vec<crate::backend::RenderCommand>, bool)> {
    use crate::backend::RenderCommand;

    let mut staging = FrameTableStaging::new();
    let lowered = commands
        .iter()
        .map(|cmd| match cmd {
            RenderCommand::BindResources { buffers: handles } => {
                let indices: Vec<u32> = handles
                    .iter()
                    .map(|h| {
                        buffers
                            .get(h)
                            .map(|b| b.arg_buffer_index)
                            .with_context(|| format!("BindResources: buffer handle {h:?} has no arg index"))
                    })
                    .collect::<Result<_>>()?;
                let frame_table_base = staging.alloc_dispatch(indices.len() as u32);
                staging.write_dispatch_indices(frame_table_base, &indices);
                Ok(RenderCommand::BindResourcesRaw {
                    indices: Vec::new(),
                    user: Vec::new(),
                    frame_table_base,
                })
            }
            other => {
                let batch = crate::frame_table::lower_render_pass_commands(&mut staging, std::slice::from_ref(other));
                Ok(batch.into_iter().next().unwrap_or_else(|| other.clone()))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let has_bindings = staging.has_bindings();
    Ok((staging.data, lowered, has_bindings))
}

/// Run CPU prologue on a logical device, blocking on `device` timeline when the ring requires it.
pub(super) fn run_prologue_for_device(
    state: &super::types::MetalState,
    device_handle: super::DeviceHandle,
    ld: &LogicalDevice,
    staging_data: &[u32],
    completed: TimelineValue,
) -> Result<u32> {
    let mut ft = ld.frame_table.lock().unwrap();
    ft.run_prologue(staging_data, completed, |tok| {
        const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);
        if !super::context::wait_until_device_seq_at_least(state, device_handle, tok, TIMEOUT) {
            anyhow::bail!("frame table prologue timed out waiting for timeline {tok}");
        }
        Ok(())
    })
}

pub(super) fn record_submission_for_device(ld: &LogicalDevice, row: u32, token: TimelineValue) {
    ld.frame_table.lock().unwrap().record_submission(row, token);
}

/// Merge graph-level staging with render-local staging (render non-zero wins).
pub(super) fn merge_staging_for_render_sync(graph: &[u32], render: &[u32]) -> Vec<u32> {
    let len = graph.len().max(render.len()).min(FRAME_TABLE_TABLE_U32S);
    let mut merged = vec![0u32; len];
    for (i, slot) in merged.iter_mut().enumerate().take(len) {
        *slot = graph.get(i).copied().unwrap_or(0);
        if render.get(i).is_some_and(|&v| v != 0) {
            *slot = render[i];
        }
    }
    merged
}

/// Refresh the active row in the shared table without advancing the ring selector.
pub(super) fn sync_table_row_to_device(ld: &LogicalDevice, data: &[u32], row: u32) -> Result<()> {
    let ft = ld.frame_table.lock().unwrap();
    let row_u32s = FRAME_TABLE_ROW_STRIDE as usize;
    let copy_u32s = data.len().min(row_u32s).min(FRAME_TABLE_TABLE_U32S);
    let src = &data[0..copy_u32s];

    let table_ptr = ft.table.contents() as *mut u32;
    anyhow::ensure!(!table_ptr.is_null(), "frame table buffer has null contents");
    let dest_ptr = unsafe {
        (table_ptr as *mut u8)
            .add(row_byte_offset(row) as usize) as *mut u32
    };
    unsafe {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dest_ptr, copy_u32s);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_cycles_rows() {
        let mut ring = FrameTableRing::new(4);
        let rows: Vec<u32> = (0..8).map(|_| ring.next_row()).collect();
        assert_eq!(rows, vec![0, 1, 2, 3, 0, 1, 2, 3]);
    }

    #[test]
    fn fresh_rows_have_no_guard() {
        let ring = FrameTableRing::new(4);
        for r in 0..4 {
            assert_eq!(ring.wait_required(r, 0), None);
        }
    }

    #[test]
    fn record_then_guard_returns_token() {
        let mut ring = FrameTableRing::new(8);
        ring.record(2, 10);
        assert_eq!(ring.wait_required(2, 9), Some(10));
        assert_eq!(ring.wait_required(2, 10), None);
    }

    #[test]
    fn reused_row_guards_on_prior_writer() {
        let mut ring = FrameTableRing::new(2);
        assert_eq!(ring.next_row(), 0);
        ring.record(0, 1);
        assert_eq!(ring.next_row(), 1);
        ring.record(1, 2);
        assert_eq!(ring.next_row(), 0);
        assert_eq!(ring.wait_required(0, 0), Some(1));
    }

    #[test]
    fn depth_one_serializes() {
        let mut ring = FrameTableRing::new(1);
        for _ in 0..3 {
            assert_eq!(ring.next_row(), 0);
        }
        ring.record(0, 5);
        assert_eq!(ring.wait_required(0, 4), Some(5));
        assert_eq!(ring.wait_required(0, 5), None);
    }

    #[test]
    fn record_only_touches_target_row() {
        let mut ring = FrameTableRing::new(4);
        ring.record(2, 99);
        assert_eq!(ring.wait_required(2, 0), Some(99));
        assert_eq!(ring.wait_required(0, 0), None);
        assert_eq!(ring.wait_required(1, 0), None);
        assert_eq!(ring.wait_required(3, 0), None);
    }

    #[test]
    fn row_byte_offset_non_overlapping_and_in_bounds() {
        for row in 0..FRAME_TABLE_MAX_ROWS {
            let off = row_byte_offset(row);
            assert_eq!(off, row as u64 * FRAME_TABLE_ROW_STRIDE as u64 * 4);
            assert!(
                off + FRAME_TABLE_ROW_STRIDE as u64 * 4 <= FRAME_TABLE_TABLE_BYTES,
                "row {row} payload overflows table buffer"
            );
            if row > 0 {
                let prev_end = row_byte_offset(row - 1) + FRAME_TABLE_ROW_STRIDE as u64 * 4;
                assert!(off >= prev_end, "row {row} overlaps row {}", row - 1);
            }
        }
    }
}
