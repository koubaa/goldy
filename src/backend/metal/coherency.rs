//! CPU/GPU coherency for host-visible Metal buffers.
//!
//! Shared storage is coherent: CPU writes through `contents()` do not call
//! `didModifyRange`, and a completed command buffer is enough before a CPU
//! read. That is the Apple silicon heap path and the standalone Shared
//! buffers used on Intel and discrete Macs.
//!
//! Managed storage (legal for buffers on discrete Macs, never for `MTLHeap`)
//! is different. After a CPU write the GPU copy is stale until
//! `didModifyRange:`. After a GPU write the CPU copy is stale until a blit
//! `synchronizeResource:` completes. These hooks are no-ops for Shared and
//! Private buffers, and they are the Managed protocol if a buffer is created
//! that way. Calling either on a non-Managed buffer is invalid.

use ::metal as mtl;

/// Tell Metal about a CPU write. Required for `MTLStorageModeManaged` only.
pub(super) fn note_cpu_write(buffer: &mtl::BufferRef, offset: u64, len: u64) {
    if len == 0 || buffer.storage_mode() != mtl::MTLStorageMode::Managed {
        return;
    }
    buffer.did_modify_range(mtl::NSRange::new(offset, len));
}

/// Make GPU writes visible to a later `contents()` read.
///
/// Issues `synchronizeResource:` and waits. No-op unless the buffer is Managed.
pub(super) fn prepare_cpu_read(queue: &mtl::CommandQueueRef, buffer: &mtl::BufferRef) {
    if buffer.storage_mode() != mtl::MTLStorageMode::Managed {
        return;
    }
    let command_buffer = queue.new_command_buffer();
    let blit = command_buffer.new_blit_command_encoder();
    blit.synchronize_resource(buffer);
    blit.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();
}
