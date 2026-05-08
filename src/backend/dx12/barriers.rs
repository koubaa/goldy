//! D3D12 enhanced barriers helpers ([`ID3D12GraphicsCommandList7::Barrier`]).
//!
//! Spec: <https://microsoft.github.io/DirectX-Specs/d3d/D3D12EnhancedBarriers.html>
//! Reference impl: <https://github.com/microsoft/DirectX-Graphics-Samples> (enhanced-barriers commits)

use std::mem::ManuallyDrop;
use windows::Win32::Graphics::Direct3D12::*;

/// Panics if `SYNC_NONE` is paired with an access other than `NO_ACCESS`, or vice versa.
/// Per the spec: "If SyncBefore is SYNC_NONE, then AccessBefore MUST be NO_ACCESS"
/// and the inverse is also expected (NO_ACCESS implies nothing to sync).
fn assert_sync_access_pairing(sync: D3D12_BARRIER_SYNC, access: D3D12_BARRIER_ACCESS, label: &str) {
    if sync == D3D12_BARRIER_SYNC_NONE {
        assert!(
            access == D3D12_BARRIER_ACCESS_NO_ACCESS,
            "{label}: SYNC_NONE requires ACCESS_NO_ACCESS, got access {:#x}",
            access.0,
        );
    }
    if access == D3D12_BARRIER_ACCESS_NO_ACCESS {
        assert!(
            sync == D3D12_BARRIER_SYNC_NONE,
            "{label}: ACCESS_NO_ACCESS requires SYNC_NONE, got sync {:#x}",
            sync.0,
        );
    }
}

/// Full-buffer range for [`D3D12_BUFFER_BARRIER`].
pub const BUFFER_BARRIER_FULL_SIZE: u64 = u64::MAX;

/// Record one or more buffer barriers.
///
/// # Safety
/// `barriers` must remain valid for the duration of the call.
pub unsafe fn barrier_buffers(cmd: &ID3D12GraphicsCommandList7, barriers: &[D3D12_BUFFER_BARRIER]) {
    if barriers.is_empty() {
        return;
    }
    let group = D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_BUFFER,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pBufferBarriers: barriers.as_ptr(),
        },
    };
    cmd.Barrier(&[group]);
}

/// Record one or more texture barriers.
///
/// # Safety
/// `barriers` must remain valid for the duration of the call.
pub unsafe fn barrier_textures(
    cmd: &ID3D12GraphicsCommandList7,
    barriers: &[D3D12_TEXTURE_BARRIER],
) {
    if barriers.is_empty() {
        return;
    }
    let group = D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_TEXTURE,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pTextureBarriers: barriers.as_ptr(),
        },
    };
    cmd.Barrier(&[group]);
}

/// Record one or more global barriers.
///
/// # Safety
/// `barriers` must remain valid for the duration of the call.
pub unsafe fn barrier_globals(cmd: &ID3D12GraphicsCommandList7, barriers: &[D3D12_GLOBAL_BARRIER]) {
    if barriers.is_empty() {
        return;
    }
    let group = D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_GLOBAL,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pGlobalBarriers: barriers.as_ptr(),
        },
    };
    cmd.Barrier(&[group]);
}

/// Release COM references inside texture barriers that were created with `resource.clone()`.
///
/// Must be called after `barrier_textures()` returns to avoid leaking `AddRef`'d resources
/// wrapped in `ManuallyDrop`.
///
/// # Safety
/// Each barrier's `pResource` must not be used after this call.
pub unsafe fn drop_texture_barriers(barriers: &mut [D3D12_TEXTURE_BARRIER]) {
    for b in barriers {
        ManuallyDrop::drop(&mut b.pResource);
    }
}

/// Release COM references inside buffer barriers that were created with `resource.clone()`.
///
/// Must be called after `barrier_buffers()` returns to avoid leaking `AddRef`'d resources
/// wrapped in `ManuallyDrop`.
///
/// # Safety
/// Each barrier's `pResource` must not be used after this call.
pub unsafe fn drop_buffer_barriers(barriers: &mut [D3D12_BUFFER_BARRIER]) {
    for b in barriers {
        ManuallyDrop::drop(&mut b.pResource);
    }
}

/// "All subresources" range for a texture barrier.
/// When `NumMipLevels == 0`, `IndexOrFirstMipLevel` is a subresource index;
/// `0xffffffff` means "all subresources" (same as `CD3DX12_BARRIER_SUBRESOURCE_RANGE(0xffffffff)`).
pub fn texture_subresource_all() -> D3D12_BARRIER_SUBRESOURCE_RANGE {
    D3D12_BARRIER_SUBRESOURCE_RANGE {
        IndexOrFirstMipLevel: 0xffffffff,
        NumMipLevels: 0,
        FirstArraySlice: 0,
        NumArraySlices: 0,
        FirstPlane: 0,
        NumPlanes: 0,
    }
}

/// Buffer covering the entire resource (typical for structured buffers).
pub fn buffer_barrier_full(
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
) -> D3D12_BUFFER_BARRIER {
    assert_sync_access_pairing(sync_before, access_before, "buffer SyncBefore/AccessBefore");
    assert_sync_access_pairing(sync_after, access_after, "buffer SyncAfter/AccessAfter");
    D3D12_BUFFER_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
        pResource: ManuallyDrop::new(Some(resource.clone())),
        Offset: 0,
        Size: BUFFER_BARRIER_FULL_SIZE,
    }
}

pub fn texture_barrier_full(
    resource: &ID3D12Resource,
    sync_before: D3D12_BARRIER_SYNC,
    sync_after: D3D12_BARRIER_SYNC,
    access_before: D3D12_BARRIER_ACCESS,
    access_after: D3D12_BARRIER_ACCESS,
    layout_before: D3D12_BARRIER_LAYOUT,
    layout_after: D3D12_BARRIER_LAYOUT,
) -> D3D12_TEXTURE_BARRIER {
    assert_sync_access_pairing(
        sync_before,
        access_before,
        "texture SyncBefore/AccessBefore",
    );
    assert_sync_access_pairing(sync_after, access_after, "texture SyncAfter/AccessAfter");
    D3D12_TEXTURE_BARRIER {
        SyncBefore: sync_before,
        SyncAfter: sync_after,
        AccessBefore: access_before,
        AccessAfter: access_after,
        LayoutBefore: layout_before,
        LayoutAfter: layout_after,
        pResource: ManuallyDrop::new(Some(resource.clone())),
        Subresources: texture_subresource_all(),
        Flags: D3D12_TEXTURE_BARRIER_FLAG_NONE,
    }
}
