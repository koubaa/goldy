//! Host read claims: public CPU ownership of a parcel between gates.
//!
//! `(&mut submission >> &parcel).take::<T>()` waits for the submission (and any later
//! GPU write on the parcel), then realizes a typed view. Host-coherent media map in
//! place; others copy through a context staging pool. CUDA fills that pool with one
//! producer-stream DtoH into cacheable pinned host memory, then `take()` copies into
//! an owned view. Eager sink copies and independently settled streaming identities are
//! not part of this path. While the view lives, a later submit that writes the parcel
//! fails rather than blocking.

use crate::backend::{GpuCommand, HostMapping};
use crate::buffer::{Allocation, BufferSource};
use crate::context::Context;
use crate::error::GoldyError;
use crate::parcel::{Parcel, ParcelStamp};
use crate::scheme::Submission;
use crate::texture::TextureBacking;
use crate::timeline::{Epoch, TimelineValue};
use crate::types::{TextureFlags, TextureKind};
use std::marker::PhantomData;
use std::ops::{Deref, Shr};
use std::sync::Arc;

/// Selected host read, realized by [`Self::take`].
///
/// Dropping without `take` is a no-op (unlike exchange claims).
#[must_use = "call take() to realize the host view"]
pub struct PendingHostRead {
    inner: Result<HostReadRequest, GoldyError>,
}

/// Selected read from a scheme-recorded [`crate::HostSink`].
///
/// Selection reserves the sink parcel immediately but does not wait. `take`
/// waits for the producing submission and reads the already-populated staging;
/// it never submits another GPU copy.
#[must_use = "call take() to realize the host view"]
pub struct PendingHostSinkRead {
    inner: Result<HostSinkReadRequest, GoldyError>,
}

struct HostSinkReadRequest {
    ctx: Context,
    ready_after: TimelineValue,
    handle: crate::backend::BufferHandle,
    byte_size: u64,
    stamp: Arc<ParcelStamp>,
    claimed: bool,
}

struct HostReadRequest {
    ctx: Context,
    ready_after: TimelineValue,
    stamp: Arc<ParcelStamp>,
    kind: HostReadKind,
}

enum HostReadKind {
    Buffer {
        handle: crate::backend::BufferHandle,
        offset: u64,
        byte_size: u64,
        keepalive: Arc<Allocation>,
    },
    Texture {
        handle: crate::handles::TextureHandle,
        keepalive: TextureBacking,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
    },
}

impl PendingHostRead {
    pub(crate) fn from_submission(submission: &Submission, parcel: &Parcel) -> Self {
        PendingHostRead {
            inner: HostReadRequest::from_submission(submission, parcel),
        }
    }

    /// Wait for the gate and return a typed view of the parcel bytes.
    pub fn take<T: bytemuck::Pod>(self) -> Result<HostView<T>, GoldyError> {
        let req = self.inner?;
        req.realize::<T>()
    }

    /// Wait for the gate and return an untyped byte view.
    pub fn take_bytes(self) -> Result<HostView<u8>, GoldyError> {
        self.take::<u8>()
    }
}

impl PendingHostSinkRead {
    pub(crate) fn from_submission(submission: &Submission, sink: &crate::HostSink) -> Self {
        let result = (|| {
            if submission.scheme_id() != sink.inner.scheme_id {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "HostSink belongs to a different scheme than this submission"
                )));
            }
            if !Arc::ptr_eq(&submission.context().inner, &sink.inner.ctx.inner) {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "HostSink belongs to a different context than this submission"
                )));
            }
            sink.inner.stamp.acquire_host_claim();
            Ok(HostSinkReadRequest {
                ctx: sink.inner.ctx.clone(),
                ready_after: submission.timeline_value(),
                handle: sink.inner.handle,
                byte_size: sink.inner.byte_size,
                stamp: Arc::clone(&sink.inner.stamp),
                claimed: true,
            })
        })();
        Self { inner: result }
    }

    /// Wait for the recorded sink copy and return its typed contents.
    pub fn take<T: bytemuck::Pod>(self) -> Result<HostView<T>, GoldyError> {
        self.inner?.realize::<T>()
    }

    /// Wait for the recorded sink copy and return its bytes.
    pub fn take_bytes(self) -> Result<HostView<u8>, GoldyError> {
        self.take::<u8>()
    }
}

impl std::fmt::Debug for PendingHostSinkRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingHostSinkRead")
            .field("ok", &self.inner.is_ok())
            .finish()
    }
}

impl HostSinkReadRequest {
    fn realize<T: bytemuck::Pod>(mut self) -> Result<HostView<T>, GoldyError> {
        let elem = std::mem::size_of::<T>();
        if elem == 0 || !(self.byte_size as usize).is_multiple_of(elem) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host sink byte size {} is not a multiple of {elem}",
                self.byte_size
            )));
        }
        self.ctx.wait_until(self.ready_after)?;
        let last_write = self.stamp.sync.lock().unwrap().last_write.clone();
        for (context, value) in last_write.iter() {
            self.ctx.wait_until_epoch(Epoch { context, value })?;
        }
        let mut bytes = vec![0u8; self.byte_size as usize];
        {
            let backend = self.ctx.runtime().inner.backend.lock().unwrap();
            backend
                .read_readback_buffer(self.handle, &mut bytes)
                .map_err(|e| self.ctx.classify(e))?;
        }
        let values = cast_bytes::<T>(bytes)?;
        self.claimed = false;
        Ok(HostView {
            stamp: Arc::clone(&self.stamp),
            ctx: None,
            mapped_handle: None,
            backing: HostViewBacking::Owned(values),
            _ty: PhantomData,
        })
    }
}

impl Drop for HostSinkReadRequest {
    fn drop(&mut self) {
        if self.claimed {
            self.stamp.release_host_claim();
        }
    }
}

impl std::fmt::Debug for PendingHostRead {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingHostRead")
            .field("ok", &self.inner.is_ok())
            .finish()
    }
}

impl HostReadRequest {
    fn from_submission(submission: &Submission, parcel: &Parcel) -> Result<Self, GoldyError> {
        let ctx = submission.context().clone();
        if !parcel.is_homed_on(&ctx) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "parcel home device does not match submission context"
            )));
        }
        let kind = if parcel.buffer_handle().is_some() {
            let byte_size = parcel.byte_size();
            if byte_size == 0 {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "host claim requires a non-zero buffer byte size"
                )));
            }
            let keepalive = parcel.grant_buffer_keepalive().map_err(|e| ctx.classify(e))?;
            HostReadKind::Buffer {
                handle: parcel.buffer_handle().expect("buffer parcel"),
                offset: parcel.source_offset(),
                byte_size,
                keepalive,
            }
        } else if parcel.texture_handle().is_some() {
            let keepalive = parcel.grant_texture_keepalive().map_err(|e| ctx.classify(e))?;
            let (width, height, format, access, flags) = parcel.texture_descriptor().ok_or_else(|| {
                GoldyError::Backend(anyhow::anyhow!("host claim requires a buffer or texture parcel"))
            })?;
            if !flags.contains(TextureFlags::COPY_SRC) {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "host claim on a texture requires TextureFlags::COPY_SRC"
                )));
            }
            if matches!(access, TextureKind::Interpolated) {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "host claim on a texture requires TextureKind::Direct or DirectInterpolated"
                )));
            }
            if width == 0 || height == 0 {
                return Err(GoldyError::Backend(anyhow::anyhow!(
                    "host claim on a texture requires non-zero dimensions"
                )));
            }
            HostReadKind::Texture {
                handle: parcel.texture_handle().expect("texture parcel"),
                keepalive,
                width,
                height,
                format,
            }
        } else {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host claim requires a buffer or texture parcel"
            )));
        };
        Ok(Self {
            ctx,
            ready_after: submission.timeline_value(),
            stamp: parcel.stamp_handle(),
            kind,
        })
    }

    fn wait_ready(&self) -> Result<(), GoldyError> {
        self.ctx.wait_until(self.ready_after)?;
        let last_write = self.stamp.sync.lock().unwrap().last_write.clone();
        for (c, tv) in last_write.iter() {
            self.ctx.wait_until_epoch(Epoch { context: c, value: tv })?;
        }
        Ok(())
    }

    fn realize<T: bytemuck::Pod>(self) -> Result<HostView<T>, GoldyError> {
        let elem = std::mem::size_of::<T>();
        self.wait_ready()?;
        let HostReadRequest {
            ctx,
            ready_after: _,
            stamp,
            kind,
        } = self;
        match kind {
            HostReadKind::Buffer {
                handle,
                offset,
                byte_size,
                keepalive,
            } => {
                if elem == 0 || !(byte_size as usize).is_multiple_of(elem) {
                    return Err(GoldyError::Backend(anyhow::anyhow!(
                        "host claim: parcel byte size {byte_size} is not a multiple of {}",
                        elem
                    )));
                }
                Self::realize_buffer(ctx, stamp, handle, offset, byte_size, keepalive)
            }
            HostReadKind::Texture {
                handle,
                keepalive,
                width,
                height,
                format,
            } => {
                let _ = keepalive;
                Self::realize_texture::<T>(ctx, stamp, handle, width, height, format, elem)
            }
        }
    }

    fn realize_buffer<T: bytemuck::Pod>(
        ctx: Context,
        stamp: Arc<ParcelStamp>,
        handle: crate::backend::BufferHandle,
        offset: u64,
        byte_size: u64,
        keepalive: Arc<Allocation>,
    ) -> Result<HostView<T>, GoldyError> {
        {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_acquire(handle).map_err(|e| ctx.classify(e))?;
        }
        let mapping = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_mapping(handle)
        };
        if let Some(mapping) = mapping {
            return Self::finish_mapped(ctx, stamp, handle, offset, byte_size, keepalive, mapping);
        }
        {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_release(handle);
        }
        let twin = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_twin_mapping(handle)
        };
        if twin.is_some() {
            return Self::finish_twin(ctx, stamp, handle, offset, byte_size, keepalive);
        }
        Self::finish_staged_buffer(ctx, stamp, handle, offset, byte_size, keepalive)
    }

    fn finish_mapped<T: bytemuck::Pod>(
        ctx: Context,
        stamp: Arc<ParcelStamp>,
        handle: crate::backend::BufferHandle,
        offset: u64,
        byte_size: u64,
        keepalive: Arc<Allocation>,
        mapping: HostMapping,
    ) -> Result<HostView<T>, GoldyError> {
        if offset.saturating_add(byte_size) > mapping.len {
            {
                let mut backend = ctx.runtime().inner.backend.lock().unwrap();
                backend.host_read_release(handle);
            }
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host claim: [{offset}..{}] exceeds mapped length {}",
                offset + byte_size,
                mapping.len
            )));
        }
        let ptr = unsafe { mapping.ptr.add(offset as usize) };
        stamp.acquire_host_claim();
        Ok(HostView {
            stamp,
            ctx: Some(ctx),
            mapped_handle: Some(handle),
            backing: HostViewBacking::Mapped {
                ptr,
                len: (byte_size as usize) / std::mem::size_of::<T>(),
                _keepalive: keepalive,
            },
            _ty: PhantomData,
        })
    }

    fn finish_twin<T: bytemuck::Pod>(
        ctx: Context,
        stamp: Arc<ParcelStamp>,
        handle: crate::backend::BufferHandle,
        offset: u64,
        byte_size: u64,
        keepalive: Arc<Allocation>,
    ) -> Result<HostView<T>, GoldyError> {
        let copy_tv = {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend
                .submit_standalone(
                    ctx.backend_handle(),
                    &[GpuCommand::CopyToCpuReadableTwin {
                        src: handle,
                        src_offset: offset,
                        size: byte_size,
                    }],
                    None,
                )
                .map_err(|e| ctx.classify(e))?
        };
        ctx.advance_high_water_timeline(copy_tv);
        ctx.wait_until(copy_tv)?;
        let mapping = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_twin_mapping(handle)
        };
        let Some(mapping) = mapping else {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host claim: CPU_READABLE twin mapping missing after copy"
            )));
        };
        if offset.saturating_add(byte_size) > mapping.len {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host claim: twin mapping shorter than parcel"
            )));
        }
        let ptr = unsafe { mapping.ptr.add(offset as usize) };
        stamp.acquire_host_claim();
        Ok(HostView {
            stamp,
            ctx: None,
            mapped_handle: None,
            backing: HostViewBacking::Mapped {
                ptr,
                len: (byte_size as usize) / std::mem::size_of::<T>(),
                _keepalive: keepalive,
            },
            _ty: PhantomData,
        })
    }

    fn finish_staged_buffer<T: bytemuck::Pod>(
        ctx: Context,
        stamp: Arc<ParcelStamp>,
        handle: crate::backend::BufferHandle,
        offset: u64,
        byte_size: u64,
        keepalive: Arc<Allocation>,
    ) -> Result<HostView<T>, GoldyError> {
        let pool = ctx.host_read_pool();
        let staging = pool.take_or_alloc_buffer(&ctx, byte_size)?;
        let copy_tv = {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend
                .submit_standalone(
                    ctx.backend_handle(),
                    &[GpuCommand::CopyBuffer {
                        src: handle,
                        src_offset: offset,
                        dst: staging,
                        dst_offset: 0,
                        size: byte_size,
                    }],
                    None,
                )
                .map_err(|e| ctx.classify(e))?
        };
        ctx.advance_high_water_timeline(copy_tv);
        ctx.wait_until(copy_tv)?;
        let mut bytes = vec![0u8; byte_size as usize];
        let read_result = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.read_readback_buffer(staging, &mut bytes)
        };
        pool.return_handle(staging, copy_tv, byte_size, false);
        if let Err(e) = read_result {
            return Err(ctx.classify(e));
        }
        drop(keepalive);
        stamp.acquire_host_claim();
        Ok(HostView {
            stamp,
            ctx: None,
            mapped_handle: None,
            backing: HostViewBacking::Owned(cast_bytes::<T>(bytes)?),
            _ty: PhantomData,
        })
    }

    fn realize_texture<T: bytemuck::Pod>(
        ctx: Context,
        stamp: Arc<ParcelStamp>,
        handle: crate::handles::TextureHandle,
        width: u32,
        height: u32,
        format: crate::types::TextureFormat,
        elem: usize,
    ) -> Result<HostView<T>, GoldyError> {
        let layout = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend
                .query_texture_copy_footprint(ctx.runtime().inner.handle, width, height, format)
                .map_err(|e| ctx.classify(e))?
        };
        if elem == 0 || !(layout.logical_bytes as usize).is_multiple_of(elem) {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "host claim: texture logical size {} is not a multiple of {elem}",
                layout.logical_bytes
            )));
        }
        let pool = ctx.host_read_pool();
        let staging = pool.take_or_alloc_texture(&ctx, layout)?;
        let copy_tv = {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend
                .submit_standalone(
                    ctx.backend_handle(),
                    &[GpuCommand::CopyTextureToReadback {
                        src: handle,
                        dst: staging,
                        layout,
                    }],
                    None,
                )
                .map_err(|e| ctx.classify(e))?
        };
        ctx.advance_high_water_timeline(copy_tv);
        ctx.wait_until(copy_tv)?;
        let mut bytes = vec![0u8; layout.logical_bytes as usize];
        let read_result = {
            let backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.read_texture_readback_staging(staging, layout, &mut bytes)
        };
        pool.return_handle(staging, copy_tv, layout.staging_bytes, true);
        if let Err(e) = read_result {
            return Err(ctx.classify(e));
        }
        stamp.acquire_host_claim();
        Ok(HostView {
            stamp,
            ctx: None,
            mapped_handle: None,
            backing: HostViewBacking::Owned(cast_bytes::<T>(bytes)?),
            _ty: PhantomData,
        })
    }
}

fn cast_bytes<T: bytemuck::Pod>(bytes: Vec<u8>) -> Result<Vec<T>, GoldyError> {
    let n = bytes.len() / std::mem::size_of::<T>();
    let mut out = vec![T::zeroed(); n];
    out.copy_from_slice(bytemuck::cast_slice(&bytes));
    Ok(out)
}

enum HostViewBacking<T> {
    Owned(Vec<T>),
    Mapped {
        ptr: *const u8,
        len: usize,
        _keepalive: Arc<Allocation>,
    },
}

/// Typed host view of a parcel, held as a public CPU claim until drop.
pub struct HostView<T: bytemuck::Pod> {
    stamp: Arc<ParcelStamp>,
    ctx: Option<Context>,
    mapped_handle: Option<crate::backend::BufferHandle>,
    backing: HostViewBacking<T>,
    _ty: PhantomData<T>,
}

unsafe impl<T: bytemuck::Pod + Send> Send for HostView<T> {}
unsafe impl<T: bytemuck::Pod + Sync> Sync for HostView<T> {}

impl<T: bytemuck::Pod> HostView<T> {
    /// Owned copy of the view contents.
    pub fn to_vec(&self) -> Vec<T> {
        self.deref().to_vec()
    }

    /// Consume into an owned `Vec`.
    pub fn into_vec(mut self) -> Vec<T> {
        match std::mem::replace(&mut self.backing, HostViewBacking::Owned(Vec::new())) {
            HostViewBacking::Owned(v) => v,
            HostViewBacking::Mapped { ptr, len, _keepalive } => unsafe {
                std::slice::from_raw_parts(ptr.cast::<T>(), len).to_vec()
            },
        }
    }
}

impl<T: bytemuck::Pod> Deref for HostView<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        match &self.backing {
            HostViewBacking::Owned(v) => v.as_slice(),
            HostViewBacking::Mapped { ptr, len, .. } => unsafe { std::slice::from_raw_parts(ptr.cast::<T>(), *len) },
        }
    }
}

impl<T: bytemuck::Pod> std::fmt::Debug for HostView<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostView")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<T: bytemuck::Pod> Drop for HostView<T> {
    fn drop(&mut self) {
        if let (Some(ctx), Some(handle)) = (self.ctx.take(), self.mapped_handle.take()) {
            let mut backend = ctx.runtime().inner.backend.lock().unwrap();
            backend.host_read_release(handle);
        }
        self.stamp.release_host_claim();
    }
}

impl Shr<&Parcel> for &mut Submission {
    type Output = PendingHostRead;

    fn shr(self, parcel: &Parcel) -> Self::Output {
        PendingHostRead::from_submission(self, parcel)
    }
}

impl Shr<&crate::parcel::Buffer> for &mut Submission {
    type Output = PendingHostRead;

    fn shr(self, buffer: &crate::parcel::Buffer) -> Self::Output {
        PendingHostRead::from_submission(self, buffer)
    }
}

impl Shr<&crate::parcel::Texture> for &mut Submission {
    type Output = PendingHostRead;

    fn shr(self, texture: &crate::parcel::Texture) -> Self::Output {
        PendingHostRead::from_submission(self, texture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::mock_runtime;
    use crate::types::{BufferFlags, BufferKind};
    use crate::Scheme;

    fn take_u32(submission: &mut Submission, parcel: &Parcel) -> Vec<u32> {
        (submission >> parcel).take::<u32>().expect("take").to_vec()
    }

    #[test]
    fn staged_host_claim_reads_written_bytes() {
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device
            .acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let mut submission = scheme.submit().unwrap();
        assert_eq!(take_u32(&mut submission, &*buf), vec![1, 2, 3, 4]);
        assert!(ctx.host_read_staging_alloc_count() >= 1);
    }

    #[test]
    fn mapped_host_claim_skips_staging_when_cpu_readable() {
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device
            .acquire_buffer_with_data_and_flags(&[9u32, 8, 7, 6], BufferKind::Scattered, BufferFlags::CPU_READABLE)
            .unwrap();
        let allocs_before = ctx.host_read_staging_alloc_count();
        let mut scheme = Scheme::new(&ctx);
        let mut submission = scheme.submit().unwrap();
        assert_eq!(take_u32(&mut submission, &*buf), vec![9, 8, 7, 6]);
        assert_eq!(ctx.host_read_staging_alloc_count(), allocs_before);
    }

    #[test]
    fn live_host_view_blocks_writer_submit() {
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device.acquire_buffer_with_data(&[1u32], BufferKind::Scattered).unwrap();
        let mut scheme = Scheme::new(&ctx);
        let mut submission = scheme.submit().unwrap();
        let view = (&mut submission >> &*buf).take::<u32>().unwrap();
        let mut writer = Scheme::new(&ctx);
        writer.clear_parcel(&*buf, 0, 0).unwrap();
        let err = writer.submit().expect_err("write while host view is live");
        assert!(err.detail().contains("host view"), "{err:?}");
        drop(view);
        writer.submit().expect("submit after view drop");
    }

    #[test]
    fn pending_host_read_drop_is_noop() {
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device.acquire_buffer_with_data(&[1u32], BufferKind::Scattered).unwrap();
        let mut scheme = Scheme::new(&ctx);
        let mut submission = scheme.submit().unwrap();
        drop(&mut submission >> &*buf);
        (&mut submission >> &*buf).take::<u32>().unwrap();
    }

    #[test]
    fn size_mismatch_is_error() {
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device
            .acquire_buffer(3, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let mut scheme = Scheme::new(&ctx);
        let mut submission = scheme.submit().unwrap();
        let err = (&mut submission >> &*buf).take::<u32>().expect_err("unaligned");
        assert!(err.detail().contains("multiple"), "{err:?}");
    }

    #[test]
    fn wrong_context_is_error() {
        let buf = mock_runtime()
            .acquire_buffer_with_data(&[1u32], BufferKind::Scattered)
            .unwrap();
        let device_b = mock_runtime();
        let ctx_b = device_b.create_context().unwrap();
        let mut submission = Scheme::new(&ctx_b).submit().unwrap();
        let err = (&mut submission >> &*buf).take::<u32>().expect_err("wrong home");
        assert!(
            err.detail().contains("home device") || err.detail().contains("context"),
            "{err:?}"
        );
    }

    #[test]
    fn texture_take_reads_logical_bytes() {
        use crate::types::{TextureFlags, TextureFormat, TextureKind};
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let tex = device
            .acquire_texture(
                2,
                2,
                TextureFormat::Rgba8Unorm,
                TextureKind::Direct,
                TextureFlags::COPY_SRC,
                Some(&data),
            )
            .unwrap();
        let mut submission = Scheme::new(&ctx).submit().unwrap();
        let view = (&mut submission >> &*tex).take::<u8>().expect("texture take");
        assert_eq!(view.len(), 16);
        assert_eq!(&*view, &data);
    }

    #[test]
    fn range_parcel_take_uses_field_offset() {
        use crate::parcel::{field, Init};
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device
            .acquire_record([field("a", Init::data(&[1u32, 2])), field("b", Init::data(&[9u32, 8]))])
            .unwrap();
        let mut submission = Scheme::new(&ctx).submit().unwrap();
        assert_eq!(
            (&mut submission >> &buf["b"])
                .take::<u32>()
                .expect("field take")
                .to_vec(),
            vec![9, 8]
        );
    }

    #[test]
    fn gpu_read_allowed_while_host_view_live() {
        use crate::{ComputePipeline, NodeAccess, ShaderModule};
        let device = mock_runtime();
        let ctx = device.create_context().unwrap();
        let buf = device
            .acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)
            .unwrap();
        let out = device
            .acquire_buffer(16, BufferKind::Scattered, None, BufferFlags::empty(), None)
            .unwrap();
        let shader = ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1,1,1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#,
        )
        .unwrap();
        let pipeline = ComputePipeline::new(&device, &shader).unwrap();
        let mut submission = Scheme::new(&ctx).submit().unwrap();
        let view = (&mut submission >> &*buf).take::<u32>().unwrap();
        let mut reader = Scheme::new(&ctx);
        reader
            .node("r", &pipeline)
            .with_parcel(&*buf, NodeAccess::Read)
            .with_parcel(&*out, NodeAccess::Write)
            .dispatch(1, 1, 1);
        reader.submit().expect("GPU read while host view is live");
        assert_eq!(&*view, &[1, 2, 3, 4]);
    }
}
