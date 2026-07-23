//! Scheme render-pass helpers for integration and screenshot tests.
//!
//! Included from multiple integration test binaries; not every entry point is used in each crate.
#![allow(dead_code)]

use goldy::{
    Context, DepthFormat, Device, MemoryExchange, Parcel, Scheme, Submission, Texture, TextureFlags, TextureFormat,
    TextureKind, WithdrawTransaction,
};
use std::sync::Arc;

/// Acquire a texture parcel suitable as a copy destination and withdraw source.
pub fn acquire_readback_texture(
    pool: &mut goldy::RetainedPool,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Texture {
    pool.acquire_texture(
        width,
        height,
        format,
        TextureKind::Direct,
        TextureFlags::COPY_SRC | TextureFlags::COPY_DST,
        None,
    )
    .expect("acquire readback texture")
}

pub fn read_grant_texture(grant: &WithdrawTransaction, submission: &mut Submission) -> Vec<u8> {
    grant
        .claim(submission)
        .expect("claim")
        .consume()
        .expect("withdraw consume")
        .to_vec()
}

/// Record render pass → copy-to-texture → withdraw once on a new scheme.
pub fn scheme_record_readback(
    ctx: &Context,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    readback: &Parcel,
    label: &'static str,
    color_load: goldy::TargetLoad,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> (Scheme, WithdrawTransaction) {
    let mut scheme = Scheme::new(ctx);
    let rt = scheme
        .lease_render_target(width, height, format, depth_format)
        .expect("render target lease");
    {
        let mut pass = scheme.render_pass(label, &rt, color_load);
        record(&mut pass);
        pass.finish();
    }
    scheme.copy_to_texture(&rt, readback).expect("copy_to_texture");
    let grant = MemoryExchange::new(scheme.context())
        .bind_withdraw(&mut scheme, readback)
        .expect("withdraw");
    (scheme, grant)
}

/// Record once, submit once, consume withdraw claim, and return pixels.
pub fn scheme_render_and_readback(
    ctx: &Context,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    readback: &Parcel,
    label: &'static str,
    color_load: goldy::TargetLoad,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> Vec<u8> {
    let (mut scheme, grant) = scheme_record_readback(
        ctx,
        width,
        height,
        format,
        depth_format,
        readback,
        label,
        color_load,
        record,
    );
    let mut frame = scheme.submit().expect("submit");
    read_grant_texture(&grant, &mut frame)
}

pub fn make_device() -> Option<Device> {
    let instance = goldy::Instance::new().ok()?;
    instance
        .request_adapter(&goldy::RequestAdapterOptions::default())
        .ok()?
        .request_device(&goldy::DeviceDescriptor::default())
        .ok()
}

pub fn device_and_pool() -> Option<(Device, goldy::RetainedPool)> {
    let device = make_device()?;
    Some((device.clone(), goldy::RetainedPool::new(Arc::new(device))))
}
