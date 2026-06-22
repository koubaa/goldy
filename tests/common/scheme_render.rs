//! Scheme render-pass helpers for integration and screenshot tests.
//!
//! Included from multiple integration test binaries; not every entry point is used in each crate.
#![allow(dead_code)]

use goldy::{
    Context, DepthFormat, Device, Grant, GrantTexture, Parcel, ReadGrant, Scheme, Submission, Texture, TextureFlags,
    TextureFormat, TextureKind,
};
use std::sync::Arc;

/// Acquire a texture parcel suitable as a copy destination and grant-read source.
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

pub fn read_grant_texture(grant: &ReadGrant<GrantTexture>, submission: &Submission) -> Vec<u8> {
    grant.consume(submission).expect("grant consume").to_vec()
}

/// Record render pass → copy-to-texture → grant-read once on a new scheme.
pub fn scheme_record_readback(
    ctx: &Context,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    readback: &Parcel,
    label: &'static str,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> (Scheme, ReadGrant<GrantTexture>) {
    let mut scheme = Scheme::new(ctx);
    let rt = scheme
        .lease_render_target(width, height, format, depth_format)
        .expect("render target lease");
    {
        let mut pass = scheme.render_pass(label, &rt);
        record(&mut pass);
        pass.finish();
    }
    scheme.copy_to_texture(&rt, readback).expect("copy_to_texture");
    let grant = scheme.grant_read_texture(readback).expect("grant_read_texture");
    (scheme, grant)
}

/// Record once, submit once, consume grant, and return pixels.
pub fn scheme_render_and_readback(
    ctx: &Context,
    width: u32,
    height: u32,
    format: TextureFormat,
    depth_format: Option<DepthFormat>,
    readback: &Parcel,
    label: &'static str,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> Vec<u8> {
    let (mut scheme, grant) = scheme_record_readback(ctx, width, height, format, depth_format, readback, label, record);
    let frame = scheme.submit().expect("submit");
    read_grant_texture(&grant, &frame)
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
