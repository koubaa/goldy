//! Scheme render-pass helpers for integration and screenshot tests.

use goldy::{
    Context, Device, Grant, GrantTexture, Parcel, ReadGrant, RenderTarget, Scheme, Submission, TextureFlags,
    TextureFormat, TextureKind,
};
use std::sync::Arc;

/// Acquire a texture parcel suitable as a copy destination and grant-read source.
pub fn acquire_readback_texture(
    pool: &mut goldy::RetainedPool,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Parcel {
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

/// Record one render pass, copy to `readback`, grant-read, submit, and return pixels.
pub fn scheme_render_and_readback(
    ctx: &Context,
    target: &RenderTarget,
    readback: &Parcel,
    label: &'static str,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> Vec<u8> {
    let mut scheme = Scheme::new(ctx);
    {
        let mut pass = scheme.render_pass(label, target);
        record(&mut pass);
        pass.finish();
    }
    scheme.copy_to_texture(target, readback);
    let grant = scheme.grant_read_texture(readback).expect("grant_read_texture");
    let frame = scheme.submit().expect("submit");
    read_grant_texture(&grant, &frame)
}

/// Record one render pass and submit without CPU readback.
pub fn scheme_render_pass_only(
    ctx: &Context,
    target: &RenderTarget,
    label: &'static str,
    record: impl FnOnce(&mut goldy::SchemeRenderPassBuilder<'_>),
) -> Scheme {
    let mut scheme = Scheme::new(ctx);
    let mut pass = scheme.render_pass(label, target);
    record(&mut pass);
    pass.finish();
    scheme.submit().expect("submit");
    scheme
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
