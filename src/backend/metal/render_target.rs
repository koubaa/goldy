//! Render target management logic.

use super::super::{DeviceHandle, RenderTargetHandle};
use super::render_commands::{create_render_pass, record};
use super::types::{MetalState, RenderTargetState};
use super::utils::format_to_mtl;
use crate::types::{DepthFormat, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLOrigin, MTLSize, MTLStorageMode, MTLTextureUsage, TextureDescriptor};

/// Create a render target.
pub(super) fn create(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    format: TextureFormat,
) -> Result<RenderTargetHandle> {
    create_with_depth(state, device_handle, width, height, format, None)
}

/// Create a render target with optional depth buffer.
pub(super) fn create_with_depth(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    depth_format: Option<DepthFormat>,
) -> Result<RenderTargetHandle> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let descriptor = TextureDescriptor::new();
    descriptor.set_width(width as u64);
    descriptor.set_height(height as u64);
    descriptor.set_pixel_format(format_to_mtl(color_format));
    descriptor.set_usage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
    descriptor.set_storage_mode(MTLStorageMode::Private);

    let texture = logical_device.device.new_texture(&descriptor);

    let depth_texture = depth_format.map(|df| {
        let depth_desc = TextureDescriptor::new();
        depth_desc.set_width(width as u64);
        depth_desc.set_height(height as u64);
        depth_desc.set_pixel_format(super::utils::depth_format_to_mtl(df));
        depth_desc.set_usage(MTLTextureUsage::RenderTarget);
        depth_desc.set_storage_mode(MTLStorageMode::Private);
        logical_device.device.new_texture(&depth_desc)
    });

    let handle = state.next_render_target_handle;
    state.next_render_target_handle += 1;

    state.render_targets.insert(
        handle,
        RenderTargetState {
            device_handle,
            width,
            height,
            format: color_format,
            texture,
            depth_texture,
            has_rendered: false,
        },
    );

    tracing::debug!(
        "Created render target {} ({}x{}, {:?})",
        handle,
        width,
        height,
        color_format
    );
    Ok(handle)
}

/// Destroy a render target.
pub(super) fn destroy(state: &mut MetalState, target: RenderTargetHandle) {
    state.render_targets.remove(&target);
}

/// Render commands to an offscreen render target.
pub(super) fn render_to(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    commands: &[super::super::RenderCommand],
) -> Result<()> {
    let logical_device = state
        .devices
        .get(&device_handle)
        .context("Invalid device handle")?;

    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target")?;

    let mut clear_color = None;
    let mut clear_depth = None;
    for cmd in commands {
        match cmd {
            super::super::RenderCommand::Clear(color) => clear_color = Some(*color),
            super::super::RenderCommand::ClearDepth(depth) => clear_depth = Some(*depth),
            _ => {}
        }
    }

    let render_pass = create_render_pass(
        &render_target.texture,
        render_target.depth_texture.as_deref(),
        clear_color,
        clear_depth,
    );

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let encoder = command_buffer.new_render_command_encoder(render_pass);

    let render_stages = mtl::MTLRenderStages::Vertex | mtl::MTLRenderStages::Fragment;
    if logical_device.heap_buffer_count > 0 {
        encoder.use_heap_at(&logical_device.buffer_heap, render_stages);
    }
    if logical_device.heap_texture_count > 0 {
        encoder.use_heap_at(&logical_device.texture_heap, render_stages);
    }

    encoder.set_viewport(mtl::MTLViewport {
        originX: 0.0,
        originY: 0.0,
        width: render_target.width as f64,
        height: render_target.height as f64,
        znear: 0.0,
        zfar: 1.0,
    });
    encoder.set_scissor_rect(mtl::MTLScissorRect {
        x: 0,
        y: 0,
        width: render_target.width as u64,
        height: render_target.height as u64,
    });

    record(encoder, commands, &state.pipelines, &state.buffers);

    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    if let Some(rt) = state.render_targets.get_mut(&target) {
        rt.has_rendered = true;
    }

    Ok(())
}

/// Read render target contents back to CPU.
pub(super) fn read_to_cpu(
    state: &MetalState,
    target: RenderTargetHandle,
    output: &mut [u8],
) -> Result<()> {
    let render_target = state
        .render_targets
        .get(&target)
        .context("Invalid render target")?;

    if !render_target.has_rendered {
        anyhow::bail!("Cannot read from render target that hasn't been rendered to");
    }

    let logical_device = state
        .devices
        .get(&render_target.device_handle)
        .context("Device no longer valid")?;

    let width = render_target.width;
    let height = render_target.height;
    let bytes_per_pixel = render_target.format.bytes_per_pixel();
    let bytes_per_row = width * bytes_per_pixel;
    let expected_size = (bytes_per_row * height) as usize;

    if output.len() < expected_size {
        anyhow::bail!(
            "Output buffer too small: need {} bytes, got {}",
            expected_size,
            output.len()
        );
    }

    let staging_buffer = logical_device.device.new_buffer(
        expected_size as u64,
        mtl::MTLResourceOptions::StorageModeShared,
    );

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let blit_encoder = command_buffer.new_blit_command_encoder();

    blit_encoder.copy_from_texture_to_buffer(
        &render_target.texture,
        0,
        0,
        MTLOrigin { x: 0, y: 0, z: 0 },
        MTLSize {
            width: width as u64,
            height: height as u64,
            depth: 1,
        },
        &staging_buffer,
        0,
        bytes_per_row as u64,
        (bytes_per_row * height) as u64,
        mtl::MTLBlitOption::empty(),
    );

    blit_encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    unsafe {
        let ptr = staging_buffer.contents();
        std::ptr::copy_nonoverlapping(ptr as *const u8, output.as_mut_ptr(), expected_size);
    }

    Ok(())
}
