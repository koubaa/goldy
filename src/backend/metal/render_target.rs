//! Render target management logic.

use super::super::{DeviceHandle, RenderTargetHandle};
use super::render_commands::{create_render_pass, record};
use super::types::{MetalState, RenderTargetState};
use super::utils::format_to_mtl;
use crate::types::{DepthFormat, TextureFormat};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{MTLStorageMode, MTLTextureUsage, TextureDescriptor};

/// Create a render target with optional depth buffer.
pub(super) fn create_with_depth(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    width: u32,
    height: u32,
    color_format: TextureFormat,
    depth_format: Option<DepthFormat>,
) -> Result<RenderTargetHandle> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

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
            texture,
            depth_texture,
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

/// Destroy a Metal offscreen render target (ARC drop frees GPU objects).
pub(super) fn destroy(state: &mut MetalState, target: RenderTargetHandle) {
    let _ = state.render_targets.remove(&target);
}

/// Render into an offscreen render target.
pub(super) fn render_to(
    state: &mut MetalState,
    device_handle: DeviceHandle,
    target: RenderTargetHandle,
    color_load: crate::types::TargetLoad,
    commands: &[super::super::RenderCommand],
) -> Result<()> {
    let logical_device = state.devices.get(&device_handle).context("Invalid device handle")?;

    let (staging_data, lowered_commands, has_bindings) =
        super::frame_table::prepare_render_commands(&state.buffers, &state.pipelines, commands)?;

    let completed = super::context::device_retired(state, device_handle);
    let prologue_row = if has_bindings {
        Some(super::frame_table::run_prologue_for_device(
            state,
            device_handle,
            logical_device,
            &staging_data,
            completed,
        )?)
    } else {
        None
    };

    let render_target = state.render_targets.get(&target).context("Invalid render target")?;

    let clear_depth = commands.iter().find_map(|cmd| match cmd {
        super::super::RenderCommand::ClearDepth(depth) => Some(*depth),
        _ => None,
    });

    let render_pass = create_render_pass(
        &render_target.texture,
        render_target.depth_texture.as_deref(),
        color_load,
        clear_depth,
    );

    let command_buffer = logical_device.command_queue.new_command_buffer();
    let encoder = command_buffer.new_render_command_encoder(render_pass);

    let is_mesh = super::render_commands::commands_use_mesh(commands);
    super::render_commands::declare_pass_resources(
        encoder,
        logical_device,
        &state.buffers,
        render_target.device_handle,
        is_mesh,
    )?;

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

    record(
        encoder,
        &lowered_commands,
        &state.pipelines,
        &state.buffers,
        prologue_row,
    )?;

    if is_mesh {
        super::objc_catch::catch_objc("end_encoding(mesh)", || encoder.end_encoding())?;
    } else {
        encoder.end_encoding();
    }
    command_buffer.commit();
    command_buffer.wait_until_completed();

    // GPU is done — record the ring row as retired so the ring guard knows it is
    // safe to reuse.  We record `completed` (the timeline value at prologue time);
    // future `wait_required` checks will see tok <= device_retired() and not stall.
    if let Some(row) = prologue_row {
        if let Some(ld) = state.devices.get(&device_handle) {
            super::frame_table::record_submission_for_device(ld, row, completed);
        }
    }

    Ok(())
}
