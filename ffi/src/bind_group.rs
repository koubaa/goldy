//! FFI bindings for BindGroup and BindGroupLayout.

use crate::buffer::GoldyBuffer;
use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::sampler::GoldySampler;
use crate::texture::GoldyTexture;
use crate::types::{GoldyBindingType, GoldyShaderStages};
use std::ptr;
use std::slice;

/// Opaque handle to a Goldy BindGroupLayout.
pub struct GoldyBindGroupLayout {
    pub(crate) inner: goldy::BindGroupLayout,
}

/// Opaque handle to a Goldy BindGroup.
pub struct GoldyBindGroup {
    pub(crate) inner: goldy::BindGroup,
}

/// Bind group layout binding descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyBindGroupLayoutBinding {
    pub binding: u32,
    pub visibility: GoldyShaderStages,
    pub binding_type: GoldyBindingType,
}

/// Buffer binding descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyBufferBinding {
    pub binding: u32,
    pub buffer: *const GoldyBuffer,
    pub offset: u64,
    /// Size in bytes, or 0 for entire buffer.
    pub size: u64,
}

/// Texture binding descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldyTextureBinding {
    pub binding: u32,
    pub texture: *const GoldyTexture,
}

/// Sampler binding descriptor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GoldySamplerBinding {
    pub binding: u32,
    pub sampler: *const GoldySampler,
}

/// Create a bind group layout.
///
/// Returns a pointer to the layout, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_bind_group_layout_create(
    device: *const GoldyDevice,
    bindings: *const GoldyBindGroupLayoutBinding,
    binding_count: u32,
) -> *mut GoldyBindGroupLayout {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    
    let bindings_slice = if binding_count > 0 && !bindings.is_null() {
        slice::from_raw_parts(bindings, binding_count as usize)
    } else {
        &[]
    };
    
    let layout_bindings: Vec<goldy::BindGroupLayoutBinding> = bindings_slice
        .iter()
        .map(|b| goldy::BindGroupLayoutBinding {
            binding: b.binding,
            visibility: b.visibility.into(),
            ty: b.binding_type.into(),
        })
        .collect();
    
    match goldy::BindGroupLayout::new(&(*device).inner, &layout_bindings) {
        Ok(layout) => Box::into_raw(Box::new(GoldyBindGroupLayout { inner: layout })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a bind group layout.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_bind_group_layout_destroy(layout: *mut GoldyBindGroupLayout) {
    if !layout.is_null() {
        drop(Box::from_raw(layout));
    }
}

/// Create a bind group with buffer bindings only.
///
/// Returns a pointer to the bind group, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_bind_group_create(
    device: *const GoldyDevice,
    layout: *const GoldyBindGroupLayout,
    buffer_bindings: *const GoldyBufferBinding,
    buffer_binding_count: u32,
) -> *mut GoldyBindGroup {
    goldy_bind_group_create_with_resources(
        device,
        layout,
        buffer_bindings,
        buffer_binding_count,
        ptr::null(),
        0,
        ptr::null(),
        0,
    )
}

/// Create a bind group with buffers, textures, and samplers.
///
/// Returns a pointer to the bind group, or null on failure.
///
/// # Safety
/// All pointers must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_bind_group_create_with_resources(
    device: *const GoldyDevice,
    layout: *const GoldyBindGroupLayout,
    buffer_bindings: *const GoldyBufferBinding,
    buffer_binding_count: u32,
    texture_bindings: *const GoldyTextureBinding,
    texture_binding_count: u32,
    sampler_bindings: *const GoldySamplerBinding,
    sampler_binding_count: u32,
) -> *mut GoldyBindGroup {
    if device.is_null() || layout.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device or layout is null"));
        return ptr::null_mut();
    }
    
    // Process buffer bindings
    let buffer_bindings_slice = if buffer_binding_count > 0 && !buffer_bindings.is_null() {
        slice::from_raw_parts(buffer_bindings, buffer_binding_count as usize)
    } else {
        &[]
    };
    
    let buffer_bindings_vec: Vec<goldy::BufferBinding> = buffer_bindings_slice
        .iter()
        .filter(|b| !b.buffer.is_null())
        .map(|b| {
            if b.size == 0 {
                goldy::BufferBinding::new(b.binding, &(*b.buffer).inner)
            } else {
                goldy::BufferBinding::with_range(b.binding, &(*b.buffer).inner, b.offset, b.size)
            }
        })
        .collect();
    
    // Process texture bindings
    let texture_bindings_slice = if texture_binding_count > 0 && !texture_bindings.is_null() {
        slice::from_raw_parts(texture_bindings, texture_binding_count as usize)
    } else {
        &[]
    };
    
    let texture_bindings_vec: Vec<goldy::TextureBinding> = texture_bindings_slice
        .iter()
        .filter(|t| !t.texture.is_null())
        .map(|t| goldy::TextureBinding::new(t.binding, &(*t.texture).inner))
        .collect();
    
    // Process sampler bindings
    let sampler_bindings_slice = if sampler_binding_count > 0 && !sampler_bindings.is_null() {
        slice::from_raw_parts(sampler_bindings, sampler_binding_count as usize)
    } else {
        &[]
    };
    
    let sampler_bindings_vec: Vec<goldy::SamplerBinding> = sampler_bindings_slice
        .iter()
        .filter(|s| !s.sampler.is_null())
        .map(|s| goldy::SamplerBinding::new(s.binding, &(*s.sampler).inner))
        .collect();
    
    match goldy::BindGroup::with_resources(
        &(*device).inner,
        &(*layout).inner,
        &buffer_bindings_vec,
        &texture_bindings_vec,
        &sampler_bindings_vec,
    ) {
        Ok(bind_group) => Box::into_raw(Box::new(GoldyBindGroup { inner: bind_group })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a bind group.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_bind_group_destroy(bind_group: *mut GoldyBindGroup) {
    if !bind_group.is_null() {
        drop(Box::from_raw(bind_group));
    }
}

