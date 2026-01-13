//! FFI bindings for Texture.

use crate::device::GoldyDevice;
use crate::error::set_last_error_from_anyhow;
use crate::types::{GoldyTextureFormat, GoldyTextureUsage};
use std::ptr;

/// Opaque handle to a Goldy Texture.
pub struct GoldyTexture {
    pub(crate) inner: goldy::Texture,
}

/// Create a new texture.
///
/// Returns a pointer to the texture, or null on failure.
///
/// # Safety
/// The device pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_texture_create(
    device: *const GoldyDevice,
    width: u32,
    height: u32,
    format: GoldyTextureFormat,
    usage: GoldyTextureUsage,
) -> *mut GoldyTexture {
    if device.is_null() {
        set_last_error_from_anyhow(&anyhow::anyhow!("Device is null"));
        return ptr::null_mut();
    }
    
    match goldy::Texture::new(&(*device).inner, width, height, format.into(), usage.into()) {
        Ok(texture) => Box::into_raw(Box::new(GoldyTexture { inner: texture })),
        Err(e) => {
            set_last_error_from_anyhow(&e);
            ptr::null_mut()
        }
    }
}

/// Destroy a texture.
///
/// # Safety
/// The pointer must be valid and not used after this call.
#[no_mangle]
pub unsafe extern "C" fn goldy_texture_destroy(texture: *mut GoldyTexture) {
    if !texture.is_null() {
        drop(Box::from_raw(texture));
    }
}

/// Get the texture width.
///
/// # Safety
/// The texture pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_texture_width(texture: *const GoldyTexture) -> u32 {
    if texture.is_null() {
        return 0;
    }
    (*texture).inner.width()
}

/// Get the texture height.
///
/// # Safety
/// The texture pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_texture_height(texture: *const GoldyTexture) -> u32 {
    if texture.is_null() {
        return 0;
    }
    (*texture).inner.height()
}

/// Get the texture format.
///
/// # Safety
/// The texture pointer must be valid.
#[no_mangle]
pub unsafe extern "C" fn goldy_texture_format(texture: *const GoldyTexture) -> GoldyTextureFormat {
    if texture.is_null() {
        return GoldyTextureFormat::Rgba8Unorm;
    }
    (*texture).inner.format().into()
}

