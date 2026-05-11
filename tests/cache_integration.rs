//! Pipeline and shader bytecode cache integration checks.
#![cfg(any(feature = "vulkan", feature = "dx12"))]

use goldy::shader_cache::{ShaderBytecodeDiskCache, GOLDY_SHADER_CACHE_MAGIC};
use goldy::{types::BackendType, ComputePipeline, DeviceType, Instance, ShaderModule};
/// Simple compute shader (same intent as [`compute_integration::DOUBLE_SHADER`]).
const CACHE_TEST_COMPUTE: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

#[cfg(feature = "vulkan")]
fn try_vulkan_gpu() -> Option<(Instance, goldy::Device)> {
    let instance = Instance::new().ok()?;
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .ok()?;
    (device.backend_type() == BackendType::Vulkan).then_some((instance, device))
}

/// [`VkPipelineCache`] is serialized on Vulkan device teardown.
#[cfg(feature = "vulkan")]
#[test]
fn vk_pipeline_cache_file_written_on_drop() {
    let Some(cache_root) = dirs::cache_dir() else {
        return;
    };
    let Some((instance, device)) = try_vulkan_gpu() else {
        return;
    };
    let adapter_id = device.adapter_id();
    let expected = cache_root
        .join("goldy")
        .join(format!("pipeline_cache_{adapter_id}.bin"));

    let shader =
        ShaderModule::from_slang(&device, CACHE_TEST_COMPUTE).expect("compile CACHE_TEST_COMPUTE");
    let _pipeline = ComputePipeline::new(&device, &shader).expect("compute pipeline");

    // Drop pipeline and shader first so their Arc<DeviceInner> refs are released,
    // allowing `drop(device)` to actually trigger DeviceInner::drop → backend destroy.
    drop(_pipeline);
    drop(shader);
    drop(device);

    assert!(
        expected.exists(),
        "expected Vulkan pipeline cache at {:?}",
        expected
    );
    drop(instance);
}

/// Loading a serialized [`VkPipelineCache`] on a subsequent device must not fail.
#[cfg(feature = "vulkan")]
#[test]
fn vk_pipeline_cache_survives_reload() {
    let Some((instance, device)) = try_vulkan_gpu() else {
        return;
    };
    let shader =
        ShaderModule::from_slang(&device, CACHE_TEST_COMPUTE).expect("compile CACHE_TEST_COMPUTE");
    let _pipeline = ComputePipeline::new(&device, &shader).expect("compute pipeline");
    drop(_pipeline);
    drop(shader);
    drop(device);

    let device2 = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("second device create");
    assert_eq!(
        device2.backend_type(),
        BackendType::Vulkan,
        "second device backend mismatch"
    );
    let shader2 = ShaderModule::from_slang(&device2, CACHE_TEST_COMPUTE).expect("compile phase 2");
    let _pipeline2 =
        ComputePipeline::new(&device2, &shader2).expect("compute pipeline after reload");
}

/// Compiled Slang shaders flush `shader_cache.bin.zst` on [`Device`] / compiler teardown.
#[test]
fn shader_cache_file_written_after_compile() {
    let Some(cache_root) = dirs::cache_dir() else {
        return;
    };
    let path = cache_root.join("goldy").join("shader_cache.bin.zst");

    let instance = match Instance::new() {
        Ok(i) => i,
        Err(_) => return,
    };
    let device = match instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
    {
        Ok(d) => d,
        Err(_) => return,
    };

    let shader =
        ShaderModule::from_slang(&device, CACHE_TEST_COMPUTE).expect("compile CACHE_TEST_COMPUTE");
    let _pipeline = ComputePipeline::new(&device, &shader).expect("compute pipeline");

    drop(_pipeline);
    drop(shader);
    drop(device);
    drop(instance);

    assert!(
        path.exists(),
        "expected Slang bytecode cache file at {:?}",
        path
    );
}

/// Wrong version line in envelope → cold [`ShaderBytecodeDiskCache`] (does not corrupt process cache).
#[test]
fn shader_cache_stale_version_is_ignored() {
    let td = tempfile::TempDir::new().expect("tempdir");
    let path = td.path().join("shader_cache.bin.zst");
    let mut uncompressed = Vec::new();
    uncompressed.extend_from_slice(GOLDY_SHADER_CACHE_MAGIC.as_slice());
    uncompressed.extend_from_slice(b"fictional-stale-goldy-version\n");
    uncompressed.extend_from_slice(&0u64.to_le_bytes());
    let compressed = zstd::encode_all(uncompressed.as_slice(), 10).expect("zstd encode");
    std::fs::write(&path, compressed).expect("write test cache blob");

    let cache = ShaderBytecodeDiskCache::new_at_path(path);
    assert!(
        !cache.version_ok_on_disk(),
        "stale envelope must not flag version_ok"
    );
    assert!(cache.is_empty(), "cold map when version mismatches build");
}
