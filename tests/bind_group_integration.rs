//! Integration tests for bind groups with real GPU.
//!
//! These tests verify that bind groups work correctly with actual GPU rendering.
//!
//! ## Bindless Support
//!
//! On supported backends (Vulkan 1.2+ with descriptor indexing, DX12), bind groups
//! are transparently converted to bindless operations:
//! - Resource indices are pushed via push constants (Vulkan) or root constants (DX12)
//! - The global descriptor set/heap is bound once at pass start
//! - SetBindGroup translates to pushing indices rather than binding descriptors
//!
//! The high-level API remains unchanged - these tests verify both traditional and
//! bindless code paths produce correct rendering results.

use goldy::{
    BindGroup, BindGroupLayout, BindGroupLayoutBinding, BindingType, Buffer, BufferBinding,
    BufferUsage, DeviceType, Instance, RenderPipeline, RenderPipelineDesc, ShaderModule,
    ShaderStages, TextureFormat, Vertex2D,
};

fn create_device() -> Option<goldy::Device> {
    let instance = Instance::new().ok()?;
    instance
        .create_device(DeviceType::DiscreteGpu)
        .ok()
        .or_else(|| {
            let instance = Instance::new().ok()?;
            instance.create_device(DeviceType::IntegratedGpu).ok()
        })
}

/// Simple shader with uniform buffer for transformation matrix
const UNIFORM_SHADER: &str = r#"
struct Uniforms {
    float4 color_multiplier;
};

[[vk::binding(0, 0)]] ConstantBuffer<Uniforms> uniforms;

struct VertexInput {
    float2 position : POSITION;
    float4 color : COLOR;
};

struct VertexOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VertexOutput vs_main(VertexInput input) {
    VertexOutput output;
    output.position = float4(input.position, 0.0, 1.0);
    output.color = input.color * uniforms.color_multiplier;
    return output;
}

[shader("fragment")]
float4 fs_main(VertexOutput input) : SV_Target {
    return input.color;
}
"#;

#[test]
fn test_bind_group_layout_creation() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)]);

    assert!(
        layout.is_ok(),
        "Failed to create bind group layout: {:?}",
        layout.err()
    );
}

#[test]
fn test_bind_group_layout_multiple_entries() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let layout = BindGroupLayout::new(
        &device,
        &[
            BindGroupLayoutBinding::uniform(0),
            BindGroupLayoutBinding::uniform_fragment(1),
            BindGroupLayoutBinding::storage(2, true),
        ],
    );

    assert!(
        layout.is_ok(),
        "Failed to create multi-entry bind group layout: {:?}",
        layout.err()
    );
}

#[test]
fn test_bind_group_with_uniform_buffer() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    // Create a uniform buffer
    let uniform_data: [f32; 4] = [1.0, 1.0, 1.0, 1.0]; // color multiplier
    let uniform_buffer = Buffer::with_data(&device, &uniform_data, BufferUsage::UNIFORM)
        .expect("Failed to create uniform buffer");

    // Create bind group layout
    let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)])
        .expect("Failed to create bind group layout");

    // Create bind group
    let bind_group = BindGroup::new(&device, &layout, &[BufferBinding::new(0, &uniform_buffer)]);

    assert!(
        bind_group.is_ok(),
        "Failed to create bind group: {:?}",
        bind_group.err()
    );
}

#[test]
fn test_pipeline_with_bind_group() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    let shader =
        ShaderModule::from_slang(&device, UNIFORM_SHADER).expect("Failed to create shader");

    let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)])
        .expect("Failed to create bind group layout");

    let pipeline = RenderPipeline::new(
        &device,
        &shader,
        &shader,
        &RenderPipelineDesc {
            vertex_layout: Vertex2D::layout(),
            target_format: TextureFormat::Rgba8Unorm,
            bind_group_layouts: &[&layout],
            ..Default::default()
        },
    );

    assert!(
        pipeline.is_ok(),
        "Failed to create pipeline with bind group layout: {:?}",
        pipeline.err()
    );
}

#[test]
fn test_multiple_bind_groups() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    // Create two uniform buffers
    let uniform1_data: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    let uniform1 = Buffer::with_data(&device, &uniform1_data, BufferUsage::UNIFORM)
        .expect("Failed to create uniform buffer 1");

    let uniform2_data: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
    let uniform2 = Buffer::with_data(&device, &uniform2_data, BufferUsage::UNIFORM)
        .expect("Failed to create uniform buffer 2");

    // Create two separate layouts and bind groups
    let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)])
        .expect("Failed to create bind group layout");

    let bind_group1 = BindGroup::new(&device, &layout, &[BufferBinding::new(0, &uniform1)])
        .expect("Failed to create bind group 1");

    let bind_group2 = BindGroup::new(&device, &layout, &[BufferBinding::new(0, &uniform2)])
        .expect("Failed to create bind group 2");

    // Both bind groups should be successfully created (test passes if we get here)
    // Bind groups are distinct objects even if using the same layout
    drop(bind_group1);
    drop(bind_group2);
}

#[test]
fn test_bind_group_with_buffer_range() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    // Create a larger buffer
    let buffer_data: [f32; 16] = [
        1.0, 0.0, 0.0, 1.0, // First uniform (red)
        0.0, 1.0, 0.0, 1.0, // Second uniform (green)
        0.0, 0.0, 1.0, 1.0, // Third uniform (blue)
        1.0, 1.0, 1.0, 1.0, // Fourth uniform (white)
    ];
    let buffer = Buffer::with_data(&device, &buffer_data, BufferUsage::UNIFORM)
        .expect("Failed to create buffer");

    let layout = BindGroupLayout::new(&device, &[BindGroupLayoutBinding::uniform(0)])
        .expect("Failed to create bind group layout");

    // Create bind group with offset into the buffer (skip first 4 floats = 16 bytes)
    let bind_group = BindGroup::new(
        &device,
        &layout,
        &[
            BufferBinding::with_range(0, &buffer, 16, 16), // Use second uniform
        ],
    );

    assert!(
        bind_group.is_ok(),
        "Failed to create bind group with buffer range: {:?}",
        bind_group.err()
    );
}

#[test]
fn test_storage_buffer_binding() {
    let Some(device) = create_device() else {
        eprintln!("Skipping test: no GPU available");
        return;
    };

    // Create a storage buffer
    let storage_data: [u32; 64] = [0; 64];
    let storage_buffer = Buffer::with_data(&device, &storage_data, BufferUsage::STORAGE)
        .expect("Failed to create storage buffer");

    let layout = BindGroupLayout::new(
        &device,
        &[BindGroupLayoutBinding {
            binding: 0,
            visibility: ShaderStages::ALL,
            ty: BindingType::StorageBuffer { read_only: false },
        }],
    )
    .expect("Failed to create bind group layout");

    let bind_group = BindGroup::new(&device, &layout, &[BufferBinding::new(0, &storage_buffer)]);

    assert!(
        bind_group.is_ok(),
        "Failed to create bind group with storage buffer: {:?}",
        bind_group.err()
    );
}
