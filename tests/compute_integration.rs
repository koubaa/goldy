//! Compute pipeline integration tests
//!
//! These tests verify compute pipeline functionality with actual GPU backends.

use goldy::{
    BindGroup, BindGroupLayout, BindGroupLayoutBinding, BindingType, Buffer, BufferBinding,
    BufferUsage, ComputeEncoder, ComputePipeline, ComputePipelineDesc, DeviceType, Instance,
    ShaderModule, ShaderStages,
};

/// Simple compute shader that doubles each value in a buffer.
const DOUBLE_SHADER: &str = r#"
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    data[id.x] = data[id.x] * 2;
}
"#;

/// Compute shader that reads from one buffer and writes to another.
const COPY_SHADER: &str = r#"
[[vk::binding(0, 0)]] StructuredBuffer<uint> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    output[id.x] = input[id.x];
}
"#;

#[test]
fn test_compute_pipeline_creation() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER)
        .expect("Failed to compile shader");

    let bind_layout = BindGroupLayout::new(
        &device,
        &[BindGroupLayoutBinding {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageBuffer { read_only: false },
        }],
    )
    .expect("Failed to create bind group layout");

    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[&bind_layout],
        },
    );

    assert!(pipeline.is_ok(), "Failed to create compute pipeline: {:?}", pipeline.err());
}

#[test]
fn test_compute_pipeline_no_bindings() {
    // A minimal compute shader with no bindings
    const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    // Do nothing - just test pipeline creation
}
"#;

    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, MINIMAL_SHADER)
        .expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc::default(),
    );

    assert!(pipeline.is_ok(), "Failed to create minimal compute pipeline: {:?}", pipeline.err());
}

#[test]
fn test_compute_dispatch_empty() {
    // Test dispatching a compute shader with no resources
    const MINIMAL_SHADER: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
}
"#;

    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, MINIMAL_SHADER)
        .expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc::default(),
    )
    .expect("Failed to create compute pipeline");

    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.dispatch(1, 1, 1);
    }

    let result = encoder.dispatch(&device);
    assert!(result.is_ok(), "Failed to dispatch: {:?}", result.err());
}

#[test]
fn test_compute_with_uav_buffer() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, DOUBLE_SHADER)
        .expect("Failed to compile shader");

    // Create buffer with initial data
    let initial_data: Vec<u32> = (0..64).collect();
    let buffer = Buffer::with_data(&device, &initial_data, BufferUsage::STORAGE)
        .expect("Failed to create buffer");

    // Create bind group
    let bind_layout = BindGroupLayout::new(
        &device,
        &[BindGroupLayoutBinding {
            binding: 0,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::StorageBuffer { read_only: false },
        }],
    )
    .expect("Failed to create bind group layout");

    let bind_group = BindGroup::new(&device, &bind_layout, &[BufferBinding::new(0, &buffer)])
        .expect("Failed to create bind group");

    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[&bind_layout],
        },
    )
    .expect("Failed to create compute pipeline");

    // Dispatch
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group);
        pass.dispatch(1, 1, 1); // 64 threads total
    }

    let result = encoder.dispatch(&device);
    assert!(result.is_ok(), "Failed to dispatch: {:?}", result.err());

    // Note: We can't easily read back the buffer without mapping support
    // This test just verifies the dispatch doesn't crash
}

#[test]
fn test_compute_with_srv_and_uav() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader = ShaderModule::from_slang(&device, COPY_SHADER)
        .expect("Failed to compile shader");

    // Create input buffer (read-only)
    let input_data: Vec<u32> = (0..64).collect();
    let input_buffer = Buffer::with_data(&device, &input_data, BufferUsage::STORAGE)
        .expect("Failed to create input buffer");

    // Create output buffer (read-write)
    let output_data: Vec<u32> = vec![0; 64];
    let output_buffer = Buffer::with_data(&device, &output_data, BufferUsage::STORAGE)
        .expect("Failed to create output buffer");

    // Create bind group with both SRV and UAV
    let bind_layout = BindGroupLayout::new(
        &device,
        &[
            BindGroupLayoutBinding {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageBuffer { read_only: true }, // SRV
            },
            BindGroupLayoutBinding {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::StorageBuffer { read_only: false }, // UAV
            },
        ],
    )
    .expect("Failed to create bind group layout");

    let bind_group = BindGroup::new(
        &device,
        &bind_layout,
        &[
            BufferBinding::new(0, &input_buffer),
            BufferBinding::new(1, &output_buffer),
        ],
    )
    .expect("Failed to create bind group");

    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[&bind_layout],
        },
    )
    .expect("Failed to create compute pipeline");

    // Dispatch
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group);
        pass.dispatch(1, 1, 1); // 64 threads
    }

    let result = encoder.dispatch(&device);
    assert!(result.is_ok(), "Failed to dispatch with SRV+UAV: {:?}", result.err());
}

