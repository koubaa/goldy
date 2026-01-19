//! Compute pipeline integration tests (Fully Bindless)
//!
//! These tests verify compute pipeline functionality with actual GPU backends
//! using the fully bindless resource binding model.

use goldy::{
    Buffer, BufferUsage, ComputeEncoder, ComputePipeline, ComputePipelineDesc, DeviceType,
    Instance, ShaderModule,
};

/// Simple compute shader that doubles each value in a buffer (fully bindless).
const DOUBLE_SHADER: &str = r#"
#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct ComputeResources {
    RWStructuredBuffer<uint> data;
};
ParameterBlock<ComputeResources> gResources;
#define DATA gResources.data

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.bindless_indices;
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> g_StorageBuffers[];
#define DATA g_StorageBuffers[getBindlessIndex(0)]

#elif defined(__HLSL__) || defined(__DX12__)
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0, space0) {
    uint dataBufferIndex;
};
#define DATA (*DescriptorHandle<RWStructuredBuffer<uint>>(uint2(dataBufferIndex, 0)))

#endif

#else
// Traditional binding mode (fallback)
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> data;
#define DATA data
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    DATA[id.x] = DATA[id.x] * 2;
}
"#;

/// Compute shader that reads from one buffer and writes to another (fully bindless).
const COPY_SHADER: &str = r#"
#ifdef __BINDLESS__

#if defined(__METAL__)
// Metal: Use ParameterBlock for argument buffer
struct ComputeResources {
    StructuredBuffer<uint> input;
    RWStructuredBuffer<uint> output;
};
ParameterBlock<ComputeResources> gResources;
#define INPUT gResources.input
#define OUTPUT gResources.output

#elif defined(__SPIRV__)
// Vulkan: Push constants for indices + global descriptor arrays
import goldy_exp.bindless_indices;
[[vk::binding(0, 0)]] StructuredBuffer<uint> g_ReadBuffers[];
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> g_StorageBuffers[];
#define INPUT g_ReadBuffers[getBindlessIndex(0)]
#define OUTPUT g_StorageBuffers[getBindlessIndex(1)]

#elif defined(__HLSL__) || defined(__DX12__)
// DX12: Bindless via root constants + ResourceDescriptorHeap
cbuffer BindlessIndices : register(b0, space0) {
    uint inputBufferIndex;
    uint outputBufferIndex;
};
#define INPUT (*DescriptorHandle<StructuredBuffer<uint>>(uint2(inputBufferIndex, 0)))
#define OUTPUT (*DescriptorHandle<RWStructuredBuffer<uint>>(uint2(outputBufferIndex, 0)))

#endif

#else
// Traditional binding mode (fallback)
[[vk::binding(0, 0)]] StructuredBuffer<uint> input;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> output;
#define INPUT input
#define OUTPUT output
#endif

[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uint3 id : SV_DispatchThreadID) {
    OUTPUT[id.x] = INPUT[id.x];
}
"#;

#[test]
fn test_compute_pipeline_creation() {
    let instance = Instance::new().expect("Failed to create instance");
    let device = instance
        .create_device(DeviceType::DiscreteGpu)
        .or_else(|_| instance.create_device(DeviceType::IntegratedGpu))
        .expect("Failed to create device");

    let shader =
        ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("Failed to compile shader");

    // Fully bindless: no bind group layouts needed
    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[],
        },
    );

    assert!(
        pipeline.is_ok(),
        "Failed to create compute pipeline: {:?}",
        pipeline.err()
    );
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

    let shader =
        ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(&device, &shader, &ComputePipelineDesc::default());

    assert!(
        pipeline.is_ok(),
        "Failed to create minimal compute pipeline: {:?}",
        pipeline.err()
    );
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

    let shader =
        ShaderModule::from_slang(&device, MINIMAL_SHADER).expect("Failed to compile shader");

    let pipeline = ComputePipeline::new(&device, &shader, &ComputePipelineDesc::default())
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

    let shader =
        ShaderModule::from_slang(&device, DOUBLE_SHADER).expect("Failed to compile shader");

    // Create buffer with initial data
    let initial_data: Vec<u32> = (0..64).collect();
    let buffer = Buffer::with_data(&device, &initial_data, BufferUsage::STORAGE)
        .expect("Failed to create buffer");

    // Fully bindless: no bind group layouts needed
    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[],
        },
    )
    .expect("Failed to create compute pipeline");

    // Dispatch using fully bindless push constants
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        // Fully bindless: pass buffer indices directly via push constants
        pass.set_push_constants(&[&buffer]);
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

    let shader = ShaderModule::from_slang(&device, COPY_SHADER).expect("Failed to compile shader");

    // Create input buffer (read-only)
    let input_data: Vec<u32> = (0..64).collect();
    let input_buffer = Buffer::with_data(&device, &input_data, BufferUsage::STORAGE)
        .expect("Failed to create input buffer");

    // Create output buffer (read-write)
    let output_data: Vec<u32> = vec![0; 64];
    let output_buffer = Buffer::with_data(&device, &output_data, BufferUsage::STORAGE)
        .expect("Failed to create output buffer");

    // Fully bindless: no bind group layouts needed
    let pipeline = ComputePipeline::new(
        &device,
        &shader,
        &ComputePipelineDesc {
            bind_group_layouts: &[],
        },
    )
    .expect("Failed to create compute pipeline");

    // Dispatch using fully bindless push constants
    let mut encoder = ComputeEncoder::new();
    {
        let mut pass = encoder.begin_compute_pass();
        pass.set_pipeline(&pipeline);
        // Fully bindless: pass buffer indices directly via push constants
        // Order matches shader slots: [input (slot 0), output (slot 1)]
        pass.set_push_constants(&[&input_buffer, &output_buffer]);
        pass.dispatch(1, 1, 1); // 64 threads
    }

    let result = encoder.dispatch(&device);
    assert!(
        result.is_ok(),
        "Failed to dispatch with SRV+UAV: {:?}",
        result.err()
    );
}
