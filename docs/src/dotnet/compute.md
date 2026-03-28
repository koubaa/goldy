# .NET Compute Shaders

Goldy's `ComputePipeline` and `ComputeEncoder` expose GPU compute to C# with the same bindless resource model as the Rust API.

## Basic Compute Dispatch

```csharp
using Goldy;

using var instance = new Instance();
using var device = instance.CreateDevice(DeviceType.DiscreteGpu);

// Shader doubles every float in a buffer
var source = """
    #include "goldy_exp.slang"

    struct PushConstants { uint buffer_idx; };
    [[vk::push_constant]] PushConstants pc;

    [shader("compute")]
    [numthreads(64, 1, 1)]
    void cs_main(uint3 id : SV_DispatchThreadID) {
        float val = asfloat(g_StorageBuffers[pc.buffer_idx].Load(id.x * 4));
        g_StorageBuffers[pc.buffer_idx].Store(id.x * 4, asuint(val * 2.0));
    }
    """;

using var shader = new ShaderModule(device, source);
using var pipeline = new ComputePipeline(device, shader);

// Upload data
float[] data = Enumerable.Range(0, 1024).Select(i => (float)i).ToArray();
using var buffer = Buffer.WithData(device, data, DataAccess.Scattered);

// Record and dispatch
var encoder = new ComputeEncoder();
encoder.SetPipeline(pipeline);
encoder.SetPushConstants(buffer);
encoder.Dispatch(16, 1, 1); // 16 * 64 = 1024 threads

encoder.Dispatch(device); // blocking
```

## Non-blocking Dispatch with GpuFuture

```csharp
// Submit without blocking
var future = encoder.Submit(device);

// Do CPU work while GPU runs
DoSomeCpuWork();

// Wait for completion
future.Wait();

// Or poll non-blocking
while (!future.IsComplete)
{
    Thread.Sleep(1);
}
```

## Ping-Pong Buffers

A common pattern for iterative GPU algorithms (e.g. Game of Life, fluid simulation):

```csharp
using var bufferA = Buffer.WithData(device, initialData, DataAccess.Scattered);
using var bufferB = Buffer.New(device, byteSize, DataAccess.Scattered);

for (int step = 0; step < iterations; step++)
{
    bool even = (step % 2) == 0;
    var src = even ? bufferA : bufferB;
    var dst = even ? bufferB : bufferA;

    var encoder = new ComputeEncoder();
    encoder.SetPipeline(pipeline);
    encoder.SetPushConstants(src, dst);
    encoder.Dispatch(workgroups, 1, 1);
    encoder.Dispatch(device);
}
```
