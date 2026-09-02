# Pipelines

Pipelines combine compiled shaders with fixed-function rendering state. Goldy provides `RenderPipeline` for graphics and `ComputePipeline` for compute workloads.

## Render Pipelines

A `RenderPipeline` pairs vertex and fragment shaders with a `RenderPipelineDesc` that configures vertex input, primitive assembly, depth testing, and the output format.

### Creating a Render Pipeline

```rust
use goldy::{
    RenderPipeline, RenderPipelineDesc, ShaderModule,
    Vertex2D, TextureFormat, PrimitiveTopology,
};

let vs = ShaderModule::from_slang(&device, include_str!("shaders/tri.vs.slang"))?;
let fs = ShaderModule::from_slang(&device, include_str!("shaders/tri.fs.slang"))?;

let pipeline = RenderPipeline::new(&device, &vs, &fs, &RenderPipelineDesc {
    vertex_layout: Vertex2D::layout(),
    topology: PrimitiveTopology::TriangleList,
    target_format: surface.format(),
    depth_stencil: None,
})?;
```

### RenderPipelineDesc

```rust
pub struct RenderPipelineDesc {
    pub vertex_layout: VertexBufferLayout,
    pub topology: PrimitiveTopology,
    pub target_format: TextureFormat,
    pub depth_stencil: Option<DepthStencilState>,
}
```

| Field | Purpose | Default |
|-------|---------|---------|
| `vertex_layout` | Describes vertex buffer stride and attributes | Empty (no vertex input) |
| `topology` | How vertices are assembled into primitives | `TriangleList` |
| `target_format` | Pixel format of the render target — **must match** `surface.format()` or the format passed to `RenderTarget::new()` | `Rgba8Unorm` |
| `depth_stencil` | Depth/stencil test configuration, or `None` to disable | `None` |

The default descriptor is valid for fullscreen passes that generate geometry from `SV_VertexID` and render to an `Rgba8Unorm` target without depth testing.

### Format Matching

The pipeline's `target_format` must match the render target it will draw into. Mismatched formats produce backend errors or undefined output.

```rust
let desc = RenderPipelineDesc {
    target_format: surface.format(),
    ..Default::default()
};
```

### Vertex Buffer Layouts

A `VertexBufferLayout` tells the pipeline how to interpret vertex buffer memory. For passes that do not use vertex buffers (fullscreen triangles, quad instancing), the default empty layout is correct.

For typed vertex input, use the `from_formats` builder or a built-in type's `layout()` method. See [Vertex Types and Layouts](vertices.md) for details.

```rust
let layout = VertexBufferLayout::from_formats::<MyVertex>(&[
    VertexFormat::Float32x3, // position
    VertexFormat::Float32x2, // uv
]);
```

### Primitive Topology

Controls how the vertex stream is assembled into geometric primitives:

```rust
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,   // default
    TriangleStrip,
}
```

```
PointList:     •  •  •  •
LineList:      •——•  •——•
LineStrip:     •——•——•——•
TriangleList:  △  △
TriangleStrip: △▽△▽
```

### Depth/Stencil State

Enable depth testing by setting `depth_stencil`. The surface or render target must have been created with a matching depth format.

```rust
use goldy::{DepthStencilState, DepthFormat, CompareFunction};

let pipeline = RenderPipeline::new(&device, &vs, &fs, &RenderPipelineDesc {
    vertex_layout: Vertex2D::layout(),
    target_format: surface.format(),
    topology: PrimitiveTopology::TriangleList,
    depth_stencil: Some(DepthStencilState {
        format: DepthFormat::Depth32Float,
        depth_write_enabled: true,
        depth_compare: CompareFunction::Less,
    }),
})?;
```

`DepthStencilState` fields:

| Field | Purpose | Default |
|-------|---------|---------|
| `format` | Depth texture format (`Depth16Unorm`, `Depth24Plus`, `Depth32Float`, etc.) | `Depth24Plus` |
| `depth_write_enabled` | Whether fragments write to the depth buffer | `true` |
| `depth_compare` | Comparison function — `Less`, `LessEqual`, `Greater`, `Always`, etc. | `Less` |

Available depth formats:

| Format | Bits | Stencil |
|--------|------|---------|
| `Depth16Unorm` | 16-bit | No |
| `Depth24Plus` | 24-bit (may use 32 internally) | No |
| `Depth24PlusStencil8` | 24-bit + 8-bit stencil | Yes |
| `Depth32Float` | 32-bit float | No |
| `Depth32FloatStencil8` | 32-bit float + 8-bit stencil | Yes |

For reverse-Z rendering, use `CompareFunction::Greater` and clear depth to `0.0`.

## Compute Pipelines

`ComputePipeline` wraps a single compute shader. See the [compute documentation](../compute/overview.md) for the full compute API.

```rust
use goldy::{ComputePipeline, ShaderModule};

let cs = ShaderModule::from_slang(&device, include_str!("shaders/sim.cs.slang"))?;
let pipeline = ComputePipeline::new(&device, &cs)?;
```

## Why Goldy Has Fewer Pipelines

Pipeline State Object (PSO) explosion is one of the biggest pain points in modern graphics. Engines routinely manage thousands of pipeline permutations and ship massive shader caches. Goldy eliminates most combinatorial dimensions:

| Dimension | Traditional Vulkan/DX12 | Goldy |
|-----------|------------------------|-------|
| Render pass compatibility | N render passes × M subpasses | Eliminated — dynamic rendering |
| Descriptor set layouts | Per-material layout permutations | One global bindless layout |
| Pipeline layouts | Per-material | One shared layout |
| Viewport / scissor | Baked into PSO | Dynamic state |
| Vertex format | Baked | Baked (unavoidable) |
| Target format | Baked | Baked (unavoidable) |

`RenderPipelineDesc` has exactly four fields. The permutation space is `vertex_layouts × topologies × target_formats × depth_configs` — deliberately small.

## Performance

Pipelines are expensive to create (shader compilation, PSO allocation) but cheap to bind during rendering. Create them once at startup and reuse across frames.

```rust
struct Renderer {
    scene_pipeline: RenderPipeline,
    ui_pipeline: RenderPipeline,
    wireframe_pipeline: RenderPipeline,
}

impl Renderer {
    fn new(device: &Device, surface: &Surface) -> Result<Self> {
        // Create all pipelines upfront
        Ok(Self {
            scene_pipeline: create_scene_pipeline(device, surface.format())?,
            ui_pipeline: create_ui_pipeline(device, surface.format())?,
            wireframe_pipeline: create_wireframe_pipeline(device, surface.format())?,
        })
    }
}
```

## Mesh pipelines

`MeshPipeline` replaces the vertex stage with a mesh shader (`[goldy_mesh]` / `mesh_main`) and a fragment shader. Amplification / task shaders are optional (`[goldy_amplification]` / `amp_main`) when `DeviceCapabilities::amplification_shaders` is set.

Create the pipeline only when `device.capabilities().mesh_shaders` is true. Record with `set_mesh_pipeline` and `dispatch_mesh` instead of `draw`. Vulkan and DX12 implement this; Metal is not wired yet.

```rust
let pipeline = MeshPipeline::new(&device, &MeshPipelineDesc {
    mesh: &shader,
    fragment: &shader,
    amplification: None,
    target_format: rt_format,
    depth_stencil: None,
})?;

let mut pass = scheme.render_pass("mesh", &rt, TargetLoad::Discard);
pass.set_mesh_pipeline(&pipeline);
pass.dispatch_mesh(1, 1, 1);
pass.finish();
```

