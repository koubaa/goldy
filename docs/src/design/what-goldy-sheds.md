# What Goldy Sheds

Goldy's bindless model and modern-hardware baseline make several traditional GPU programming concepts unnecessary. These aren't missing features — they're intentional design choices that keep the API small and the programming model coherent.

## No Descriptor Set Management

Traditional APIs require you to declare descriptor set layouts, allocate descriptor pools, write descriptor sets, and bind them before each draw or dispatch. A typical Vulkan pipeline touches three to four descriptor set objects before anything reaches the GPU.

Goldy replaces all of this with a flat bindless heap. Resources get a slot index when created, and shaders access them by that index. There are no layouts, no pools, no binding calls.

```hlsl
// Shader receives resources by index — no descriptor sets
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<Particle> particles, ThreadId id) {
    particles[id.x].position += particles[id.x].velocity;
}
```

This also eliminates **pipeline layouts as objects**. In Vulkan, each unique combination of descriptor set layouts produces a pipeline layout, which is baked into the pipeline at creation time. Goldy's single global bindless layout means one pipeline layout for all pipelines.

## No Manual Barrier Insertion

In Vulkan and DX12, you manually insert memory barriers and image layout transitions to tell the GPU when a resource changes from "written by compute" to "read by fragment" (or any other transition). Missing a barrier is a silent correctness bug; inserting too many is a performance bug.

Goldy's scheme (dependency graph) handles this automatically. You declare what each node reads and writes; Goldy derives the minimal set of barriers and transitions. This is both safer and typically more efficient than hand-placed barriers, because the scheme has a global view of the frame.

## No Shader Permutation Systems

Traditional engines maintain thousands of shader variants — combinations of feature flags, render pass compatibility, descriptor set layout versions, and pipeline state. Some ship dedicated cloud infrastructure just to compile and cache them all.

Goldy collapses most of the dimensions that drive permutation counts:

| Traditional dimension | Goldy equivalent |
|-----------------------|------------------|
| Render pass compatibility | Dynamic rendering — no render pass objects |
| Descriptor set layout | One global bindless layout |
| Pipeline layout | Implicit from the global layout |
| Viewport/scissor state | Dynamic state, not baked into PSO |

What remains — shader source × vertex format × target format × depth config — is a small, manageable space. Goldy addresses pipeline variety by having fewer pipelines, not by building infrastructure to manage many variants.

## Minimal Pipeline State Management

A Vulkan `VkGraphicsPipelineCreateInfo` touches blend state, depth/stencil state, rasterizer state, multisample state, input assembly, viewport/scissor, dynamic state flags, render pass, subpass, pipeline layout, and shader stages. Many of these are baked in at pipeline creation time, producing the combinatorial explosion that drives PSO caches.

Goldy uses dynamic rendering and dynamic state to move viewport, scissor, and render target configuration out of the pipeline object. The remaining pipeline state is intentionally minimal:

```rust
let pipeline = RenderPipeline::new(&device, &shader, &shader, &desc)?;
```

Blend mode, depth testing, and vertex format are still part of the pipeline — they represent genuine hardware configuration. But the many compatibility dimensions that traditional APIs bake in are gone.

## No Separate Compute API

OpenCL introduced compute to GPUs as an entirely separate API with its own device model, memory model, and dispatch semantics. Even "unified" APIs like Vulkan treat compute as a second-class citizen — compute pipelines and graphics pipelines share almost no code paths.

In Goldy, compute is a first-class citizen on the same footing as graphics. Compute shaders use the same bindless resource model, the same buffer types, and the same scheme. A compute dispatch that writes to a buffer and a draw call that reads from it are just nodes in the same dependency graph.

```rust
// Compute updates particles, render draws them — same scheme
scheme.node("update", &compute_pipeline)
    .with_parcel(&particle_buf, NodeAccess::ReadWrite)
    .dispatch(workgroups, 1, 1);
let mut pass = scheme.render_pass("draw", &scene_rt);
pass.with_parcel(&particle_buf, NodeAccess::Read);
// ...
```

## The Design Principle

Each of these omissions follows the same logic: if modern hardware doesn't need a concept for correctness or performance, Goldy doesn't expose it. The result is an API where the concepts that remain — buffers, textures, shaders, pipelines, scheme — each carry their weight.
