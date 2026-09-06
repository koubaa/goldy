# Summary

[Introduction](./introduction.md)

# Tutorial

- [Installation](./tutorial/installation.md)
- [Your First Triangle](./tutorial/first-triangle.md)
- [Your First Compute Shader](./tutorial/first-compute.md)

# Programming Model

- [Parcels](./programming-model/parcels.md)
- [Virtual Entry Points](./programming-model/virtual-entry-points.md)
- [Rust Compute Kernels](./programming-model/rust-kernels.md)
- [CPU Dispatches](./programming-model/cpu-dispatch.md)
- [Yielding Scripts](./programming-model/yielding-scripts.md)
- [Slang in One Source](./programming-model/slang.md)

# Compute Workflows

- [Settlement](./compute/settlement.md)
- [Pipelined Frames](./compute/pipelined-frames.md)
- [Compute to Surface](./compute/compute-to-surface.md)

# Graphics Workflows

- [Pipelines](./graphics/pipelines.md)
- [Render Pass Nodes](./graphics/commands.md)
- [Vertex Types and Layouts](./graphics/vertices.md)

# Surfaces and Render Targets

- [Rendering Outputs](./surfaces/overview.md)

# Resources at Scale

- [Buffers](./resources/buffers.md)
- [RetainedPool and Parcel](./resources/retained-pool.md)
- [Textures and Samplers](./resources/textures.md)
- [Pooling and Sub-Allocation](./resources/pooling.md)
- [Transient Allocation](./resources/transient-allocation.md)
- [VRAM Allocator](./resources/vram-allocator.md)

# Backends

- [Backend Architecture](./backends/overview.md)
- [Conditional Compilation](./backends/conditional-compilation.md)

# Debugging and Observability

- [Debugging and Observability](./debugging/overview.md)
- [CPU host-callable shaders](./debugging/cpu-host-callable.md)

# Bindings

- [Python](./bindings/python.md)
- [.NET](./bindings/dotnet.md)
- [C++](./bindings/cpp.md)
- [Rust FFI Client](./bindings/rust-ffi-client.md)

# Examples

- [Examples Gallery](./examples/gallery.md)
  - [triangle](./examples/triangle.md)
  - [mesh_triangle](./examples/mesh_triangle.md)
  - [gradient](./examples/gradient.md)
  - [checkerboard](./examples/checkerboard.md)
  - [compute_particles](./examples/compute_particles.md)
  - [game_of_life](./examples/game_of_life.md)
  - [compute_to_surface](./examples/compute_to_surface.md)
  - [ray_query](./examples/ray_query.md)
  - [solid_cube](./examples/solid_cube.md)
  - [spinning_cube](./examples/spinning_cube.md)
  - [depth_quads](./examples/depth_quads.md)
  - [textured_quad](./examples/textured_quad.md)
  - [instancing](./examples/instancing.md)
  - [bouncing_lines](./examples/bouncing_lines.md)
  - [waveform](./examples/waveform.md)
  - [plasma](./examples/plasma.md)
  - [tunnel](./examples/tunnel.md)
  - [metaballs](./examples/metaballs.md)
  - [mandelbrot](./examples/mandelbrot.md)
  - [digital_clock](./examples/digital_clock.md)
  - [starfield](./examples/starfield.md)
  - [particles](./examples/particles.md)
  - [multi_window](./examples/multi_window.md)
  - [Shared Helpers](./examples/shared-helpers.md)

# Design & Philosophy

- [Motivation](./design/motivation.md)
- [What Goldy Sheds](./design/what-goldy-sheds.md)
- [Shader Specialization Prediction](./design/shader-specialization.md)
- [Static Shader Bounds Analysis](./design/shader-bounds-analysis.md)
- [Goldy vs wgpu](./design/comparison.md)
- [Target Hardware](./design/hardware.md)

# Fondaco Machine (research)

- [Overview](./design/fondaco.md)
- [Terminology](./fondaco/terminology.md)
- [Machine Specification](./fondaco/specification.md)
- [Goldy Runtime Mapping](./fondaco/goldy-runtime.md)
- [Design Thesis](./fondaco/design-thesis.md)

# Appendix

- [Slang Quick Reference](./appendix/slang-reference.md)
- [Environment Variables](./appendix/environment-variables.md)

---

[License](./license.md)
