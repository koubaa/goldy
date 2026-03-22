// Goldy shader utility module (EXPERIMENTAL)
// Import with: import goldy_exp;
//
// ⚠️ This library's API is unstable and may change significantly
// as we learn what abstractions work best for shader development.
//
// == Cross-Platform Resource Binding ==
//
// Use access pattern functions for unified resource binding:
//
//   // Uniforms (ConstantBuffer)
//   float t = goldy_broadcast<TimeUniforms>(0).time;
//
//   // Storage buffers (RWStructuredBuffer)
//   RWStructuredBuffer<Particle> buf = goldy_scattered<Particle>(0);
//
// These work on SPIRV, DX12, and Metal (Argument Buffers Tier 2 required).
//
// Math/color also provide portable helpers shared with Ekrano: 2D `Transform`,
// `bbox_intersect`, gradient `extend_mode*`, `erf7`, `unpack4x8unorm` / `pack4x8unorm`, `premul_alpha`.
// `signed_atomic_min` / `signed_atomic_max` implement signed min/max on RWByteAddressBuffer (DXIL unsigned atomics workaround).

module goldy_exp;

__include "goldy_exp/algebra";
__include "goldy_exp/collectives";
__include "goldy_exp/math";
__include "goldy_exp/atomics";
__include "goldy_exp/color";
__include "goldy_exp/vertex";
__include "goldy_exp/types";
__include "goldy_exp/primitives";
__include "goldy_exp/descriptor_handle";
__include "goldy_exp/bindless_resources";
__include "goldy_exp/access";
