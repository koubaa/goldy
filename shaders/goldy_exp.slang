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
//   float t = goldy_broadcast<TimeUniforms>(0).time;     // Uniform data
//   Particle p = goldy_scattered_read<Particle>(0, idx); // Storage buffers
//   float4 c = goldy_interpolated<float4>(0).Sample(s, uv); // Textures
//
// These work on SPIRV, DX12, and Metal Tier 2. For Metal Tier 1, use
// traditional bindings with preprocessor fallback (see access.slang).
//
// == Alternative: DescriptorHandle<T> ==
//
// For direct handle-based access, use Slang's DescriptorHandle<T>.
// See plasma.slang for a complete example with Goldy's Vulkan binding override.
//
// == Legacy Resource Access ==
//
// For manual control with preprocessor directives, import:
//   import goldy_exp.buffer_indices;  // For push/root constants indices
//   import goldy_exp.bindless;        // For Metal ParameterBlock documentation

module goldy_exp;

__include "goldy_exp/math";
__include "goldy_exp/color";
__include "goldy_exp/vertex";
__include "goldy_exp/types";
__include "goldy_exp/primitives";
__include "goldy_exp/descriptor_handle";
__include "goldy_exp/bindless_resources";
__include "goldy_exp/access";

// Note: buffer_indices and bindless are NOT auto-included.
// Import them explicitly when needed.
