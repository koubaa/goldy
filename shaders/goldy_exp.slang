// Goldy shader utility module (EXPERIMENTAL)
// Import with: import goldy_exp;
//
// ⚠️ This library's API is unstable and may change significantly
// as we learn what abstractions work best for shader development.
//
// == Cross-Platform Resource Binding ==
//
// For preprocessor-free resource binding, use Slang's DescriptorHandle<T>.
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

// Note: buffer_indices and bindless are NOT auto-included.
// Import them explicitly when needed.
