// Goldy shader utility module (EXPERIMENTAL)
// Import with: import goldy_exp;
//
// ⚠️ This library's API is unstable and may change significantly
// as we learn what abstractions work best for shader development.
//
// For buffer index access, shaders should explicitly import:
//   import goldy_exp.buffer_indices;  // For push/root constants indices
//   import goldy_exp.bindless;        // For Metal ParameterBlock documentation

module goldy_exp;

__include "goldy_exp/math";
__include "goldy_exp/color";
__include "goldy_exp/vertex";

// Note: bindless and buffer_indices are NOT auto-included.
// Shaders should import them explicitly when needed to avoid b0 register conflicts.

