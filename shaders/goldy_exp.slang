// Goldy shader utility module (EXPERIMENTAL)
// Import with: import goldy_exp;
//
// ⚠️ This library's API is unstable and may change significantly
// as we learn what abstractions work best for shader development.
//
// For bindless support, shaders should explicitly import:
//   import goldy_exp.bindless_indices;  // For push/root constants indices
//   import goldy_exp.bindless;          // For ParameterBlock documentation

module goldy_exp;

__include "goldy_exp/math";
__include "goldy_exp/color";
__include "goldy_exp/vertex";

// Note: bindless and bindless_indices are NOT auto-included.
// Shaders should import them explicitly when needed in their
// platform-specific bindless code paths to avoid b0 register conflicts.

