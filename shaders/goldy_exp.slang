// Goldy shader utility module (EXPERIMENTAL)
// Import with: import goldy_exp;
//
// ⚠️ This library's API is unstable and may change significantly
// as we learn what abstractions work best for shader development.
//
// == Cross-Platform Resource Binding ==
//
// Tag your entry point with [goldy_compute], [goldy_vertex], or
// [goldy_fragment] and declare resources as typed parameters:
//
//   [goldy_compute]
//   [numthreads(64, 1, 1)]
//   void cs_main(TimeUniforms cfg, Scattered<uint> data, ThreadId id) {
//       data[id.x] = data[id.x] + cfg.base;
//   }
//
// Works on SPIRV, DX12, and Metal (Argument Buffers Tier 2 required).
//

module goldy_exp;

__include "goldy_exp/algebra";
__include "goldy_exp/collectives";
__include "goldy_exp/math";
__include "goldy_exp/color";
__include "goldy_exp/vertex";
__include "goldy_exp/types";
__include "goldy_exp/primitives";
__include "goldy_exp/descriptor_handle";
__include "goldy_exp/bindless_resources";
__include "goldy_exp/access";
