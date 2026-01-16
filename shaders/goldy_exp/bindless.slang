// Bindless resource access for Metal (EXPERIMENTAL)
// Part of the goldy_exp module
//
// This module documents the ParameterBlock pattern for bindless resource access
// using Metal's Argument Buffers Tier 2.
//
// == How It Works ==
//
// Slang's ParameterBlock<T> type generates Metal code that:
// 1. Creates a pointer to the struct type: `T constant*`
// 2. For buffers inside T: generates `ElementType device*` pointers
// 3. For textures inside T: generates `MTLResourceID` values
//
// The host (Rust) must:
// 1. Allocate an argument buffer
// 2. Write GPU addresses (buffer.gpuAddress) for buffers
// 3. Write resource IDs (texture.gpuResourceID) for textures
// 4. Bind the argument buffer at the slot Slang expects
//
// == Usage Pattern ==
//
// In your shader:
//
//   #ifdef __BINDLESS__
//   
//   // Define your resource struct
//   struct MyResources {
//       StructuredBuffer<MyData> dataBuffer;
//       RWStructuredBuffer<MyOutput> outputBuffer;
//       Texture2D<float4> myTexture;
//       SamplerState mySampler;
//   };
//   
//   // Declare a ParameterBlock with your resources
//   ParameterBlock<MyResources> gResources;
//   
//   // Access resources naturally:
//   MyData d = gResources.dataBuffer[idx];
//   gResources.outputBuffer[idx] = result;
//   float4 color = gResources.myTexture.Sample(gResources.mySampler, uv);
//   
//   #else
//   // Traditional binding fallback
//   [[vk::binding(0, 0)]] StructuredBuffer<MyData> dataBuffer;
//   // ...
//   #endif
//
// == Metal Code Generation ==
//
// For the above example, Slang generates Metal code like:
//
//   struct MyResources {
//       MyData device* dataBuffer;      // 64-bit GPU address
//       MyOutput device* outputBuffer;  // 64-bit GPU address  
//       texture2d<float> myTexture;     // In argument buffer: 64-bit MTLResourceID
//       sampler mySampler;              // In argument buffer: 64-bit MTLResourceID
//   };
//   
//   [[kernel]] void cs_main(MyResources constant* gResources [[buffer(N)]]) {
//       // Access gResources->dataBuffer[i] dereferences the pointer
//   }
//
// The pointer dereference happens automatically in Metal when you access
// gResources->dataBuffer[i].

implementing goldy_exp;

// Re-export common types for convenience
// (Shaders can also use these directly from Slang's stdlib)

// Note: ParameterBlock is a built-in Slang type, no need to define it here.
// Just use: ParameterBlock<YourResourceStruct> gResources;

