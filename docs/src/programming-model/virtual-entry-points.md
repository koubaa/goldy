# Virtual Entry Points

Goldy's **virtual entry points** let you write shader entry points with clean, typed parameters instead of raw `uniform uint` slots and `SV_*` semantics. You annotate your function with `[goldy_compute]`, `[goldy_vertex]`, or `[goldy_fragment]`, and a source-to-source transform generates the real Slang `[shader("...")]` entry point with all the bindless plumbing wired up.

## The Attributes

| Attribute | Stage | Generated Slang Attribute |
|---|---|---|
| `[goldy_compute]` | Compute | `[shader("compute")]` |
| `[goldy_vertex]` | Vertex | `[shader("vertex")]` |
| `[goldy_fragment]` | Fragment | `[shader("fragment")]` |

A minimal example:

```hlsl
import goldy_exp;

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
```

This is equivalent to manually writing a `[shader("compute")]` entry point with `uniform uint` push-constant parameters, descriptor heap lookups, and `SV_DispatchThreadID` — but without any of that boilerplate.

## What Virtual Entry Points Accept

### Resource Parameters

Each resource parameter occupies one **bindless slot** (a 16-bit index packed into push constants). The generated wrapper calls the corresponding `goldy_*` free function to resolve the slot to a live GPU handle.

| Parameter Type | Resolves Via | Description |
|---|---|---|
| `Scattered<T>` | `goldy_scattered<T>(slot)` | Read/write storage buffer |
| `BufRO<T>` | `goldy_buf_ro<T>(slot)` | Read-only storage buffer |
| `Interpolated<T>` | `goldy_interpolated<T>(slot)` | Sampled 2D texture |
| `DirectSpatial<T>` | `goldy_direct_spatial<T>(slot)` | Read/write 2D texture |
| `ByteAddress` | `goldy_byte_address(slot)` | Raw byte-address buffer |
| `Filter` | `goldy_filter(slot)` | Sampler state |
| `Accel` | `goldy_accel(slot)` | Top-level acceleration structure (`RayQuery`, when the device sets `GOLDY_RAY_QUERY`) |

### Broadcast Parameters

Any user-defined struct type that isn't a recognized resource or system-value type is treated as a **broadcast** (constant buffer). The generated code calls `goldy_broadcast<T>(slot)` to fetch the entire struct from a uniform buffer:

```hlsl
struct SimParams { float dt; uint count; };

[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(SimParams params, Scattered<Particle> data, ThreadId id) {
    // params is fetched from a constant buffer automatically
}
```

In vertex and fragment shaders, the **last** unrecognized struct is treated as the stage input (vertex attributes or fragment varyings) rather than a broadcast. All preceding unrecognized structs are broadcasts.

### System-Value Parameters

System-value wrapper types are mapped to `SV_*` semantics. The generated entry point declares the raw semantic parameter and constructs the wrapper:

| Wrapper Type | Maps To | Available Fields |
|---|---|---|
| `ThreadId` | `SV_DispatchThreadID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |
| `GroupThreadId` | `SV_GroupThreadID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |
| `GroupId` | `SV_GroupID` | `.x`, `.y`, `.z`, `.xy`, `.xyz` |
| `VertexId` | `SV_VertexID` | `.value` |
| `InstanceId` | `SV_InstanceID` | `.value` |
| `IsFrontFace` | `SV_IsFrontFace` | `.value` |

### Scalar Parameters

Plain scalar types (`uint`, `float`, `int`, `bool`, and vector variants) become **user parameters** — full-precision `u32` words in a separate region of the push constants. Bind them with `with_param` on the compute node builder (after `with_parcel` calls for resource params):

```rust
scheme
    .node("offset", &pipeline)
    .with_parcel(&data, NodeAccess::ReadWrite)
    .with_param(offset)
    .dispatch(1, 1, 1);
```

### Pass-Through Parameters

In vertex and fragment shaders, the last unrecognized struct parameter passes through as a stage input (vertex attributes or interpolated varyings). It appears directly in the generated entry point signature without bindless resolution:

```hlsl
[goldy_fragment]
float4 fs_main(MyUniforms cfg, FullscreenVarying input) : SV_Target {
    // cfg → broadcast (slot 0)
    // input → pass-through stage input (interpolated varyings)
    return float4(cfg.time, 0, 0, 1);
}
```

## The Source-to-Source Transform

The transform (implemented in `slang/virtual_main.rs`) runs before Slang compilation and performs three operations:

1. **Generates a wrapper function** with the real `[shader("...")]` attribute and a fixed 16-word push-constant signature.
2. **Renames the user function** from `cs_main` to `_goldy_user_cs_main` so both can coexist.
3. **Removes the `[goldy_*]` attribute** and `[numthreads]` from the renamed user function (they live on the generated wrapper).

### Push Constant Layout

The generated entry point always declares a fixed signature regardless of how many parameters the user function has:

```
Words  0–7:  _bw0.._bw7   — 16 × u16 bindless indices packed 2 per word
Words  8–15: _uw0.._uw7   — 8 × u32 user scalar parameters
```

Bindless indices are packed as pairs into 32-bit words: the low 16 bits of `_bw0` hold slot 0, the high 16 bits hold slot 1, and so on. This fits up to 16 resource/broadcast parameters and 8 scalar parameters in 64 bytes of push constants.

### Before and After

**What you write:**

```hlsl
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(TimeUniforms cfg, Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + cfg.base;
}
```

**What gets compiled** (generated wrapper prepended, user function renamed):

```hlsl
[shader("compute")]
[numthreads(64, 1, 1)]
void cs_main(uniform uint _bw0, ..., uniform uint _bw7,
             uniform uint _uw0, ..., uniform uint _uw7,
             uint3 _sv0 : SV_DispatchThreadID) {
    TimeUniforms cfg = goldy_broadcast<TimeUniforms>(_bw0 & 0xFFFFu);
    Scattered<uint> data = goldy_scattered<uint>((_bw0 >> 16u) & 0xFFFFu);
    ThreadId id = ThreadId(_sv0);
    _goldy_user_cs_main(cfg, data, id);
}

// Original function, renamed:
void _goldy_user_cs_main(TimeUniforms cfg, Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + cfg.base;
}
```

The `#line 1` directive is inserted between the generated wrapper and the user source so that compiler diagnostics report correct line numbers.

### Vertex/Fragment Example

```hlsl
[goldy_vertex]
VSOutput vs_main(SceneUniforms scene, Scattered<Instance> instances, VertexId vid, InstanceId iid) {
    // scene → broadcast (slot 0)
    // instances → scattered (slot 1)
    // vid → SV_VertexID
    // iid → SV_InstanceID
    Instance inst = instances[iid.value];
    VSOutput out;
    // ... transform vertex ...
    return out;
}

[goldy_fragment]
float4 fs_main(SceneUniforms scene, Interpolated<float4> albedo, Filter samp,
               VSOutput input) : SV_Target {
    // scene → broadcast (slot 0)
    // albedo → texture (slot 1)
    // samp → sampler (slot 2)
    // input → pass-through stage varying
    return albedo.Sample(samp, input.uv) * scene.tint;
}
```

Both entry points share the same push-constant layout. Fragment shader slot expectations take precedence when Goldy extracts category metadata (since resource binding typically lives there in a vertex+fragment pair).

## Preprocessor Conditionals

Virtual entry points support `#ifdef`/`#else`/`#endif` blocks directly inside the parameter list. This is useful for shader variants like MSAA:

```hlsl
[goldy_compute]
[numthreads(4, 16, 1)]
void cs_main(BufRO<uint> config,
#ifdef msaa
             BufRO<uint> mask_lut, DirectSpatial<float4> out_tex,
#else
             DirectSpatial<float4> out_tex,
#endif
             ThreadId tid) {
    // ...
}
```

The transform generates conditional blocks in the wrapper's signature, body, and call arguments so that the correct branch is selected at compile time based on preprocessor defines.
