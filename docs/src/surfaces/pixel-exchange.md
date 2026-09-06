# Pixel Exchange

The CPU compute backend (`GOLDY_BACKEND=cpu`) has no textures, samplers, or
surfaces. A pixmap is a **buffer parcel**. [`PixelExchange`](https://docs.rs/goldy/latest/goldy/struct.PixelExchange.html)
withdraws that parcel and, on claim consume, copies the bytes into a
[`PixelSink`](https://docs.rs/goldy/latest/goldy/trait.PixelSink.html).

The sink is a **foreign subsystem** in the Fondaco sense: not a Goldy
`Device`, not a second `GOLDY_BACKEND`. Scheme submissions stay on the compute
device. Graphics (when present) is reached only through the consume verb.

This is the shape of Vello's unmerged CPU fine writeback (`CpuTexture` + a blit
into a GPU view), minus a Goldy graphics backend.

## PixelSink

```rust
pub trait PixelSink: Send + Sync {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError>;
    fn generation(&self) -> u64;
    fn size(&self) -> (u32, u32);
}
```

Implementations serialise internally. Resize is a verb that bumps
`generation`; in-flight [`PixelTransaction`](https://docs.rs/goldy/latest/goldy/struct.PixelTransaction.html)
claims become stale.

| Sink | Feature | What `blit` does |
|------|---------|------------------|
| [`HostPixelSink`](https://docs.rs/goldy/latest/goldy/struct.HostPixelSink.html) | always | Copy into a `Vec<u8>` |
| `foreign::vulkan::ForeignSurface` | `vulkan` | `vkCmdCopyBufferToImage` on a process-wide Vulkan singleton (offscreen image today) |
| `foreign::dx12::ForeignSurface` | `dx12` (Windows) | `CopyTextureRegion` on a process-wide D3D12 singleton (hardware, then WARP) |
| `foreign::metal::ForeignSurface` | `metal` (macOS/iOS) | `MTLBlitCommandEncoder` on a process-wide Metal singleton |

Each `goldy::foreign::*` adapter creates its own graphics device. Vulkan and DX12
share only the process-wide instance/factory lock with the matching Goldy backend
(`vkCreateInstance`, `CreateDXGIFactory2`); Metal has no equivalent constructor
lock. None of them create a Goldy `Instance`. Windowed swapchain present is a
later verb on the same singleton.

## Frame cycle

```rust
let sink = Arc::new(HostPixelSink::new(width, height, TextureFormat::Rgba8Unorm)?);
let exchange = PixelExchange::new(&ctx, sink.clone());
let layout = PixmapLayout::tight(width, height, TextureFormat::Rgba8Unorm);

let mut scheme = Scheme::new(&ctx);
scheme.node("fine", &pipeline).with_parcel(pixmap.whole(), NodeAccess::ReadWrite).dispatch(...);
let tx = exchange.bind_source(&mut scheme, pixmap.whole(), layout)?;

let mut submission = scheme.submit()?;
tx.claim(&mut submission)?.consume()?;
```

`bind_source` requires a **buffer** parcel whose byte size equals
`layout.staging_bytes()`. Texture parcels are rejected.

`consume` waits for the withdraw (CPU timeline on the CPU device), reads
staging, then calls `sink.blit`. `discard` recycles staging without blitting.

## PixmapLayout

`row_pitch == 0` means tightly packed (`width * bytes_per_pixel`). Otherwise
`row_pitch` is the byte stride between rows and must be at least the tight row.

The source parcel is `height * row_pitch` bytes. The sink stores tightly packed
pixels (`HostPixelSink::pixels`, `ForeignSurface::snapshot`).
