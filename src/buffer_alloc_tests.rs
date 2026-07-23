#[cfg(test)]
mod buffer_alloc_tests {
    use crate::parcel::Parcel;
    use crate::test_support::SerialGpuDevice;
    use crate::types::{BufferFlags, ResourceAccess};
    use crate::{BufferKind, Context, MemoryExchange, Scheme};
    use std::sync::Arc;

    fn make_device() -> SerialGpuDevice {
        SerialGpuDevice::new()
    }

    fn withdraw_bytes(ctx: &Context, device: &crate::Device, arc: &Arc<crate::buffer::Allocation>) -> Vec<u8> {
        let parcel = Parcel::from_whole_buffer(Arc::clone(arc), Arc::downgrade(&device.inner));
        let mut scheme = Scheme::new(ctx);
        let tx = MemoryExchange::new(ctx)
            .bind_withdraw(&mut scheme, &parcel)
            .expect("bind_withdraw");
        let mut sub = scheme.submit().expect("submit");
        tx.claim(&mut sub)
            .expect("claim")
            .consume()
            .expect("consume")
            .to_vec()
    }

    // ─── Buffer resize (Phase 1: stable handles, realloc-copy fallback) ───────────

    #[test]
    fn resize_preserves_contents() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)
                .expect("buf"),
        );
        Arc::get_mut(&mut arc).unwrap().resize_to(32).expect("resize");
        let out = withdraw_bytes(&ctx, &device, &arc);
        let words: &[u32] = bytemuck::cast_slice(&out);
        assert_eq!(&words[..4], &[1u32, 2, 3, 4]);
        for w in &words[4..8] {
            assert_eq!(*w, 0, "new region should be zero");
        }
    }

    #[test]
    fn resize_preserves_bindless_index() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer(16, BufferKind::Scattered, None, BufferFlags::empty())
            .expect("buf");
        let idx = buf.resource_index(ResourceAccess::Write).expect("bindless");
        buf.resize_to(256).expect("resize");
        assert_eq!(buf.resource_index(ResourceAccess::Write), Some(idx));
    }

    #[test]
    fn resize_down_truncates() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_data(&[10u32, 20, 30, 40], BufferKind::Scattered)
                .expect("buf"),
        );
        Arc::get_mut(&mut arc).unwrap().resize_to(8).expect("resize down");
        let out = withdraw_bytes(&ctx, &device, &arc);
        let words: &[u32] = bytemuck::cast_slice(&out);
        assert_eq!(words, &[10u32, 20]);
    }

    #[test]
    fn resize_uninitialized_skips_copy() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_data(&[0xABCD_BEEFu32], BufferKind::Scattered)
                .expect("buf"),
        );
        Arc::get_mut(&mut arc)
            .unwrap()
            .resize_to_uninitialized(8)
            .expect("resize uni");
        let _ = withdraw_bytes(&ctx, &device, &arc);
    }

    #[test]
    fn new_with_capacity_hint_smoke() {
        let device = make_device();
        let b = device
            .alloc_buffer_with_capacity(16, 4096, BufferKind::Scattered, BufferFlags::empty())
            .expect("b");
        assert_eq!(b.size(), 16);
        assert!(b.allocated_size() >= 4096, "expected oversize allocation");
    }

    #[test]
    fn oversize_resize_within_capacity_preserves_and_zeros_tail() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_capacity(16, 4096, BufferKind::Scattered, BufferFlags::empty())
                .expect("buf"),
        );
        let idx = arc.resource_index(ResourceAccess::Write).expect("bindless");
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, &[0xabu8; 16])
            .expect("seed");
        Arc::get_mut(&mut arc).unwrap().resize_to(256).expect("grow within cap");
        assert_eq!(arc.resource_index(ResourceAccess::Write), Some(idx));
        assert!(arc.size() >= 256);
        let got = withdraw_bytes(&ctx, &device, &arc);
        assert_eq!(&got[..16], &[0xabu8; 16]);
        assert!(got[16..].iter().all(|&x| x == 0));
    }

    #[test]
    fn oversize_resize_beyond_capacity_falls_back_and_preserves() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_capacity(16, 256, BufferKind::Scattered, BufferFlags::empty())
                .expect("buf"),
        );
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, &[7u8; 16])
            .expect("w");
        Arc::get_mut(&mut arc).unwrap().resize_to(512).expect("grow past cap");
        assert!(arc.allocated_size() >= 512);
        let got = withdraw_bytes(&ctx, &device, &arc);
        assert_eq!(&got[..16], &[7u8; 16]);
    }

    #[test]
    fn hint_unused_above_does_not_corrupt_prefix() {
        let device = make_device();
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_capacity(64, 4096, BufferKind::Scattered, BufferFlags::empty())
                .expect("buf"),
        );
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, &[0x11u8; 64])
            .expect("w");
        Arc::get_mut(&mut arc).unwrap().hint_unused_above(32);
        let got = withdraw_bytes(&ctx, &device, &arc);
        assert_eq!(&got[..32], &[0x11u8; 32]);
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    #[test]
    fn device_capabilities_metal_reports_constant_resize() {
        use crate::{types::BufferResizeCost, BackendType};
        let device = SerialGpuDevice::preferring(crate::DeviceType::IntegratedGpu);
        let _ctx = device.create_context().expect("context");
        assert_eq!(device.backend_type(), BackendType::Metal);
        let caps = device.capabilities();
        assert_eq!(caps.buffer_resize_cost, BufferResizeCost::Constant);
        assert!(caps.buffer_decommit_supported);
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn device_capabilities_vulkan_reports_pagebind_when_sparse() {
        use crate::{types::BufferResizeCost, BackendType};
        let device = SerialGpuDevice::preferring(crate::DeviceType::DiscreteGpu);
        if device.backend_type() != BackendType::Vulkan {
            return;
        }
        let caps = device.capabilities();
        if caps.buffer_resize_cost == BufferResizeCost::PageBind {
            assert_eq!(caps.buffer_page_size, 64 * 1024);
            assert!(caps.buffer_decommit_supported);
        }
    }

    #[cfg(feature = "dx12")]
    #[test]
    fn device_capabilities_dx12_reports_pagebind_when_reserved_supported() {
        use crate::{types::BufferResizeCost, BackendType};
        // With multiple backends enabled, `GOLDY_BACKEND` may select a non-DX12 API; skip in that case.
        let device = SerialGpuDevice::preferring(crate::DeviceType::DiscreteGpu);
        if device.backend_type() != BackendType::Dx12 {
            return;
        }
        let caps = device.capabilities();
        if caps.buffer_resize_cost == BufferResizeCost::PageBind {
            assert_eq!(caps.buffer_page_size, 64 * 1024);
            assert!(caps.buffer_decommit_supported);
        }
    }

    #[cfg(any(feature = "vulkan", feature = "dx12"))]
    #[test]
    fn sparse_backend_oversize_resize_and_hint_within_capacity() {
        use crate::types::BufferResizeCost;
        let device = SerialGpuDevice::preferring(crate::DeviceType::DiscreteGpu);
        if device.capabilities().buffer_resize_cost != BufferResizeCost::PageBind {
            return;
        }
        let ctx = device.create_context().expect("context");
        let mut arc = Arc::new(
            device
                .alloc_buffer_with_capacity(64, 4096, BufferKind::Scattered, BufferFlags::empty())
                .expect("buf"),
        );
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, &[0x11u8; 64])
            .expect("w");
        Arc::get_mut(&mut arc).unwrap().resize_to(256).expect("grow within cap");
        let got = withdraw_bytes(&ctx, &device, &arc);
        assert_eq!(&got[..64], &[0x11u8; 64]);
        assert!(got[64..].iter().all(|&x| x == 0));

        Arc::get_mut(&mut arc).unwrap().hint_unused_above(32);
        let prefix = withdraw_bytes(&ctx, &device, &arc);
        assert_eq!(&prefix[..32], &[0x11u8; 32]);
    }

    /// DX12 reserved-buffer path: cross tile boundaries, `hint_unused_above`, regrowth, stable bindless, compute.
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    #[test]
    fn dx12_reserved_buffer_resize_compute_smoke() {
        use crate::types::BufferResizeCost;
        use crate::{BackendType, ComputePipeline, DeviceType, NodeAccess, ShaderModule};
        const SMOKY_SHADER: &str = r#"
    import goldy_exp;

    [goldy_compute]
    [numthreads(16, 1, 1)]
    void cs_main(Scattered<uint> data, ThreadId id) {
        data[id.x] = data[id.x] * 2;
    }
    "#;
        let device = SerialGpuDevice::preferring(DeviceType::DiscreteGpu);
        let ctx = device.create_context().expect("context");
        if device.backend_type() != BackendType::Dx12 {
            return;
        }
        if device.capabilities().buffer_resize_cost != BufferResizeCost::PageBind {
            return;
        }

        let mut arc = Arc::new(
            device
                .alloc_buffer_with_capacity(256, 4 * 64 * 1024, BufferKind::Scattered, BufferFlags::empty())
                .expect("buf"),
        );
        let bindless = arc.resource_index(ResourceAccess::ReadWrite).expect("bindless");

        let initial: Vec<u32> = (1..=16).collect();
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, bytemuck::cast_slice(&initial))
            .expect("w");

        Arc::get_mut(&mut arc).unwrap().resize_to(200 * 1024).expect("grow across tiles");
        assert_eq!(arc.resource_index(ResourceAccess::ReadWrite), Some(bindless));

        let out = withdraw_bytes(&ctx, &device, &arc);
        let read: &[u32] = bytemuck::cast_slice(&out[..64]);
        assert_eq!(read, initial.as_slice());

        let shader = ShaderModule::from_slang(&device, SMOKY_SHADER).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        // Scope scheme/parcel so withdraw keepalive Arc clones drop before resize_to.
        {
            let mut scheme = Scheme::new(&ctx);
            scheme
                .node("n0", &pipeline)
                .with_parcel(arc.as_ref(), NodeAccess::ReadWrite)
                .dispatch(1, 1, 1);
            let parcel = Parcel::from_whole_buffer(Arc::clone(&arc), Arc::downgrade(&device.inner));
            let grant = MemoryExchange::new(&ctx)
                .bind_withdraw(&mut scheme, &parcel)
                .expect("withdraw");
            let mut sub = scheme.submit().expect("dispatch");
            let loan = grant.claim(&mut sub).expect("claim").consume().expect("consume");
            let read: &[u32] = bytemuck::cast_slice(&loan[..64]);
            for (i, &v) in read.iter().enumerate() {
                assert_eq!(v, (i as u32 + 1) * 2, "after first dispatch[{i}]");
            }
        }

        // Shrink to one tile, decommit reserved tail with `hint_unused_above`, then grow again.
        Arc::get_mut(&mut arc).unwrap().resize_to(64 * 1024).expect("shrink to one tile");
        assert_eq!(arc.resource_index(ResourceAccess::ReadWrite), Some(bindless));
        Arc::get_mut(&mut arc).unwrap().hint_unused_above(64 * 1024);
        Arc::get_mut(&mut arc).unwrap().resize_to(200 * 1024).expect("grow after decommit hint");
        assert_eq!(arc.resource_index(ResourceAccess::ReadWrite), Some(bindless));

        let initial2: Vec<u32> = (0..16).collect();
        Arc::get_mut(&mut arc)
            .unwrap()
            .write(0, bytemuck::cast_slice(&initial2))
            .expect("w2");
        // Fresh scheme: avoid retained-CB replay against Allocation-backed parcels.
        let mut scheme2 = Scheme::new(&ctx);
        scheme2
            .node("n1", &pipeline)
            .with_parcel(arc.as_ref(), NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let parcel2 = Parcel::from_whole_buffer(Arc::clone(&arc), Arc::downgrade(&device.inner));
        let grant2 = MemoryExchange::new(&ctx)
            .bind_withdraw(&mut scheme2, &parcel2)
            .expect("withdraw2");
        let mut sub2 = scheme2.submit().expect("dispatch2");
        let loan2 = grant2.claim(&mut sub2).expect("claim").consume().expect("consume");
        let read: &[u32] = bytemuck::cast_slice(&loan2[..64]);
        for (i, &v) in read.iter().enumerate() {
            assert_eq!(v, (i as u32) * 2, "after second dispatch[{i}]");
        }
    }

    #[test]
    fn hint_unused_above_smoke() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer(64, BufferKind::Scattered, None, BufferFlags::empty())
            .expect("buf");
        buf.hint_unused_above(32);
    }
}
