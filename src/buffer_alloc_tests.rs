#[cfg(test)]
mod buffer_alloc_tests {
    use crate::test_support::SerialGpuDevice;
    use crate::types::{BufferFlags, ResourceAccess};
    use crate::BufferKind;

    fn make_device() -> SerialGpuDevice {
        SerialGpuDevice::new()
    }

    // ─── Buffer resize (Phase 1: stable handles, realloc-copy fallback) ───────────

    #[test]
    fn resize_preserves_contents() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)
            .expect("buf");
        buf.resize_to(32).expect("resize");
        let mut out = vec![0u8; 32];
        buf.read_to_cpu(&device, &mut out).expect("read");
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
        let mut buf = device
            .alloc_buffer_with_data(&[10u32, 20, 30, 40], BufferKind::Scattered)
            .expect("buf");
        buf.resize_to(8).expect("resize down");
        let mut out = vec![0u8; 8];
        buf.read_to_cpu(&device, &mut out).expect("read");
        let words: &[u32] = bytemuck::cast_slice(&out);
        assert_eq!(words, &[10u32, 20]);
    }

    #[test]
    fn resize_uninitialized_skips_copy() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer_with_data(&[0xABCD_BEEFu32], BufferKind::Scattered)
            .expect("buf");
        buf.resize_to_uninitialized(8).expect("resize uni");
        let mut out = vec![0u8; 8];
        buf.read_to_cpu(&device, &mut out).expect("read");
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
        let mut buf = device
            .alloc_buffer_with_capacity(16, 4096, BufferKind::Scattered, BufferFlags::empty())
            .expect("buf");
        let idx = buf.resource_index(ResourceAccess::Write).expect("bindless");
        buf.write(0, &[0xabu8; 16]).expect("seed");
        buf.resize_to(256).expect("grow within cap");
        assert_eq!(buf.resource_index(ResourceAccess::Write), Some(idx));
        assert!(buf.size() >= 256);
        let mut got = vec![0u8; 256];
        buf.read_to_cpu(&device, &mut got).expect("read");
        assert_eq!(&got[..16], &[0xabu8; 16]);
        assert!(got[16..].iter().all(|&x| x == 0));
    }

    #[test]
    fn oversize_resize_beyond_capacity_falls_back_and_preserves() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer_with_capacity(16, 256, BufferKind::Scattered, BufferFlags::empty())
            .expect("buf");
        buf.write(0, &[7u8; 16]).expect("w");
        buf.resize_to(512).expect("grow past cap");
        assert!(buf.allocated_size() >= 512);
        let mut got = vec![0u8; 512];
        buf.read_to_cpu(&device, &mut got).expect("r");
        assert_eq!(&got[..16], &[7u8; 16]);
    }

    #[test]
    fn hint_unused_above_does_not_corrupt_prefix() {
        let device = make_device();
        let mut buf = device
            .alloc_buffer_with_capacity(64, 4096, BufferKind::Scattered, BufferFlags::empty())
            .expect("buf");
        buf.write(0, &[0x11u8; 64]).expect("w");
        buf.hint_unused_above(32);
        let mut got = vec![0u8; 32];
        buf.read_to_cpu(&device, &mut got).expect("r");
        assert_eq!(&got[..], &[0x11u8; 32]);
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
        let mut buf = device
            .alloc_buffer_with_capacity(64, 4096, BufferKind::Scattered, BufferFlags::empty())
            .expect("buf");
        buf.write(0, &[0x11u8; 64]).expect("w");
        buf.resize_to(256).expect("grow within cap");
        let mut got = vec![0u8; 256];
        buf.read_to_cpu(&device, &mut got).expect("r");
        assert_eq!(&got[..64], &[0x11u8; 64]);
        assert!(got[64..].iter().all(|&x| x == 0));

        buf.hint_unused_above(32);
        let mut prefix = vec![0u8; 32];
        buf.read_to_cpu(&device, &mut prefix).expect("r2");
        assert_eq!(&prefix[..], &[0x11u8; 32]);
    }

    /// DX12 reserved-buffer path: cross tile boundaries, `hint_unused_above`, regrowth, stable bindless, compute.
    #[cfg(all(feature = "dx12", target_os = "windows"))]
    #[test]
    fn dx12_reserved_buffer_resize_compute_smoke() {
        use crate::types::BufferResizeCost;
        use crate::{BackendType, ComputePipeline, DeviceType, NodeAccess, Scheme, ShaderModule};
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

        let mut buf = device
            .alloc_buffer_with_capacity(256, 4 * 64 * 1024, BufferKind::Scattered, BufferFlags::empty())
            .expect("buf");
        let bindless = buf.resource_index(ResourceAccess::ReadWrite).expect("bindless");

        let initial: Vec<u32> = (1..=16).collect();
        buf.write(0, bytemuck::cast_slice(&initial)).expect("w");

        buf.resize_to(200 * 1024).expect("grow across tiles");
        assert_eq!(buf.resource_index(ResourceAccess::ReadWrite), Some(bindless));

        let mut read = vec![0u32; 16];
        buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
            .expect("read");
        assert_eq!(read, initial);

        let shader = ShaderModule::from_slang(&device, SMOKY_SHADER).expect("shader");
        let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");
        let mut scheme = Scheme::new(&ctx);
        scheme
            .node("n0", &pipeline)
            .with_parcel(&buf, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv = scheme.submit().expect("dispatch").timeline_value();
        ctx.wait_until(tv).expect("wait dispatch");

        buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
            .expect("read2");
        for (i, &v) in read.iter().enumerate() {
            assert_eq!(v, (i as u32 + 1) * 2, "after first dispatch[{i}]");
        }

        // Shrink to one tile, decommit reserved tail with `hint_unused_above`, then grow again.
        buf.resize_to(64 * 1024).expect("shrink to one tile");
        assert_eq!(buf.resource_index(ResourceAccess::ReadWrite), Some(bindless));
        buf.hint_unused_above(64 * 1024);
        buf.resize_to(200 * 1024).expect("grow after decommit hint");
        assert_eq!(buf.resource_index(ResourceAccess::ReadWrite), Some(bindless));

        let initial2: Vec<u32> = (0..16).collect();
        buf.write(0, bytemuck::cast_slice(&initial2)).expect("w2");
        // Fresh scheme: avoid retained-CB replay against Allocation-backed parcels.
        let mut scheme2 = Scheme::new(&ctx);
        scheme2
            .node("n1", &pipeline)
            .with_parcel(&buf, NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let tv2 = scheme2.submit().expect("dispatch2").timeline_value();
        ctx.wait_until(tv2).expect("wait dispatch2");

        buf.read_to_cpu(&device, bytemuck::cast_slice_mut(&mut read))
            .expect("read3");
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
