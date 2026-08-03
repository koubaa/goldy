//! CUDA + DX12 presentation companion tests (Windows).
//!
//! Skip only when no CUDA backend/adapters exist, or when a Win32 window cannot be
//! created (headless). If CUDA adapters are present under `cuda+graphics+dx12`,
//! companion attach and present must succeed — soft-skipping those failures would
//! hide regressions on capable CI machines.

#![cfg(all(feature = "cuda", feature = "graphics", feature = "dx12", target_os = "windows"))]

use goldy::types::BackendType;
use goldy::{
    ComputePipeline, DeviceDescriptor, Instance, PresentMode, RequestAdapterOptions, Scheme,
    ShaderModule, SurfaceConfig, SurfaceExchange,
};
use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle,
    Win32WindowHandle, WindowHandle, WindowsDisplayHandle,
};
use std::num::NonZeroIsize;
use std::sync::{Arc, Mutex};
use windows::core::w;
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DestroyWindow, CW_USEDEFAULT, WS_OVERLAPPEDWINDOW,
};

/// CUDA+DX12 companion / DXGI present is not safe to run concurrently in-process:
/// parallel `request_device` + present against the same adapter deadlocks (observed as
/// `cuda_compute_to_present_multi_frame` hanging under the default libtest thread pool).
static CUDA_DX12_PRESENT_TEST_LOCK: Mutex<()> = Mutex::new(());

fn try_cuda_instance() -> Option<Instance> {
    // SAFETY: test process; GOLDY_BACKEND is read during Instance::new.
    unsafe { std::env::set_var("GOLDY_BACKEND", "cuda") };
    Instance::new().ok().filter(|i| {
        i.enumerate_adapters()
            .into_iter()
            .any(|a| a.get_info().backend == BackendType::Cuda)
    })
}

struct TestWindow {
    hwnd: HWND,
}

impl TestWindow {
    fn create(width: i32, height: i32) -> windows::core::Result<Self> {
        // Built-in STATIC class avoids Win32_Graphics_Gdi (RegisterClassW).
        let hwnd = unsafe {
            CreateWindowExW(
                Default::default(),
                w!("STATIC"),
                w!("goldy cuda present test"),
                // Hidden: unit tests must not pop visible windows/dialogs.
                WS_OVERLAPPEDWINDOW,
                CW_USEDEFAULT,
                CW_USEDEFAULT,
                width,
                height,
                None,
                None,
                None,
                None,
            )
        }?;
        Ok(Self { hwnd })
    }
}

impl Drop for TestWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.hwnd);
        }
    }
}

impl HasWindowHandle for TestWindow {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        let mut handle = Win32WindowHandle::new(
            NonZeroIsize::new(self.hwnd.0 as isize).ok_or(HandleError::Unavailable)?,
        );
        handle.hinstance = None;
        Ok(unsafe { WindowHandle::borrow_raw(RawWindowHandle::Win32(handle)) })
    }
}

impl HasDisplayHandle for TestWindow {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        Ok(unsafe {
            DisplayHandle::borrow_raw(RawDisplayHandle::Windows(WindowsDisplayHandle::new()))
        })
    }
}

const FILL_SHADER: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(8, 8, 1)]
void cs_main(DirectSpatial<float4> output, ThreadId tid) {
    output[tid.xy] = float4(0.1, 0.2, 0.3, 1.0);
}
"#;

#[test]
fn cuda_device_attaches_dx12_companion_or_skips() {
    let _guard = CUDA_DX12_PRESENT_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("CUDA adapter must be available when CUDA backends enumerate");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("DX12 companion must attach under cuda+graphics+dx12 on a matching NVIDIA adapter"),
    );
    assert_eq!(device.backend_type(), BackendType::Cuda);
    let _ctx = device.create_context().expect("create_context");
}

#[test]
fn cuda_compute_to_present_multi_frame() {
    let _guard = CUDA_DX12_PRESENT_TEST_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(instance) = try_cuda_instance() else {
        eprintln!("skip: no CUDA backend / adapters");
        return;
    };
    let adapter = instance
        .request_adapter(&RequestAdapterOptions::default())
        .expect("CUDA adapter");
    let device = Arc::new(
        adapter
            .request_device(&DeviceDescriptor::default())
            .expect("DX12 companion must attach under cuda+graphics+dx12"),
    );
    assert_eq!(device.backend_type(), BackendType::Cuda);
    let ctx = device.create_context().expect("context");

    let Ok(window) = TestWindow::create(128, 128) else {
        eprintln!("skip: CreateWindowExW failed (headless / no interactive session)");
        return;
    };
    let surface = SurfaceExchange::new_with_depth(
        &ctx,
        &window,
        2,
        SurfaceConfig {
            present_mode: PresentMode::Immediate,
            depth_format: None,
        },
    )
    .expect("surface create");

    let shader = ShaderModule::from_slang(&device, FILL_SHADER).expect("shader compile");
    let pipeline = ComputePipeline::new(&device, &shader).expect("pipeline");

    for frame_i in 0..3u32 {
        let mut scheme = Scheme::new(&ctx);
        let (lease, present_tx) = surface.bind_destination(&mut scheme).expect("bind");
        let (w, h) = surface.size();
        scheme
            .node("fill", &pipeline)
            .with_present(&lease)
            .dispatch(w.div_ceil(8), h.div_ceil(8), 1);
        let mut submission = scheme.submit().expect("submit");
        let compute_tv = goldy::test_support::submission_epoch(&submission);
        assert!(compute_tv > 0, "frame {frame_i}: compute must advance timeline");

        let claim = present_tx.claim(&mut submission).expect("claim");
        claim.consume().expect("present");

        // Present publishes completion on the Goldy/CUDA timeline. Waiting on the
        // compute submit value must succeed (same namespace; present >= compute).
        goldy::test_support::wait_until(&ctx, compute_tv).unwrap_or_else(|e| {
            panic!("frame {frame_i}: wait_until({compute_tv}) failed: {e:#}")
        });
        assert!(
            goldy::test_support::gpu_progress(&ctx) >= compute_tv,
            "frame {frame_i}: progress {} < compute {compute_tv}",
            goldy::test_support::gpu_progress(&ctx)
        );
    }

    // New Scheme each frame above is correctness-only; still assert the present path
    // no longer requires CUDA handoff/flush when submit tails signal the fence.
    let stats = device.cuda_path_stats_for_test().expect("CUDA stats");
    assert_eq!(
        stats.present_handoffs, 0,
        "compute-to-present should signal in the submit tail, not via present handoff"
    );
    assert_eq!(stats.worker_flushes, 0, "present must not flush the submission worker");
}
