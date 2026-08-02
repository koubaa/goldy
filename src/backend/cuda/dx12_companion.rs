//! D3D12 companion device for CUDA presentation (Windows only).
//!
//! Compiled only when `cuda`, `graphics`, and `dx12` are all enabled. Pairs a CUDA
//! ordinal with the matching DXGI adapter by LUID, then owns a DIRECT queue plus a
//! shareable fence imported into CUDA as an external semaphore.

use anyhow::{bail, Context as _, Result};
use cudarc::driver::{sys, CudaContext};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE, LUID};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::{
    D3D12CreateDevice, ID3D12CommandAllocator, ID3D12CommandList, ID3D12CommandQueue, ID3D12Device,
    ID3D12Fence, ID3D12GraphicsCommandList, D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC,
    D3D12_COMMAND_QUEUE_FLAG_NONE, D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, D3D12_FENCE_FLAG_SHARED,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory2, IDXGIAdapter1, IDXGIFactory4, IDXGIFactory5, DXGI_ADAPTER_FLAG,
    DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, DXGI_FEATURE_PRESENT_ALLOW_TEARING,
};
use windows::Win32::System::Threading::{CreateEventA, WaitForSingleObject, INFINITE};

/// 8-byte Windows LUID shared by DXGI and `cuDeviceGetLuid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct AdapterLuid {
    pub bytes: [u8; 8],
}

impl AdapterLuid {
    pub fn from_windows(luid: LUID) -> Self {
        let mut bytes = [0u8; 8];
        bytes[..4].copy_from_slice(&luid.LowPart.to_le_bytes());
        bytes[4..].copy_from_slice(&luid.HighPart.to_le_bytes());
        Self { bytes }
    }

    pub fn from_cuda_device(cu_device: sys::CUdevice) -> Result<Self> {
        let mut luid = [0i8; 8];
        let mut node_mask: u32 = 0;
        let r = unsafe {
            sys::cuDeviceGetLuid(
                luid.as_mut_ptr(),
                &mut node_mask,
                cu_device,
            )
        };
        if r != sys::CUresult::CUDA_SUCCESS {
            bail!("CUDA: cuDeviceGetLuid failed: {r:?}");
        }
        let mut bytes = [0u8; 8];
        for (i, b) in luid.iter().enumerate() {
            bytes[i] = *b as u8;
        }
        Ok(Self { bytes })
    }
}

/// Match a CUDA context's physical GPU to a DXGI hardware adapter by LUID.
pub(super) fn find_matching_dxgi_adapter(
    factory: &IDXGIFactory4,
    cuda_luid: AdapterLuid,
) -> Result<(IDXGIAdapter1, u32, AdapterLuid)> {
    let mut index = 0u32;
    loop {
        let adapter = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(_) => break,
        };
        let desc = unsafe { adapter.GetDesc1() }.context("DXGI: GetDesc1 failed")?;
        let flags = DXGI_ADAPTER_FLAG(desc.Flags as i32);
        if flags.contains(DXGI_ADAPTER_FLAG_SOFTWARE) {
            index += 1;
            continue;
        }
        let luid = AdapterLuid::from_windows(desc.AdapterLuid);
        if luid == cuda_luid {
            let name = String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_string();
            tracing::info!(
                "CUDA↔DX12 LUID match: adapter {index} ({name}) bytes={:02x?}",
                luid.bytes
            );
            return Ok((adapter, index, luid));
        }
        index += 1;
    }
    bail!(
        "CUDA: no DXGI hardware adapter matches CUDA LUID {:02x?} \
         (linked-node / TCC / headless NVIDIA devices cannot present via DX12)",
        cuda_luid.bytes
    )
}

/// Minimal D3D12 device + DIRECT queue + shared fence for CUDA presentation.
pub(super) struct Dx12Companion {
    pub factory: IDXGIFactory4,
    pub allow_tearing: bool,
    pub adapter_luid: AdapterLuid,
    pub dxgi_adapter_id: u32,
    pub device: ID3D12Device,
    pub queue: ID3D12CommandQueue,
    pub queue_lock: std::sync::Mutex<()>,
    /// Shareable fence; CUDA imports it as an external semaphore.
    pub fence: ID3D12Fence,
    pub fence_value: AtomicU64,
    /// CUDA import of [`Self::fence`].
    pub cuda_semaphore: sys::CUexternalSemaphore,
    pub cuda_ctx: Arc<CudaContext>,
    /// Scratch allocator/list pool for present copy + blit (one per in-flight slot).
    pub present_slots: Vec<PresentCommandSlot>,
    /// Dedicated command allocator/list for one-shot resource-state init.
    /// Must not share present slots — those may still be GPU-busy when scratch is created.
    pub init_allocator: ID3D12CommandAllocator,
    pub init_list: ID3D12GraphicsCommandList,
    /// Fence value of the last submit that used [`Self::init_allocator`].
    pub init_fence: AtomicU64,
}

pub(super) struct PresentCommandSlot {
    pub allocator: ID3D12CommandAllocator,
    pub list: ID3D12GraphicsCommandList,
    /// Last fence value submitted with this slot (0 = never used).
    #[allow(dead_code)]
    pub fence_value: u64,
}

impl Dx12Companion {
    pub fn create(cuda_ctx: &Arc<CudaContext>) -> Result<Self> {
        let cuda_luid = AdapterLuid::from_cuda_device(cuda_ctx.cu_device())?;
        let factory: IDXGIFactory4 = unsafe { CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)) }
            .context("CUDA/DX12: CreateDXGIFactory2 failed")?;
        let allow_tearing = factory
            .cast::<IDXGIFactory5>()
            .ok()
            .and_then(|f5| {
                let mut allow: i32 = 0;
                let hr = unsafe {
                    f5.CheckFeatureSupport(
                        DXGI_FEATURE_PRESENT_ALLOW_TEARING,
                        &mut allow as *mut _ as *mut _,
                        std::mem::size_of::<i32>() as u32,
                    )
                };
                hr.ok().map(|()| allow != 0)
            })
            .unwrap_or(false);

        let (adapter, dxgi_adapter_id, adapter_luid) =
            find_matching_dxgi_adapter(&factory, cuda_luid)?;

        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) }
            .context("CUDA/DX12: D3D12CreateDevice failed")?;
        let device = device.context("CUDA/DX12: D3D12CreateDevice returned null")?;

        let node_count = unsafe { device.GetNodeCount() };
        if node_count != 1 {
            bail!(
                "CUDA/DX12: linked-node adapters are not supported yet (GetNodeCount={node_count})"
            );
        }

        let queue_desc = D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            Priority: D3D12_COMMAND_QUEUE_PRIORITY_NORMAL.0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        };
        let queue: ID3D12CommandQueue = unsafe { device.CreateCommandQueue(&queue_desc) }
            .context("CUDA/DX12: CreateCommandQueue failed")?;

        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_SHARED) }
            .context("CUDA/DX12: CreateFence(SHARED) failed")?;

        let fence_handle: HANDLE = unsafe {
            device.CreateSharedHandle(&fence, None, windows::Win32::Foundation::GENERIC_ALL.0, None)
        }
        .context("CUDA/DX12: CreateSharedHandle(fence) failed")?;

        let cuda_semaphore = import_d3d12_fence(cuda_ctx, fence_handle)?;
        // CUDA does not take ownership of the Win32 NT handle.
        unsafe {
            let _ = CloseHandle(fence_handle);
        }

        let mut present_slots = Vec::with_capacity(MAX_FRAMES);
        for _ in 0..MAX_FRAMES {
            let allocator: ID3D12CommandAllocator =
                unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                    .context("CUDA/DX12: CreateCommandAllocator failed")?;
            let list: ID3D12GraphicsCommandList = unsafe {
                device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
            }
            .context("CUDA/DX12: CreateCommandList failed")?;
            unsafe { list.Close() }.context("CUDA/DX12: Close initial command list")?;
            present_slots.push(PresentCommandSlot {
                allocator,
                list,
                fence_value: 0,
            });
        }

        // Dedicated allocator/list for one-shot state init (must not steal present slots).
        let init_allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT) }
                .context("CUDA/DX12: CreateCommandAllocator(init) failed")?;
        let init_list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &init_allocator, None)
        }
        .context("CUDA/DX12: CreateCommandList(init) failed")?;
        unsafe { init_list.Close() }.context("CUDA/DX12: Close init command list")?;

        Ok(Self {
            factory,
            allow_tearing,
            adapter_luid,
            dxgi_adapter_id,
            device,
            queue,
            queue_lock: std::sync::Mutex::new(()),
            fence,
            fence_value: AtomicU64::new(1),
            cuda_semaphore,
            cuda_ctx: Arc::clone(cuda_ctx),
            present_slots,
            init_allocator,
            init_list,
            init_fence: AtomicU64::new(0),
        })
    }

    /// Allocate the next fence value (monotonic, starts at 1).
    pub fn next_fence_value(&self) -> u64 {
        self.fence_value.fetch_add(1, Ordering::AcqRel)
    }

    pub fn signal_queue(&self, value: u64) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.Signal(&self.fence, value) }
            .context("CUDA/DX12: queue Signal failed")?;
        Ok(())
    }

    pub fn wait_queue(&self, value: u64) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.Wait(&self.fence, value) }.context("CUDA/DX12: queue Wait failed")?;
        Ok(())
    }

    pub fn execute_and_signal(
        &self,
        lists: &[Option<windows::Win32::Graphics::Direct3D12::ID3D12CommandList>],
        signal_value: u64,
    ) -> Result<()> {
        let _guard = self.queue_lock.lock().unwrap();
        unsafe { self.queue.ExecuteCommandLists(lists) };
        unsafe { self.queue.Signal(&self.fence, signal_value) }
            .context("CUDA/DX12: Signal after ExecuteCommandLists failed")?;
        Ok(())
    }

    pub fn cpu_wait(&self, value: u64) -> Result<()> {
        if unsafe { self.fence.GetCompletedValue() } >= value {
            return Ok(());
        }
        let event = unsafe { CreateEventA(None, false, false, None) }
            .context("CUDA/DX12: CreateEventA failed")?;
        unsafe { self.fence.SetEventOnCompletion(value, event) }
            .context("CUDA/DX12: SetEventOnCompletion failed")?;
        unsafe { WaitForSingleObject(event, INFINITE) };
        unsafe {
            let _ = CloseHandle(event);
        }
        Ok(())
    }

    pub fn wait_idle(&self) -> Result<()> {
        let v = self.next_fence_value();
        self.signal_queue(v)?;
        self.cpu_wait(v)
    }

    /// Record and submit a one-shot barrier list on the dedicated init allocator.
    pub fn submit_init_list(&self) -> Result<()> {
        let prev = self.init_fence.load(Ordering::Acquire);
        if prev > 0 {
            self.cpu_wait(prev)?;
        }
        unsafe { self.init_allocator.Reset() }.context("CUDA/DX12: reset init allocator")?;
        unsafe { self.init_list.Reset(&self.init_allocator, None) }
            .context("CUDA/DX12: reset init list")?;
        Ok(())
    }

    pub fn finish_init_list(&self) -> Result<()> {
        unsafe { self.init_list.Close() }.context("CUDA/DX12: close init list")?;
        let cmd: ID3D12CommandList = self.init_list.cast().context("cast init list")?;
        let signal = self.next_fence_value();
        self.execute_and_signal(&[Some(cmd)], signal)?;
        self.cpu_wait(signal)?;
        self.init_fence.store(signal, Ordering::Release);
        Ok(())
    }
}

impl Drop for Dx12Companion {
    fn drop(&mut self) {
        let _ = self.wait_idle();
        let _ = self.cuda_ctx.bind_to_thread();
        let sem = std::mem::replace(&mut self.cuda_semaphore, std::ptr::null_mut());
        if !sem.is_null() {
            let _ = unsafe { sys::cuDestroyExternalSemaphore(sem) };
        }
    }
}

// SAFETY: COM objects + CUDA semaphore are used under Goldy's backend / queue locks.
unsafe impl Send for Dx12Companion {}
unsafe impl Sync for Dx12Companion {}

pub(super) const MAX_FRAMES: usize = 3;

fn import_d3d12_fence(
    cuda_ctx: &Arc<CudaContext>,
    handle: HANDLE,
) -> Result<sys::CUexternalSemaphore> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind context for fence import")?;
    let desc = sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC {
        type_: sys::CUexternalSemaphoreHandleType::CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE,
        handle: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
            win32: sys::CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
                handle: handle.0,
                name: std::ptr::null(),
            },
        },
        flags: 0,
        reserved: [0; 16],
    };
    let mut sem: sys::CUexternalSemaphore = std::ptr::null_mut();
    let r = unsafe { sys::cuImportExternalSemaphore(&mut sem, &desc) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuImportExternalSemaphore(D3D12_FENCE) failed: {r:?}");
    }
    Ok(sem)
}

/// Signal the imported D3D12 fence from a CUDA stream.
pub(super) fn cuda_signal_fence(
    cuda_ctx: &CudaContext,
    sem: sys::CUexternalSemaphore,
    stream: sys::CUstream,
    value: u64,
) -> Result<()> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind for external signal")?;
    let params = sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 { value },
            nvSciSync: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
                reserved: 0,
            },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
            },
            reserved: [0; 12],
        },
        flags: 0,
        reserved: [0; 16],
    };
    let r = unsafe { sys::cuSignalExternalSemaphoresAsync(&sem, &params, 1, stream) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuSignalExternalSemaphoresAsync failed: {r:?}");
    }
    Ok(())
}

/// Wait on the imported D3D12 fence from a CUDA stream.
pub(super) fn cuda_wait_fence(
    cuda_ctx: &CudaContext,
    sem: sys::CUexternalSemaphore,
    stream: sys::CUstream,
    value: u64,
) -> Result<()> {
    cuda_ctx
        .bind_to_thread()
        .context("CUDA/DX12: bind for external wait")?;
    let params = sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS {
        params: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
            fence: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 { value },
            nvSciSync: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
                reserved: 0,
            },
            keyedMutex: sys::CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
                key: 0,
                timeoutMs: 0,
            },
            reserved: [0; 10],
        },
        flags: 0,
        reserved: [0; 16],
    };
    let r = unsafe { sys::cuWaitExternalSemaphoresAsync(&sem, &params, 1, stream) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("CUDA: cuWaitExternalSemaphoresAsync failed: {r:?}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luid_roundtrip_windows_layout() {
        let luid = LUID {
            LowPart: 0x11223344,
            HighPart: 0x55667788u32 as i32,
        };
        let a = AdapterLuid::from_windows(luid);
        assert_eq!(&a.bytes[..4], &0x11223344u32.to_le_bytes());
        assert_eq!(&a.bytes[4..], &(0x55667788u32 as i32).to_le_bytes());
    }

    #[test]
    fn companion_create_matches_cuda_luid_or_skips() {
        let Ok(ctx) = CudaContext::new(0) else {
            eprintln!("skip: no CUDA device 0");
            return;
        };
        let cuda_luid = match AdapterLuid::from_cuda_device(ctx.cu_device()) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("skip: cuDeviceGetLuid failed: {e:#}");
                return;
            }
        };
        match Dx12Companion::create(&ctx) {
            Ok(companion) => {
                assert_eq!(companion.adapter_luid, cuda_luid);
                // WARP uses a sentinel in the main DX12 backend; companion rejects software.
                let stream = ctx.default_stream();
                let v = companion.next_fence_value();
                cuda_signal_fence(&ctx, companion.cuda_semaphore, stream.cu_stream(), v)
                    .expect("cuda signal");
                stream.synchronize().expect("stream sync after signal");
                companion.cpu_wait(v).expect("cpu wait on shared fence");
                let _ = companion.wait_idle();
            }
            Err(e) => eprintln!("skip: Dx12Companion::create failed: {e:#}"),
        }
    }
}
