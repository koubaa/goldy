//! Compute-only CUDA backend prototype.
//!
//! Slang compiles `[goldy_compute]` (and plain compute) shaders to PTX. Launch
//! arguments use Slang's CUDA `StructuredBuffer` ABI (`{T* data; size_t count}`)
//! interleaved with bare `uniform uint` scalars from [`GpuCommand::BindResourcesRaw::user`].
//! Single-dispatch registry keys come from [`GpuCommand::BindResourcesRaw`]; batched
//! dispatches resolve keys from [`GpuCommand::FrameTableStaging`] in shader
//! parameter order — there is no bindless heap or device-side frame-table routing.

use super::*;
use crate::backend::shared::{PushLayout, DISPATCH_BATCH_STRIDE, MAX_USER_SLOTS, TOTAL_PUSH_BYTES};
use crate::frame_table::dispatch_table_base_word_index;
use crate::slang::virtual_main::CudaLaunchArgKind;
use crate::types::{BufferResizeCost, DeviceType};
use anyhow::{Context as _, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Slang CUDA structured-buffer descriptor: `{ T* data; size_t count }`.
#[repr(C)]
#[derive(Clone, Copy)]
struct CudaBufferArg {
    data: u64,
    count: usize,
}

// SAFETY: plain POD matching Slang's CUDA StructuredBuffer / RWStructuredBuffer ABI.
unsafe impl DeviceRepr for CudaBufferArg {}

pub(crate) struct CudaBackend {
    adapter_info: Vec<AdapterInfo>,
    devices: HashMap<DeviceHandle, CudaDevice>,
    contexts: HashMap<ContextHandle, Arc<CudaSubmitContext>>,
    buffers: HashMap<BufferHandle, CudaBuffer>,
    buffer_slots: HashMap<u32, BufferHandle>,
    shaders: HashMap<ShaderHandle, CudaShader>,
    compute_pipelines: HashMap<ComputePipelineHandle, CudaComputePipeline>,
    next_device: DeviceHandle,
    next_context: ContextHandle,
    next_buffer: BufferHandle,
    next_slot: u32,
    next_shader: ShaderHandle,
    next_compute_pipeline: ComputePipelineHandle,
}

struct CudaDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    next_timeline: Arc<AtomicU64>,
    retired: Arc<AtomicU64>,
}

struct CudaSubmitContext {
    device: DeviceHandle,
    completed: AtomicU64,
    signal_queue: crate::signal::SignalQueue,
}

struct CudaProgress {
    context: Arc<CudaSubmitContext>,
}

impl ContextGpuProgress for CudaProgress {
    fn gpu_progress(&self) -> crate::timeline::TimelineValue {
        self.context.completed.load(Ordering::Acquire)
    }
}

struct CudaDestroyContext;

impl ContextDestroyHandle for CudaDestroyContext {
    fn wait(&self) -> Result<()> {
        Ok(())
    }

    fn finish(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

struct CudaBuffer {
    device: DeviceHandle,
    memory: Arc<Mutex<CudaSlice<u8>>>,
    offset: u64,
    size: u64,
    capacity: u64,
    element_stride: Option<u32>,
    slot: Option<u32>,
    readback: bool,
}

struct CudaShader {
    device: DeviceHandle,
    source: String,
    search_paths: Vec<String>,
    defines: Vec<(String, String)>,
    optimization_level: crate::types::OptimizationLevel,
}

struct CudaComputePipeline {
    device: DeviceHandle,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    function: CudaFunction,
    workgroup_size: [u32; 3],
    slot_access: Vec<Option<ResourceAccess>>,
    /// Author param order for `[goldy_compute]`; empty for plain Slang compute (all-buffer fallback).
    launch_layout: Vec<CudaLaunchArgKind>,
}

/// Host-side values pushed to `cuLaunchKernel` in shader parameter order.
enum CudaLaunchArg {
    Buffer(CudaBufferArg),
    Scalar(u32),
}

impl CudaBackend {
    pub(crate) fn new() -> Result<Self> {
        ensure_cuda_toolkit_on_path();
        cudarc::driver::result::init().context("CUDA: driver init failed")?;
        let count = CudaContext::device_count().context("CUDA: enumerate devices")?;
        if count <= 0 {
            anyhow::bail!("CUDA: no devices found");
        }
        let mut adapter_info = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let ctx = CudaContext::new(ordinal as usize).with_context(|| format!("CUDA: open device {ordinal}"))?;
            let name = ctx.name().unwrap_or_else(|_| format!("CUDA device {ordinal}"));
            adapter_info.push(AdapterInfo {
                id: ordinal as u32,
                name,
                vendor: "NVIDIA".to_string(),
                backend: BackendType::Cuda,
                device_type: DeviceType::DiscreteGpu,
            });
        }
        Ok(Self {
            adapter_info,
            devices: HashMap::new(),
            contexts: HashMap::new(),
            buffers: HashMap::new(),
            buffer_slots: HashMap::new(),
            shaders: HashMap::new(),
            compute_pipelines: HashMap::new(),
            next_device: 1,
            next_context: 1,
            next_buffer: 1,
            next_slot: 0,
            next_shader: 1,
            next_compute_pipeline: 1,
        })
    }

    fn device(&self, handle: DeviceHandle) -> Result<&CudaDevice> {
        self.devices.get(&handle).context("CUDA: invalid device handle")
    }

    fn context(&self, handle: ContextHandle) -> Result<&Arc<CudaSubmitContext>> {
        self.contexts.get(&handle).context("CUDA: invalid context handle")
    }

    fn unsupported<T>(operation: &str) -> Result<T> {
        anyhow::bail!("CUDA compute-only backend does not support {operation}")
    }

    fn create_storage_buffer(
        &mut self,
        device: DeviceHandle,
        logical_size: u64,
        capacity: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let capacity = capacity.max(logical_size).max(4);
        let gpu = self.device(device)?;
        let memory = Arc::new(Mutex::new(
            gpu.stream
                .alloc_zeros::<u8>(capacity as usize)
                .context("CUDA: alloc buffer")?,
        ));
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot = self
            .next_slot
            .checked_add(1)
            .context("CUDA buffer registry exhausted")?;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            CudaBuffer {
                device,
                memory,
                offset: 0,
                size: logical_size,
                capacity,
                element_stride,
                slot: Some(slot),
                readback: false,
            },
        );
        Ok(handle)
    }

    fn compile_compute_ptx(
        &self,
        shader: &CudaShader,
    ) -> Result<(String, Vec<Option<ResourceAccess>>, [u32; 3], Vec<CudaLaunchArgKind>)> {
        ensure_cuda_toolkit_on_path();
        let compiler = crate::slang::SlangCompiler::new().context("CUDA: initialize Slang")?;
        let paths: Vec<&str> = shader.search_paths.iter().map(String::as_str).collect();
        let defines: Vec<(&str, &str)> = shader
            .defines
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        let launch_layout = crate::slang::virtual_main::extract_cuda_compute_launch_layout(&shader.source)
            .map_err(|error| anyhow::anyhow!("CUDA launch layout failed: {error}"))?;
        let cuda_source = crate::slang::virtual_main::transform_virtual_main_cuda_compute(&shader.source)
            .map_err(|error| anyhow::anyhow!("CUDA shader lowering failed: {error}"))?;
        let workgroup_size = crate::slang::parse_numthreads(&shader.source).unwrap_or([1, 1, 1]);
        let compiled = compiler.compile_bindless_with_reflection_and_defines(
            &cuda_source,
            crate::slang::ShaderTarget::Ptx,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &paths,
            &defines,
            &[],
            shader.optimization_level,
        )?;
        let mut ptx = compiled
            .shader
            .as_str()
            .context("CUDA: Slang returned non-text PTX output")?
            .to_owned();
        while ptx.ends_with('\0') {
            ptx.pop();
        }
        let access = crate::slang::virtual_main::extract_push_constant_categories(&shader.source)
            .iter()
            .map(|category| {
                category.map(|category| match category {
                    crate::types::ResourceCategory::Broadcast
                    | crate::types::ResourceCategory::Texture
                    | crate::types::ResourceCategory::Sampler => ResourceAccess::Read,
                    crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::StorageImage => {
                        ResourceAccess::ReadWrite
                    }
                })
            })
            .collect();
        Ok((ptx, access, workgroup_size, launch_layout))
    }

    fn buffer_arg(&self, stream: &Arc<CudaStream>, buffer: &CudaBuffer) -> Result<CudaBufferArg> {
        let memory = buffer.memory.lock().unwrap();
        let start = buffer.offset as usize;
        let end = (buffer.offset + buffer.size) as usize;
        let view = memory.try_slice(start..end).context("CUDA: buffer view out of range")?;
        let (ptr, _sync) = view.device_ptr(stream);
        let stride = buffer.element_stride.unwrap_or(1).max(1) as u64;
        let count = if buffer.size == 0 {
            0
        } else {
            (buffer.size / stride) as usize
        };
        Ok(CudaBufferArg { data: ptr, count })
    }

    fn resolve_buffer_arg(&self, stream: &Arc<CudaStream>, binding: usize, index: u32) -> Result<CudaBufferArg> {
        let handle = self
            .buffer_slots
            .get(&index)
            .with_context(|| format!("CUDA: binding {binding} references unknown registry key {index}"))?;
        let buffer = self
            .buffers
            .get(handle)
            .with_context(|| format!("CUDA: registry key {index} references a destroyed buffer"))?;
        self.buffer_arg(stream, buffer)
    }

    /// Build launch args in shader parameter order.
    ///
    /// Empty `launch_layout` means plain (non-`[goldy_compute]`) Slang: one buffer arg
    /// per registry index and no scalars.
    fn build_launch_args(
        &self,
        stream: &Arc<CudaStream>,
        launch_layout: &[CudaLaunchArgKind],
        indices: &[u32],
        user: &[u32],
    ) -> Result<Vec<CudaLaunchArg>> {
        if launch_layout.is_empty() {
            if !user.is_empty() {
                anyhow::bail!(
                    "CUDA: scalar user params require a [goldy_compute] entry; got {} user word(s)",
                    user.len()
                );
            }
            let mut args = Vec::with_capacity(indices.len());
            for (binding, index) in indices.iter().copied().enumerate() {
                args.push(CudaLaunchArg::Buffer(self.resolve_buffer_arg(stream, binding, index)?));
            }
            return Ok(args);
        }

        let expected_buffers = launch_layout
            .iter()
            .filter(|kind| matches!(kind, CudaLaunchArgKind::Buffer))
            .count();
        let expected_scalars = launch_layout
            .iter()
            .filter(|kind| matches!(kind, CudaLaunchArgKind::Scalar))
            .count();
        if indices.len() != expected_buffers {
            anyhow::bail!(
                "CUDA: dispatch bound {} buffer(s) but shader expects {expected_buffers}",
                indices.len()
            );
        }
        if user.len() != expected_scalars {
            anyhow::bail!(
                "CUDA: dispatch provided {} scalar user word(s) but shader expects {expected_scalars}",
                user.len()
            );
        }

        let mut args = Vec::with_capacity(launch_layout.len());
        let mut index_i = 0usize;
        let mut user_i = 0usize;
        for kind in launch_layout {
            match kind {
                CudaLaunchArgKind::Buffer => {
                    let index = indices[index_i];
                    args.push(CudaLaunchArg::Buffer(self.resolve_buffer_arg(stream, index_i, index)?));
                    index_i += 1;
                }
                CudaLaunchArgKind::Scalar => {
                    args.push(CudaLaunchArg::Scalar(user[user_i]));
                    user_i += 1;
                }
            }
        }
        Ok(args)
    }

    fn write_buffer_region(stream: &Arc<CudaStream>, buffer: &CudaBuffer, offset: u64, data: &[u8]) -> Result<()> {
        if offset + data.len() as u64 > buffer.size {
            anyhow::bail!("CUDA: write exceeds logical buffer size");
        }
        let mut memory = buffer.memory.lock().unwrap();
        let start = (buffer.offset + offset) as usize;
        let end = start + data.len();
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: write range out of bounds")?;
        stream.memcpy_htod(data, &mut view).context("CUDA: HtoD write failed")
    }

    fn clear_buffer_region(stream: &Arc<CudaStream>, buffer: &CudaBuffer, offset: u64, size: u64) -> Result<()> {
        let clear_size = if size == 0 {
            buffer.size.saturating_sub(offset)
        } else {
            size
        };
        let mut memory = buffer.memory.lock().unwrap();
        let start = (buffer.offset + offset) as usize;
        let end = start + clear_size as usize;
        let mut view = memory
            .try_slice_mut(start..end)
            .context("CUDA: clear range out of bounds")?;
        stream.memset_zeros(&mut view).context("CUDA: memset failed")
    }

    fn copy_buffer_region(
        stream: &Arc<CudaStream>,
        src: &CudaBuffer,
        src_offset: u64,
        dst: &CudaBuffer,
        dst_offset: u64,
        size: u64,
    ) -> Result<()> {
        if size == 0 {
            return Ok(());
        }
        if src.device != dst.device {
            anyhow::bail!("CUDA: copy across devices is not supported");
        }
        if src_offset + size > src.size {
            anyhow::bail!("CUDA: copy source range exceeds logical buffer size");
        }
        if dst_offset + size > dst.size {
            anyhow::bail!("CUDA: copy destination range exceeds logical buffer size");
        }

        let src_abs = src.offset + src_offset;
        let dst_abs = dst.offset + dst_offset;
        let byte_len = size as usize;

        if Arc::ptr_eq(&src.memory, &dst.memory) {
            // Same allocation: avoid simultaneous &/&mut CudaSlice views. A device temp
            // keeps both overlapping and non-overlapping self-copies memmove-safe.
            let mut temp = stream
                .alloc_zeros::<u8>(byte_len)
                .context("CUDA: alloc overlapping-copy scratch")?;
            {
                let memory = src.memory.lock().unwrap();
                let src_view = memory
                    .try_slice(src_abs as usize..src_abs as usize + byte_len)
                    .context("CUDA: copy source out of bounds")?;
                stream
                    .memcpy_dtod(&src_view, &mut temp)
                    .context("CUDA: same-alloc copy to scratch")?;
            }
            {
                let mut memory = dst.memory.lock().unwrap();
                let mut dst_view = memory
                    .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                    .context("CUDA: copy destination out of bounds")?;
                stream
                    .memcpy_dtod(&temp, &mut dst_view)
                    .context("CUDA: same-alloc copy from scratch")?;
            }
            return Ok(());
        }

        // Distinct allocations: lock in pointer order to avoid AB/BA deadlocks.
        let src_arc = Arc::clone(&src.memory);
        let dst_arc = Arc::clone(&dst.memory);
        let src_ptr = Arc::as_ptr(&src_arc);
        let dst_ptr = Arc::as_ptr(&dst_arc);
        if src_ptr < dst_ptr {
            let src_guard = src_arc.lock().unwrap();
            let mut dst_guard = dst_arc.lock().unwrap();
            let src_view = src_guard
                .try_slice(src_abs as usize..src_abs as usize + byte_len)
                .context("CUDA: copy source out of bounds")?;
            let mut dst_view = dst_guard
                .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                .context("CUDA: copy destination out of bounds")?;
            stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .context("CUDA: device-to-device copy failed")?;
        } else {
            let mut dst_guard = dst_arc.lock().unwrap();
            let src_guard = src_arc.lock().unwrap();
            let src_view = src_guard
                .try_slice(src_abs as usize..src_abs as usize + byte_len)
                .context("CUDA: copy source out of bounds")?;
            let mut dst_view = dst_guard
                .try_slice_mut(dst_abs as usize..dst_abs as usize + byte_len)
                .context("CUDA: copy destination out of bounds")?;
            stream
                .memcpy_dtod(&src_view, &mut dst_view)
                .context("CUDA: device-to-device copy failed")?;
        }
        Ok(())
    }

    fn launch_compute(
        &self,
        stream: &Arc<CudaStream>,
        pipeline: &CudaComputePipeline,
        indices: &[u32],
        user: &[u32],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let launch_args = self.build_launch_args(stream, &pipeline.launch_layout, indices, user)?;
        let cfg = LaunchConfig {
            grid_dim: workgroups,
            block_dim: (
                pipeline.workgroup_size[0],
                pipeline.workgroup_size[1],
                pipeline.workgroup_size[2],
            ),
            shared_mem_bytes: 0,
        };
        // SAFETY: argument order/types match the Slang CUDA entry signature.
        unsafe {
            let mut builder = stream.launch_builder(&pipeline.function);
            for arg in &launch_args {
                match arg {
                    CudaLaunchArg::Buffer(buffer) => {
                        builder.arg(buffer);
                    }
                    CudaLaunchArg::Scalar(word) => {
                        builder.arg(word);
                    }
                }
            }
            builder.launch(cfg).context("CUDA: cuLaunchKernel failed")?;
        }
        Ok(())
    }

    fn launch_layout_buffer_count(launch_layout: &[CudaLaunchArgKind]) -> Option<usize> {
        if launch_layout.is_empty() {
            None
        } else {
            Some(
                launch_layout
                    .iter()
                    .filter(|kind| matches!(kind, CudaLaunchArgKind::Buffer))
                    .count(),
            )
        }
    }

    fn launch_layout_scalar_count(launch_layout: &[CudaLaunchArgKind]) -> usize {
        launch_layout
            .iter()
            .filter(|kind| matches!(kind, CudaLaunchArgKind::Scalar))
            .count()
    }

    fn execute_dispatch_batch(
        &self,
        stream: &Arc<CudaStream>,
        pipeline: &CudaComputePipeline,
        frame_table: Option<&[u32]>,
        arg_data: &[u8],
        count: u32,
    ) -> Result<()> {
        let entry_count = count as usize;
        if entry_count == 0 {
            return Ok(());
        }
        let needed = entry_count
            .checked_mul(DISPATCH_BATCH_STRIDE)
            .context("CUDA: DispatchBatch stride overflow")?;
        anyhow::ensure!(
            arg_data.len() >= needed,
            "CUDA: DispatchBatch arg_data len {} < {} entries × stride {}",
            arg_data.len(),
            entry_count,
            DISPATCH_BATCH_STRIDE
        );

        let mut bases = Vec::with_capacity(entry_count);
        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            bases.push(layout._reserved[dispatch_table_base_word_index()]);
        }

        let n_buffers = match Self::launch_layout_buffer_count(&pipeline.launch_layout) {
            Some(n) => n,
            None => {
                // Plain (non-[goldy_compute]) kernels: infer buffer arity from the
                // contiguous frame-table bases allocated for this batch (count >= 2).
                anyhow::ensure!(
                    entry_count >= 2,
                    "CUDA: DispatchBatch with empty launch layout requires at least 2 entries"
                );
                let delta = bases[1]
                    .checked_sub(bases[0])
                    .context("CUDA: invalid frame-table bases")?;
                for window in bases.windows(2) {
                    anyhow::ensure!(
                        window[1].saturating_sub(window[0]) == delta,
                        "CUDA: DispatchBatch frame-table bases are not uniformly spaced ({bases:?})"
                    );
                }
                delta as usize
            }
        };
        let n_scalars = Self::launch_layout_scalar_count(&pipeline.launch_layout);

        if n_buffers > 0 {
            let table =
                frame_table.context("CUDA: DispatchBatch requires FrameTableStaging when bindings are present")?;
            for (i, &table_base) in bases.iter().enumerate() {
                let start = table_base as usize;
                let end = start
                    .checked_add(n_buffers)
                    .context("CUDA: frame-table range overflow")?;
                anyhow::ensure!(
                    end <= table.len(),
                    "CUDA: DispatchBatch entry {i} frame-table range [{start}, {end}) exceeds staging len {}",
                    table.len()
                );
            }
        } else if n_scalars == 0 && !pipeline.launch_layout.is_empty() {
            // goldy entry with only system-value params — nothing to bind.
        }

        for i in 0..entry_count {
            let base = i * DISPATCH_BATCH_STRIDE;
            let layout: PushLayout = *bytemuck::from_bytes(&arg_data[base..base + TOTAL_PUSH_BYTES]);
            let wg_off = base + TOTAL_PUSH_BYTES;
            let wg_x = u32::from_ne_bytes(arg_data[wg_off..wg_off + 4].try_into().unwrap());
            let wg_y = u32::from_ne_bytes(arg_data[wg_off + 4..wg_off + 8].try_into().unwrap());
            let wg_z = u32::from_ne_bytes(arg_data[wg_off + 8..wg_off + 12].try_into().unwrap());

            let indices: &[u32] = if n_buffers == 0 {
                &[]
            } else {
                let table = frame_table.expect("validated above");
                let start = bases[i] as usize;
                &table[start..start + n_buffers]
            };
            let user = if n_scalars == 0 {
                &[][..]
            } else {
                anyhow::ensure!(
                    n_scalars <= MAX_USER_SLOTS,
                    "CUDA: DispatchBatch entry {i} expects {n_scalars} scalars (max {MAX_USER_SLOTS})"
                );
                &layout.user[..n_scalars]
            };

            self.launch_compute(stream, pipeline, indices, user, (wg_x, wg_y, wg_z))
                .with_context(|| format!("CUDA: DispatchBatch entry {i} launch failed"))?;
        }
        Ok(())
    }

    fn submit_commands(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
    ) -> Result<crate::timeline::TimelineValue> {
        let context = Arc::clone(self.context(ctx)?);
        let device_handle = context.device;
        let stream = Arc::clone(&self.device(device_handle)?.stream);
        let next_timeline = Arc::clone(&self.device(device_handle)?.next_timeline);
        let retired = Arc::clone(&self.device(device_handle)?.retired);

        let mut current_pipeline: Option<ComputePipelineHandle> = None;
        let mut current_indices: Vec<u32> = Vec::new();
        let mut current_user: Vec<u32> = Vec::new();
        let mut frame_table: Option<Arc<[u32]>> = None;

        for command in commands {
            match command {
                GpuCommand::SetPipeline(pipeline) => current_pipeline = Some(*pipeline),
                GpuCommand::BindResourcesRaw { indices, user, .. } => {
                    if user.len() > MAX_USER_SLOTS {
                        anyhow::bail!(
                            "CUDA: at most {MAX_USER_SLOTS} scalar user params per dispatch, got {}",
                            user.len()
                        );
                    }
                    current_indices.clone_from(indices);
                    current_user.clone_from(user);
                }
                GpuCommand::Dispatch {
                    workgroups_x,
                    workgroups_y,
                    workgroups_z,
                    ..
                } => {
                    let pipeline_handle = current_pipeline.context("CUDA: dispatch without a compute pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("CUDA: invalid compute pipeline")?;
                    self.launch_compute(
                        &stream,
                        pipeline,
                        &current_indices,
                        &current_user,
                        (*workgroups_x, *workgroups_y, *workgroups_z),
                    )?;
                }
                GpuCommand::DispatchIndirect { .. } => {
                    anyhow::bail!(
                        "CUDA compute-only PoC does not support indirect dispatch; \
                         use the graphics-companion fallback in the full CUDA backend"
                    )
                }
                GpuCommand::ClearBuffer { buffer, offset, size } => {
                    let buffer = self.buffers.get(buffer).context("CUDA: invalid clear buffer")?;
                    Self::clear_buffer_region(&stream, buffer, *offset, *size)?;
                }
                GpuCommand::WriteBuffer { buffer, offset, data } => {
                    let buffer = self.buffers.get(buffer).context("CUDA: invalid write buffer")?;
                    Self::write_buffer_region(&stream, buffer, *offset, data)?;
                }
                GpuCommand::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    size,
                } => {
                    let src_buf = self.buffers.get(src).context("CUDA: invalid copy source")?.clone_meta();
                    let dst_buf = self
                        .buffers
                        .get(dst)
                        .context("CUDA: invalid copy destination")?
                        .clone_meta();
                    Self::copy_buffer_region(&stream, &src_buf, *src_offset, &dst_buf, *dst_offset, *size)?;
                }
                GpuCommand::FrameTableStaging { data } => {
                    frame_table = Some(Arc::clone(data));
                }
                GpuCommand::ResourceBarrier { .. } => {
                    // Same-stream FIFO ordering is sufficient for the compute-only PoC.
                }
                GpuCommand::DispatchBatch { arg_data, count, .. } => {
                    let pipeline_handle = current_pipeline.context("CUDA: DispatchBatch without a compute pipeline")?;
                    let pipeline = self
                        .compute_pipelines
                        .get(&pipeline_handle)
                        .context("CUDA: invalid compute pipeline")?;
                    self.execute_dispatch_batch(&stream, pipeline, frame_table.as_deref(), arg_data.as_ref(), *count)?;
                }
                GpuCommand::WriteTexture { .. }
                | GpuCommand::WriteTextureRegion { .. }
                | GpuCommand::CopyTexture { .. }
                | GpuCommand::CopyRenderTarget { .. }
                | GpuCommand::CopyBufferToTexture { .. }
                | GpuCommand::CopyTextureToReadback { .. } => {
                    anyhow::bail!("CUDA compute-only backend: texture command is not supported")
                }
            }
        }

        stream.synchronize().context("CUDA: stream synchronize failed")?;

        let value = next_timeline.fetch_add(1, Ordering::AcqRel);
        context.completed.store(value, Ordering::Release);
        retired.fetch_max(value, Ordering::AcqRel);
        context.signal_queue.push_boundary_crossed(value);
        Ok(value)
    }
}

/// Soft clone of buffer metadata + shared allocation (for copy that needs both ends).
impl CudaBuffer {
    fn clone_meta(&self) -> Self {
        Self {
            device: self.device,
            memory: Arc::clone(&self.memory),
            offset: self.offset,
            size: self.size,
            capacity: self.capacity,
            element_stride: self.element_stride,
            slot: self.slot,
            readback: self.readback,
        }
    }
}

fn ensure_cuda_toolkit_on_path() {
    let path = std::env::var_os("PATH").unwrap_or_default();
    let candidates = [
        std::env::var_os("CUDA_PATH")
            .map(PathBuf::from)
            .map(|p| p.join("bin/x64")),
        std::env::var_os("CUDA_PATH").map(PathBuf::from).map(|p| p.join("bin")),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
        )),
        Some(PathBuf::from(
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin",
        )),
        Some(PathBuf::from("/usr/local/cuda/bin")),
    ];
    for cand in candidates.into_iter().flatten() {
        if !cand.is_dir() {
            continue;
        }
        let cand_os = cand.as_os_str();
        if path.to_string_lossy().contains(cand.to_string_lossy().as_ref()) {
            return;
        }
        let mut new_path = cand_os.to_os_string();
        #[cfg(windows)]
        new_path.push(";");
        #[cfg(not(windows))]
        new_path.push(":");
        new_path.push(&path);
        // SAFETY: called before concurrent Slang/NVRTC work in this process for the PoC.
        unsafe { std::env::set_var("PATH", new_path) };
        return;
    }
}

impl GpuBackendSubmitSession for CudaBackend {
    fn clone_context_submit_session(
        &self,
        _ctx: ContextHandle,
        backend: std::sync::Arc<std::sync::Mutex<Box<dyn GpuBackend>>>,
    ) -> std::sync::Arc<dyn ContextSubmitSession> {
        LockedSubmitSession::with_backend_type(backend, BackendType::Cuda)
    }
}

impl GpuBackendTimelineWait for CudaBackend {
    fn take_timeline_submission_epoch_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<submission_worker::SubmissionEpochWait>> {
        Ok(None)
    }

    fn take_timeline_blocking_wait(
        &self,
        _ctx: ContextHandle,
        _value: crate::timeline::TimelineValue,
    ) -> Result<Option<Box<dyn TimelineBlockingWait>>> {
        Ok(None)
    }

    fn finish_timeline_wait(&mut self, ctx: ContextHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        if self.gpu_progress(ctx) < value {
            anyhow::bail!("CUDA: timeline value {value} was not submitted on context {ctx}");
        }
        Ok(())
    }
}

impl GpuBackendPresentSplit for CudaBackend {
    fn take_present_gpu_work(
        &mut self,
        _frame: FrameToken,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<Box<dyn PresentGpuWork>> {
        Self::unsupported("presentation")
    }

    fn finish_present(
        &mut self,
        _finish: PresentFinishState,
        _submit_tv: crate::timeline::TimelineValue,
    ) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("presentation")
    }
}

impl GpuBackend for CudaBackend {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn backend_type(&self) -> BackendType {
        BackendType::Cuda
    }

    fn enumerate_adapters(&self) -> Vec<AdapterInfo> {
        self.adapter_info.clone()
    }

    fn adapter_capabilities(&self, _adapter_id: u32) -> crate::device::DeviceCapabilities {
        crate::device::DeviceCapabilities {
            has_zero_copy_storage_readback: false,
            buffer_resize_cost: BufferResizeCost::Copy,
            buffer_decommit_supported: false,
            host_sidecar_on_submit_worker: false,
            split_compute_partitions_on_barrier_cost: false,
            fuse_upload_with_compute_partitions: true,
            ..crate::device::DeviceCapabilities::default()
        }
    }

    fn create_device(&mut self, adapter_id: u32) -> Result<DeviceHandle> {
        ensure_cuda_toolkit_on_path();
        let ctx = CudaContext::new(adapter_id as usize)
            .with_context(|| format!("CUDA: create device for adapter {adapter_id}"))?;
        let stream = ctx.default_stream();
        let handle = self.next_device;
        self.next_device += 1;
        self.devices.insert(
            handle,
            CudaDevice {
                ctx,
                stream,
                next_timeline: Arc::new(AtomicU64::new(1)),
                retired: Arc::new(AtomicU64::new(0)),
            },
        );
        Ok(handle)
    }

    fn destroy_device(&mut self, device: DeviceHandle) {
        if let Some(gpu) = self.devices.remove(&device) {
            let _ = gpu.stream.synchronize();
        }
        self.contexts.retain(|_, context| context.device != device);
        self.buffers.retain(|_, buffer| buffer.device != device);
        self.shaders.retain(|_, shader| shader.device != device);
        self.compute_pipelines.retain(|_, pipeline| pipeline.device != device);
        self.buffer_slots.retain(|_, handle| self.buffers.contains_key(handle));
    }

    fn is_device_valid(&self, device: DeviceHandle) -> bool {
        self.devices.contains_key(&device)
    }

    fn device_wait_idle(&mut self, device: DeviceHandle) -> Result<()> {
        self.device(device)?
            .stream
            .synchronize()
            .context("CUDA: device wait idle failed")
    }

    fn create_context(&mut self, device: DeviceHandle) -> Result<ContextHandle> {
        self.device(device)?;
        if self.contexts.values().any(|context| context.device == device) {
            anyhow::bail!("CUDA prototype supports one submission context per device");
        }
        let handle = self.next_context;
        self.next_context += 1;
        self.contexts.insert(
            handle,
            Arc::new(CudaSubmitContext {
                device,
                completed: AtomicU64::new(0),
                signal_queue: crate::signal::SignalQueue::new(),
            }),
        );
        Ok(handle)
    }

    fn detach_context_for_destroy(&mut self, ctx: ContextHandle) -> Option<Box<dyn ContextDestroyHandle>> {
        self.contexts.remove(&ctx)?;
        Some(Box::new(CudaDestroyContext))
    }

    fn clone_context_deletion_flush(
        &self,
        ctx: ContextHandle,
    ) -> Option<std::sync::Arc<dyn ContextDeferredDeletionFlush>> {
        self.contexts
            .contains_key(&ctx)
            .then(|| Arc::new(NoOpDeferredDeletionFlush) as Arc<dyn ContextDeferredDeletionFlush>)
    }

    fn clone_context_gpu_progress(&self, ctx: ContextHandle) -> Option<std::sync::Arc<dyn ContextGpuProgress>> {
        Some(Arc::new(CudaProgress {
            context: Arc::clone(self.contexts.get(&ctx)?),
        }))
    }

    fn context_device(&self, ctx: ContextHandle) -> DeviceHandle {
        self.contexts.get(&ctx).map(|context| context.device).unwrap_or(0)
    }

    fn create_buffer(
        &mut self,
        device: DeviceHandle,
        size: u64,
        _access: BufferKind,
        element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<BufferHandle> {
        self.create_storage_buffer(device, size, size, element_stride)
    }

    fn create_buffer_with_capacity(
        &mut self,
        device: DeviceHandle,
        initial_size: u64,
        capacity: u64,
        _access: BufferKind,
        element_stride: Option<u32>,
        _flags: BufferFlags,
    ) -> Result<(BufferHandle, u64)> {
        let capacity = capacity.max(initial_size);
        Ok((
            self.create_storage_buffer(device, initial_size, capacity, element_stride)?,
            capacity,
        ))
    }

    fn destroy_buffer(&mut self, buffer: BufferHandle) {
        if let Some(buffer) = self.buffers.remove(&buffer) {
            if let Some(slot) = buffer.slot {
                self.buffer_slots.remove(&slot);
            }
        }
    }

    fn write_buffer(&mut self, buffer: BufferHandle, offset: u64, data: &[u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
        let stream = Arc::clone(&self.device(buffer.device)?.stream);
        Self::write_buffer_region(&stream, buffer, offset, data)
    }

    fn alloc_readback_buffer(&mut self, device: DeviceHandle, size: u64) -> Result<BufferHandle> {
        let gpu = self.device(device)?;
        let capacity = size.max(4);
        let memory = Arc::new(Mutex::new(
            gpu.stream
                .alloc_zeros::<u8>(capacity as usize)
                .context("CUDA: alloc readback")?,
        ));
        let handle = self.next_buffer;
        self.next_buffer += 1;
        self.buffers.insert(
            handle,
            CudaBuffer {
                device,
                memory,
                offset: 0,
                size,
                capacity,
                element_stride: None,
                slot: None,
                readback: true,
            },
        );
        Ok(handle)
    }

    fn read_readback_buffer(&self, buffer: BufferHandle, output: &mut [u8]) -> Result<()> {
        let buffer = self.buffers.get(&buffer).context("CUDA: invalid readback buffer")?;
        if !buffer.readback {
            anyhow::bail!("CUDA: buffer is not readback staging");
        }
        if output.len() as u64 > buffer.size {
            anyhow::bail!("CUDA: read exceeds readback buffer size");
        }
        let stream = Arc::clone(&self.device(buffer.device)?.stream);
        let memory = buffer.memory.lock().unwrap();
        let view = memory
            .try_slice(buffer.offset as usize..(buffer.offset as usize + output.len()))
            .context("CUDA: readback range out of bounds")?;
        stream.memcpy_dtoh(&view, output).context("CUDA: DtoH readback failed")
    }

    fn free_readback_buffer(&mut self, buffer: BufferHandle) {
        self.destroy_buffer(buffer);
    }

    fn query_texture_copy_footprint(
        &self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
    ) -> Result<TextureCopyFootprint> {
        Self::unsupported("texture readback")
    }

    fn alloc_texture_readback_staging(
        &mut self,
        _device: DeviceHandle,
        _layout: TextureCopyFootprint,
    ) -> Result<BufferHandle> {
        Self::unsupported("texture readback")
    }

    fn read_texture_readback_staging(
        &self,
        _buffer: BufferHandle,
        _layout: TextureCopyFootprint,
        _output: &mut [u8],
    ) -> Result<()> {
        Self::unsupported("texture readback")
    }

    fn texture_copy_retention_tag(&self, _texture: TextureHandle) -> u64 {
        0
    }

    fn clear_buffer(&mut self, device: DeviceHandle, buffer: BufferHandle, offset: u64, size: u64) -> Result<()> {
        let stream = Arc::clone(&self.device(device)?.stream);
        let target = self.buffers.get(&buffer).context("CUDA: invalid buffer handle")?;
        Self::clear_buffer_region(&stream, target, offset, size)
    }

    fn buffer_size(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.size).unwrap_or(0)
    }

    fn buffer_capacity(&self, buffer: BufferHandle) -> u64 {
        self.buffers.get(&buffer).map(|buffer| buffer.capacity).unwrap_or(0)
    }

    fn set_buffer_logical_size(
        &mut self,
        _device: DeviceHandle,
        buffer: BufferHandle,
        new_logical_size: u64,
    ) -> Result<()> {
        let buffer = self.buffers.get_mut(&buffer).context("CUDA: invalid buffer handle")?;
        if new_logical_size == 0 || new_logical_size > buffer.capacity {
            anyhow::bail!("CUDA: logical size must be in 1..=capacity");
        }
        buffer.size = new_logical_size;
        Ok(())
    }

    fn buffer_bindless_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffers.get(&buffer)?.slot
    }

    fn buffer_bindless_srv_index(&self, buffer: BufferHandle) -> Option<u32> {
        self.buffer_bindless_index(buffer)
    }

    fn create_buffer_view(
        &mut self,
        parent: BufferHandle,
        offset: u64,
        size: u64,
        element_stride: Option<u32>,
    ) -> Result<BufferHandle> {
        let parent = self
            .buffers
            .get(&parent)
            .context("CUDA: invalid parent buffer")?
            .clone_meta();
        if offset + size > parent.size {
            anyhow::bail!("CUDA: buffer view exceeds parent");
        }
        let handle = self.next_buffer;
        self.next_buffer += 1;
        let slot = self.next_slot;
        self.next_slot += 1;
        self.buffer_slots.insert(slot, handle);
        self.buffers.insert(
            handle,
            CudaBuffer {
                device: parent.device,
                memory: parent.memory,
                offset: parent.offset + offset,
                size,
                capacity: size,
                element_stride: element_stride.or(parent.element_stride),
                slot: Some(slot),
                readback: false,
            },
        );
        Ok(handle)
    }

    fn resize_buffer(
        &mut self,
        device: DeviceHandle,
        buffer: BufferHandle,
        new_size: u64,
        preserve_contents: bool,
    ) -> Result<()> {
        let old = self
            .buffers
            .get(&buffer)
            .context("CUDA: invalid buffer handle")?
            .clone_meta();
        if old.device != device {
            anyhow::bail!("CUDA: buffer belongs to another device");
        }
        let stream = Arc::clone(&self.device(device)?.stream);
        let capacity = new_size.max(4);
        let mut replacement = stream
            .alloc_zeros::<u8>(capacity as usize)
            .context("CUDA: resize alloc")?;
        if preserve_contents {
            let copy_size = old.size.min(new_size);
            if copy_size > 0 {
                let memory = old.memory.lock().unwrap();
                let src = memory
                    .try_slice(old.offset as usize..(old.offset + copy_size) as usize)
                    .context("CUDA: resize src")?;
                let mut dst = replacement
                    .try_slice_mut(0..copy_size as usize)
                    .context("CUDA: resize dst")?;
                stream
                    .memcpy_dtod(&src, &mut dst)
                    .context("CUDA: resize device-to-device copy")?;
            }
        }
        let target = self.buffers.get_mut(&buffer).expect("validated above");
        target.memory = Arc::new(Mutex::new(replacement));
        target.offset = 0;
        target.size = new_size;
        target.capacity = capacity;
        Ok(())
    }

    fn create_shader_with_paths(
        &mut self,
        device: DeviceHandle,
        slang_source: &str,
        search_paths: &[&str],
        defines: &[(&str, &str)],
        optimization_level: crate::types::OptimizationLevel,
    ) -> Result<ShaderHandle> {
        self.device(device)?;
        let handle = self.next_shader;
        self.next_shader += 1;
        self.shaders.insert(
            handle,
            CudaShader {
                device,
                source: slang_source.to_owned(),
                search_paths: search_paths.iter().map(|value| (*value).to_owned()).collect(),
                defines: defines
                    .iter()
                    .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                    .collect(),
                optimization_level,
            },
        );
        Ok(handle)
    }

    fn destroy_shader(&mut self, shader: ShaderHandle) {
        self.shaders.remove(&shader);
    }

    fn create_pipeline(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    fn destroy_pipeline(&mut self, _pipeline: PipelineHandle) {}

    fn create_pipeline_with_depth(
        &mut self,
        _device: DeviceHandle,
        _vertex_shader: ShaderHandle,
        _fragment_shader: ShaderHandle,
        _vertex_layout: &VertexBufferLayout,
        _topology: PrimitiveTopology,
        _target_format: TextureFormat,
        _depth_stencil: Option<&DepthStencilState>,
    ) -> Result<PipelineHandle> {
        Self::unsupported("graphics pipelines")
    }

    fn create_render_target_with_depth(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _color_format: TextureFormat,
        _depth_format: Option<DepthFormat>,
    ) -> Result<RenderTargetHandle> {
        Self::unsupported("render targets")
    }

    fn render_to_target(
        &mut self,
        _device: DeviceHandle,
        _target: RenderTargetHandle,
        _color_load: crate::types::TargetLoad,
        _commands: &[RenderCommand],
    ) -> Result<()> {
        Self::unsupported("rendering")
    }

    fn create_texture(
        &mut self,
        _device: DeviceHandle,
        _width: u32,
        _height: u32,
        _format: TextureFormat,
        _access: TextureKind,
        _flags: TextureFlags,
    ) -> Result<TextureHandle> {
        Self::unsupported("textures")
    }

    fn write_texture(&mut self, _texture: TextureHandle, _data: &[u8], _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("textures")
    }

    fn write_texture_region(
        &mut self,
        _texture: TextureHandle,
        _x: u32,
        _y: u32,
        _width: u32,
        _height: u32,
        _data: &[u8],
    ) -> Result<()> {
        Self::unsupported("textures")
    }

    fn destroy_texture(&mut self, _texture: TextureHandle) {}

    fn texture_bindless_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn texture_bindless_sampled_index(&self, _texture: TextureHandle) -> Option<u32> {
        None
    }

    fn create_sampler(&mut self, _device: DeviceHandle, _desc: &SamplerDesc) -> Result<SamplerHandle> {
        Self::unsupported("samplers")
    }

    fn destroy_sampler(&mut self, _sampler: SamplerHandle) {}

    fn sampler_bindless_index(&self, _sampler: SamplerHandle) -> Option<u32> {
        None
    }

    fn create_surface(
        &mut self,
        _device: DeviceHandle,
        _window: &dyn raw_window_handle::HasWindowHandle,
        _display: &dyn raw_window_handle::HasDisplayHandle,
        _depth_format: Option<DepthFormat>,
    ) -> Result<SurfaceHandle> {
        Self::unsupported("surfaces")
    }

    fn destroy_surface(&mut self, _surface: SurfaceHandle) {}

    fn surface_resize(&mut self, _surface: SurfaceHandle, _width: u32, _height: u32) -> Result<()> {
        Self::unsupported("surfaces")
    }

    fn surface_size(&self, _surface: SurfaceHandle) -> (u32, u32) {
        (0, 0)
    }

    fn surface_format(&self, _surface: SurfaceHandle) -> TextureFormat {
        TextureFormat::Bgra8UnormSrgb
    }

    fn gpu_progress(&self, ctx: ContextHandle) -> crate::timeline::TimelineValue {
        self.contexts
            .get(&ctx)
            .map(|context| context.completed.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    fn device_timeline_retired(&self, device: DeviceHandle) -> crate::timeline::TimelineValue {
        self.devices
            .get(&device)
            .map(|device| device.retired.load(Ordering::Acquire))
            .unwrap_or(0)
    }

    fn device_wait_until(&mut self, device: DeviceHandle, value: crate::timeline::TimelineValue) -> Result<()> {
        self.device_wait_idle(device)?;
        if self.device_timeline_retired(device) < value {
            anyhow::bail!("CUDA: timeline value {value} has not been submitted");
        }
        Ok(())
    }

    fn poll_signals(
        &mut self,
        ctx: ContextHandle,
        _progress: crate::timeline::TimelineValue,
    ) -> Vec<crate::signal::QueuedSignal> {
        self.contexts
            .get(&ctx)
            .map(|context| crate::signal::drain_all_queued_signals(&context.signal_queue))
            .unwrap_or_default()
    }

    fn submit_standalone(
        &mut self,
        ctx: ContextHandle,
        commands: &[GpuCommand],
        sync: Option<&SubmitSync>,
    ) -> Result<crate::timeline::TimelineValue> {
        if let Some(sync) = sync {
            for epoch in sync
                .waits
                .iter()
                .chain(sync.cpu_waits.iter())
                .chain(sync.host_observed_waits.iter())
            {
                self.device_wait_until(self.context_device(ctx), epoch.value)?;
            }
            for write in &sync.deferred_host_writes {
                self.write_buffer(write.buffer, write.offset, &write.data)?;
            }
        }
        let effective = commands_with_sync_prologue(commands, sync);
        self.submit_commands(ctx, &effective)
    }

    fn begin_frame(&mut self, _surface: SurfaceHandle, _ctx: ContextHandle) -> Result<(FrameToken, TextureHandle)> {
        Self::unsupported("frames")
    }

    fn submit_frame(&mut self, _frame: &FrameToken) -> Result<crate::timeline::TimelineValue> {
        Self::unsupported("frames")
    }

    fn create_compute_pipeline(
        &mut self,
        device: DeviceHandle,
        compute_shader: ShaderHandle,
        _debug_name: Option<&str>,
    ) -> Result<ComputePipelineHandle> {
        let shader = self
            .shaders
            .get(&compute_shader)
            .context("CUDA: invalid shader handle")?;
        if shader.device != device {
            anyhow::bail!("CUDA: shader belongs to another device");
        }
        let (ptx, slot_access, workgroup_size, launch_layout) = self.compile_compute_ptx(shader)?;
        let gpu = self.device(device)?;
        let module = gpu
            .ctx
            .load_module(Ptx::from_src(ptx))
            .context("CUDA: cuModuleLoadData failed")?;
        let function = module
            .load_function("cs_main")
            .context("CUDA: cuModuleGetFunction(cs_main) failed")?;
        let handle = self.next_compute_pipeline;
        self.next_compute_pipeline += 1;
        self.compute_pipelines.insert(
            handle,
            CudaComputePipeline {
                device,
                module,
                function,
                workgroup_size,
                slot_access,
                launch_layout,
            },
        );
        Ok(handle)
    }

    fn destroy_compute_pipeline(&mut self, pipeline: ComputePipelineHandle) {
        self.compute_pipelines.remove(&pipeline);
    }

    fn compute_pipeline_slot_access(&self, pipeline: ComputePipelineHandle) -> Vec<Option<ResourceAccess>> {
        self.compute_pipelines
            .get(&pipeline)
            .map(|pipeline| pipeline.slot_access.clone())
            .unwrap_or_default()
    }

    fn max_bindless_slots_per_category(&self, _device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        if matches!(
            category,
            crate::types::ResourceCategory::Scattered | crate::types::ResourceCategory::Broadcast
        ) {
            4096
        } else {
            0
        }
    }

    fn available_bindless_slots(&self, device: DeviceHandle, category: crate::types::ResourceCategory) -> u32 {
        self.max_bindless_slots_per_category(device, category).saturating_sub(
            self.buffers
                .values()
                .filter(|buffer| buffer.device == device && buffer.slot.is_some())
                .count() as u32,
        )
    }

    fn max_submission_contexts(&self, _device: DeviceHandle) -> u32 {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DOUBLE_SLANG: &str = r#"
[shader("compute")]
[numthreads(1, 1, 1)]
void cs_main(uniform RWStructuredBuffer<uint> values, uint3 id : SV_DispatchThreadID) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * 2;
}
"#;

    const DOUBLE_GOLDY_TWO_BUFFER_SLANG: &str = r#"
import goldy_exp;

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(BufRO<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

    fn run_compute_dispatch_and_readback(shader_source: &str) -> Result<()> {
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA compute test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))?;
        let shader = backend.create_shader_with_paths(
            device,
            shader_source,
            &[],
            &[],
            crate::types::OptimizationLevel::Default,
        )?;
        let pipeline = backend.create_compute_pipeline(device, shader, Some("double"))?;
        let slot = backend.buffer_bindless_index(buffer).context("missing registry key")?;
        let submitted = backend.submit_standalone(
            ctx,
            &[
                GpuCommand::SetPipeline(pipeline),
                GpuCommand::BindResourcesRaw {
                    indices: vec![slot],
                    user: vec![],
                    frame_table_base: 0,
                },
                GpuCommand::Dispatch {
                    label: Some("double"),
                    workgroups_x: 4,
                    workgroups_y: 1,
                    workgroups_z: 1,
                },
            ],
            None,
        )?;
        assert_eq!(backend.gpu_progress(ctx), submitted);

        let readback = backend.alloc_readback_buffer(device, 16)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 16,
            }],
            None,
        )?;
        let mut bytes = [0u8; 16];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn slang_compute_dispatch_and_readback() -> Result<()> {
        run_compute_dispatch_and_readback(DOUBLE_SLANG)
    }

    fn run_scheme_compute_and_withdraw(shader_source: &str) -> Result<()> {
        let backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let buffer = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, shader_source)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&buffer, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buffer)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    #[test]
    fn scheme_dispatches_goldy_virtual_compute_and_withdraws() -> Result<()> {
        run_scheme_compute_and_withdraw(DOUBLE_GOLDY_SLANG)
    }

    #[test]
    fn scheme_binds_two_goldy_buffers_in_parameter_order() -> Result<()> {
        let backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                return Ok(());
            }
        };
        let device = Arc::new(crate::Device::from_backend(Box::new(backend))?);
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let input = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let output = pool.acquire_buffer_sized::<u32>(4, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, DOUBLE_GOLDY_TWO_BUFFER_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &pipeline)
            .with_parcel(&input, crate::NodeAccess::Read)
            .with_parcel(&output, crate::NodeAccess::Write)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &output)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[2, 4, 6, 8]);
        Ok(())
    }

    fn try_cuda_device() -> Result<Option<Arc<crate::Device>>> {
        match CudaBackend::new() {
            Ok(backend) => Ok(Some(Arc::new(crate::Device::from_backend(Box::new(backend))?))),
            Err(error) => {
                eprintln!("skipping CUDA scheme test: {error:#}");
                Ok(None)
            }
        }
    }

    const WITH_PARAM_UINT_SLANG: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint value, ThreadId id) {
    out[0] = value;
}
"#;

    #[test]
    fn scheme_with_param_uint_roundtrip() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const EXPECTED: u32 = 42;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_uint", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(EXPECTED)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[EXPECTED]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_uint_zero() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let out = pool.acquire_buffer_with_data(&[0xDEAD_BEEFu32], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_zero", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(0u32)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[0]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_uint_max() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, WITH_PARAM_UINT_SLANG)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_max", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(u32::MAX)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[u32::MAX]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_float_reinterpret() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let out = pool.acquire_buffer_sized::<u32>(1, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<float> out, float value, ThreadId id) {
    out[0] = value;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        #[allow(clippy::approx_constant)]
        let value: f32 = 3.14159;
        let bits = value.to_bits();

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_float", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(bits)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[bits]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_two_scalars() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let out = pool.acquire_buffer_sized::<u32>(2, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Scattered<uint> out, uint a, uint b, ThreadId id) {
    out[0] = a;
    out[1] = b;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const A: u32 = 0xABCD;
        const B: u32 = 0x1234;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_two", &pipeline)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(A)
            .with_param(B)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[A, B]);
        Ok(())
    }

    #[test]
    fn scheme_with_param_after_two_buffers() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        const N: usize = 64;
        let input: Vec<u32> = (0..N as u32).collect();
        let inp = pool.acquire_buffer_with_data(&input, BufferKind::Scattered)?;
        let out = pool.acquire_buffer_sized::<u32>(N as u64, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> inp, Scattered<uint> out, uint offset, ThreadId id) {
    out[id.x] = inp[id.x] + offset;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        const OFFSET: u32 = 100;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("uniform_offset", &pipeline)
            .with_parcel(&inp, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .with_param(OFFSET)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let expected: Vec<u32> = input.iter().map(|v| v + OFFSET).collect();
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), expected.as_slice());
        Ok(())
    }

    #[test]
    fn scheme_broadcast_parcel_struct_mul() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            mul: u32,
        }
        impl crate::StructuredBufferElement for Params {}
        let cfg = pool.acquire_buffer_with_data(&[Params { mul: 3 }], BufferKind::Broadcast)?;
        let values = pool.acquire_buffer_with_data(&[1u32, 2, 3, 4], BufferKind::Scattered)?;
        let shader = crate::ShaderModule::from_slang(
            &device,
            r#"
import goldy_exp;

struct Params { uint mul; };

[goldy_compute]
[numthreads(1, 1, 1)]
void cs_main(Params cfg, Scattered<uint> values, ThreadId id) {
    values[id.x] = values[id.x] * cfg.mul;
}
"#,
        )?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("broadcast_mul", &pipeline)
            .with_parcel(&cfg, crate::NodeAccess::Read)
            .with_parcel(&values, crate::NodeAccess::ReadWrite)
            .dispatch(4, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &values)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[3, 6, 9, 12]);
        Ok(())
    }

    #[test]
    fn slang_emits_ptx_for_compute() -> Result<()> {
        ensure_cuda_toolkit_on_path();
        let compiler = match crate::slang::SlangCompiler::new() {
            Ok(compiler) => compiler,
            Err(error) => {
                eprintln!("skipping CUDA PTX emission test: {error:#}");
                return Ok(());
            }
        };
        let compiled = match compiler.compile_bindless_with_reflection(
            DOUBLE_SLANG,
            crate::slang::ShaderTarget::Ptx,
            &[("cs_main", crate::slang::SlangStage::Compute)],
            &[],
        ) {
            Ok(compiled) => compiled,
            Err(error) => {
                eprintln!("skipping CUDA PTX emission test (Slang/NVRTC): {error:#}");
                return Ok(());
            }
        };
        let ptx = compiled.shader.as_str().context("expected text PTX")?;
        assert!(
            ptx.contains(".entry") || ptx.contains("cs_main"),
            "Slang output did not look like PTX:\n{ptx}"
        );
        Ok(())
    }

    #[test]
    fn ptx_cache_key_differs_from_wgsl() {
        use crate::shader_cache::compile_cache_key;
        use crate::slang::{ffi::SlangStage, ShaderTarget};
        use crate::types::OptimizationLevel;

        let src = "void cs_main() {}";
        let eps = [("cs_main", SlangStage::Compute)];
        let ptx = compile_cache_key(src, ShaderTarget::Ptx, &eps, &[], &[], &[], OptimizationLevel::Default);
        let wgsl = compile_cache_key(src, ShaderTarget::Wgsl, &eps, &[], &[], &[], OptimizationLevel::Default);
        assert_ne!(ptx, wgsl);
    }

    // ─── M2: multi-node command coverage ───────────────────────────────────

    const M2_FILL_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, uint value, ThreadId id) {
    data[id.x] = value;
}
"#;

    const M2_DOUBLE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x] * 2;
}
"#;

    const M2_ADD_TEN_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] + 10;
}
"#;

    const M2_IN_PLACE_DOUBLE_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = data[id.x] * 2;
}
"#;

    const M2_COPY_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> input, Scattered<uint> output, ThreadId id) {
    output[id.x] = input[id.x];
}
"#;

    const M2_SUM_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> a, Scattered<uint> b, Scattered<uint> out, ThreadId id) {
    out[id.x] = a[id.x] + b[id.x];
}
"#;

    const M2_FILL_INDEX_SHADER: &str = r#"
import goldy_exp;
[goldy_compute]
[numthreads(64, 1, 1)]
void cs_main(Scattered<uint> data, ThreadId id) {
    data[id.x] = id.x;
}
"#;

    #[test]
    fn scheme_same_pipeline_batch_two_fills() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let a = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let b = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let shader = crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?;
        let pipeline = crate::ComputePipeline::new(&device, &shader)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_a", &pipeline)
            .with_parcel(&a, crate::NodeAccess::Write)
            .with_param(7u32)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &pipeline)
            .with_parcel(&b, crate::NodeAccess::Write)
            .with_param(9u32)
            .dispatch(1, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        let bytes_a = withdraw_a.claim(&mut submission)?.consume()?;
        let bytes_b = withdraw_b.claim(&mut submission)?.consume()?;
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes_a).iter().all(|&v| v == 7));
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes_b).iter().all(|&v| v == 9));
        Ok(())
    }

    #[test]
    fn scheme_graph_linear_chain() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let src = pool.acquire_buffer_with_data(&(0..64u32).collect::<Vec<_>>(), BufferKind::Scattered)?;
        let dst = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let double_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_DOUBLE_SHADER)?)?;
        let add_pipe =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_ADD_TEN_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double", &double_pipe)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&dst, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("add_ten", &add_pipe)
            .with_parcel(&dst, crate::NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &dst)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, (i as u32) * 2 + 10, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_graph_independent_dispatches() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let a = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let b = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        // Distinct pipeline objects so analysis emits two Dispatch commands, not DispatchBatch.
        let fill_a = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?)?;
        let fill_b = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_FILL_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_a", &fill_a)
            .with_parcel(&a, crate::NodeAccess::Write)
            .with_param(42u32)
            .dispatch(1, 1, 1);
        scheme
            .node("fill_b", &fill_b)
            .with_parcel(&b, crate::NodeAccess::Write)
            .with_param(99u32)
            .dispatch(1, 1, 1);
        let withdraw_a = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &a)?;
        let withdraw_b = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &b)?;
        let mut submission = scheme.submit()?;
        assert!(
            bytemuck::cast_slice::<u8, u32>(&withdraw_a.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 42)
        );
        assert!(
            bytemuck::cast_slice::<u8, u32>(&withdraw_b.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 99)
        );
        Ok(())
    }

    #[test]
    fn scheme_graph_diamond_dependency() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let src = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let y = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let z = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let out = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let fill = crate::ComputePipeline::new(
            &device,
            &crate::ShaderModule::from_slang(&device, M2_FILL_INDEX_SHADER)?,
        )?;
        let double =
            crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_DOUBLE_SHADER)?)?;
        let sum = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_SUM_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("fill_src", &fill)
            .with_parcel(&src, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_y", &double)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&y, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("double_to_z", &double)
            .with_parcel(&src, crate::NodeAccess::Read)
            .with_parcel(&z, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        scheme
            .node("sum_yz", &sum)
            .with_parcel(&y, crate::NodeAccess::Read)
            .with_parcel(&z, crate::NodeAccess::Read)
            .with_parcel(&out, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &out)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, (i as u32) * 4, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_clear_then_dispatch() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let buf = pool.acquire_buffer_with_data(&vec![0xDEAD_BEEFu32; 64], BufferKind::Scattered)?;
        let pipe = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_ADD_TEN_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 0, 64 * 4)?;
        scheme
            .node("add_ten", &pipe)
            .with_parcel(&buf, crate::NodeAccess::ReadWrite)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buf)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert!(bytemuck::cast_slice::<u8, u32>(&bytes).iter().all(|&v| v == 10));
        Ok(())
    }

    #[test]
    fn scheme_write_copy_dispatch_chain() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let src = pool.acquire_buffer_with_data(&(0..64u32).collect::<Vec<_>>(), BufferKind::Scattered)?;
        let mid = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let dst = pool.acquire_buffer_sized::<u32>(64, BufferKind::Scattered, BufferFlags::empty())?;
        let copy = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_COPY_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme.copy_buffer_parcel(&src, 0, &mid, 0, 64 * 4)?;
        scheme
            .node("copy_mid_to_dst", &copy)
            .with_parcel(&mid, crate::NodeAccess::Read)
            .with_parcel(&dst, crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &dst)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&bytes).iter().enumerate() {
            assert_eq!(val, i as u32, "element {i}");
        }
        Ok(())
    }

    #[test]
    fn scheme_buffer_view_copy_and_isolation() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        const N: usize = 64;
        let src: Vec<u32> = (1..=N as u32).collect();
        let dst = vec![0u32; N];
        let cells = pool.acquire_record([
            crate::ordinal(crate::Init::data(&src)),
            crate::ordinal(crate::Init::data(&dst)),
        ])?;
        let copy = crate::ComputePipeline::new(&device, &crate::ShaderModule::from_slang(&device, M2_COPY_SHADER)?)?;

        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("copy_fields", &copy)
            .with_parcel(&cells[0], crate::NodeAccess::Read)
            .with_parcel(&cells[1], crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[1])?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), src.as_slice());

        // Isolation: doubling one field must leave the sibling untouched.
        let sentinel = vec![100u32; N];
        let work: Vec<u32> = (1..=N as u32).collect();
        let cells = pool.acquire_record([
            crate::ordinal(crate::Init::data(&sentinel)),
            crate::ordinal(crate::Init::data(&work)),
        ])?;
        let double = crate::ComputePipeline::new(
            &device,
            &crate::ShaderModule::from_slang(&device, M2_IN_PLACE_DOUBLE_SHADER)?,
        )?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme
            .node("double_work", &double)
            .with_parcel(&cells[1], crate::NodeAccess::Write)
            .dispatch(1, 1, 1);
        let grant_sentinel = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[0])?;
        let grant_work = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &cells[1])?;
        let mut submission = scheme.submit()?;
        assert!(
            bytemuck::cast_slice::<u8, u32>(&grant_sentinel.claim(&mut submission)?.consume()?)
                .iter()
                .all(|&v| v == 100)
        );
        for (i, &val) in bytemuck::cast_slice::<u8, u32>(&grant_work.claim(&mut submission)?.consume()?)
            .iter()
            .enumerate()
        {
            assert_eq!(val, (i as u32 + 1) * 2, "work[{i}]");
        }
        Ok(())
    }

    #[test]
    fn overlapping_self_copy_is_memmove_safe() -> Result<()> {
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA overlapping copy test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            32,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        // [1,2,3,4,5,6,7,8] → copy first 16 bytes onto offset 8 (overlap).
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4, 5, 6, 7, 8]))?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: buffer,
                dst_offset: 8,
                size: 16,
            }],
            None,
        )?;
        let readback = backend.alloc_readback_buffer(device, 32)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 32,
            }],
            None,
        )?;
        let mut bytes = [0u8; 32];
        backend.read_readback_buffer(readback, &mut bytes)?;
        assert_eq!(bytemuck::cast_slice::<u8, u32>(&bytes), &[1, 2, 1, 2, 3, 4, 7, 8]);
        Ok(())
    }

    #[test]
    fn resize_buffer_preserves_contents_on_device() -> Result<()> {
        let mut backend = match CudaBackend::new() {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping CUDA resize test: {error:#}");
                return Ok(());
            }
        };
        let device = backend.create_device(0)?;
        let ctx = backend.create_context(device)?;
        let buffer = backend.create_buffer(
            device,
            16,
            BufferKind::Scattered,
            Some(4),
            BufferFlags::COPY_SRC | BufferFlags::COPY_DST,
        )?;
        backend.write_buffer(buffer, 0, bytemuck::cast_slice(&[10u32, 20, 30, 40]))?;
        backend.resize_buffer(device, buffer, 32, true)?;
        assert_eq!(backend.buffer_size(buffer), 32);
        assert!(backend.buffer_capacity(buffer) >= 32);

        let readback = backend.alloc_readback_buffer(device, 32)?;
        backend.submit_standalone(
            ctx,
            &[GpuCommand::CopyBuffer {
                src: buffer,
                src_offset: 0,
                dst: readback,
                dst_offset: 0,
                size: 32,
            }],
            None,
        )?;
        let mut bytes = [0u8; 32];
        backend.read_readback_buffer(readback, &mut bytes)?;
        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
        assert_eq!(&words[..4], &[10, 20, 30, 40]);
        // Newly grown tail is zero-filled by alloc_zeros.
        assert_eq!(&words[4..], &[0, 0, 0, 0]);
        Ok(())
    }

    #[test]
    fn scheme_clear_parcel_partial_preserves_edges() -> Result<()> {
        let Some(device) = try_cuda_device()? else {
            return Ok(());
        };
        let ctx = device.create_context()?;
        const N: usize = 64;
        let mut init = Vec::with_capacity(N);
        for i in 0..N {
            if i < 16 {
                init.push(0xAAAA_AAAAu32);
            } else if i < 48 {
                init.push(0xBBBB_BBBBu32);
            } else {
                init.push(0xCCCC_CCCCu32);
            }
        }
        let mut pool = crate::RetainedPool::new(Arc::clone(&device));
        let buf = pool.acquire_buffer_with_data(&init, BufferKind::Scattered)?;
        let mut scheme = crate::Scheme::new(&ctx);
        scheme.clear_parcel(&buf, 16 * 4, 32 * 4)?;
        let withdraw = crate::MemoryExchange::new(&ctx).bind_withdraw(&mut scheme, &buf)?;
        let mut submission = scheme.submit()?;
        let bytes = withdraw.claim(&mut submission)?.consume()?;
        let words = bytemuck::cast_slice::<u8, u32>(&bytes);
        assert!(words[..16].iter().all(|&v| v == 0xAAAA_AAAA));
        assert!(words[16..48].iter().all(|&v| v == 0));
        assert!(words[48..].iter().all(|&v| v == 0xCCCC_CCCC));
        Ok(())
    }
}
