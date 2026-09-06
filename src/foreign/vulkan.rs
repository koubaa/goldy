//! Vulkan as a foreign graphics object: no Goldy device, verbs under one lock.
//!
//! Creates its own `vkInstance` / `vkDevice` (serialised with
//! [`crate::backend::vulkan::VK_INSTANCE_LOCK`]). Offscreen surfaces hold a
//! transfer-dst image plus persistently mapped staging. [`ForeignSurface::blit`]
//! copies host pixels through `vkCmdCopyBufferToImage` and a copy-back into a
//! host-visible readback so [`ForeignSurface::snapshot`] can assert GPU contents.
//!
//! Windowed WSI (`vkCreateSwapchainKHR` against a raw window) is a later verb
//! on this same singleton.

use crate::backend::vulkan::{find_memory_type, format_to_vk, VK_INSTANCE_LOCK};
use crate::pixel::{PixelSink, PixmapLayout};
use crate::types::TextureFormat;
use crate::GoldyError;
use ash::vk;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::{Arc, Mutex, OnceLock};

/// Process-wide Vulkan adapter. Lazily created on [`try_adapter`].
pub struct ForeignVulkan {
    state: Mutex<AdapterState>,
}

struct AdapterState {
    _entry: ash::Entry,
    instance: ash::Instance,
    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family: u32,
    next_id: u32,
    surfaces: HashMap<u32, SurfaceSlot>,
}

struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: usize,
}

// Mapped host pointers are only used while `AdapterState` is locked.
// SAFETY: `HostBuffer` is stored only in that mutex; the raw pointer is never
// sent without the lock that owns the mapping.
unsafe impl Send for HostBuffer {}

struct DeviceImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
}

struct SurfaceSlot {
    width: u32,
    height: u32,
    format: TextureFormat,
    generation: u64,
    image: DeviceImage,
    staging: HostBuffer,
    readback: HostBuffer,
    cmd_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    dropped: bool,
}

struct SurfaceHandle {
    adapter: Arc<ForeignVulkan>,
    id: u32,
}

impl Drop for SurfaceHandle {
    fn drop(&mut self) {
        self.adapter.release(self.id);
    }
}

/// Offscreen Vulkan image owned by the foreign singleton.
#[derive(Clone)]
pub struct ForeignSurface {
    inner: Arc<SurfaceHandle>,
}

static ADAPTER: OnceLock<Result<Arc<ForeignVulkan>, String>> = OnceLock::new();

/// Return the process-wide adapter, creating it on first success.
///
/// Returns `None` when the Vulkan loader or a suitable device is missing.
/// Failures are cached: later calls do not retry.
pub fn try_adapter() -> Option<Arc<ForeignVulkan>> {
    match ADAPTER.get_or_init(init_adapter) {
        Ok(a) => Some(Arc::clone(a)),
        Err(e) => {
            tracing::debug!("foreign Vulkan adapter unavailable: {e}");
            None
        }
    }
}

fn init_adapter() -> Result<Arc<ForeignVulkan>, String> {
    init_adapter_inner().map_err(|e| e.detail())
}

fn init_adapter_inner() -> Result<Arc<ForeignVulkan>, GoldyError> {
    let entry = unsafe { ash::Entry::load() }.map_err(|e| GoldyError::Backend(anyhow::anyhow!("{e}")))?;
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"goldy-foreign")
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_name(c"goldy-foreign")
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::API_VERSION_1_1);
    let create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
    let instance = {
        let _guard = VK_INSTANCE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        unsafe { entry.create_instance(&create_info, None) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkCreateInstance: {e}")))?
    };
    let phys_list = match unsafe { instance.enumerate_physical_devices() } {
        Ok(list) if !list.is_empty() => list,
        Ok(_) => {
            destroy_instance(&entry, &instance);
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Vulkan: no physical devices"
            )));
        }
        Err(e) => {
            destroy_instance(&entry, &instance);
            return Err(GoldyError::Backend(anyhow::anyhow!("enumerate_physical_devices: {e}")));
        }
    };

    let mut chosen: Option<(vk::PhysicalDevice, u32, vk::PhysicalDeviceType)> = None;
    for phys in phys_list {
        let props = unsafe { instance.get_physical_device_properties(phys) };
        let families = unsafe { instance.get_physical_device_queue_family_properties(phys) };
        let Some(family) = families.iter().position(|f| {
            f.queue_flags
                .contains(vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS)
                || f.queue_flags.contains(vk::QueueFlags::TRANSFER)
                || f.queue_flags.contains(vk::QueueFlags::GRAPHICS)
        }) else {
            continue;
        };
        let ty = props.device_type;
        let better = match chosen {
            None => true,
            Some((_, _, prev)) => rank_device(ty) > rank_device(prev),
        };
        if better {
            chosen = Some((phys, family as u32, ty));
        }
    }
    let Some((phys, queue_family, ty)) = chosen else {
        destroy_instance(&entry, &instance);
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign Vulkan: no queue with TRANSFER or GRAPHICS"
        )));
    };
    let name = unsafe { CStr::from_ptr(instance.get_physical_device_properties(phys).device_name.as_ptr()) };
    tracing::info!(
        device = %name.to_string_lossy(),
        ?ty,
        queue_family,
        "foreign Vulkan adapter"
    );

    let queue_prio = [1.0f32];
    let queue_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_prio);
    let queues = [queue_info];
    let device_info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
    let device = unsafe { instance.create_device(phys, &device_info, None) }.map_err(|e| {
        destroy_instance(&entry, &instance);
        GoldyError::Backend(anyhow::anyhow!("vkCreateDevice: {e}"))
    })?;
    let queue = unsafe { device.get_device_queue(queue_family, 0) };

    Ok(Arc::new(ForeignVulkan {
        state: Mutex::new(AdapterState {
            _entry: entry,
            instance,
            phys,
            device,
            queue,
            queue_family,
            next_id: 1,
            surfaces: HashMap::new(),
        }),
    }))
}

fn rank_device(ty: vk::PhysicalDeviceType) -> u8 {
    match ty {
        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
        vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
        _ => 0,
    }
}

fn destroy_instance(_entry: &ash::Entry, instance: &ash::Instance) {
    let _guard = VK_INSTANCE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    unsafe { instance.destroy_instance(None) };
}

impl ForeignVulkan {
    /// Offscreen `width × height` image. No window, no swapchain.
    pub fn offscreen(
        self: &Arc<Self>,
        width: u32,
        height: u32,
        format: TextureFormat,
    ) -> Result<ForeignSurface, GoldyError> {
        let layout = PixmapLayout::tight(width, height, format);
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let slot = SurfaceSlot::create(&state, layout)?;
        let id = state.next_id;
        state.next_id += 1;
        state.surfaces.insert(id, slot);
        Ok(ForeignSurface {
            inner: Arc::new(SurfaceHandle {
                adapter: Arc::clone(self),
                id,
            }),
        })
    }
}

impl AdapterState {
    fn reap(&mut self) {
        let ids: Vec<u32> = self
            .surfaces
            .iter()
            .filter(|(_, s)| s.dropped)
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            if let Some(slot) = self.surfaces.remove(&id) {
                slot.destroy(&self.device);
            }
        }
    }
}

impl SurfaceSlot {
    fn create(state: &AdapterState, layout: PixmapLayout) -> Result<Self, GoldyError> {
        let vk_format = format_to_vk(layout.format);
        let staging_size = layout.staging_bytes() as usize;
        let staging = alloc_host_buffer(&state.instance, state.phys, &state.device, staging_size)?;
        let readback = alloc_host_buffer(&state.instance, state.phys, &state.device, staging_size)?;
        let image = alloc_image(
            &state.instance,
            state.phys,
            &state.device,
            layout.width,
            layout.height,
            vk_format,
        )?;
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(state.queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let cmd_pool = unsafe { state.device.create_command_pool(&pool_info, None) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkCreateCommandPool: {e}")))?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let cmd = match unsafe { state.device.allocate_command_buffers(&alloc) } {
            Ok(mut v) => v.remove(0),
            Err(e) => {
                unsafe { state.device.destroy_command_pool(cmd_pool, None) };
                return Err(GoldyError::Backend(anyhow::anyhow!("vkAllocateCommandBuffers: {e}")));
            }
        };
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        let fence = match unsafe { state.device.create_fence(&fence_info, None) } {
            Ok(f) => f,
            Err(e) => {
                unsafe { state.device.destroy_command_pool(cmd_pool, None) };
                return Err(GoldyError::Backend(anyhow::anyhow!("vkCreateFence: {e}")));
            }
        };
        Ok(Self {
            width: layout.width,
            height: layout.height,
            format: layout.format,
            generation: 1,
            image,
            staging,
            readback,
            cmd_pool,
            cmd,
            fence,
            dropped: false,
        })
    }

    fn destroy(self, device: &ash::Device) {
        unsafe {
            let _ = device.wait_for_fences(&[self.fence], true, u64::MAX);
            device.destroy_fence(self.fence, None);
            device.destroy_command_pool(self.cmd_pool, None);
            device.destroy_image(self.image.image, None);
            device.free_memory(self.image.memory, None);
            device.destroy_buffer(self.staging.buffer, None);
            device.free_memory(self.staging.memory, None);
            device.destroy_buffer(self.readback.buffer, None);
            device.free_memory(self.readback.memory, None);
        }
    }

    fn wait(&self, device: &ash::Device) -> Result<(), GoldyError> {
        unsafe { device.wait_for_fences(&[self.fence], true, u64::MAX) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkWaitForFences: {e}")))
    }
}

fn alloc_host_buffer(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    device: &ash::Device,
    size: usize,
) -> Result<HostBuffer, GoldyError> {
    let info = vk::BufferCreateInfo::default()
        .size(size as u64)
        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let buffer = unsafe { device.create_buffer(&info, None) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkCreateBuffer: {e}")))?;
    let req = unsafe { device.get_buffer_memory_requirements(buffer) };
    let Some(ty) = find_memory_type(
        instance,
        phys,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) else {
        unsafe { device.destroy_buffer(buffer, None) };
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign Vulkan: no HOST_VISIBLE | HOST_COHERENT memory type"
        )));
    };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(ty);
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(GoldyError::Backend(anyhow::anyhow!("vkAllocateMemory: {e}")));
        }
    };
    if let Err(e) = unsafe { device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        return Err(GoldyError::Backend(anyhow::anyhow!("vkBindBufferMemory: {e}")));
    }
    let ptr = unsafe { device.map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty()) }.map_err(|e| {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_buffer(buffer, None);
        }
        GoldyError::Backend(anyhow::anyhow!("vkMapMemory: {e}"))
    })? as *mut u8;
    Ok(HostBuffer {
        buffer,
        memory,
        ptr,
        size,
    })
}

fn alloc_image(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    device: &ash::Device,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<DeviceImage, GoldyError> {
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&info, None) }
        .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkCreateImage: {e}")))?;
    let req = unsafe { device.get_image_memory_requirements(image) };
    let Some(ty) = find_memory_type(
        instance,
        phys,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| find_memory_type(instance, phys, req.memory_type_bits, vk::MemoryPropertyFlags::empty())) else {
        unsafe { device.destroy_image(image, None) };
        return Err(GoldyError::Backend(anyhow::anyhow!(
            "foreign Vulkan: no memory type for transfer image"
        )));
    };
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(req.size)
        .memory_type_index(ty);
    let memory = match unsafe { device.allocate_memory(&alloc, None) } {
        Ok(m) => m,
        Err(e) => {
            unsafe { device.destroy_image(image, None) };
            return Err(GoldyError::Backend(anyhow::anyhow!("vkAllocateMemory(image): {e}")));
        }
    };
    if let Err(e) = unsafe { device.bind_image_memory(image, memory, 0) } {
        unsafe {
            device.free_memory(memory, None);
            device.destroy_image(image, None);
        }
        return Err(GoldyError::Backend(anyhow::anyhow!("vkBindImageMemory: {e}")));
    }
    Ok(DeviceImage { image, memory })
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

fn image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(color_range())
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
}

fn copy_region(layout: PixmapLayout) -> vk::BufferImageCopy {
    let bpp = layout.bytes_per_pixel();
    let row_texels = if layout.row_pitch == 0 {
        0
    } else {
        (layout.row_pitch_bytes() / u64::from(bpp)) as u32
    };
    vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: row_texels,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: layout.width,
            height: layout.height,
            depth: 1,
        },
    }
}

impl ForeignVulkan {
    fn blit(&self, id: u32, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        layout.validate()?;
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.reap();
        let device = state.device.clone();
        let queue = state.queue;
        let slot = state
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Vulkan surface {id} is gone")))?;
        if slot.dropped {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Vulkan surface {id} has been dropped"
            )));
        }
        if layout.width != slot.width || layout.height != slot.height || layout.format != slot.format {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Vulkan blit layout {}x{} {:?} does not match surface {}x{} {:?}",
                layout.width,
                layout.height,
                layout.format,
                slot.width,
                slot.height,
                slot.format
            )));
        }
        if pixels.len() < layout.staging_bytes() as usize {
            return Err(GoldyError::Backend(anyhow::anyhow!(
                "foreign Vulkan blit: {} source bytes, need {}",
                pixels.len(),
                layout.staging_bytes()
            )));
        }
        let cmd = slot.cmd;
        let fence = slot.fence;
        let image = slot.image.image;
        let staging_buf = slot.staging.buffer;
        let staging_ptr = slot.staging.ptr;
        let readback_buf = slot.readback.buffer;
        slot.wait(&device)?;
        unsafe { device.reset_fences(&[fence]) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkResetFences: {e}")))?;
        let n = layout.staging_bytes() as usize;
        // SAFETY: `staging_ptr` is a persistently mapped HOST_VISIBLE allocation of
        // `slot.staging.size` bytes, exclusive under `AdapterState`'s mutex.
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging_ptr, n);
        }
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkBeginCommandBuffer: {e}")))?;
        let to_dst = image_barrier(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_dst],
            );
        }
        let region = copy_region(layout);
        unsafe {
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        let to_src = image_barrier(
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::TRANSFER_READ,
        );
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_src],
            );
        }
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf,
                &[region],
            );
        }
        unsafe { device.end_command_buffer(cmd) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkEndCommandBuffer: {e}")))?;
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default().command_buffers(&cmds);
        unsafe { device.queue_submit(queue, &[submit], fence) }
            .map_err(|e| GoldyError::Backend(anyhow::anyhow!("vkQueueSubmit: {e}")))?;
        Ok(())
    }

    fn snapshot(&self, id: u32, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let device = state.device.clone();
        let slot = state
            .surfaces
            .get_mut(&id)
            .ok_or_else(|| GoldyError::Backend(anyhow::anyhow!("foreign Vulkan surface {id} is gone")))?;
        slot.wait(&device)?;
        // SAFETY: `readback.ptr` is mapped for `readback.size` bytes; exclusive under the adapter lock.
        let staging = unsafe { std::slice::from_raw_parts(slot.readback.ptr, slot.readback.size) };
        let mut tight = vec![0u8; layout.logical_bytes() as usize];
        layout.unpack_into(&staging[..layout.staging_bytes() as usize], &mut tight)?;
        Ok(tight)
    }

    fn generation(&self, id: u32) -> u64 {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| s.generation).unwrap_or(0)
    }

    fn size(&self, id: u32) -> (u32, u32) {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.surfaces.get(&id).map(|s| (s.width, s.height)).unwrap_or((0, 0))
    }

    fn release(&self, id: u32) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(slot) = state.surfaces.get_mut(&id) {
            slot.dropped = true;
        }
        state.reap();
    }
}

impl PixelSink for ForeignSurface {
    fn blit(&self, pixels: &[u8], layout: PixmapLayout) -> Result<(), GoldyError> {
        self.inner.adapter.blit(self.inner.id, pixels, layout)
    }

    fn generation(&self) -> u64 {
        self.inner.adapter.generation(self.inner.id)
    }

    fn size(&self) -> (u32, u32) {
        self.inner.adapter.size(self.inner.id)
    }
}

impl ForeignSurface {
    /// Tightly packed pixels after the last blit (GPU copy-back).
    pub fn snapshot(&self, layout: PixmapLayout) -> Result<Vec<u8>, GoldyError> {
        self.inner.adapter.snapshot(self.inner.id, layout)
    }
}
