//! Signal-driven, incrementally patched `MTLResource` residency sets for Metal encoders.
//!
//! Instead of scanning and sorting `MetalState::{buffers, textures, accels}` behind a global
//! generation counter, each [`LogicalDevice`](super::types::LogicalDevice) owns a
//! [`DeviceResidency`] subscriber wired to its `DeviceHandle` scope on the three observable
//! resource tables:
//! - Resource additions, replacements, and removals patch [`RefCountedSet`] tiers in `O(1)`
//!   (`rw`, `ro`, `accel`).
//! - Sub-buffer views (`BufferView` / `TensorView`) sharing an already-resident `MTLBuffer`,
//!   logical extent updates keeping the same `MTLBuffer`, withdraw-staging buffers, and
//!   mutations on other `LogicalDevice`s do **not** alter the unique `MTLResource` set and
//!   emit no downstream invalidation.
//! - Each [`MetalSubmissionContext`](super::types::MetalSubmissionContext) subscribes a
//!   context-private [`ContextResidencySlot`] to [`DeviceResidency::changed`], caching an
//!   immutable `Arc<ResidencyLists>` locally so steady-state submissions across multiple
//!   contexts never contend on a shared lock or shared atomic counter.

use super::super::{AccelerationStructureHandle, BufferHandle, DeviceHandle, TextureHandle};
use super::types::{AccelState, BufferState, TextureState};
use crate::observer::{ObservableTable, RefCountedSet, Signal, Subscription, TableEvent};
use ::metal as mtl;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const _: () = assert!(std::mem::size_of::<&mtl::ResourceRef>() == std::mem::size_of::<usize>());

#[inline]
fn buffer_resource_ptr(state: &BufferState) -> Option<usize> {
    if state.is_withdraw_staging {
        return None;
    }
    let buf_ref: &mtl::BufferRef = &state.buffer;
    let res_ref = unsafe { std::mem::transmute::<&mtl::BufferRef, &mtl::ResourceRef>(buf_ref) };
    Some(std::ptr::from_ref(res_ref) as usize)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TextureTier {
    ReadWrite,
    ReadOnly,
}

#[inline]
fn texture_tier_and_ptr(state: &TextureState) -> (TextureTier, usize) {
    let tex_ref: &mtl::TextureRef = &state.texture;
    let res_ref = unsafe { std::mem::transmute::<&mtl::TextureRef, &mtl::ResourceRef>(tex_ref) };
    let tier = if state.is_storage_image {
        TextureTier::ReadWrite
    } else {
        TextureTier::ReadOnly
    };
    (tier, std::ptr::from_ref(res_ref) as usize)
}

#[inline]
fn accel_resource_ptr(state: &AccelState) -> usize {
    let as_ref: &mtl::AccelerationStructureRef = &state.accel;
    let res_ref = unsafe { std::mem::transmute::<&mtl::AccelerationStructureRef, &mtl::ResourceRef>(as_ref) };
    std::ptr::from_ref(res_ref) as usize
}

/// Immutable snapshot of deduplicated `MTLResource` pointers grouped by `MTLResourceUsage`.
#[derive(Default)]
pub(crate) struct ResidencyLists {
    rw_ptrs: Box<[usize]>,
    ro_ptrs: Box<[usize]>,
    accel_ptrs: Box<[usize]>,
}

impl ResidencyLists {
    /// Declare all resident device resources on `encoder` in at most three `use_resources` calls.
    ///
    /// # Safety
    ///
    /// Caller must hold a shared borrow of `MetalState` (guaranteeing no concurrent table
    /// removals have released any backing Objective-C `MTLResource`) and `self` must be the
    /// current snapshot for `encoder`'s device.
    #[inline]
    pub(super) unsafe fn declare_for_compute(&self, encoder: &mtl::ComputeCommandEncoderRef) {
        if !self.rw_ptrs.is_empty() {
            let rw_refs = unsafe {
                std::slice::from_raw_parts(self.rw_ptrs.as_ptr().cast::<&mtl::ResourceRef>(), self.rw_ptrs.len())
            };
            encoder.use_resources(rw_refs, mtl::MTLResourceUsage::Read | mtl::MTLResourceUsage::Write);
        }
        if !self.ro_ptrs.is_empty() {
            let ro_refs = unsafe {
                std::slice::from_raw_parts(self.ro_ptrs.as_ptr().cast::<&mtl::ResourceRef>(), self.ro_ptrs.len())
            };
            encoder.use_resources(ro_refs, mtl::MTLResourceUsage::Read);
        }
        if !self.accel_ptrs.is_empty() {
            let accel_refs = unsafe {
                std::slice::from_raw_parts(
                    self.accel_ptrs.as_ptr().cast::<&mtl::ResourceRef>(),
                    self.accel_ptrs.len(),
                )
            };
            encoder.use_resources(accel_refs, mtl::MTLResourceUsage::Read);
        }
    }
}

#[derive(Default)]
struct ResidencySets {
    rw: RefCountedSet<usize>,
    ro: RefCountedSet<usize>,
    accel: RefCountedSet<usize>,
    published: Option<Arc<ResidencyLists>>,
}

impl ResidencySets {
    fn insert_texture(&mut self, tier: TextureTier, ptr: usize) -> bool {
        match tier {
            TextureTier::ReadWrite => self.rw.insert(ptr),
            TextureTier::ReadOnly => self.ro.insert(ptr),
        }
    }

    fn remove_texture(&mut self, tier: TextureTier, ptr: usize) -> bool {
        match tier {
            TextureTier::ReadWrite => self.rw.remove(ptr),
            TextureTier::ReadOnly => self.ro.remove(ptr),
        }
    }

    fn snapshot(&mut self) -> Arc<ResidencyLists> {
        Arc::clone(self.published.get_or_insert_with(|| {
            Arc::new(ResidencyLists {
                rw_ptrs: self.rw.as_slice().into(),
                ro_ptrs: self.ro.as_slice().into(),
                accel_ptrs: self.accel.as_slice().into(),
            })
        }))
    }
}

struct DeviceResidencyCore {
    sets: Mutex<ResidencySets>,
    changed: Signal<()>,
}

impl DeviceResidencyCore {
    fn finish_mutation(&self, mut guard: std::sync::MutexGuard<'_, ResidencySets>, changed: bool) {
        if changed {
            guard.published = None;
            drop(guard);
            self.changed.emit(&());
        }
    }

    fn on_buffer_event(&self, event: TableEvent<'_, BufferHandle, BufferState>) {
        let mut guard = self.sets.lock().unwrap();
        let changed = match event {
            TableEvent::Inserted { value, .. } => buffer_resource_ptr(value).is_some_and(|ptr| guard.rw.insert(ptr)),
            TableEvent::Removed { value, .. } => buffer_resource_ptr(value).is_some_and(|ptr| guard.rw.remove(ptr)),
            TableEvent::Replaced { old, new, .. } => {
                guard.rw.replace(buffer_resource_ptr(old), buffer_resource_ptr(new))
            }
        };
        self.finish_mutation(guard, changed);
    }

    fn on_texture_event(&self, event: TableEvent<'_, TextureHandle, TextureState>) {
        let mut guard = self.sets.lock().unwrap();
        let changed = match event {
            TableEvent::Inserted { value, .. } => {
                let (tier, ptr) = texture_tier_and_ptr(value);
                guard.insert_texture(tier, ptr)
            }
            TableEvent::Removed { value, .. } => {
                let (tier, ptr) = texture_tier_and_ptr(value);
                guard.remove_texture(tier, ptr)
            }
            TableEvent::Replaced { old, new, .. } => {
                let old_entry = texture_tier_and_ptr(old);
                let new_entry = texture_tier_and_ptr(new);
                if old_entry == new_entry {
                    false
                } else {
                    let mut c = guard.remove_texture(old_entry.0, old_entry.1);
                    c |= guard.insert_texture(new_entry.0, new_entry.1);
                    c
                }
            }
        };
        self.finish_mutation(guard, changed);
    }

    fn on_accel_event(&self, event: TableEvent<'_, AccelerationStructureHandle, AccelState>) {
        let mut guard = self.sets.lock().unwrap();
        let changed = match event {
            TableEvent::Inserted { value, .. } => guard.accel.insert(accel_resource_ptr(value)),
            TableEvent::Removed { value, .. } => guard.accel.remove(accel_resource_ptr(value)),
            TableEvent::Replaced { old, new, .. } => guard
                .accel
                .replace(Some(accel_resource_ptr(old)), Some(accel_resource_ptr(new))),
        };
        self.finish_mutation(guard, changed);
    }
}

/// Per-`LogicalDevice` incremental residency subscriber.
pub(crate) struct DeviceResidency {
    core: Arc<DeviceResidencyCore>,
    _subscriptions: [Subscription; 3],
}

impl DeviceResidency {
    pub(super) fn connect(
        device_handle: DeviceHandle,
        buffers: &ObservableTable<BufferHandle, BufferState, DeviceHandle>,
        textures: &ObservableTable<TextureHandle, TextureState, DeviceHandle>,
        accels: &ObservableTable<AccelerationStructureHandle, AccelState, DeviceHandle>,
    ) -> Self {
        let core = Arc::new(DeviceResidencyCore {
            sets: Mutex::new(ResidencySets::default()),
            changed: Signal::new(),
        });
        let buf_sub = buffers.connect_scoped_weak(device_handle, &core, DeviceResidencyCore::on_buffer_event);
        let tex_sub = textures.connect_scoped_weak(device_handle, &core, DeviceResidencyCore::on_texture_event);
        let acc_sub = accels.connect_scoped_weak(device_handle, &core, DeviceResidencyCore::on_accel_event);
        Self {
            core,
            _subscriptions: [buf_sub, tex_sub, acc_sub],
        }
    }

    /// Return the current immutable residency lists for this device, materializing at most once
    /// after a batch of mutations.
    pub(super) fn snapshot(&self) -> Arc<ResidencyLists> {
        self.core.sets.lock().unwrap().snapshot()
    }

    fn changed(&self) -> &Signal<()> {
        &self.core.changed
    }
}

/// Per-`MetalSubmissionContext` residency slot that caches [`ResidencyLists`] locally and
/// refreshes only when [`DeviceResidency`] emits a change signal for that device.
pub(crate) struct ContextResidencySlot {
    dirty: Arc<AtomicBool>,
    cached: Option<Arc<ResidencyLists>>,
    _subscription: Subscription,
}

impl ContextResidencySlot {
    pub(super) fn connect(device_residency: &DeviceResidency) -> Self {
        let dirty = Arc::new(AtomicBool::new(true));
        let subscription = device_residency.changed().connect_weak(&dirty, |flag, _| {
            flag.store(true, Ordering::Release);
        });
        Self {
            dirty,
            cached: None,
            _subscription: subscription,
        }
    }

    /// Resolve the up-to-date [`ResidencyLists`] for this context without touching shared device
    /// locks in steady state.
    pub(super) fn resolve(&mut self, device_residency: &DeviceResidency) -> Arc<ResidencyLists> {
        if self.dirty.swap(false, Ordering::AcqRel) || self.cached.is_none() {
            let fresh = device_residency.snapshot();
            self.cached = Some(Arc::clone(&fresh));
            fresh
        } else {
            Arc::clone(self.cached.as_ref().expect("checked is_some"))
        }
    }
}
