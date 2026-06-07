//! Opt-in VRAM accounting observers for [`Device`](crate::device::Device).
//!
//! Clients that want byte-level telemetry or budget enforcement register a
//! [`VramObserver`] via [`Device::add_vram_observer`](crate::device::Device::add_vram_observer).
//! When no observers are registered, the hot path skips observer notification entirely.
//!
//! [`TrackingVramAllocator`](crate::vram_allocator::TrackingVramAllocator) composes
//! [`VramByteTracker`] internally for the legacy `with_vram_allocator` wrapper path.

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

use anyhow::Result;

use crate::buffer::Buffer;
use crate::texture::Texture;
use crate::vram_allocator::{ParcelType, VramAllocator};

/// Stable handle returned by [`Device::add_vram_observer`](crate::device::Device::add_vram_observer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VramObserverId(u64);

impl VramObserverId {
    pub(crate) fn from_raw(id: u64) -> Self {
        Self(id)
    }

    /// Raw id value (diagnostics only).
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Allocation event emitted after a deed-holding parcel is created through the `Device::alloc_*` helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramAllocEvent {
    /// Reserved backing size (`Buffer::allocated_size` / texture byte size).
    pub reserved: u64,
    /// Committed logical size handed to the runtime.
    pub committed: u64,
    pub kind: ParcelType,
}

impl VramAllocEvent {
    pub fn from_buffer(buf: &Buffer) -> Self {
        Self {
            reserved: buf.allocated_size(),
            committed: buf.size(),
            kind: ParcelType::Buffer,
        }
    }

    pub fn from_texture(tex: &Texture) -> Self {
        let byte_size = tex.byte_size() as u64;
        Self {
            reserved: byte_size,
            committed: byte_size,
            kind: ParcelType::Texture,
        }
    }
}

/// Free event emitted when a deed-holding parcel is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VramFreeEvent {
    pub reserved: u64,
    pub committed: u64,
    pub kind: ParcelType,
}

/// Opt-in observer for VRAM allocation and free events on a [`Device`](crate::device::Device).
pub trait VramObserver: Send + Sync {
    /// Called after a successful allocation. May reject (e.g. budget enforcement).
    fn on_allocated(&self, event: &VramAllocEvent) -> Result<()>;

    /// Called when a deed-holding parcel is dropped.
    fn on_freed(&self, event: &VramFreeEvent);
}

/// Byte-level VRAM tracker suitable as a [`VramObserver`] or composed by
/// [`TrackingVramAllocator`](crate::vram_allocator::TrackingVramAllocator).
pub struct VramByteTracker {
    live_bytes: AtomicI64,
    budget_bytes: Option<u64>,
}

impl VramByteTracker {
    /// Track allocations with no budget cap.
    pub fn new() -> Self {
        Self {
            live_bytes: AtomicI64::new(0),
            budget_bytes: None,
        }
    }

    /// Track allocations and reject those that would exceed `budget_bytes`.
    pub fn with_budget(budget_bytes: u64) -> Self {
        Self {
            live_bytes: AtomicI64::new(0),
            budget_bytes: Some(budget_bytes),
        }
    }

    /// Net bytes currently tracked (allocations minus frees).
    pub fn allocated_bytes(&self) -> u64 {
        self.live_bytes.load(Ordering::Relaxed).max(0) as u64
    }

    /// Optional byte budget, if configured.
    pub fn budget(&self) -> Option<u64> {
        self.budget_bytes
    }

    fn check_budget(&self, additional: u64) -> Result<()> {
        if let Some(cap) = self.budget_bytes {
            let current = self.live_bytes.load(Ordering::Relaxed) as u64;
            if current.saturating_add(additional) > cap {
                anyhow::bail!(
                    "VramObserver budget exceeded: {current} + {additional} > {cap} \
                     (budget={})",
                    crate::vram_allocator::bytesize(cap),
                );
            }
        }
        Ok(())
    }

    /// Budget check used by [`TrackingVramAllocator`] before delegating allocation.
    pub(crate) fn check_budget_for_alloc(&self, additional: u64) -> Result<()> {
        self.check_budget(additional)
    }

    pub(crate) fn add_allocated(&self, reserved: u64) {
        self.live_bytes.fetch_add(reserved as i64, Ordering::Relaxed);
    }

    pub(crate) fn sub_freed(&self, reserved: u64) {
        self.live_bytes.fetch_sub(reserved as i64, Ordering::Relaxed);
    }

    /// Record an allocation (budget check + increment).
    pub fn record_allocated(&self, reserved: u64) -> Result<()> {
        self.check_budget(reserved)?;
        self.add_allocated(reserved);
        Ok(())
    }

    /// Record a free (decrement).
    pub fn record_freed(&self, reserved: u64) {
        self.sub_freed(reserved);
    }
}

impl Default for VramByteTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl VramObserver for VramByteTracker {
    fn on_allocated(&self, event: &VramAllocEvent) -> Result<()> {
        self.record_allocated(event.reserved)
    }

    fn on_freed(&self, event: &VramFreeEvent) {
        self.record_freed(event.reserved);
    }
}

struct ObserverEntry {
    id: VramObserverId,
    observer: Arc<dyn VramObserver>,
}

impl Clone for ObserverEntry {
    fn clone(&self) -> Self {
        Self {
            id: self.id,
            observer: Arc::clone(&self.observer),
        }
    }
}

/// Event aggregator on [`Device`](crate::device::Device) for resource lifecycle signals.
///
/// Today this fans out deed (alloc/free) events to [`VramObserver`]s and the installed
/// [`VramAllocator`]. The hub is intentionally generic so additional event kinds can be
/// wired through it later without changing parcel drop paths.
pub(crate) struct ResourceHub {
    next_id: AtomicU64,
    observers: Mutex<Arc<[ObserverEntry]>>,
}

impl ResourceHub {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            next_id: AtomicU64::new(1),
            observers: Mutex::new(Arc::from([])),
        })
    }

    pub fn has_observers(&self) -> bool {
        !self.observers.lock().unwrap().is_empty()
    }

    pub fn add_observer(&self, observer: Arc<dyn VramObserver>) -> VramObserverId {
        let id = VramObserverId::from_raw(self.next_id.fetch_add(1, Ordering::Relaxed));
        let mut guard = self.observers.lock().unwrap();
        let mut entries: Vec<ObserverEntry> = guard.iter().cloned().collect();
        entries.push(ObserverEntry { id, observer });
        *guard = entries.into();
        id
    }

    pub fn remove_observer(&self, id: VramObserverId) -> bool {
        let mut guard = self.observers.lock().unwrap();
        let entries: Vec<ObserverEntry> = guard.iter().filter(|e| e.id != id).cloned().collect();
        let removed = entries.len() != guard.len();
        *guard = entries.into();
        removed
    }

    pub fn notify_allocated(&self, event: &VramAllocEvent) -> Result<()> {
        let observers = Arc::clone(&*self.observers.lock().unwrap());
        for entry in observers.iter() {
            entry.observer.on_allocated(event)?;
        }
        Ok(())
    }

    pub fn notify_freed(&self, event: &VramFreeEvent, allocator: Option<&dyn VramAllocator>) {
        let observers = Arc::clone(&*self.observers.lock().unwrap());
        for entry in observers.iter() {
            entry.observer.on_freed(event);
        }
        if let Some(alloc) = allocator {
            alloc.notify_freed(event.reserved, event.committed, event.kind);
        }
    }
}

/// Deed attached to GPU parcels allocated through [`Device::alloc_*`].
#[derive(Clone)]
pub(crate) struct ParcelDeed {
    pub hub: Weak<ResourceHub>,
    pub allocator: Weak<dyn VramAllocator>,
}

impl ParcelDeed {
    pub fn notify_freed(&self, event: &VramFreeEvent) {
        if let Some(hub) = self.hub.upgrade() {
            let allocator = self.allocator.upgrade();
            hub.notify_freed(event, allocator.as_deref());
        } else if let Some(alloc) = self.allocator.upgrade() {
            alloc.notify_freed(event.reserved, event.committed, event.kind);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    struct CountingObserver {
        allocs: AtomicU32,
        frees: AtomicU32,
    }

    impl CountingObserver {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                allocs: AtomicU32::new(0),
                frees: AtomicU32::new(0),
            })
        }
    }

    impl VramObserver for CountingObserver {
        fn on_allocated(&self, _event: &VramAllocEvent) -> Result<()> {
            self.allocs.fetch_add(1, AtomicOrdering::Relaxed);
            Ok(())
        }

        fn on_freed(&self, _event: &VramFreeEvent) {
            self.frees.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    #[test]
    fn resource_hub_notifies_observers() {
        let hub = ResourceHub::new();
        let counter = CountingObserver::new();
        let _id = hub.add_observer(counter.clone());

        let event = VramAllocEvent {
            reserved: 1024,
            committed: 1024,
            kind: ParcelType::Buffer,
        };
        hub.notify_allocated(&event).unwrap();
        assert_eq!(counter.allocs.load(AtomicOrdering::Relaxed), 1);

        let free = VramFreeEvent {
            reserved: 1024,
            committed: 1024,
            kind: ParcelType::Buffer,
        };
        hub.notify_freed(&free, None);
        assert_eq!(counter.frees.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn byte_tracker_budget() {
        let tracker = VramByteTracker::with_budget(8192);
        tracker
            .on_allocated(&VramAllocEvent {
                reserved: 4096,
                committed: 4096,
                kind: ParcelType::Buffer,
            })
            .unwrap();
        assert_eq!(tracker.allocated_bytes(), 4096);
        let err = tracker.on_allocated(&VramAllocEvent {
            reserved: 8192,
            committed: 8192,
            kind: ParcelType::Buffer,
        });
        assert!(err.is_err());
    }
}
