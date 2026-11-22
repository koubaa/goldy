//! Metal heap and host-visible storage selection.
//!
//! Apple's `MTLHeapDescriptor.storageMode` allows Shared heaps only on Apple
//! silicon. Intel iGPUs report `hasUnifiedMemory` and still abort
//! `newHeap:` with `Shared storage mode disallowed` (issue #330). macOS heaps
//! do not accept Managed either.
//!
//! The decision is a pure function of device capabilities so it can be tested
//! without a Metal device. Shared heaps and Shared textures require
//! `supportsFamily(Apple1)`. `hasUnifiedMemory` is recorded separately: it
//! says whether a host mapping is free, and it does not select heap storage.
//! `MTLGPUFamilyMac2` is true on Apple silicon and on Intel and AMD Macs.

/// Capabilities read from `MTLDevice`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalMemoryFeatures {
    /// `MTLDevice.hasUnifiedMemory`. Intel iGPUs report this and still cannot
    /// use a Shared heap. Not the heap-storage predicate.
    pub has_unified_memory: bool,
    /// `supportsFamily(MTLGPUFamilyApple1)`. Later Apple families imply this one.
    pub apple_family: bool,
    /// `supportsFamily(MTLGPUFamilyMac2)`.
    pub mac2_family: bool,
}

/// Storage mode used for every `MTLHeap` this backend creates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeapStorage {
    /// `MTLStorageModeShared`. Apple-family GPUs only.
    Shared,
    /// `MTLStorageModePrivate`. Intel and discrete Macs.
    Private,
}

/// Where a CPU-mapped buffer is allocated.
///
/// Both placements use Shared storage. Shared buffers are legal on discrete
/// Macs; Managed buffers would also be legal but require `didModifyRange`
/// after every CPU write and `synchronize` before every CPU read. Goldy's
/// `contents()` contract (including `CPU_WRITABLE`) is the Shared one, so
/// discrete GPUs get a standalone Shared buffer instead of a Managed one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostVisibleBufferPlacement {
    /// Suballocated from the Shared heap.
    SharedHeap,
    /// `device.newBuffer` with Shared storage. Not placed in a heap, because
    /// the heap on this device is Private and the storage modes must match.
    SharedStandalone,
}

/// Storage mode for sampled and storage textures allocated by Goldy.
///
/// Uploads are blit copies from Shared staging buffers, so discrete GPUs can
/// keep the texture Private inside the Private texture heap. Managed is not
/// used: heaps reject it, and nothing maps texture bytes on the CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TextureStorage {
    Shared,
    Private,
}

/// Placement chosen for one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalStoragePlan {
    pub heap: HeapStorage,
    pub host_visible_buffer: HostVisibleBufferPlacement,
    pub texture: TextureStorage,
}

impl MetalMemoryFeatures {
    /// Shared `MTLHeap`s and Shared textures are legal on Apple-family GPUs.
    pub fn allows_shared_heaps(self) -> bool {
        self.apple_family
    }

    pub fn plan(self) -> MetalStoragePlan {
        if self.allows_shared_heaps() {
            MetalStoragePlan {
                heap: HeapStorage::Shared,
                host_visible_buffer: HostVisibleBufferPlacement::SharedHeap,
                texture: TextureStorage::Shared,
            }
        } else {
            MetalStoragePlan {
                heap: HeapStorage::Private,
                host_visible_buffer: HostVisibleBufferPlacement::SharedStandalone,
                texture: TextureStorage::Private,
            }
        }
    }

    /// Short label for logs. Uses every capability bit so a Mac2 discrete GPU
    /// is not confused with an Apple-family GPU that failed to report unified
    /// memory.
    pub fn family_summary(self) -> &'static str {
        match (self.apple_family, self.mac2_family, self.has_unified_memory) {
            (true, _, true) => "apple-unified",
            (true, _, false) => "apple-non-unified",
            (false, true, true) => "mac2-unified",
            (false, true, false) => "mac2-discrete",
            (false, false, true) => "unified",
            (false, false, false) => "discrete",
        }
    }
}

impl MetalStoragePlan {
    /// Host-visible buffers are suballocated from the heap only on the Shared
    /// heap path. Apple silicon returns true; Intel and discrete Macs return false.
    pub fn host_visible_buffers_from_heap(self) -> bool {
        matches!(self.host_visible_buffer, HostVisibleBufferPlacement::SharedHeap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn features(unified: bool, apple: bool, mac2: bool) -> MetalMemoryFeatures {
        MetalMemoryFeatures {
            has_unified_memory: unified,
            apple_family: apple,
            mac2_family: mac2,
        }
    }

    #[test]
    fn apple_silicon_keeps_shared_heaps() {
        // M-series: unified memory, Apple family, and Mac2 (compatibility).
        let plan = features(true, true, true).plan();
        assert_eq!(plan.heap, HeapStorage::Shared);
        assert_eq!(plan.host_visible_buffer, HostVisibleBufferPlacement::SharedHeap);
        assert_eq!(plan.texture, TextureStorage::Shared);
        assert!(plan.host_visible_buffers_from_heap());
        assert_eq!(features(true, true, true).family_summary(), "apple-unified");
    }

    #[test]
    fn intel_mac2_uses_private_heaps_and_standalone_shared_buffers() {
        // Intel MacBook / AMD discrete: Mac2, no Apple family, no unified memory.
        // This is the `Shared storage mode disallowed` case.
        let feats = features(false, false, true);
        assert!(!feats.allows_shared_heaps());
        assert_eq!(feats.family_summary(), "mac2-discrete");
        let plan = feats.plan();
        assert_eq!(plan.heap, HeapStorage::Private);
        assert_eq!(plan.host_visible_buffer, HostVisibleBufferPlacement::SharedStandalone);
        assert_eq!(plan.texture, TextureStorage::Private);
        assert!(!plan.host_visible_buffers_from_heap());
    }

    #[test]
    fn intel_igpu_reports_unified_memory_and_still_uses_private_heaps() {
        // Iris / UHD: Mac2, hasUnifiedMemory, not an Apple-family GPU.
        // This is the single-GPU Intel MacBook that still aborted under a
        // hasUnifiedMemory predicate.
        let feats = features(true, false, true);
        assert!(!feats.allows_shared_heaps());
        assert_eq!(feats.family_summary(), "mac2-unified");
        let plan = feats.plan();
        assert_eq!(plan.heap, HeapStorage::Private);
        assert_eq!(plan.host_visible_buffer, HostVisibleBufferPlacement::SharedStandalone);
        assert_eq!(plan.texture, TextureStorage::Private);
        assert!(!plan.host_visible_buffers_from_heap());
    }

    #[test]
    fn apple_family_allows_shared_heaps_without_the_unified_bit() {
        // The heap gate is supportsFamily(Apple1), not hasUnifiedMemory.
        let feats = features(false, true, true);
        assert!(feats.allows_shared_heaps());
        assert_eq!(feats.family_summary(), "apple-non-unified");
        assert_eq!(feats.plan().heap, HeapStorage::Shared);
        assert_eq!(feats.plan().texture, TextureStorage::Shared);
        assert!(feats.plan().host_visible_buffers_from_heap());
    }

    #[test]
    fn older_discrete_mac_matches_intel_plan() {
        let feats = features(false, false, false);
        assert_eq!(feats.family_summary(), "discrete");
        assert_eq!(feats.plan(), features(false, false, true).plan());
    }
}
