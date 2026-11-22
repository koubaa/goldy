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
    /// Non-Apple placement, or `GOLDY_METAL_PLACEMENT=discrete` on any GPU.
    pub discrete: bool,
    pub heap: HeapStorage,
    pub host_visible_buffer: HostVisibleBufferPlacement,
    pub texture: TextureStorage,
}

/// Where one buffer lives after allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetalBufferPlacement {
    /// Shared, suballocated from the Shared heap. `contents()` is valid.
    SharedHeap,
    /// Shared `newBuffer`. `contents()` is valid.
    SharedDevice,
    /// Private, from the Private heap. `contents()` is null.
    PrivateHeap,
    /// Private `newBuffer`. `contents()` is null.
    PrivateDevice,
}

impl MetalBufferPlacement {
    pub fn host_mapped(self) -> bool {
        matches!(self, Self::SharedHeap | Self::SharedDevice)
    }

    pub fn from_heap(self) -> bool {
        matches!(self, Self::SharedHeap | Self::PrivateHeap)
    }
}

/// `GOLDY_METAL_PLACEMENT=discrete` forces the non-Apple path on Apple silicon
/// so CI can run it. `apple` forces the Apple path. Anything else is unset.
pub fn forced_discrete() -> bool {
    static FORCE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    FORCE
        .get_or_init(|| match std::env::var("GOLDY_METAL_PLACEMENT") {
            Ok(value) if value.eq_ignore_ascii_case("discrete") => Some(true),
            Ok(value) if value.eq_ignore_ascii_case("apple") => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

/// Host-mapped vs device-local, before the jumbo-size fallback.
pub fn buffer_is_host_visible(
    kind: crate::types::BufferKind,
    flags: crate::types::BufferFlags,
    discrete: bool,
) -> bool {
    if flags.contains(crate::types::BufferFlags::GPU_ONLY) {
        return false;
    }
    if !discrete {
        return true;
    }
    match kind {
        crate::types::BufferKind::Broadcast => true,
        crate::types::BufferKind::Scattered => {
            flags.contains(crate::types::BufferFlags::CPU_READABLE)
                || flags.contains(crate::types::BufferFlags::CPU_WRITABLE)
        }
    }
}

/// Facts the floor and mesh checks need. Pure so profiles can be tested off-device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalDeviceFacts {
    pub apple_family: bool,
    pub apple7: bool,
    pub mac2_family: bool,
    pub metal3: bool,
    pub has_unified_memory: bool,
    pub argument_buffers_tier2: bool,
    pub read_write_texture_tier2: bool,
    pub os_major: u32,
    pub paravirtual: bool,
}

impl MetalDeviceFacts {
    pub fn memory_features(self) -> MetalMemoryFeatures {
        MetalMemoryFeatures {
            has_unified_memory: self.has_unified_memory,
            apple_family: self.apple_family,
            mac2_family: self.mac2_family,
        }
    }

    /// `None` when the device meets the floor. The string is a clean error, not a panic.
    pub fn floor_rejection(self) -> Option<&'static str> {
        // `os_major == 0` means the host version could not be read. Do not
        // reject Apple silicon CI on that alone.
        if self.os_major > 0 && self.os_major < 13 {
            return Some("Goldy Metal requires macOS 13 or newer (C-layout argument buffers)");
        }
        if !self.metal3 {
            return Some("Goldy Metal requires a Metal 3 GPU (Polaris and older are outside the floor)");
        }
        if !self.argument_buffers_tier2 {
            return Some("Goldy Metal requires Argument Buffers Tier 2");
        }
        // GitHub's Apple paravirtual GPU runs storage images but can under-report
        // the read-write tier. Non-Apple GPUs still have to advertise tier 2.
        if !self.apple_family && !self.read_write_texture_tier2 {
            return Some("Goldy Metal requires read-write texture tier 2");
        }
        None
    }

    pub fn meets_floor(self) -> bool {
        self.floor_rejection().is_none()
    }

    /// Mesh shaders need Metal 3 and Apple7 or Mac2, and not a paravirtual device.
    /// Polaris is Mac2 but not Metal 3, so it does not advertise mesh shaders.
    pub fn mesh_shaders(self) -> bool {
        !self.paravirtual && self.metal3 && (self.apple7 || self.mac2_family)
    }
}

impl MetalMemoryFeatures {
    /// Shared `MTLHeap`s and Shared textures are legal on Apple-family GPUs.
    pub fn allows_shared_heaps(self) -> bool {
        self.apple_family
    }

    pub fn plan(self) -> MetalStoragePlan {
        self.plan_with(forced_discrete())
    }

    pub fn plan_with(self, force_discrete: bool) -> MetalStoragePlan {
        let discrete = force_discrete || !self.allows_shared_heaps();
        if discrete {
            MetalStoragePlan {
                discrete: true,
                heap: HeapStorage::Private,
                host_visible_buffer: HostVisibleBufferPlacement::SharedStandalone,
                texture: TextureStorage::Private,
            }
        } else {
            MetalStoragePlan {
                discrete: false,
                heap: HeapStorage::Shared,
                host_visible_buffer: HostVisibleBufferPlacement::SharedHeap,
                texture: TextureStorage::Shared,
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
        !self.discrete && matches!(self.host_visible_buffer, HostVisibleBufferPlacement::SharedHeap)
    }
}

/// Refuse a Shared heap or Shared texture while discrete placement is active.
pub fn debug_refuse_shared_resource(discrete: bool, shared: bool, what: &str) {
    if discrete && shared {
        panic!("discrete Metal placement forbids a Shared {what}");
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
        let plan = features(true, true, true).plan_with(false);
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
        let plan = feats.plan_with(false);
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
        let plan = feats.plan_with(false);
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
        assert_eq!(feats.plan_with(false).heap, HeapStorage::Shared);
        assert_eq!(feats.plan_with(false).texture, TextureStorage::Shared);
        assert!(feats.plan_with(false).host_visible_buffers_from_heap());
    }

    #[test]
    fn older_discrete_mac_matches_intel_plan() {
        let feats = features(false, false, false);
        assert_eq!(feats.family_summary(), "discrete");
        assert_eq!(feats.plan_with(false), features(false, false, true).plan_with(false));
    }

    fn base_ok() -> MetalDeviceFacts {
        MetalDeviceFacts {
            apple_family: false,
            apple7: false,
            mac2_family: true,
            metal3: true,
            has_unified_memory: false,
            argument_buffers_tier2: true,
            read_write_texture_tier2: true,
            os_major: 14,
            paravirtual: false,
        }
    }

    #[test]
    fn profiles_select_floor_and_placement() {
        use crate::types::{BufferFlags, BufferKind};

        let m1 = MetalDeviceFacts {
            apple_family: true,
            apple7: true,
            has_unified_memory: true,
            ..base_ok()
        };
        assert!(m1.meets_floor());
        assert!(m1.mesh_shaders());
        assert!(!m1.memory_features().plan_with(false).discrete);
        assert!(buffer_is_host_visible(
            BufferKind::Scattered,
            BufferFlags::empty(),
            false
        ));

        let uhd630 = MetalDeviceFacts {
            has_unified_memory: true,
            ..base_ok()
        };
        assert!(uhd630.meets_floor());
        assert!(uhd630.memory_features().plan_with(false).discrete);
        assert!(!buffer_is_host_visible(
            BufferKind::Scattered,
            BufferFlags::empty(),
            true
        ));
        assert!(buffer_is_host_visible(
            BufferKind::Scattered,
            BufferFlags::CPU_WRITABLE,
            true
        ));
        assert!(buffer_is_host_visible(
            BufferKind::Broadcast,
            BufferFlags::empty(),
            true
        ));
        assert!(!buffer_is_host_visible(
            BufferKind::Scattered,
            BufferFlags::GPU_ONLY,
            true
        ));

        let iris = MetalDeviceFacts {
            has_unified_memory: true,
            ..base_ok()
        };
        assert!(iris.meets_floor());

        let amd_5500 = base_ok();
        assert!(amd_5500.meets_floor());
        assert!(amd_5500.memory_features().plan_with(false).discrete);
        assert!(amd_5500.mesh_shaders());

        let w6800 = base_ok();
        assert!(w6800.meets_floor());

        let polaris = MetalDeviceFacts {
            metal3: false,
            ..base_ok()
        };
        assert_eq!(
            polaris.floor_rejection(),
            Some("Goldy Metal requires a Metal 3 GPU (Polaris and older are outside the floor)")
        );
        assert!(!polaris.mesh_shaders());

        let para_arm = MetalDeviceFacts {
            apple_family: true,
            apple7: true,
            has_unified_memory: true,
            paravirtual: true,
            ..base_ok()
        };
        assert!(para_arm.meets_floor());
        assert!(!para_arm.mesh_shaders());

        let para_x86 = MetalDeviceFacts {
            metal3: false,
            mac2_family: false,
            argument_buffers_tier2: false,
            read_write_texture_tier2: false,
            ..base_ok()
        };
        assert!(para_x86.floor_rejection().is_some());

        let old_os = MetalDeviceFacts {
            os_major: 12,
            ..base_ok()
        };
        assert!(old_os.floor_rejection().unwrap().contains("macOS 13"));

        let forced = m1.memory_features().plan_with(true);
        assert!(forced.discrete);
        assert_eq!(forced.heap, HeapStorage::Private);
        assert_eq!(forced.texture, TextureStorage::Private);
    }
}
