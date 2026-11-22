import Foundation
import Metal

/// Prints Metal device capabilities and, with a flag, tries one heap storage mode.
///
/// `--shared-heap` aborts on non-unified GPUs (`Shared storage mode disallowed`).
/// `--managed-heap` aborts on macOS: heaps accept Private and, only with unified
/// memory, Shared. `--private-heap` is the mode discrete GPUs allow.
/// Without a flag this only prints capabilities and exits 0.

guard let device = MTLCreateSystemDefaultDevice() else {
    fputs("METAL_DEVICE=none\n", stderr)
    exit(2)
}

let tier2 = device.argumentBuffersSupport == .tier2
let summary = [
    "METAL_DEVICE=\(device.name)",
    "unified=\(device.hasUnifiedMemory)",
    "apple1=\(device.supportsFamily(.apple1))",
    "mac2=\(device.supportsFamily(.mac2))",
    "lowPower=\(device.isLowPower)",
    "headless=\(device.isHeadless)",
    "tier2=\(tier2)",
].joined(separator: " ")
fputs(summary + "\n", stderr)
print(summary)

func modeName(_ mode: MTLStorageMode) -> String {
    switch mode {
    case .shared: return "shared"
    case .managed: return "managed"
    case .private: return "private"
    case .memoryless: return "memoryless"
    default: return "other(\(mode.rawValue))"
    }
}

let mode: MTLStorageMode?
if CommandLine.arguments.contains("--shared-heap") {
    mode = .shared
} else if CommandLine.arguments.contains("--managed-heap") {
    mode = .managed
} else if CommandLine.arguments.contains("--private-heap") {
    mode = .private
} else {
    mode = nil
}

guard let mode else { exit(0) }
let name = modeName(mode)

// Match Goldy's heap descriptor (tracked, automatic, default cache). 4 MB is
// large enough for devices that reject a 64 KB heap as too small; Goldy's
// runtime heaps are 64 MB. An illegal storage mode aborts inside makeHeap
// instead of returning nil.
let desc = MTLHeapDescriptor()
desc.size = 4 * 1024 * 1024
desc.storageMode = mode
desc.cpuCacheMode = .defaultCache
desc.type = .automatic
desc.hazardTrackingMode = .tracked

guard let heap = device.makeHeap(descriptor: desc) else {
    fputs("HEAP=\(name) nil\n", stderr)
    exit(3)
}
fputs("HEAP=\(name) ok size=\(heap.size) storage=\(modeName(heap.storageMode))\n", stderr)
exit(0)
