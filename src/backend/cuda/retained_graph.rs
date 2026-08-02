//! Retained CUDA GraphExec artifacts for Goldy partition replay.
//!
//! Graph objects are not thread-safe; the submission worker owns the registry and
//! is the only thread that creates, launches, or destroys [`CudaGraph`] values while
//! the worker is alive. After worker shutdown, the registry is dropped exclusively
//! on the teardown path.

use anyhow::{Context as _, Result};
use cudarc::driver::{sys, CudaGraph, CudaModule, CudaSlice, CudaStream};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Process-visible counters for capture / launch / fallback / eviction.
#[derive(Debug, Default)]
pub(super) struct CudaGraphStats {
    pub captures: AtomicU64,
    pub launches: AtomicU64,
    pub fallbacks: AtomicU64,
    pub evictions: AtomicU64,
}

impl CudaGraphStats {
    pub fn snapshot(&self) -> CudaGraphStatsSnapshot {
        CudaGraphStatsSnapshot {
            captures: self.captures.load(Ordering::Relaxed),
            launches: self.launches.load(Ordering::Relaxed),
            fallbacks: self.fallbacks.load(Ordering::Relaxed),
            evictions: self.evictions.load(Ordering::Relaxed),
        }
    }

    pub fn reset(&self) {
        self.captures.store(0, Ordering::Relaxed);
        self.launches.store(0, Ordering::Relaxed);
        self.fallbacks.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CudaGraphStatsSnapshot {
    pub captures: u64,
    pub launches: u64,
    pub fallbacks: u64,
    pub evictions: u64,
}

/// One captured partition graph plus pinned resources it references.
pub(super) struct CudaRetainedPartition {
    pub graph: CudaGraph,
    /// Keep buffer allocations alive for the lifetime of the graph (baked device pointers).
    #[allow(dead_code)]
    pub buffers: Vec<Arc<Mutex<CudaSlice<u8>>>>,
    /// Keep PTX modules alive for the lifetime of the graph.
    #[allow(dead_code)]
    pub modules: Vec<Arc<CudaModule>>,
    pub last_launch_tv: u64,
}

// SAFETY: `CudaGraph` is only created, launched, and dropped while exclusive access is
// held by the goldy-submit worker (via this registry's mutex), or after that worker has
// been flushed and shut down during device/context teardown.
unsafe impl Send for CudaRetainedPartition {}

/// Worker-owned map of retained CUDA graphs keyed by `(context, partition_key)`.
#[derive(Default)]
pub(super) struct GraphRegistry {
    graphs: HashMap<(crate::backend::ContextHandle, u64), CudaRetainedPartition>,
    /// Graphs removed from `graphs` but still referenced by in-flight launches.
    pending_drops: Vec<(u64, CudaRetainedPartition)>,
}

impl GraphRegistry {
    pub fn insert(
        &mut self,
        ctx: crate::backend::ContextHandle,
        key: u64,
        partition: CudaRetainedPartition,
    ) {
        if let Some(old) = self.graphs.insert((ctx, key), partition) {
            drop(old);
        }
    }

    pub fn get_mut(
        &mut self,
        ctx: crate::backend::ContextHandle,
        key: u64,
    ) -> Option<&mut CudaRetainedPartition> {
        self.graphs.get_mut(&(ctx, key))
    }

    pub fn remove(
        &mut self,
        ctx: crate::backend::ContextHandle,
        key: u64,
    ) -> Option<CudaRetainedPartition> {
        self.graphs.remove(&(ctx, key))
    }

    pub fn remove_context(
        &mut self,
        ctx: crate::backend::ContextHandle,
    ) -> Vec<CudaRetainedPartition> {
        let keys: Vec<_> = self
            .graphs
            .keys()
            .filter_map(|(c, k)| (*c == ctx).then_some((*c, *k)))
            .collect();
        let mut out = Vec::with_capacity(keys.len());
        for key in keys {
            if let Some(p) = self.graphs.remove(&key) {
                out.push(p);
            }
        }
        out
    }

    pub fn defer_drop(&mut self, retire_at: u64, partition: CudaRetainedPartition) {
        self.pending_drops.push((retire_at, partition));
    }

    /// Drop any pending graphs whose last launch has retired.
    pub fn drain_retired(&mut self, retired: u64) {
        self.pending_drops.retain(|(retire_at, _)| *retire_at > retired);
    }
}

/// True when the driver is in launch-blocking mode (incompatible with stream capture).
pub(super) fn cuda_launch_blocking_active() -> bool {
    match std::env::var_os("CUDA_LAUNCH_BLOCKING") {
        Some(v) => v != "0",
        None => false,
    }
}

/// Capture `record` into a new [`CudaGraph`] on `stream` and instantiate it.
pub(super) fn capture_ops_to_graph(
    stream: &Arc<CudaStream>,
    record: impl FnOnce() -> Result<()>,
) -> Result<CudaGraph> {
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("CUDA: begin_capture failed")?;
    let capture_result = record();
    // Prefer a defined instantiate flag; cudarc's enum has no zero variant.
    let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
    let end = stream.end_capture(flags);
    if let Err(error) = &capture_result {
        // Best-effort stream recovery after a failed/invalidated capture.
        let _ = stream.synchronize();
        return Err(anyhow::anyhow!("CUDA: graph capture recording failed: {error:#}"));
    }
    let graph = end
        .context("CUDA: end_capture / instantiate failed")?
        .context("CUDA: end_capture returned an empty graph")?;
    Ok(graph)
}
