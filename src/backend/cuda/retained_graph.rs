//! Retained CUDA GraphExec artifacts for Goldy partition replay.
//!
//! Graph objects are not thread-safe; the submission worker owns the registry and
//! is the only thread that creates, launches, or destroys [`OwnedCudaGraph`] values while
//! the worker is alive. After worker shutdown, the registry is dropped exclusively
//! on the teardown path.

use anyhow::{Context as _, Result};
use cudarc::driver::{sys, CudaModule, CudaSlice, CudaStream};
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

/// Worker-owned graph + GraphExec pair (mirrors cudarc's [`CudaGraph`] with a public
/// end-capture → mutate → instantiate seam for device-updatable nodes).
pub(super) struct OwnedCudaGraph {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
    stream: Arc<CudaStream>,
}

impl OwnedCudaGraph {
    pub fn launch(&self) -> Result<()> {
        self.stream
            .context()
            .bind_to_thread()
            .context("CUDA: bind context for graph launch")?;
        unsafe { cudarc::driver::result::graph::launch(self.cu_graph_exec, self.stream.cu_stream()) }
            .context("CUDA: cuGraphLaunch failed")
    }

    pub fn upload(&self) -> Result<()> {
        self.stream
            .context()
            .bind_to_thread()
            .context("CUDA: bind context for graph upload")?;
        unsafe { cudarc::driver::result::graph::upload(self.cu_graph_exec, self.stream.cu_stream()) }
            .context("CUDA: cuGraphUpload failed")
    }
}

impl Drop for OwnedCudaGraph {
    fn drop(&mut self) {
        let _ = self.stream.context().bind_to_thread();
        let exec = std::mem::replace(&mut self.cu_graph_exec, std::ptr::null_mut());
        if !exec.is_null() {
            let _ = unsafe { cudarc::driver::result::graph::exec_destroy(exec) };
        }
        let graph = std::mem::replace(&mut self.cu_graph, std::ptr::null_mut());
        if !graph.is_null() {
            let _ = unsafe { cudarc::driver::result::graph::destroy(graph) };
        }
    }
}

// SAFETY: created, launched, and dropped only while exclusive access is held by the
// goldy-submit worker (via this registry's mutex), or after that worker has been flushed
// and shut down during device/context teardown.
unsafe impl Send for OwnedCudaGraph {}

/// One captured partition graph plus pinned resources it references.
pub(super) struct CudaRetainedPartition {
    pub graph: OwnedCudaGraph,
    /// Keep buffer allocations alive for the lifetime of the graph (baked device pointers).
    #[allow(dead_code)]
    pub buffers: Vec<Arc<Mutex<CudaSlice<u8>>>>,
    /// Keep PTX modules alive for the lifetime of the graph.
    #[allow(dead_code)]
    pub modules: Vec<Arc<CudaModule>>,
    /// Keep CUDA texture arrays / tex/surf objects alive for baked handles.
    #[allow(dead_code)]
    pub textures: Vec<Arc<super::texture::CudaTextureResource>>,
    pub last_launch_tv: u64,
}

// SAFETY: see [`OwnedCudaGraph`].
unsafe impl Send for CudaRetainedPartition {}

/// Worker-owned map of retained CUDA graphs keyed by `(context, partition_key)`.
#[derive(Default)]
pub(super) struct GraphRegistry {
    graphs: HashMap<(crate::backend::ContextHandle, u64), CudaRetainedPartition>,
    /// Graphs removed from `graphs` but still referenced by in-flight launches.
    pending_drops: Vec<(u64, CudaRetainedPartition)>,
}

impl GraphRegistry {
    pub fn insert(&mut self, ctx: crate::backend::ContextHandle, key: u64, partition: CudaRetainedPartition) {
        if let Some(old) = self.graphs.insert((ctx, key), partition) {
            drop(old);
        }
    }

    pub fn get_mut(&mut self, ctx: crate::backend::ContextHandle, key: u64) -> Option<&mut CudaRetainedPartition> {
        self.graphs.get_mut(&(ctx, key))
    }

    pub fn remove(&mut self, ctx: crate::backend::ContextHandle, key: u64) -> Option<CudaRetainedPartition> {
        self.graphs.remove(&(ctx, key))
    }

    pub fn remove_context(&mut self, ctx: crate::backend::ContextHandle) -> Vec<CudaRetainedPartition> {
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

/// Capture `record` into a new [`OwnedCudaGraph`] on `stream` and instantiate it.
/// Prefer [`pending_submit::capture_partition_graph`] for the production path (it
/// also finalizes device-updatable indirect consumers).
#[cfg(test)]
pub(super) fn capture_ops_to_graph(
    stream: &Arc<CudaStream>,
    record: impl FnOnce() -> Result<()>,
) -> Result<OwnedCudaGraph> {
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)
        .context("CUDA: begin_capture failed")?;
    let capture_result = record();
    let end = unsafe { cudarc::driver::result::stream::end_capture(stream.cu_stream()) };
    if let Err(error) = &capture_result {
        let _ = stream.synchronize();
        return Err(anyhow::anyhow!("CUDA: graph capture recording failed: {error:#}"));
    }
    let cu_graph = end.context("CUDA: end_capture failed")?;
    if cu_graph.is_null() {
        anyhow::bail!("CUDA: end_capture returned an empty graph");
    }
    instantiate_owned(stream, cu_graph)
}

pub(super) fn instantiate_owned(stream: &Arc<CudaStream>, cu_graph: sys::CUgraph) -> Result<OwnedCudaGraph> {
    let flags = sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
    let cu_graph_exec = match unsafe { cudarc::driver::result::graph::instantiate(cu_graph, flags) } {
        Ok(exec) => exec,
        Err(error) => {
            let _ = unsafe { cudarc::driver::result::graph::destroy(cu_graph) };
            return Err(error).context("CUDA: cuGraphInstantiate failed");
        }
    };
    Ok(OwnedCudaGraph {
        cu_graph,
        cu_graph_exec,
        stream: Arc::clone(stream),
    })
}

/// Opt the `consumer_ordinal`-th kernel node (0-based among kernel nodes) into
/// device-updatable mode and return its [`sys::CUgraphDeviceNode`] handle.
pub(super) fn make_kernel_node_device_updatable(
    cu_graph: sys::CUgraph,
    consumer_ordinal: usize,
) -> Result<sys::CUgraphDeviceNode> {
    let mut num_nodes: usize = 0;
    let r = unsafe { sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut num_nodes) };
    if r != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuGraphGetNodes(count) failed: {r:?}");
    }
    let mut nodes = vec![std::ptr::null_mut(); num_nodes];
    let r = unsafe { sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut num_nodes) };
    if r != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuGraphGetNodes failed: {r:?}");
    }
    let mut kernel_nodes = Vec::new();
    for node in nodes.into_iter().take(num_nodes) {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        let r = unsafe { sys::cuGraphNodeGetType(node, &mut ty) };
        if r != sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("CUDA: cuGraphNodeGetType failed: {r:?}");
        }
        if ty == sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL {
            kernel_nodes.push(node);
        }
    }
    let node = *kernel_nodes.get(consumer_ordinal).with_context(|| {
        format!(
            "CUDA: device-updatable consumer ordinal {consumer_ordinal} out of range \
             ({} kernel nodes)",
            kernel_nodes.len()
        )
    })?;
    let mut attr_value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
    attr_value.deviceUpdatableKernelNode.deviceUpdatable = 1;
    attr_value.deviceUpdatableKernelNode.devNode = std::ptr::null_mut();
    let r = unsafe {
        sys::cuGraphKernelNodeSetAttribute(
            node,
            sys::CUkernelNodeAttrID::CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE,
            &attr_value,
        )
    };
    if r != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuGraphKernelNodeSetAttribute(DEVICE_UPDATABLE) failed: {r:?}");
    }
    let mut out_value: sys::CUlaunchAttributeValue = unsafe { std::mem::zeroed() };
    let r = unsafe {
        sys::cuGraphKernelNodeGetAttribute(
            node,
            sys::CUkernelNodeAttrID::CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE,
            &mut out_value,
        )
    };
    if r != sys::CUresult::CUDA_SUCCESS {
        anyhow::bail!("CUDA: cuGraphKernelNodeGetAttribute(DEVICE_UPDATABLE) failed: {r:?}");
    }
    let mut dev_node = unsafe { out_value.deviceUpdatableKernelNode.devNode };
    if dev_node.is_null() {
        dev_node = unsafe { attr_value.deviceUpdatableKernelNode.devNode };
    }
    if dev_node.is_null() {
        anyhow::bail!("CUDA: device-updatable kernel node returned a null CUgraphDeviceNode");
    }
    Ok(dev_node)
}
