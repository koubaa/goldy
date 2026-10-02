//! Per-dispatch GPU timing for Metal, enabled by `GOLDY_METAL_DISPATCH_TIMING=<path>`.
//!
//! Apple GPUs sample counters only at encoder (stage) boundaries, and the driver
//! merges adjacent compute encoders in Instruments, so neither Metal System Trace
//! nor `GOLDY_GPU_PROFILE` can attribute time inside one encoder. While this
//! variable is set, every dispatch gets its own compute encoder with
//! start/end-of-encoder timestamp samples. When the command buffer completes, one
//! NDJSON line per dispatch is appended to `<path>`:
//!
//! ```text
//! {"cb":12,"i":3,"label":"layer0/attn/wq","us":8.42}
//! ```
//!
//! Splitting encoders serializes dispatches and adds per-encoder cost, so absolute
//! step times differ from normal runs; per-dispatch durations are what this is for.

use metal as mtl;
use objc::{msg_send, sel, sel_impl};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex};

static PATH: LazyLock<Option<PathBuf>> = LazyLock::new(|| {
    std::env::var_os("GOLDY_METAL_DISPATCH_TIMING")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
});
static FILE: LazyLock<Option<Mutex<std::fs::File>>> = LazyLock::new(|| {
    let path = PATH.as_ref()?;
    match std::fs::File::create(path) {
        Ok(f) => Some(Mutex::new(f)),
        Err(e) => {
            tracing::warn!("GOLDY_METAL_DISPATCH_TIMING: cannot create {}: {e}", path.display());
            None
        }
    }
});
static NEXT_CB: AtomicU64 = AtomicU64::new(0);

pub(super) fn enabled() -> bool {
    PATH.is_some()
}

/// Timestamp samples for the dispatches of one command buffer.
pub(super) struct DispatchTimer {
    samples: mtl::CounterSampleBuffer,
    capacity: usize,
    labels: Vec<String>,
}

impl DispatchTimer {
    /// A timer for up to `dispatches` encoders, or `None` when timing is off or the
    /// device cannot sample at stage boundaries.
    pub(super) fn new(device: &mtl::DeviceRef, dispatches: usize) -> Option<Self> {
        if !enabled() || dispatches == 0 {
            return None;
        }
        if !device.supports_counter_sampling(mtl::MTLCounterSamplingPoint::AtStageBoundary) {
            tracing::warn!("GOLDY_METAL_DISPATCH_TIMING: device cannot sample at stage boundaries");
            return None;
        }
        let sets = device.counter_sets();
        let timestamp = sets.iter().find(|s| s.name().eq_ignore_ascii_case("timestamp"))?;
        let desc = mtl::CounterSampleBufferDescriptor::new();
        desc.set_counter_set(timestamp);
        desc.set_storage_mode(mtl::MTLStorageMode::Shared);
        desc.set_sample_count((2 * dispatches) as u64);
        let samples = device.new_counter_sample_buffer_with_descriptor(&desc).ok()?;
        Some(Self {
            samples,
            capacity: dispatches,
            labels: Vec::with_capacity(dispatches),
        })
    }

    /// Descriptor for the next dispatch's encoder, or `None` once capacity is used.
    pub(super) fn next_pass(&mut self) -> Option<mtl::ComputePassDescriptor> {
        let n = self.labels.len();
        if n >= self.capacity {
            return None;
        }
        let pass = mtl::ComputePassDescriptor::new().to_owned();
        let attachment = pass.sample_buffer_attachments().object_at(0)?;
        attachment.set_sample_buffer(&self.samples);
        attachment.set_start_of_encoder_sample_index((2 * n) as u64);
        attachment.set_end_of_encoder_sample_index((2 * n + 1) as u64);
        self.labels.push(String::new());
        Some(pass)
    }

    /// Name the dispatch in the encoder opened by the last [`Self::next_pass`].
    pub(super) fn label_current(&mut self, label: Option<&str>) {
        if let Some(last) = self.labels.last_mut() {
            *last = label.unwrap_or("<unlabeled>").to_owned();
        }
    }

    /// Write this command buffer's dispatch durations once it completes.
    pub(super) fn attach(self, command_buffer: &mtl::CommandBufferRef) {
        if self.labels.is_empty() {
            return;
        }
        let cb_index = NEXT_CB.fetch_add(1, Ordering::Relaxed);
        let timer = Arc::new(self);
        let handler = block::ConcreteBlock::new(move |cb: &mtl::CommandBufferRef| {
            timer.report(cb, cb_index);
        })
        .copy();
        command_buffer.add_completed_handler(&handler);
    }

    fn report(&self, cb: &mtl::CommandBufferRef, cb_index: u64) {
        let Some(file) = FILE.as_ref() else { return };
        let n = self.labels.len();
        let ticks = resolve(&self.samples, 2 * n);
        if ticks.len() < 2 * n {
            return;
        }
        // Timestamp ticks are device units; scale them onto the command buffer's GPU
        // span, which Metal reports in seconds.
        let (gpu_start, gpu_end): (f64, f64) = unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        let valid = |t: u64| t != 0 && t != u64::MAX;
        let lo = ticks.iter().copied().filter(|&t| valid(t)).min().unwrap_or(0);
        let hi = ticks.iter().copied().filter(|&t| valid(t)).max().unwrap_or(0);
        let us_per_tick = if hi > lo { (gpu_end - gpu_start) * 1e6 / (hi - lo) as f64 } else { 0.0 };
        let mut out = String::new();
        for (i, label) in self.labels.iter().enumerate() {
            let (s, e) = (ticks[2 * i], ticks[2 * i + 1]);
            let us = if valid(s) && valid(e) && e >= s { (e - s) as f64 * us_per_tick } else { -1.0 };
            let label = label.replace('\\', "\\\\").replace('"', "\\\"");
            out.push_str(&format!("{{\"cb\":{cb_index},\"i\":{i},\"label\":\"{label}\",\"us\":{us:.3}}}\n"));
        }
        if let Ok(mut f) = file.lock() {
            let _ = f.write_all(out.as_bytes());
        }
    }
}

/// `resolveCounterRange:` as raw ticks. metal-rs 0.33 copies zero bytes here.
fn resolve(samples: &mtl::CounterSampleBufferRef, count: usize) -> Vec<u64> {
    let range = mtl::NSRange::new(0, count as u64);
    unsafe {
        let data: *mut objc::runtime::Object = msg_send![samples, resolveCounterRange: range];
        if data.is_null() {
            return Vec::new();
        }
        let len: usize = msg_send![data, length];
        let mut out = vec![0u64; len / 8];
        let () = msg_send![data, getBytes: out.as_mut_ptr() length: out.len() * 8];
        out
    }
}
