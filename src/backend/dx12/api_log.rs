//! DX12 native-API call trace logger.
//!
//! Enabled by setting `GOLDY_API_LOG=<path>` (e.g. `GOLDY_API_LOG=C:\tmp\goldy.ndjson`).
//! Each logged D3D12/DXGI-relevant call appends one JSON object (NDJSON) to that file:
//!
//! ```text
//! {"t_us":1432.10,"tid":"ThreadId(3)","op":"device_create","adapter_id":4294967295,"device":1}
//! {"t_us":1440.55,"tid":"ThreadId(3)","op":"context_create","device":1,"ctx":1,"is_warp":true}
//! {"t_us":1460.02,"tid":"ThreadId(7)","op":"queue_wait","queue":140701871616,"producer_fence":140701871888,"value":5}
//! {"t_us":1461.20,"tid":"ThreadId(7)","op":"execute_command_lists","queue":140701871616,"num_lists":1}
//! {"t_us":1461.40,"tid":"ThreadId(7)","op":"queue_signal","queue":140701871616,"ctx_fence":140701871904,"value":9}
//! {"t_us":1470.00,"tid":"ThreadId(7)","op":"device_removed","device":1,"hresult":-2005270496}
//! ```
//!
//! Fields present on every record: `t_us` (microseconds since process start, monotonic) and
//! `tid` (thread id). Everything else is `op`-specific. `queue`/`producer_fence`/`ctx_fence`
//! are stable per-object identifiers (the underlying COM interface pointer, via
//! [`com_identity`]) rather than Goldy context handles, since the low-level submit helper that
//! issues `Wait`/`ExecuteCommandLists`/`Signal` doesn't have a context handle in scope — grep
//! the surrounding `context_create`/`context_destroy` records for the same device to map a
//! pointer back to a context by elimination.
//!
//! Every D3D12 command queue in Goldy is externally synchronized by a `Mutex` already (see
//! `LogicalDevice::queue_lock` / `Dx12SubmissionContext::queue_lock`), so calls *to the same
//! queue* are already totally ordered; what this log lets you reconstruct is the
//! *interleaving across different queues/devices/threads*, which is exactly what's needed to
//! spot cross-context or cross-device anomalies (e.g. a queue/device being torn down
//! concurrently with another thread creating a new one on the same adapter).
//!
//! Like the Metal API logger this uses a background writer thread so the hot submit path
//! stays allocation-light: the foreground formats a `String` and `try_send`s it, dropping on
//! channel backpressure rather than blocking a queue operation. Set `GOLDY_API_LOG_SYNC=1` to
//! force synchronous, unbuffered writes (useful when correlating against a crash where the
//! writer thread might not get to flush).

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, Once, OnceLock};
use std::time::Instant;

pub struct ApiLogConfig {
    pub path: PathBuf,
    pub sync: bool,
}

impl ApiLogConfig {
    fn from_env() -> Option<Self> {
        let path = std::env::var("GOLDY_API_LOG").ok().filter(|s| !s.is_empty())?;
        let sync = std::env::var("GOLDY_API_LOG_SYNC").map(|v| v == "1").unwrap_or(false);
        Some(Self {
            path: PathBuf::from(path),
            sync,
        })
    }
}

static EPOCH: LazyLock<Instant> = LazyLock::new(Instant::now);

#[inline]
fn t_us() -> f64 {
    EPOCH.elapsed().as_secs_f64() * 1_000_000.0
}

#[inline]
fn tid_name() -> String {
    std::thread::current()
        .name()
        .map(str::to_owned)
        .unwrap_or_else(|| format!("{:?}", std::thread::current().id()))
}

const CHAN_CAP: usize = 4096;

static SENDER: OnceLock<std::sync::mpsc::SyncSender<String>> = OnceLock::new();
static SYNC_WRITER: OnceLock<std::sync::Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);
static INIT: Once = Once::new();

/// Idempotent; call at the start of every `Dx12Backend::new()`. Only the *first* call
/// process-wide actually opens/truncates the file and spawns the writer thread — `Dx12Backend`
/// is constructed once per `Device`, and with many devices created across parallel tests we
/// must not re-truncate (and thereby erase) an already-open log.
pub(super) fn init() {
    INIT.call_once(|| {
        let Some(cfg) = ApiLogConfig::from_env() else {
            return;
        };

        let file = match std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&cfg.path)
        {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("GOLDY_API_LOG: cannot open {:?}: {e}", cfg.path);
                return;
            }
        };

        tracing::info!("GOLDY_API_LOG (dx12) enabled -> {:?}", cfg.path);
        ENABLED.store(true, Ordering::Relaxed);

        if cfg.sync {
            let writer = std::io::BufWriter::new(file);
            let _ = SYNC_WRITER.set(std::sync::Mutex::new(writer));
            return;
        }

        let (tx, rx) = std::sync::mpsc::sync_channel::<String>(CHAN_CAP);
        let _ = SENDER.set(tx);

        std::thread::Builder::new()
            .name("goldy_dx12_api_log_writer".into())
            .spawn(move || {
                use std::io::Write;
                let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
                while let Ok(line) = rx.recv() {
                    let _ = writeln!(writer, "{line}");
                    loop {
                        match rx.try_recv() {
                            Ok(extra) => {
                                let _ = writeln!(writer, "{extra}");
                            }
                            Err(_) => break,
                        }
                    }
                    let _ = writer.flush();
                }
                let _ = writer.flush();
            })
            .expect("spawn goldy_dx12_api_log_writer");
    });
}

#[inline]
fn emit(line: String) {
    if let Some(sw) = SYNC_WRITER.get() {
        use std::io::Write;
        if let Ok(mut w) = sw.lock() {
            let _ = writeln!(w, "{line}");
        }
        return;
    }
    if let Some(tx) = SENDER.get() {
        let _ = tx.try_send(line);
    }
}

#[inline]
pub(super) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Stable per-object identifier for a COM interface, used to correlate distinct queues/fences
/// across log lines without threading numeric handles through every low-level call site.
pub(super) fn com_identity<I: windows::core::Interface>(obj: &I) -> u64 {
    obj.as_raw() as u64
}

/// `device_create` — `D3D12CreateDevice` succeeded.
pub(super) fn log_device_create(adapter_id: u32, device: super::DeviceHandle) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"device_create","adapter_id":{},"device":{}}}"#,
        t_us(),
        tid_name(),
        adapter_id,
        device
    ));
}

/// `device_destroy` — device teardown started (`destroy_device_inner`).
pub(super) fn log_device_destroy(device: super::DeviceHandle) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"device_destroy","device":{}}}"#,
        t_us(),
        tid_name(),
        device
    ));
}

/// `context_create` — a `Dx12SubmissionContext` (and its own `ID3D12CommandQueue`) was created.
pub(super) fn log_context_create(device: super::DeviceHandle, ctx: super::ContextHandle, is_warp: bool) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"context_create","device":{},"ctx":{},"is_warp":{}}}"#,
        t_us(),
        tid_name(),
        device,
        ctx,
        is_warp
    ));
}

/// `context_destroy` — a context's queue/fence/frame-table were torn down.
pub(super) fn log_context_destroy(device: super::DeviceHandle, ctx: super::ContextHandle) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"context_destroy","device":{},"ctx":{}}}"#,
        t_us(),
        tid_name(),
        device,
        ctx
    ));
}

/// `queue_wait` — `ID3D12CommandQueue::Wait` on a cross-context producer fence.
/// `queue` and `producer_fence` are the raw `IUnknown` COM pointers (stable per-object
/// identifiers) of the waiting queue and the fence being waited on, since the calling
/// context's numeric handle isn't threaded through this low-level submit helper.
pub(super) fn log_queue_wait(queue: u64, producer_fence: u64, value: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"queue_wait","queue":{},"producer_fence":{},"value":{}}}"#,
        t_us(),
        tid_name(),
        queue,
        producer_fence,
        value
    ));
}

/// `execute_command_lists` — `ID3D12CommandQueue::ExecuteCommandLists`.
pub(super) fn log_execute_command_lists(queue: u64, num_lists: usize) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"execute_command_lists","queue":{},"num_lists":{}}}"#,
        t_us(),
        tid_name(),
        queue,
        num_lists
    ));
}

/// `queue_signal` — `ID3D12CommandQueue::Signal` on this context's own fence.
pub(super) fn log_queue_signal(queue: u64, ctx_fence: u64, value: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"queue_signal","queue":{},"ctx_fence":{},"value":{}}}"#,
        t_us(),
        tid_name(),
        queue,
        ctx_fence,
        value
    ));
}

/// `fence_wait_cpu` — a CPU-side blocking `SetEventOnCompletion`/`WaitForSingleObject` on a
/// fence (e.g. context destroy draining in-flight GPU work, or a device-wide idle wait).
pub(super) fn log_fence_wait_cpu(location: &str, target_value: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"fence_wait_cpu","location":"{}","target_value":{}}}"#,
        t_us(),
        tid_name(),
        location,
        target_value
    ));
}

/// `device_removed` — a fence read returned `u64::MAX`; logs `GetDeviceRemovedReason()`.
pub(super) fn log_device_removed(device: super::DeviceHandle, hresult: i32) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"device_removed","device":{},"hresult":{}}}"#,
        t_us(),
        tid_name(),
        device,
        hresult
    ));
}
