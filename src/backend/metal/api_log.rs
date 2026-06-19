//! Metal native-API call trace logger.
//!
//! Enabled by setting `GOLDY_API_LOG=<path>` (e.g. `GOLDY_API_LOG=/tmp/goldy.ndjson`).
//! Each GPU API call appends one JSON object (NDJSON) to that file:
//!
//! ```text
//! {"t_us":1432.10,"tid":"TID_RENDER","op":"dispatch","label":"fine","wg":[120,68,1]}
//! {"t_us":1433.05,"tid":"TID_RENDER","op":"encoder_open","kind":"compute"}
//! {"t_us":1450.88,"tid":"TID_PRESENT","op":"commit","tv":91823}
//! ```
//!
//! Fields present on every record:
//! - `t_us`  — microseconds since process start (monotonic)
//! - `tid`   — thread name (or numeric id if unnamed)
//! - `op`    — operation name (see variants below)
//!
//! The file is written by a background thread to keep the hot path allocation-free:
//! the foreground just formats a stack string and `try_send`s it.  On channel
//! backpressure the record is dropped (visible as a gap) rather than blocking the
//! render thread.  A `GOLDY_API_LOG_SYNC=1` env var forces synchronous writes for
//! tests where every record must land.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{LazyLock, OnceLock};
use std::time::Instant;

// ── Config ────────────────────────────────────────────────────────────────────

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

// ── Process-start epoch ───────────────────────────────────────────────────────

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

// ── Background writer ─────────────────────────────────────────────────────────

// Channel capacity: 4096 records. Drops on overflow rather than stalling render.
const CHAN_CAP: usize = 4096;

static SENDER: OnceLock<std::sync::mpsc::SyncSender<String>> = OnceLock::new();
static ENABLED: AtomicBool = AtomicBool::new(false);

/// Must be called once at backend initialisation (before any GPU work).
/// Reads `GOLDY_API_LOG` and, if set, spawns the writer thread.
pub(super) fn init() {
    let Some(cfg) = ApiLogConfig::from_env() else {
        return;
    };

    // Truncate/create the output file.
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

    tracing::info!("GOLDY_API_LOG enabled → {:?}", cfg.path);
    ENABLED.store(true, Ordering::Relaxed);

    if cfg.sync {
        // Synchronous path: write directly, no background thread.
        // SYNC_WRITER is checked first in `emit`; `writeln!` brings in `Write` inline.
        let writer = std::io::BufWriter::new(file);
        let (tx, _rx) = std::sync::mpsc::sync_channel::<String>(1);
        let _ = SENDER.set(tx);
        let _ = SYNC_WRITER.set(std::sync::Mutex::new(writer));
        return;
    }

    let (tx, rx) = std::sync::mpsc::sync_channel::<String>(CHAN_CAP);
    let _ = SENDER.set(tx);

    std::thread::Builder::new()
        .name("goldy_api_log_writer".into())
        .spawn(move || {
            use std::io::Write;
            let mut writer = std::io::BufWriter::with_capacity(64 * 1024, file);
            while let Ok(line) = rx.recv() {
                let _ = writeln!(writer, "{line}");
                // Flush lazily: drain the channel first, then flush.
                loop {
                    match rx.try_recv() {
                        Ok(extra) => { let _ = writeln!(writer, "{extra}"); }
                        Err(_) => break,
                    }
                }
                let _ = writer.flush();
            }
            let _ = writer.flush();
        })
        .expect("spawn goldy_api_log_writer");
}

static SYNC_WRITER: OnceLock<std::sync::Mutex<std::io::BufWriter<std::fs::File>>> = OnceLock::new();

// ── Hot-path emit ─────────────────────────────────────────────────────────────

/// Emit a pre-formatted NDJSON line.  
/// On the async path: `try_send` (non-blocking, drops on overflow).  
/// On the sync path: writes inline.
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

// ── Public logging helpers ────────────────────────────────────────────────────

#[inline]
pub(super) fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// `encoder_open` — a new MTLComputeCommandEncoder or MTLBlitCommandEncoder was created.
pub(super) fn log_encoder_open(kind: &'static str) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"encoder_open","kind":"{}"}}"#,
        t_us(),
        tid_name(),
        kind
    ));
}

/// `encoder_end` — an encoder's `endEncoding` was called.
pub(super) fn log_encoder_end(kind: &'static str) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"encoder_end","kind":"{}"}}"#,
        t_us(),
        tid_name(),
        kind
    ));
}

/// `dispatch` — `dispatchThreadgroups` called.
pub(super) fn log_dispatch(label: Option<&str>, wg_x: u32, wg_y: u32, wg_z: u32) {
    let lbl = label.unwrap_or("?");
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"dispatch","label":"{}","wg":[{},{},{}]}}"#,
        t_us(),
        tid_name(),
        lbl,
        wg_x,
        wg_y,
        wg_z
    ));
}

/// `dispatch_indirect` — `dispatchThreadgroupsWithIndirectBuffer` called.
pub(super) fn log_dispatch_indirect(label: Option<&str>, buf: u64, offset: u64) {
    let lbl = label.unwrap_or("?");
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"dispatch_indirect","label":"{}","buf":{},"offset":{}}}"#,
        t_us(),
        tid_name(),
        lbl,
        buf,
        offset
    ));
}

/// `dispatch_batch` — batched indirect dispatches (DispatchBatch).
pub(super) fn log_dispatch_batch(label: Option<&str>, count: u32) {
    let lbl = label.unwrap_or("?");
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"dispatch_batch","label":"{}","count":{}}}"#,
        t_us(),
        tid_name(),
        lbl,
        count
    ));
}

/// `barrier` — `memoryBarrierWithScope` (global barrier).
pub(super) fn log_barrier() {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"barrier"}}"#,
        t_us(),
        tid_name()
    ));
}

/// `resource_barrier` — `memoryBarrierWithResources`.
pub(super) fn log_resource_barrier(buf_count: usize, tex_count: usize) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"resource_barrier","bufs":{},"texs":{}}}"#,
        t_us(),
        tid_name(),
        buf_count,
        tex_count
    ));
}

/// `copy_texture` — blit encoder `copyFromTexture`.
pub(super) fn log_copy_texture(src: u64, dst: u64, w: u64, h: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"copy_texture","src":{},"dst":{},"w":{},"h":{}}}"#,
        t_us(),
        tid_name(),
        src,
        dst,
        w,
        h
    ));
}

/// `copy_buffer` — blit encoder `copyFromBuffer`.
pub(super) fn log_copy_buffer(src: u64, dst: u64, size: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"copy_buffer","src":{},"dst":{},"size":{}}}"#,
        t_us(),
        tid_name(),
        src,
        dst,
        size
    ));
}

/// `fill_buffer` — blit encoder `fillBuffer` (clear).
pub(super) fn log_fill_buffer(buf: u64, size: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"fill_buffer","buf":{},"size":{}}}"#,
        t_us(),
        tid_name(),
        buf,
        size
    ));
}

/// `commit` — `MTLCommandBuffer.commit` called.
pub(super) fn log_commit(tv: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"commit","tv":{}}}"#,
        t_us(),
        tid_name(),
        tv
    ));
}

/// `next_drawable` — `nextDrawable` returned (or failed).
pub(super) fn log_next_drawable(ok: bool) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"next_drawable","ok":{}}}"#,
        t_us(),
        tid_name(),
        ok
    ));
}

/// `present_drawable` — `presentDrawable:` scheduled.
pub(super) fn log_present_drawable(tv: u64) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"present_drawable","tv":{}}}"#,
        t_us(),
        tid_name(),
        tv
    ));
}

/// `write_texture` — blit-encoder texture upload.
pub(super) fn log_write_texture(tex: u64, w: u32, h: u32, bytes: usize) {
    emit(format!(
        r#"{{"t_us":{:.3},"tid":"{}","op":"write_texture","tex":{},"w":{},"h":{},"bytes":{}}}"#,
        t_us(),
        tid_name(),
        tex,
        w,
        h,
        bytes
    ));
}
