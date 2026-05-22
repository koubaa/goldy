//! GPU profiling helpers controlled by `GOLDY_GPU_PROFILE`.
//!
//! - Any non-empty value enables structured [`tracing`] logs for GPU timings.
//! - `chrome`, `chrome:`, `chrome=<path>`, or `chrome:<path>` additionally writes a
//!   Perfetto-compatible Chrome trace JSON array (pretty-printed) to disk after each readback.

use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

/// Parsed configuration from [`std::env::var`] `"GOLDY_GPU_PROFILE"`.
#[derive(Clone, Debug)]
pub struct GpuProfileConfig {
    pub enabled: bool,
    pub chrome_path: Option<PathBuf>,
}

impl GpuProfileConfig {
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var("GOLDY_GPU_PROFILE") else {
            return Self {
                enabled: false,
                chrome_path: None,
            };
        };
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Self {
                enabled: false,
                chrome_path: None,
            };
        }

        let lower = trimmed.to_ascii_lowercase();
        let chrome_path = if lower.starts_with("chrome") {
            let rest = trimmed
                .get("chrome".len()..)
                .unwrap_or("")
                .trim_start_matches(':')
                .trim_start_matches('=')
                .trim();
            if rest.is_empty() {
                Some(PathBuf::from("goldy_gpu_trace.json"))
            } else {
                Some(PathBuf::from(rest))
            }
        } else {
            None
        };

        Self {
            enabled: true,
            chrome_path,
        }
    }
}

pub static GPU_PROFILE_CONFIG: LazyLock<GpuProfileConfig> =
    LazyLock::new(GpuProfileConfig::from_env);

#[inline]
pub fn gpu_profile_enabled() -> bool {
    GPU_PROFILE_CONFIG.enabled
}

/// GPU duration for one dispatch (nanoseconds).
#[derive(Clone, Copy, Debug)]
pub struct DispatchGpuNs {
    pub label: &'static str,
    pub gpu_ns: u64,
}

pub fn log_cb_timing(backend: &str, timeline: u64, gpu_ms: f64) {
    if !gpu_profile_enabled() {
        return;
    }
    tracing::info!("[GPU] backend={backend} timeline={timeline} gpu={gpu_ms:.3}ms");
    chrome_record_complete(backend, timeline, None, gpu_ms * 1_000_000.0);
}

pub fn log_dispatch_timings(backend: &str, timeline: u64, dispatches: &[DispatchGpuNs]) {
    if !gpu_profile_enabled() {
        return;
    }
    let mut total_ns = 0u64;
    for d in dispatches {
        let ms = d.gpu_ns as f64 / 1_000_000.0;
        total_ns = total_ns.saturating_add(d.gpu_ns);
        tracing::info!(
            "[GPU] backend={backend} timeline={timeline} dispatch={:?} gpu={ms:.3}ms",
            d.label
        );
        chrome_record_complete(backend, timeline, Some(d.label), ms * 1_000_000.0);
    }
    let total_ms = total_ns as f64 / 1_000_000.0;
    tracing::info!(
        "[GPU] backend={backend} timeline={timeline} total_dispatch_gpu={total_ms:.3}ms"
    );
}

fn wall_ts_us() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1_000_000.0)
        .unwrap_or(0.0)
}

fn escape_json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn chrome_record_complete(
    backend: &str,
    timeline: u64,
    dispatch_label: Option<&'static str>,
    dur_us: f64,
) {
    let Some(path) = GPU_PROFILE_CONFIG.chrome_path.clone() else {
        return;
    };
    let name = match dispatch_label {
        Some(l) => format!("goldy-{backend}-{l}"),
        None => format!("goldy-{backend}-cmdbuf-timeline-{timeline}"),
    };
    let ts_us = wall_ts_us();
    let tid = format!("goldy-{backend}");

    let mut guard = CHROME_EVENTS.lock().unwrap();
    guard.push(ChromeEvent {
        name,
        ts_us,
        dur_us,
        tid,
        timeline,
    });
    if let Err(e) = flush_chrome_trace(&path, &guard) {
        tracing::warn!("GOLDY_GPU_PROFILE chrome export failed: {e}");
    }
}

struct ChromeEvent {
    name: String,
    ts_us: f64,
    dur_us: f64,
    tid: String,
    timeline: u64,
}

static CHROME_EVENTS: Mutex<Vec<ChromeEvent>> = Mutex::new(Vec::new());

fn flush_chrome_trace(path: &Path, events: &[ChromeEvent]) -> std::io::Result<()> {
    let mut s = String::from("[\n");
    for (i, e) in events.iter().enumerate() {
        if i > 0 {
            s.push_str(",\n");
        }
        s.push_str(&format!(
            r#"  {{"name":"{}","cat":"gpu","ph":"X","ts":{:.3},"dur":{:.3},"pid":1,"tid":"{}","args":{{"timeline":{}}}}}"#,
            escape_json_str(&e.name),
            e.ts_us,
            e.dur_us,
            escape_json_str(&e.tid),
            e.timeline
        ));
    }
    s.push_str("\n]");
    std::fs::write(path, s)
}
