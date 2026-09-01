//! Opt-in wall-clock breakdown for shader compile / PSO create.
//!
//! Set `GOLDY_SHADER_TIMING=1` to print per-phase lines and running totals.
//! Used to validate [goldy#175](https://github.com/koubaa/goldy/issues/175).

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
#[cfg(any(feature = "vulkan", feature = "dx12"))]
use std::time::Instant;

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("GOLDY_SHADER_TIMING").is_some_and(|v| v != "0"))
}

struct Totals {
    ns: BTreeMap<&'static str, u64>,
    n: BTreeMap<&'static str, u64>,
}

static TOTALS: OnceLock<Mutex<Totals>> = OnceLock::new();

fn totals() -> &'static Mutex<Totals> {
    TOTALS.get_or_init(|| {
        Mutex::new(Totals {
            ns: BTreeMap::new(),
            n: BTreeMap::new(),
        })
    })
}

pub(crate) fn record(phase: &'static str, detail: &str, dur: Duration) {
    if !enabled() {
        return;
    }
    let ms = dur.as_secs_f64() * 1000.0;
    if detail.is_empty() {
        eprintln!("GOLDY_SHADER_TIMING {phase} {ms:.3}ms");
    } else {
        eprintln!("GOLDY_SHADER_TIMING {phase} {detail} {ms:.3}ms");
    }
    let mut t = totals().lock().unwrap_or_else(|e| e.into_inner());
    *t.ns.entry(phase).or_insert(0) += dur.as_nanos() as u64;
    *t.n.entry(phase).or_insert(0) += 1;
}

#[cfg(any(feature = "vulkan", feature = "dx12"))]
pub(crate) fn scope(phase: &'static str, detail: impl Into<String>) -> TimingScope {
    TimingScope {
        phase,
        detail: detail.into(),
        start: Instant::now(),
        active: enabled(),
    }
}

#[cfg(any(feature = "vulkan", feature = "dx12"))]
pub(crate) struct TimingScope {
    phase: &'static str,
    detail: String,
    start: Instant,
    active: bool,
}

#[cfg(any(feature = "vulkan", feature = "dx12"))]
impl Drop for TimingScope {
    fn drop(&mut self) {
        if self.active {
            record(self.phase, &self.detail, self.start.elapsed());
        }
    }
}

/// Print accumulated phase totals (no-op unless `GOLDY_SHADER_TIMING` is set).
pub fn dump_totals(label: &str) {
    if !enabled() {
        return;
    }
    let t = totals().lock().unwrap_or_else(|e| e.into_inner());
    eprintln!("GOLDY_SHADER_TIMING TOTALS {label}");
    for (phase, ns) in &t.ns {
        let n = t.n.get(phase).copied().unwrap_or(0);
        let ms = *ns as f64 / 1_000_000.0;
        eprintln!("GOLDY_SHADER_TIMING SUM {phase} n={n} {ms:.3}ms");
    }
}

/// Clear accumulators so sequential `GoldyRenderer::new` calls can be compared.
pub fn reset_totals() {
    if !enabled() {
        return;
    }
    let mut t = totals().lock().unwrap_or_else(|e| e.into_inner());
    t.ns.clear();
    t.n.clear();
}
