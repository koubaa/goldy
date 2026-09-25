//! Validation settings and the environment switches that default them (`GOLDY_VALIDATION`,
//! `GOLDY_VALIDATE_LAYOUTS`, `GOLDY_SHADER_VALIDATION`).
//!
//! [`Validation`] is the API: an instance or backend is created with one, and every check
//! reads that value rather than the environment. `GOLDY_VALIDATION` and
//! `GOLDY_VALIDATE_LAYOUTS` only supply [`Validation::from_env`], the default.
//!
//! **Semantics**
//! - `GOLDY_VALIDATE_LAYOUTS=1|true|yes` — unchanged; enables Rust/Slang layout and buffer
//!   stride checks (same family as before).
//! - `GOLDY_VALIDATION` — list of categories (comma, semicolon, or whitespace separated,
//!   case-insensitive):
//!   - `layout` / `layouts` — layout + stride checks
//!   - `api` — graphics API validation (Vulkan validation layer + `VK_EXT_debug_utils` where
//!     built; Metal `MTL_SHADER_VALIDATION=1` when unset, process-wide from the first Metal
//!     backend; CUDA Driver diagnostics: PTX JIT logs, launch-limit checks, and a stream
//!     sync after every op with graph capture off; WebGPU/wgpu validation error scopes on
//!     shader/PSO create and bind groups). For loader-only Vulkan layers, set
//!     `VK_INSTANCE_LAYERS` / `VK_LAYER_PATH` yourself.
//!   - `timeline` — WSI timeline invariants (Vulkan surface `acquire()` post-wait checks)
//!   - `scheme` / `readback` / `graph` — retained-scheme host-read staging invariants
//!     plus graph-level lifetime checks (Accel built in this scheme before TraceRay /
//!     RayQuery). Cycle detection, mesh/draw mix-ups, and BLAS/TLAS misuse always run.
//!   - `host_access` — page-protect CPU-visible GPU copies (CPU backend parcels; more backends later)
//!   - `all` — layout, GPU API, timeline, scheme, and host_access
//! - `GOLDY_VALIDATION=1|true|yes` (no list) — **GPU API only** (does not turn on layout checks,
//!   so hot-path layout validation stays opt-in). For everything, use **`GOLDY_VALIDATION=all`**
//!   or **`GOLDY_VALIDATION=layout,api`**.
//! - `GOLDY_SHADER_VALIDATION` — static checks over Slang IR at shader compile time
//!   (`all`, `bounds`, `-bounds`; see [`crate::slang::shader_validation`]). Separate from
//!   `GOLDY_VALIDATION` and **not** implied by `GOLDY_VALIDATION=all`: these cost a second
//!   compile plus a whole-program analysis per shader and report "not proven" rather than
//!   invariant violations.
//! - `GOLDY_DISABLE_CB_REUSE=1|true|yes` — disable the CB-retention facility entirely:
//!   no retention fingerprints, no backend CB store/resubmit, no retained-allocator
//!   retire waits, no topology-dirty registration for replay. Each submit re-records
//!   via ordinary `submit_graph` / `submit_standalone`. Also implied when
//!   `GOLDY_GPU_PROFILE` is set, because timestamp queries reference a per-submit
//!   query heap that must not outlive a retained list.

/// Which of Goldy's validation checks run.
///
/// The default is [`Validation::from_env`], so `GOLDY_VALIDATION` decides unless a program
/// passes its own value to [`crate::Instance::with_validation`] (or a backend constructor).
/// A backend keeps the value it was created with, and every runtime on it reports that value
/// through [`crate::Runtime::validation`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Validation {
    /// Rust/Slang layout and buffer-stride checks (`layout`, or `GOLDY_VALIDATE_LAYOUTS`).
    pub layout: bool,
    /// Graphics API validation (`api`): the Vulkan validation layer, Metal shader validation,
    /// CUDA driver diagnostics with a stream sync after every operation and no graph capture,
    /// and WebGPU error scopes.
    pub gpu_api: bool,
    /// WSI timeline invariants (`timeline`).
    pub timeline: bool,
    /// Retained-scheme host-read and graph lifetime checks (`scheme`).
    pub scheme: bool,
    /// Page-protected CPU-visible copies (`host_access`).
    pub host_access: bool,
    /// GPU API validation errors fail Goldy calls and panic on backend drop
    /// (`GOLDY_VALIDATION_FATAL`; `all` does not imply it).
    pub fatal: bool,
}

impl Validation {
    /// Every check off.
    pub const NONE: Self = Self {
        layout: false,
        gpu_api: false,
        timeline: false,
        scheme: false,
        host_access: false,
        fatal: false,
    };

    /// Every check `GOLDY_VALIDATION=all` enables. Errors stay non-fatal.
    pub const ALL: Self = Self {
        layout: true,
        gpu_api: true,
        timeline: true,
        scheme: true,
        host_access: true,
        fatal: false,
    };

    /// The checks `GOLDY_VALIDATION`, `GOLDY_VALIDATE_LAYOUTS` and `GOLDY_VALIDATION_FATAL`
    /// request.
    #[must_use]
    pub fn from_env() -> Self {
        let mut v = std::env::var("GOLDY_VALIDATION")
            .map(|s| parse_validation_list(&s))
            .unwrap_or(Self::NONE);
        v.layout |= env_truthy("GOLDY_VALIDATE_LAYOUTS");
        v.fatal = env_truthy("GOLDY_VALIDATION_FATAL");
        v
    }
}

impl Default for Validation {
    fn default() -> Self {
        Self::from_env()
    }
}

fn env_truthy(name: &str) -> bool {
    std::env::var(name)
        .map(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn legacy_gpu_only_short_form(raw: &str) -> Option<bool> {
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    if matches!(t.to_ascii_lowercase().as_str(), "1" | "true" | "yes") {
        Some(true)
    } else {
        None
    }
}

fn parse_validation_list(raw: &str) -> Validation {
    let mut out = Validation::NONE;
    if let Some(true) = legacy_gpu_only_short_form(raw) {
        out.gpu_api = true;
        return out;
    }
    let normalized = raw.replace(';', ",");
    for chunk in normalized.split(',') {
        for part in chunk.split_whitespace() {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            match p.to_ascii_lowercase().as_str() {
                "all" => out = Validation::ALL,
                "layout" | "layouts" => out.layout = true,
                "api" => out.gpu_api = true,
                "timeline" => out.timeline = true,
                "scheme" | "readback" | "graph" => out.scheme = true,
                "host_access" | "host-access" => out.host_access = true,
                _ => {}
            }
        }
    }
    out
}

/// Static checks to run over Slang IR at shader compile time (`GOLDY_SHADER_VALIDATION`).
///
/// Empty unless the variable is set; `GOLDY_VALIDATION` never turns these on. Findings are
/// warnings and never fail a compile.
#[must_use]
pub fn shader_validation_checks() -> crate::slang::ShaderChecks {
    std::env::var("GOLDY_SHADER_VALIDATION")
        .map(|s| crate::slang::ShaderChecks::parse(&s))
        .unwrap_or_default()
}

use std::cell::Cell;

// Thread-local override for `retained_cb_reuse_disabled`. When `Some`, takes precedence
// over the environment / profiler on this thread only (safe under parallel cargo tests).
// Always compiled (not `cfg(test)`-only) so integration tests under `tests/` can use it
// via [`crate::test_support::CbReuseOverride`].
thread_local! {
    static TEST_CB_REUSE_DISABLED_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static TEST_SPECIALIZATION_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
    static TEST_FUSION_COMPILE_FAULT: Cell<bool> = const { Cell::new(false) };
}

/// Make fused compiles started on this thread fail (tests of the unfused fallback).
#[doc(hidden)]
pub fn set_fusion_compile_fault(fail: bool) {
    TEST_FUSION_COMPILE_FAULT.with(|c| c.set(fail));
}

pub(crate) fn fusion_compile_fault() -> bool {
    TEST_FUSION_COMPILE_FAULT.with(|c| c.get())
}

/// Default for [`crate::Scheme::automatic_fusion`] on schemes that never called
/// [`crate::Scheme::set_automatic_fusion`].
///
/// Off unless `GOLDY_FUSION=1` (or `true` / `yes` / `on`); see
/// `docs/src/programming-model/rust-kernels.md`.
#[must_use]
pub(crate) fn fusion_enabled() -> bool {
    match std::env::var("GOLDY_FUSION") {
        Ok(v) => matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"),
        Err(_) => false,
    }
}

/// Install a thread-local override for [`specialization_enabled`].
///
/// Prefer [`crate::test_support::SpecializationOverride`] — it clears on drop.
#[doc(hidden)]
pub fn set_specialization_override(enabled: bool) {
    TEST_SPECIALIZATION_OVERRIDE.with(|c| c.set(Some(enabled)));
}

/// Clear the override installed by [`set_specialization_override`].
#[doc(hidden)]
pub fn clear_specialization_override() {
    TEST_SPECIALIZATION_OVERRIDE.with(|c| c.set(None));
}

/// Whether retained schemes may predict and swap in specialized compute pipelines.
///
/// On by default. Set `GOLDY_SPECIALIZATION=0` (or `false` / `no` / `off`) to keep every
/// dispatch on the pipeline the caller bound; the predictor then records no history and
/// compiles nothing. Backends whose pipeline layouts do not follow the shader signature
/// (WebGPU today) are excluded regardless of this variable — see
/// `docs/src/design/shader-specialization.md`.
///
/// Tests that assert exact record counts across many frames pin this with
/// [`crate::test_support::SpecializationOverride`] so a developer shell cannot flip them.
#[must_use]
pub(crate) fn specialization_enabled() -> bool {
    if let Some(enabled) = TEST_SPECIALIZATION_OVERRIDE.with(|c| c.get()) {
        return enabled;
    }
    match std::env::var("GOLDY_SPECIALIZATION") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

/// Install a thread-local override for CB reuse (see [`retained_cb_reuse_disabled`]).
///
/// Prefer [`crate::test_support::CbReuseOverride`] — it clears on drop.
#[doc(hidden)]
pub fn set_cb_reuse_override(disabled: bool) {
    TEST_CB_REUSE_DISABLED_OVERRIDE.with(|c| c.set(Some(disabled)));
}

/// Clear the override installed by [`set_cb_reuse_override`].
#[doc(hidden)]
pub fn clear_cb_reuse_override() {
    TEST_CB_REUSE_DISABLED_OVERRIDE.with(|c| c.set(None));
}

/// When true, disable the CB-retention facility entirely (not merely skip resubmit hits).
///
/// Set `GOLDY_DISABLE_CB_REUSE=1` (or `true` / `yes`), or enable `GOLDY_GPU_PROFILE`.
/// Goldy tears down any live replay ledger and routes retainable partitions through ordinary
/// `submit_graph` — no fingerprints, backend CB storage, allocator retire waits, or replay
/// topology registration.
///
/// Tests that assert retention behavior must pin the mode with
/// [`crate::test_support::CbReuseOverride`] so a developer shell exporting
/// `GOLDY_DISABLE_CB_REUSE=1` cannot flip the suite.
#[must_use]
pub(crate) fn retained_cb_reuse_disabled() -> bool {
    if let Some(disabled) = TEST_CB_REUSE_DISABLED_OVERRIDE.with(|c| c.get()) {
        return disabled;
    }
    env_truthy("GOLDY_DISABLE_CB_REUSE") || crate::gpu_profiler::gpu_profile_enabled()
}

#[cfg(test)]
mod tests {
    use super::parse_validation_list;

    #[test]
    fn parse_list_tokens() {
        let p = parse_validation_list("layout,api");
        assert!(p.layout);
        assert!(p.gpu_api);

        let p = parse_validation_list("layout");
        assert!(p.layout);
        assert!(!p.gpu_api);

        let p = parse_validation_list("all");
        assert!(p.layout);
        assert!(p.gpu_api);
        assert!(p.timeline);
        assert!(p.scheme);
        assert!(p.host_access);

        let p = parse_validation_list("api,fatal");
        assert!(p.gpu_api);
        assert!(!p.host_access);

        let p = parse_validation_list("timeline");
        assert!(!p.layout);
        assert!(!p.gpu_api);
        assert!(p.timeline);
        assert!(!p.scheme);

        let p = parse_validation_list("graph");
        assert!(p.scheme);
        assert!(!p.gpu_api);

        let p = parse_validation_list("host_access");
        assert!(p.host_access);
        assert!(!p.gpu_api);

        let p = parse_validation_list("api; api");
        assert!(!p.layout);
        assert!(p.gpu_api);
    }

    #[test]
    fn parse_legacy_truthy_is_gpu_only() {
        let p = parse_validation_list("1");
        assert!(!p.layout);
        assert!(p.gpu_api);

        let p = parse_validation_list("true");
        assert!(!p.layout);
        assert!(p.gpu_api);
    }

    #[test]
    fn parse_unknown_tokens_do_not_enable_api() {
        let p = parse_validation_list("gpu,vulkan,metal,shader,fatal");
        assert!(!p.layout);
        assert!(!p.gpu_api);
    }
}
