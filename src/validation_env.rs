//! Environment-driven validation switches (`GOLDY_VALIDATION`, `GOLDY_VALIDATE_LAYOUTS`).
//!
//! **Semantics**
//! - `GOLDY_VALIDATE_LAYOUTS=1|true|yes` — unchanged; enables Rust/Slang layout and buffer
//!   stride checks (same family as before).
//! - `GOLDY_VALIDATION` — list of categories (comma, semicolon, or whitespace separated,
//!   case-insensitive):
//!   - `layout` / `layouts` — layout + stride checks
//!   - `api` — graphics API validation (Vulkan validation layer + `VK_EXT_debug_utils` where
//!     built; Metal `MTL_SHADER_VALIDATION=1` when `GOLDY_VALIDATION` includes `api` and the
//!     variable is unset — set once before the first device is enumerated). For loader-only
//!     Vulkan layers, set `VK_INSTANCE_LAYERS` / `VK_LAYER_PATH` yourself.
//!   - `timeline` — WSI timeline invariants (Vulkan surface `acquire()` post-wait checks)
//!   - `scheme` / `readback` — retained-scheme grant readback invariants (staging pool, frame pairing)
//!   - `all` — layout, GPU API, timeline, and scheme
//! - `GOLDY_VALIDATION=1|true|yes` (no list) — **GPU API only** (does not turn on layout checks,
//!   so hot-path layout validation stays opt-in). For everything, use **`GOLDY_VALIDATION=all`**
//!   or **`GOLDY_VALIDATION=layout,api`**.
//! - `GOLDY_DISABLE_CB_REUSE=1|true|yes` — disable the CB-retention facility entirely:
//!   no retention fingerprints, no backend CB store/resubmit, no retained-allocator
//!   retire waits, no topology-dirty registration for replay. Each submit re-records
//!   via ordinary `submit_graph` / `submit_standalone`. Also implied when
//!   [`crate::gpu_profiler::gpu_profile_enabled`] is true, because timestamp queries
//!   reference a per-submit query heap that must not outlive a retained list.

#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
struct ParsedValidation {
    layout: bool,
    gpu_api: bool,
    timeline: bool,
    scheme: bool,
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

fn parse_validation_list(raw: &str) -> ParsedValidation {
    let mut out = ParsedValidation::default();
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
                "all" => {
                    out.layout = true;
                    out.gpu_api = true;
                    out.timeline = true;
                    out.scheme = true;
                }
                "layout" | "layouts" => out.layout = true,
                "api" => out.gpu_api = true,
                "timeline" => out.timeline = true,
                "scheme" | "readback" => out.scheme = true,
                _ => {}
            }
        }
    }
    out
}

fn from_goldy_validation_var() -> ParsedValidation {
    std::env::var("GOLDY_VALIDATION")
        .map(|s| parse_validation_list(&s))
        .unwrap_or_default()
}

/// Layout / struct / buffer-stride validation (Slang reflection vs Rust, dispatch-time strides).
#[must_use]
pub fn layout_validation_enabled() -> bool {
    if env_truthy("GOLDY_VALIDATE_LAYOUTS") {
        return true;
    }
    from_goldy_validation_var().layout
}

/// Vulkan Khronos validation + `VK_EXT_debug_utils`, Metal `MTL_SHADER_VALIDATION`, etc.
#[cfg(any(feature = "vulkan", all(feature = "metal", target_os = "macos")))]
#[must_use]
pub(crate) fn gpu_api_validation_enabled() -> bool {
    from_goldy_validation_var().gpu_api
}

/// WSI timeline invariants (Vulkan surface acquire post-wait checks).
#[cfg(feature = "vulkan")]
#[must_use]
pub(crate) fn timeline_validation_enabled() -> bool {
    from_goldy_validation_var().timeline
}

/// Retained-scheme grant readback invariants (frame/grant pairing, staging pool checks).
#[must_use]
pub(crate) fn scheme_validation_enabled() -> bool {
    from_goldy_validation_var().scheme
}

#[cfg(test)]
use std::cell::Cell;

// Thread-local override for `retained_cb_reuse_disabled`. When `Some`, takes precedence
// over the environment / profiler on this thread only (safe under parallel cargo tests).
#[cfg(test)]
thread_local! {
    static TEST_CB_REUSE_DISABLED_OVERRIDE: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Install a thread-local test override for CB reuse (see [`retained_cb_reuse_disabled`]).
///
/// Must be paired with [`clear_test_cb_reuse_override`] on the same thread.
#[cfg(test)]
pub(crate) fn set_test_cb_reuse_override(disabled: bool) {
    TEST_CB_REUSE_DISABLED_OVERRIDE.with(|c| c.set(Some(disabled)));
}

/// Clear the override installed by [`set_test_cb_reuse_override`].
#[cfg(test)]
pub(crate) fn clear_test_cb_reuse_override() {
    TEST_CB_REUSE_DISABLED_OVERRIDE.with(|c| c.set(None));
}

/// When true, disable the CB-retention facility entirely (not merely skip resubmit hits).
///
/// Set `GOLDY_DISABLE_CB_REUSE=1` (or `true` / `yes`), or enable [`crate::gpu_profiler::gpu_profile_enabled`].
/// Goldy tears down any live replay ledger and routes retainable partitions through ordinary
/// `submit_graph` — no fingerprints, backend CB storage, allocator retire waits, or replay
/// topology registration.
#[must_use]
pub(crate) fn retained_cb_reuse_disabled() -> bool {
    #[cfg(test)]
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

        let p = parse_validation_list("timeline");
        assert!(!p.layout);
        assert!(!p.gpu_api);
        assert!(p.timeline);
        assert!(!p.scheme);

        let p = parse_validation_list("scheme");
        assert!(!p.layout);
        assert!(!p.gpu_api);
        assert!(!p.timeline);
        assert!(p.scheme);

        let p = parse_validation_list("readback");
        assert!(p.scheme);

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
        let p = parse_validation_list("gpu,vulkan,metal,shader");
        assert!(!p.layout);
        assert!(!p.gpu_api);
    }
}
