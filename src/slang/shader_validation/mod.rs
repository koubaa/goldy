//! Static checks over Slang IR, run at shader compile time (`GOLDY_SHADER_VALIDATION`).
//!
//! These are *compile-time* analyses of the shader program, distinct from the runtime
//! invariant checks behind `GOLDY_VALIDATION`: they cost a second Slang compile (to IR, with
//! debug info) plus a whole-program analysis per shader, they can be wrong in the
//! conservative direction (a finding means "not proven", not "definitely broken"), and they
//! never fail a compile. That is why they hang off their own variable and are not implied
//! by `GOLDY_VALIDATION=all`.
//!
//! Each check is a rule under this module ([`bounds`] today) that consumes the shared IR
//! toolkit in [`crate::slang::ir`] and produces its own report; [`validate`] runs the
//! selected rules over a linked module and collects the reports into a
//! [`ShaderValidationReport`]. The `SlangCompiler` hook that compiles the IR, gathers the
//! imported modules and logs findings is `SlangCompiler::validate_shader`.
//!
//! # `GOLDY_SHADER_VALIDATION`
//!
//! A list of check names (comma, semicolon or whitespace separated, case-insensitive),
//! processed left to right:
//!
//! - `all` (or `1` / `true` / `yes`) — every check
//! - `bounds` — one check by name
//! - `-bounds` — remove a check enabled earlier in the list (`all,-bounds`)
//! - `none` / `0` / `false` / `no` / unset — nothing
//!
//! Unknown names are ignored.

pub mod bounds;

use crate::slang::ir::{self, IrError};

pub use bounds::{BoundsDiagnostic, BoundsReport};

/// Which static checks to run. Empty by default; see the module documentation for the
/// `GOLDY_SHADER_VALIDATION` grammar that [`ShaderChecks::parse`] accepts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ShaderChecks {
    /// Dynamic array indices proven inside `0 <= index < length` ([`bounds`]).
    pub bounds: bool,
}

impl ShaderChecks {
    /// Every check.
    #[must_use]
    pub const fn all() -> ShaderChecks {
        ShaderChecks { bounds: true }
    }

    /// No check.
    #[must_use]
    pub const fn none() -> ShaderChecks {
        ShaderChecks { bounds: false }
    }

    /// `true` when no check is selected.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        !self.bounds
    }

    /// Parse a `GOLDY_SHADER_VALIDATION` value.
    #[must_use]
    pub fn parse(raw: &str) -> ShaderChecks {
        let mut out = ShaderChecks::none();
        let normalized = raw.replace(';', ",");
        for chunk in normalized.split(',') {
            for part in chunk.split_whitespace() {
                let lower = part.to_ascii_lowercase();
                let (enable, name) = match lower.strip_prefix('-') {
                    Some(rest) => (false, rest),
                    None => (true, lower.as_str()),
                };
                match name {
                    "all" | "1" | "true" | "yes" => out = if enable { ShaderChecks::all() } else { out },
                    "none" | "0" | "false" | "no" => out = ShaderChecks::none(),
                    "bounds" => out.bounds = enable,
                    _ => {}
                }
            }
        }
        out
    }
}

/// Reports of the checks that ran; a check that was not selected is `None`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShaderValidationReport {
    pub bounds: Option<BoundsReport>,
}

impl ShaderValidationReport {
    /// `true` when every check that ran proved everything it looked at.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.bounds.as_ref().is_none_or(BoundsReport::is_clean)
    }

    /// Human-readable findings of every check that ran, one per line, in report order.
    pub fn findings(&self) -> impl Iterator<Item = (&'static str, String)> + '_ {
        self.bounds
            .iter()
            .flat_map(|r| r.diagnostics.iter().map(|d| ("bounds", d.to_string())))
    }
}

/// Run `checks` over the translation unit serialized in `container` (a `.slang-module` RIFF
/// blob) linked with the containers of the modules it imports.
///
/// `libraries` may be in any order and may be incomplete: calls into a missing module are
/// treated as unknown and named in the diagnostics.
pub fn validate(
    container: &[u8],
    libraries: &[&[u8]],
    checks: ShaderChecks,
) -> Result<ShaderValidationReport, IrError> {
    let mut report = ShaderValidationReport::default();
    if checks.is_empty() {
        return Ok(report);
    }
    let linked = ir::link_containers(container, libraries)?;
    if checks.bounds {
        report.bounds = Some(bounds::analyze(&linked));
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_check_lists() {
        assert_eq!(ShaderChecks::parse(""), ShaderChecks::none());
        assert_eq!(ShaderChecks::parse("all"), ShaderChecks::all());
        assert_eq!(ShaderChecks::parse("1"), ShaderChecks::all());
        assert_eq!(ShaderChecks::parse("bounds"), ShaderChecks { bounds: true });
        assert_eq!(ShaderChecks::parse("Bounds; nonsense"), ShaderChecks { bounds: true });
        assert_eq!(ShaderChecks::parse("all,-bounds"), ShaderChecks::none());
        assert_eq!(ShaderChecks::parse("-bounds,all"), ShaderChecks::all());
        assert_eq!(ShaderChecks::parse("bounds none"), ShaderChecks::none());
        assert_eq!(ShaderChecks::parse("0"), ShaderChecks::none());
        assert!(ShaderChecks::parse("-all").is_empty());
    }

    #[test]
    fn no_checks_means_no_parse() {
        // Garbage input is not even looked at when nothing is selected.
        assert_eq!(
            validate(&[1, 2, 3], &[], ShaderChecks::none()),
            Ok(ShaderValidationReport::default())
        );
        assert!(matches!(
            validate(&[1, 2, 3], &[], ShaderChecks::all()),
            Err(IrError::Malformed(_))
        ));
    }
}
