//! Shader compilation and management logic.

use super::super::shared::{ShaderDesc, ShaderStageCompileDesc};
use super::super::{DeviceHandle, ShaderHandle};
use super::types::{MetalState, ShaderState};
use crate::slang::{ShaderTarget, SlangCompiler, SlangStage};
use ::metal as mtl;
use anyhow::{Context, Result};
use mtl::{Device as MTLDevice, Library};
use std::collections::HashMap;

/// Patch compute-shader MSL to fix Slang codegen bug that surfaces on Metal:
///
/// **Bug A** — missing `constant*` / `[[buffer(1)]]` on `EntryPointParams_0`.
/// When wave intrinsics (`WaveGetLaneCount` etc.) are combined with a cross-module
/// `[ForceInline]` function that takes a fixed-size `groupshared` parameter, Slang
/// omits the `constant*` address-space qualifier and `[[buffer(1)]]` attribute from
/// the `EntryPointParams_0` kernel argument.  The ill-formed MSL signature is:
///
/// ```text
/// [[kernel]] void cs_main(..., EntryPointParams_0 entryPointParamsN)
/// ```
///
/// The patch restores the correct form:
///
/// ```text
/// [[kernel]] void cs_main(..., EntryPointParams_0 constant* entryPointParamsN [[buffer(1)]])
/// ```
///
/// and updates every member access (`VARNAME.FIELD`) to pointer syntax
/// (`VARNAME->FIELD`) plus the struct-copy assignment to a dereference
/// (`= VARNAME;` → `= *VARNAME;`).
///
/// **Bug B** — threadgroup array copied into thread-local storage.
/// Cross-module `[ForceInline]` functions that receive a `groupshared` array
/// parameter generate a per-thread copy of the array:
///
/// ```text
/// thread array<TYPE, int(N)> VAR = *kernelContext_M->FIELD;
/// ```
///
/// This means every write goes to the calling thread's private stack, so
/// cross-thread communication through the scratch buffer silently produces
/// wrong values (barriers have nothing to synchronise).  The patch replaces
/// the copy with a `threadgroup` reference:
///
/// ```text
/// threadgroup array<TYPE, int(N)>& VAR = *kernelContext_M->FIELD;
/// ```
pub(super) fn patch_compute_msl(msl: &str) -> String {
    let s = patch_compute_msl_entry_point_params(msl);
    patch_msl_threadgroup_copies(&s)
}

/// Fix Bug A: restore the `constant*` qualifier and `[[buffer(1)]]` attribute for
/// `EntryPointParams_0` when Slang omits them in compute-shader kernels.
fn patch_compute_msl_entry_point_params(msl: &str) -> String {
    if !msl.contains("EntryPointParams_0") || msl.contains("EntryPointParams_0 constant*") {
        return msl.to_string();
    }
    // Locate the ill-formed kernel parameter `EntryPointParams_0 VARNAME)`.
    // The struct definition ends with `\n{` (no space) and the KernelContext member
    // ends with `;`, so only the kernel-signature occurrence ends with `)`.
    const EP_NEEDLE: &str = "EntryPointParams_0 ";
    let mut search_from = 0usize;
    let var_name = loop {
        let rel = match msl[search_from..].find(EP_NEEDLE) {
            Some(p) => p,
            None => return msl.to_string(),
        };
        let abs = search_from + rel;
        let after = &msl[abs + EP_NEEDLE.len()..];
        let var_end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(after.len());
        let var = &after[..var_end];
        if !var.is_empty() && after[var_end..].starts_with(')') {
            break var.to_string();
        }
        search_from = abs + EP_NEEDLE.len() + 1;
    };

    // Step 1: fix the kernel-signature parameter.
    let old_param = format!("{}{}{}", EP_NEEDLE, var_name, ")");
    let new_param = format!("{}constant* {} [[buffer(1)]])", EP_NEEDLE, var_name);
    let mut s = msl.replacen(&old_param, &new_param, 1);

    // Step 2: change member-access syntax from `.FIELD` to `->FIELD`.
    let dot = format!("{}.", var_name);
    let arrow = format!("{}->", var_name);
    s = s.replace(&dot, &arrow);

    // Step 3: the struct-copy assignment must dereference the now-pointer parameter.
    let assign = format!("= {};", var_name);
    let deref = format!("= *{};", var_name);
    s = s.replace(&assign, &deref);

    s
}

/// Fix Bug B: replace per-thread copies of `threadgroup` arrays with references.
///
/// Two patterns appear depending on the collective function:
///
/// **Pattern 1** — explicit or implicit thread copy directly from a KernelContext field:
/// ```text
///     thread array<TYPE, int(N)> VAR = *kernelContext_M->FIELD;  // explicit thread
///           array<TYPE, int(N)> VAR = *kernelContext_M->FIELD;   // implicit thread
/// ```
///
/// **Pattern 2** — a secondary copy of the Pattern-1 variable (e.g. inside a loop body):
/// ```text
///     thread array<TYPE, int(N)> VAR2 = VAR1;
/// ```
/// where `VAR1` was already fixed to a `threadgroup&` reference above.
///
/// Both are replaced with `threadgroup` references so that reads/writes go to the
/// actual shared threadgroup memory rather than a per-thread stack copy.
fn patch_msl_threadgroup_copies(msl: &str) -> String {
    if !msl.contains("= *kernelContext") {
        return msl.to_string();
    }
    // Track variable names that have been turned into threadgroup references so
    // that Pattern-2 copies of them can be caught in the same single pass.
    let mut tg_ref_vars: std::collections::HashSet<String> = std::collections::HashSet::new();

    let mut result = String::with_capacity(msl.len() + 128);
    for line in msl.lines() {
        let trimmed = line.trim_start();

        // Pattern 1a — explicit `thread array<` from KernelContext.
        let is_p1_explicit = trimmed.starts_with("thread array<") && line.contains("= *kernelContext");
        // Pattern 1b — implicit thread (`array<` with no qualifier) from KernelContext.
        let is_p1_implicit =
            !trimmed.starts_with("thread") && trimmed.starts_with("array<") && line.contains("= *kernelContext");
        // Pattern 2 — explicit `thread array<` copying a known threadgroup-ref variable.
        let is_p2 = trimmed.starts_with("thread array<")
            && !line.contains("= *kernelContext")
            && line.find(" = ").is_some_and(|eq| {
                let rhs = line[eq + 3..].trim_end_matches(';').trim();
                tg_ref_vars.contains(rhs)
            });

        let patched = if is_p1_explicit || is_p1_implicit || is_p2 {
            // Normalise: ensure the line starts with `thread array<` for uniform handling.
            let s = if is_p1_implicit {
                let arr_pos = line.find("array<").unwrap();
                format!("{}thread {}", &line[..arr_pos], &line[arr_pos..])
            } else {
                line.to_string()
            };
            // Change address space: `thread` → `threadgroup`.
            let s = s.replacen("thread array<", "threadgroup array<", 1);
            // Insert `&` after the closing `>` of the array type to make it a reference.
            // `rfind("> ")` unambiguously finds the type-closer because `->` in the RHS
            // never has a space after `>`.
            if let (Some(gt), Some(eq)) = (s.rfind("> "), s.find(" = ")) {
                if gt < eq {
                    let mut out = String::with_capacity(s.len() + 1);
                    out.push_str(&s[..gt + 1]);
                    out.push('&');
                    out.push_str(&s[gt + 1..]);
                    // Record the variable so downstream Pattern-2 copies are also fixed.
                    let after_gt = &s[gt + 2..]; // skip `> `
                    if let Some(end) = after_gt.find(|c: char| !c.is_alphanumeric() && c != '_') {
                        let var = &after_gt[..end];
                        if !var.is_empty() {
                            tg_ref_vars.insert(var.to_string());
                        }
                    }
                    out
                } else {
                    s
                }
            } else {
                s
            }
        } else {
            line.to_string()
        };
        result.push_str(&patched);
        result.push('\n');
    }
    if !msl.ends_with('\n') && result.ends_with('\n') {
        result.pop();
    }
    result
}

/// Fix the Slang bug where an internal dispatch function emits `[[stage_in]]` on a
/// `thread*` pointer parameter instead of a value parameter.
///
/// For shaders compiled with `[goldy_vertex]` Slang generates two function signatures:
///
/// 1. An internal wrapper, e.g.:
///    ```text
///    StaticVarying_0 vs_main_0(const StaticVertexIn_0 thread* _pt0_0 [[stage_in]], ...)
///    ```
///    Here `[[stage_in]]` is on a **pointer** — Metal rejects this.
///
/// 2. The real entry point:
///    ```text
///    [[vertex]] vs_main_Result_0 vs_main(StaticVertexIn_0 _S1 [[stage_in]], ...)
///    ```
///    Here `[[stage_in]]` is on a **value type** — this is correct.
///
/// The fix: only strip `[[stage_in]]` when the same line also contains `thread*`
/// (i.e., the invalid pointer case).  Using the entire file prefix as the search
/// context (the original implementation) incorrectly fires on simple vertex shaders
/// where `thread*` appears in an unrelated helper function above the entry point.
pub(super) fn patch_vertex_stage_in_pointer(msl: &str) -> String {
    const STAGE_IN: &str = "[[stage_in]]";

    let Some(si_pos) = msl.find(STAGE_IN) else {
        return msl.to_string();
    };

    // Determine the line that contains this [[stage_in]] occurrence.
    let line_start = msl[..si_pos].rfind('\n').map_or(0, |p| p + 1);
    let line_end = msl[si_pos..].find('\n').map_or(msl.len(), |p| si_pos + p);
    let line = &msl[line_start..line_end];

    // Only strip when the [[stage_in]] is on a thread-pointer parameter.
    // A legitimate entry-point [[stage_in]] is always on a value type (no `*` on that line).
    if !line.contains("thread*") {
        return msl.to_string();
    }

    let before = &msl[..si_pos];
    let prefix_end = before.trim_end().len();
    let si_end = si_pos + STAGE_IN.len();

    let mut result = String::with_capacity(msl.len());
    result.push_str(&msl[..prefix_end]);
    result.push_str(&msl[si_end..]);
    result
}

/// Patch vertex-shader MSL to inject the missing EntryPointParams [[buffer(1)]] binding.
///
/// Slang's Metal backend generates `EntryPointParams_0` inside `KernelContext_0` for
/// vertex shaders with `uniform` entry-point parameters, but — unlike compute shaders —
/// it never exposes the struct as a `[[buffer(1)]]` parameter in the vertex entry point.
/// The `entryPointParams_0` pointer in `KernelContext_0` therefore remains uninitialized,
/// causing `goldy_scattered` / `goldy_broadcast` to read from garbage memory.
///
/// This function patches the generated MSL to:
///   1. Add `EntryPointParams_0 constant* _goldy_ep [[buffer(1)]]` to the entry-point
///      parameter list (right before the closing `)` after `[[buffer(0)]]`).
///   2. Set `(&kernelContextN)->entryPointParams_0 = _goldy_ep;` immediately after the
///      `gGoldy_0` assignment that already exists in the entry-point body.
///
/// Only applied when `EntryPointParams_0` appears in the source but is not yet a
/// `[[buffer(1)]]` parameter (i.e., the Slang bug is present).
pub(super) fn patch_vertex_msl_entry_point_params(msl: &str) -> String {
    // Nothing to do if there are no entry-point params.
    if !msl.contains("EntryPointParams_0") {
        return msl.to_string();
    }
    // Already correctly bound (Slang fixed the bug, or it's a compute shader).
    if msl.contains("EntryPointParams_0 constant*") && msl.contains("[[buffer(1)]]") {
        return msl.to_string();
    }

    // Step 1 — inject the missing parameter into the entry-point signature.
    // The vertex entry point ends with `[[buffer(0)]])` (the closing paren is immediately
    // after the last attribute). We replace the first occurrence only.
    const SIG_NEEDLE: &str = "[[buffer(0)]])";
    const SIG_REPLACEMENT: &str = "[[buffer(0)]], EntryPointParams_0 constant* _goldy_ep [[buffer(1)]])";
    let patched = if let Some(pos) = msl.find(SIG_NEEDLE) {
        let mut s = String::with_capacity(msl.len() + 80);
        s.push_str(&msl[..pos]);
        s.push_str(SIG_REPLACEMENT);
        s.push_str(&msl[pos + SIG_NEEDLE.len()..]);
        s
    } else {
        // Pattern not found — leave unchanged (defensive).
        return msl.to_string();
    };

    // Step 2 — inject the assignment inside the entry-point body.
    // Find the ASSIGNMENT `->gGoldy_0 = ` (as opposed to member accesses `->gGoldy_0->`).
    // Extract the kernel-context variable name from the LHS: `(&KCTX)->gGoldy_0 = `.
    const ASSIGN_NEEDLE: &str = ")->gGoldy_0 = ";
    if let Some(arrow_pos) = patched.find(ASSIGN_NEEDLE) {
        // Walk backwards from arrow_pos to find the opening `(&`.
        let prefix = &patched[..arrow_pos];
        if let Some(amp_pos) = prefix.rfind("(&") {
            let kctx_name = &prefix[amp_pos + 2..]; // from after `(&` to arrow_pos
                                                    // Find the end of the assignment statement (the semicolon).
            let after_arrow = &patched[arrow_pos + ASSIGN_NEEDLE.len()..];
            if let Some(semi_rel) = after_arrow.find(';') {
                let semi_abs = arrow_pos + ASSIGN_NEEDLE.len() + semi_rel;
                let injection = format!("\n    (&{})->entryPointParams_0 = _goldy_ep;", kctx_name);
                let mut result = String::with_capacity(patched.len() + injection.len());
                result.push_str(&patched[..=semi_abs]);
                result.push_str(&injection);
                result.push_str(&patched[semi_abs + 1..]);
                return result;
            }
        }
    }

    patched
}

/// Compile a shader stage to MSL and create a Metal library.
fn compile_stage_with_reflection(
    slang_compiler: &SlangCompiler,
    device: &MTLDevice,
    desc: &ShaderStageCompileDesc<'_>,
) -> Result<(Library, Option<crate::slang::ShaderReflection>)> {
    let compile_outcome = slang_compiler.compile_bindless_with_reflection_and_defines(
        desc.slang_source,
        ShaderTarget::Metal,
        &[(desc.entry_point, desc.stage)],
        desc.search_paths,
        desc.extra_defines,
        desc.layout_checks,
        desc.optimization_level,
    );

    let result = compile_outcome.with_context(|| format!("Failed to compile {} shader stage", desc.entry_point))?;

    if !result.reflection.parameter_blocks.is_empty() {
        tracing::debug!(
            "Shader {} has {} ParameterBlock(s):",
            desc.entry_point,
            result.reflection.parameter_blocks.len()
        );
        for pb in &result.reflection.parameter_blocks {
            tracing::debug!(
                "  - {} at slot {} (size={}, alignment={}, fields={})",
                pb.name,
                pb.binding_slot,
                pb.size,
                pb.alignment,
                pb.fields.len()
            );
            for field in &pb.fields {
                tracing::debug!(
                    "    - {}: {:?} at offset {} (size={})",
                    field.name,
                    field.resource_kind,
                    field.offset,
                    field.size
                );
            }
        }
    }

    let raw_msl = result.shader.as_str().context("Failed to get MSL source")?.to_string();

    // Apply stage-specific MSL patches for known Slang codegen bugs.
    let msl_source = if desc.stage == SlangStage::Vertex {
        let patched = patch_vertex_msl_entry_point_params(&raw_msl);
        if patched != raw_msl {
            tracing::debug!(
                "Applied vertex EntryPointParams [[buffer(1)]] patch for {}",
                desc.entry_point
            );
        }
        let patched2 = patch_vertex_stage_in_pointer(&patched);
        if patched2 != patched {
            tracing::debug!(
                "Applied vertex [[stage_in]] pointer-to-value patch for {}",
                desc.entry_point
            );
        }
        patched2
    } else if desc.stage == SlangStage::Compute {
        // Fix cross-module [ForceInline] + groupshared bugs (slang #10641; see patch_compute_msl).
        let patched = patch_compute_msl(&raw_msl);
        if patched != raw_msl {
            tracing::debug!(
                "Applied compute MSL patches (EntryPointParams / threadgroup copy) for {}",
                desc.entry_point
            );
        }
        patched
    } else {
        raw_msl
    };

    tracing::debug!("Compiled MSL {} shader ({} bytes)", desc.entry_point, msl_source.len());

    if let Ok(dump_dir) = std::env::var("GOLDY_DUMP_SHADERS") {
        use std::io::Write;
        use std::sync::atomic::{AtomicU32, Ordering};
        static DUMP_IDX: AtomicU32 = AtomicU32::new(0);
        let idx = DUMP_IDX.fetch_add(1, Ordering::Relaxed);
        let dir = std::path::Path::new(&dump_dir);
        let _ = std::fs::create_dir_all(dir);
        let filename = format!("{:03}_{}.metal", idx, desc.entry_point);
        if let Ok(mut f) = std::fs::File::create(dir.join(&filename)) {
            let _ = f.write_all(msl_source.as_bytes());
            tracing::info!("Dumped MSL to {}/{}", dump_dir, filename);
        }
    }

    let library = device
        .new_library_with_source(&msl_source, &mtl::CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("Failed to create Metal library for {}: {}", desc.entry_point, e))?;

    Ok((library, Some(result.reflection)))
}

/// Ensure a shader stage is compiled. Compiles on first access.
pub(super) fn ensure_stage_compiled(
    state: &mut MetalState,
    shader_handle: ShaderHandle,
    stage: SlangStage,
) -> Result<()> {
    struct CompileScratch {
        device_handle: DeviceHandle,
        slang_source: String,
        search_paths: Vec<String>,
        optimization_level: crate::types::OptimizationLevel,
        defines: Vec<(String, String)>,
        layout_checks: Vec<crate::slang::OwnedLayoutCheck>,
        entry_point: &'static str,
    }

    let maybe_scratch: Option<CompileScratch> = {
        let shaders = &state.shaders;
        let shader = shaders.get(&shader_handle).context("Invalid shader handle")?;
        let (entry_point, need_compile) = match stage {
            SlangStage::Vertex => ("vs_main", shader.vertex_library.is_none()),
            SlangStage::Fragment => ("fs_main", shader.fragment_library.is_none()),
            SlangStage::Compute => ("cs_main", shader.compute_library.is_none()),
            _ => anyhow::bail!("Metal backend only supports Vertex, Fragment, and Compute stages"),
        };
        if !need_compile {
            None
        } else {
            Some(CompileScratch {
                device_handle: shader.device_handle,
                slang_source: shader.slang_source.clone(),
                search_paths: shader.search_paths.clone(),
                optimization_level: shader.optimization_level,
                defines: shader.defines.clone(),
                layout_checks: shader.layout_checks.clone(),
                entry_point,
            })
        }
    };

    let Some(scratch) = maybe_scratch else {
        return Ok(());
    };

    let mtl_dev = state
        .devices
        .get(&scratch.device_handle)
        .context("Shader's device no longer valid")?
        .device
        .clone();

    let search_path_refs: Vec<&str> = scratch.search_paths.iter().map(|s| s.as_str()).collect();
    let extra_defines: Vec<(&str, &str)> = scratch.defines.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let compile_desc = ShaderStageCompileDesc {
        slang_source: &scratch.slang_source,
        search_paths: &search_path_refs,
        entry_point: scratch.entry_point,
        stage,
        extra_defines: &extra_defines,
        layout_checks: &scratch.layout_checks,
        optimization_level: scratch.optimization_level,
    };

    let compiler = state.slang_compiler_mut_or_init()?;
    let (library, reflection) = compile_stage_with_reflection(compiler, &mtl_dev, &compile_desc)?;

    let shader = state
        .shaders
        .get_mut(&shader_handle)
        .expect("shader handle must be valid after ensure_stage_compiled");
    match stage {
        SlangStage::Vertex => shader.vertex_library = Some(library),
        SlangStage::Fragment => shader.fragment_library = Some(library),
        SlangStage::Compute => shader.compute_library = Some(library),
        _ => unreachable!("stage already validated"),
    }

    if shader.reflection.is_none() {
        let reflection = reflection.map(|mut r| {
            if r.push_constant_categories.is_empty() {
                r.push_constant_categories =
                    crate::slang::virtual_main::extract_push_constant_categories(&scratch.slang_source);
            }
            r
        });
        shader.reflection = reflection;
    }

    if !scratch.layout_checks.is_empty() {
        shader.layout_checks.clear();
    }

    Ok(())
}

/// Create a shader handle (compilation deferred to pipeline creation).
pub(super) fn create(
    devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    next_shader_handle: &mut ShaderHandle,
    desc: ShaderDesc<'_>,
) -> Result<ShaderHandle> {
    devices.get(&desc.device).context("Invalid device handle")?;

    let handle = *next_shader_handle;
    *next_shader_handle += 1;

    shaders.insert(
        handle,
        ShaderState {
            device_handle: desc.device,
            slang_source: desc.slang_source.to_string(),
            search_paths: desc.search_paths.iter().map(|s| s.to_string()).collect(),
            defines: desc
                .defines
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            optimization_level: desc.optimization_level,
            vertex_library: None,
            fragment_library: None,
            compute_library: None,
            reflection: None,
            layout_checks: desc.layout_checks,
        },
    );

    tracing::debug!("Created shader handle {} (compilation deferred)", handle);
    Ok(handle)
}

/// Destroy a shader.
pub(super) fn destroy(
    _devices: &HashMap<DeviceHandle, super::types::SharedLogicalDevice>,
    shaders: &mut HashMap<ShaderHandle, ShaderState>,
    shader_handle: ShaderHandle,
) {
    shaders.remove(&shader_handle);
}

// ============================================================================
// Unit tests for MSL patching functions
// ============================================================================
//
// All patch functions are pure `&str -> String` transforms, so tests need no
// GPU device — just synthetic MSL strings derived from real Slang output.
// Snippets omit struct bodies and unrelated boilerplate; the patchers only
// inspect specific substrings and line-level patterns.

#[cfg(test)]
mod tests {
    use super::{
        patch_compute_msl, patch_compute_msl_entry_point_params, patch_msl_threadgroup_copies,
        patch_vertex_msl_entry_point_params, patch_vertex_stage_in_pointer,
    };

    // ── patch_vertex_stage_in_pointer ────────────────────────────────────────

    /// Bindless vertex shader: Slang emits [[stage_in]] on a `thread*` pointer
    /// in the internal dispatch function (vs_main_0).  That occurrence must be
    /// stripped; the legitimate [[stage_in]] on the real [[vertex]] entry point
    /// must be preserved.
    #[test]
    fn stage_in_stripped_from_internal_thread_pointer_fn() {
        // Mirrors the real Slang output for a [goldy_vertex] shader.
        // The first [[stage_in]] appears on the vs_main_0 line which also has thread*.
        let msl = concat!(
            "StaticVarying_0 vs_main_0(",
            "const StaticVertexIn_0 thread* _pt0_0 [[stage_in]], ",
            "KernelContext_0 thread* kernelContext_4)\n",
            "{\n    return _pt0_0;\n}\n",
            "[[vertex]] vs_main_Result_0 vs_main(",
            "vertexInput_0 _S19 [[stage_in]], ",
            "GoldyBindlessResources_default_0 constant* gGoldy_1 [[buffer(0)]], ",
            "EntryPointParams_0 constant* _goldy_ep [[buffer(1)]])\n",
            "{\n    return _S19;\n}\n",
        );

        let out = patch_vertex_stage_in_pointer(msl);

        // Internal fn: [[stage_in]] removed, pointer type kept intact.
        assert!(
            out.contains("thread* _pt0_0,"),
            "thread pointer parameter should remain without [[stage_in]]"
        );
        assert!(
            !out.contains("_pt0_0 [[stage_in]]"),
            "[[stage_in]] must be stripped from the thread* pointer parameter"
        );

        // Real entry point: [[stage_in]] preserved.
        assert!(
            out.contains("_S19 [[stage_in]]"),
            "[[stage_in]] on the real vertex entry point must not be touched"
        );
    }

    /// Simple vertex shader (no bindless, no KernelContext): the only helper
    /// function uses `thread*` for passing the input by pointer, but that line
    /// does NOT carry [[stage_in]].  The real entry point's [[stage_in]] on a
    /// value-type parameter must survive unchanged.
    ///
    /// This is the exact case that broke `render_target_integration` tests when
    /// the patch used a file-global `thread*` search instead of a line-scoped one.
    #[test]
    fn stage_in_preserved_when_thread_star_only_in_helper_fn() {
        // The helper carries `thread*` but no [[stage_in]] on its signature line.
        // The entry point has [[stage_in]] on a value type — must not be touched.
        let msl = concat!(
            "VertexOutput_0 _goldy_user_vs_main_0(const VertexInput_0 thread* input_0)\n",
            "{\n    return *input_0;\n}\n",
            "[[vertex]] vs_main_Result_0 vs_main(vertexInput_0 _S1 [[stage_in]])\n",
            "{\n    return _S1;\n}\n",
        );

        let out = patch_vertex_stage_in_pointer(msl);

        assert_eq!(
            out, msl,
            "MSL must be unchanged when [[stage_in]] is only on a value-type entry point"
        );
    }

    /// No [[stage_in]] anywhere: patch must be a no-op.
    #[test]
    fn stage_in_noop_when_absent() {
        let msl = "[[kernel]] void cs_main(device uint* buf [[buffer(0)]])\n{\n}\n";
        let out = patch_vertex_stage_in_pointer(msl);
        assert_eq!(out, msl, "MSL without [[stage_in]] must pass through unchanged");
    }

    // ── patch_vertex_msl_entry_point_params ──────────────────────────────────

    /// Slang bug present: vertex shader has EntryPointParams_0 inside
    /// KernelContext_0 but the entry point exposes only [[buffer(0)]].
    /// The patch must inject [[buffer(1)]] into the signature and wire the
    /// entryPointParams_0 field in the body.
    #[test]
    fn vertex_ep_params_injected_when_buffer1_missing() {
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "struct KernelContext_0 {\n",
            "    GoldyBindlessResources_default_0 constant* gGoldy_0;\n",
            "    EntryPointParams_0 constant* entryPointParams_0;\n",
            "};\n",
            "[[vertex]] vs_main_Result_0 vs_main(",
            "vertexInput_0 _S19 [[stage_in]], ",
            "GoldyBindlessResources_default_0 constant* gGoldy_1 [[buffer(0)]])\n",
            "{\n",
            "    KernelContext_0 kernelContext_5;\n",
            "    (&kernelContext_5)->gGoldy_0 = gGoldy_1;\n",
            "    return _S19;\n",
            "}\n",
        );

        let out = patch_vertex_msl_entry_point_params(msl);

        assert!(
            out.contains("EntryPointParams_0 constant* _goldy_ep [[buffer(1)]])"),
            "[[buffer(1)]] parameter must be injected into the entry-point signature"
        );
        assert!(
            out.contains("(&kernelContext_5)->entryPointParams_0 = _goldy_ep;"),
            "entryPointParams_0 field must be wired to _goldy_ep in the entry-point body"
        );
        // Original [[buffer(0)]] binding must still be present.
        assert!(
            out.contains("[[buffer(0)]]"),
            "[[buffer(0)]] binding must remain after the patch"
        );
    }

    /// Already patched (both markers present): must be a no-op.
    #[test]
    fn vertex_ep_params_noop_when_already_correct() {
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "[[vertex]] vs_main_Result_0 vs_main(",
            "vertexInput_0 _S19 [[stage_in]], ",
            "GoldyBindlessResources_default_0 constant* gGoldy_1 [[buffer(0)]], ",
            "EntryPointParams_0 constant* _goldy_ep [[buffer(1)]])\n",
            "{\n    return _S19;\n}\n",
        );

        let out = patch_vertex_msl_entry_point_params(msl);
        assert_eq!(out, msl, "MSL with correct [[buffer(1)]] must pass through unchanged");
    }

    /// No EntryPointParams_0 at all (simple shader): must be a no-op.
    #[test]
    fn vertex_ep_params_noop_when_no_entry_point_params() {
        let msl = concat!(
            "[[vertex]] vs_main_Result_0 vs_main(vertexInput_0 _S1 [[stage_in]])\n",
            "{\n    return _S1;\n}\n",
        );

        let out = patch_vertex_msl_entry_point_params(msl);
        assert_eq!(out, msl, "MSL without EntryPointParams_0 must pass through unchanged");
    }

    /// Fragment shader: Slang correctly emits [[buffer(1)]] for fragment stages,
    /// so the patch must recognise this and leave the MSL unchanged.
    #[test]
    fn vertex_ep_params_noop_for_correctly_bound_fragment_shader() {
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "[[fragment]] pixelOutput_0 fs_main(",
            "pixelInput_0 _S11 [[stage_in]], ",
            "GoldyBindlessResources_default_0 constant* gGoldy_1 [[buffer(0)]], ",
            "EntryPointParams_0 constant* entryPointParams_1 [[buffer(1)]])\n",
            "{\n    return _S11;\n}\n",
        );

        let out = patch_vertex_msl_entry_point_params(msl);
        assert_eq!(
            out, msl,
            "Fragment shader with correct [[buffer(1)]] must pass through unchanged"
        );
    }

    // ── patch_compute_msl_entry_point_params ─────────────────────────────────

    /// Slang Bug A: EntryPointParams_0 emitted as a bare value parameter
    /// (no `constant*`, no `[[buffer(1)]]`).  The patch must:
    ///   1. Fix the signature to `constant* VAR [[buffer(1)]])`.
    ///   2. Rewrite member accesses from `VAR.field` to `VAR->field`.
    ///   3. Rewrite the struct-copy assignment `= VAR;` to `= *VAR;`.
    #[test]
    fn compute_ep_params_patched_when_missing_constant_ptr() {
        // The KernelContext struct is intentionally omitted: it would contain
        // `EntryPointParams_0 constant*` which would trip the no-op guard.
        // Only the kernel signature (where the bug manifests) and usage sites
        // are needed to exercise the patch.
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "[[kernel]] void cs_main(\n",
            "    device uint* buf [[buffer(0)]],\n",
            "    EntryPointParams_0 epVar0)\n",
            "{\n",
            "    uint v = epVar0._bw0_0;\n",
            "    uint w = epVar0._bw0_0;\n",
            "    KernelContext_0 kc2 = epVar0;\n",
            "}\n",
        );

        let out = patch_compute_msl_entry_point_params(msl);

        // Signature: bare value param replaced with constant pointer + [[buffer(1)]].
        assert!(
            out.contains("EntryPointParams_0 constant* epVar0 [[buffer(1)]])"),
            "entry-point parameter must be fixed to `constant* epVar0 [[buffer(1)]]`"
        );
        // Member access: `.field` → `->field`.
        assert!(
            out.contains("epVar0->_bw0_0"),
            "member access must be rewritten from . to ->"
        );
        // Struct-copy assignment: `= epVar0;` → `= *epVar0;`.
        assert!(
            out.contains("= *epVar0;"),
            "struct-copy assignment must be rewritten to dereference the pointer"
        );
    }

    /// `EntryPointParams_0 constant*` already present: no-op.
    #[test]
    fn compute_ep_params_noop_when_already_correct() {
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "[[kernel]] void cs_main(\n",
            "    device uint* buf [[buffer(0)]],\n",
            "    EntryPointParams_0 constant* epVar0 [[buffer(1)]])\n",
            "{\n}\n",
        );

        let out = patch_compute_msl_entry_point_params(msl);
        assert_eq!(out, msl, "MSL with correct constant* must pass through unchanged");
    }

    /// No EntryPointParams_0 at all: no-op.
    #[test]
    fn compute_ep_params_noop_when_absent() {
        let msl = concat!(
            "[[kernel]] void cs_main(device uint* buf [[buffer(0)]])\n",
            "{\n    buf[0] = 42;\n}\n",
        );

        let out = patch_compute_msl_entry_point_params(msl);
        assert_eq!(out, msl, "MSL without EntryPointParams_0 must pass through unchanged");
    }

    // ── patch_msl_threadgroup_copies ─────────────────────────────────────────

    /// Pattern 1a: explicit `thread array<...>` copy from a KernelContext field.
    /// Must become a `threadgroup array<...>&` reference.
    #[test]
    fn threadgroup_copies_explicit_thread_patched() {
        let msl = concat!(
            "[[kernel]] void cs_main()\n",
            "{\n",
            "    thread array<uint, int(64)> scratch = *kernelContext_0->shared_data;\n",
            "    scratch[0] = 1;\n",
            "}\n",
        );

        let out = patch_msl_threadgroup_copies(msl);

        assert!(
            out.contains("threadgroup array<uint, int(64)>& scratch"),
            "explicit thread copy must become a threadgroup reference"
        );
        assert!(
            !out.contains("thread array<uint"),
            "thread address space must be replaced by threadgroup"
        );
    }

    /// Pattern 1b: implicit thread qualifier (bare `array<...>` with no explicit
    /// address space) copied from a KernelContext field.
    /// Must become a `threadgroup array<...>&` reference.
    #[test]
    fn threadgroup_copies_implicit_thread_patched() {
        let msl = concat!(
            "[[kernel]] void cs_main()\n",
            "{\n",
            "    array<uint, int(64)> scratch = *kernelContext_0->shared_data;\n",
            "    scratch[0] = 1;\n",
            "}\n",
        );

        let out = patch_msl_threadgroup_copies(msl);

        assert!(
            out.contains("threadgroup array<uint, int(64)>& scratch"),
            "implicit thread copy must become a threadgroup reference"
        );
    }

    /// Pattern 2: secondary copy of a variable that was already made a threadgroup
    /// reference by Pattern 1.  The secondary copy must also be patched.
    #[test]
    fn threadgroup_copies_secondary_copy_also_patched() {
        let msl = concat!(
            "[[kernel]] void cs_main()\n",
            "{\n",
            // Pattern 1: scratch becomes a threadgroup ref.
            "    thread array<uint, int(64)> scratch = *kernelContext_0->shared_data;\n",
            // Pattern 2: local is a thread copy of scratch (already a tg ref).
            "    thread array<uint, int(64)> local = scratch;\n",
            "    local[0] = 1;\n",
            "}\n",
        );

        let out = patch_msl_threadgroup_copies(msl);

        // Both scratch and local must be threadgroup references.
        assert!(
            out.contains("threadgroup array<uint, int(64)>& scratch"),
            "pattern-1 variable must become a threadgroup reference"
        );
        assert!(
            out.contains("threadgroup array<uint, int(64)>& local"),
            "pattern-2 copy of a tg-ref variable must also become a threadgroup reference"
        );
        assert!(
            !out.contains("thread array<uint"),
            "no thread array copies must remain after patching"
        );
    }

    /// No `= *kernelContext` pattern present: must be a no-op.
    #[test]
    fn threadgroup_copies_noop_when_no_kernelcontext_copy() {
        let msl = concat!(
            "[[kernel]] void cs_main()\n",
            "{\n",
            "    thread uint x = 0;\n",
            "    x = 1;\n",
            "}\n",
        );

        let out = patch_msl_threadgroup_copies(msl);
        assert_eq!(out, msl, "MSL without kernelContext copies must pass through unchanged");
    }

    // ── patch_compute_msl (orchestrator) ─────────────────────────────────────

    /// Both Bug A (missing EntryPointParams constant*) and Bug B (thread array
    /// copy from kernelContext) are present.  A single call to the orchestrator
    /// must fix both.
    #[test]
    fn patch_compute_msl_fixes_both_bugs() {
        // KernelContext struct definition omitted for the same reason as
        // `compute_ep_params_patched_when_missing_constant_ptr`: it would
        // contain `EntryPointParams_0 constant*` and trigger the no-op guard.
        let msl = concat!(
            "struct EntryPointParams_0 { uint _bw0_0; };\n",
            "[[kernel]] void cs_main(\n",
            "    device uint* buf [[buffer(0)]],\n",
            "    EntryPointParams_0 epVar0)\n",
            "{\n",
            "    thread array<uint, int(32)> scratch = *kernelContext_0->shared_data;\n",
            "    buf[0] = epVar0._bw0_0 + scratch[0];\n",
            "}\n",
        );

        let out = patch_compute_msl(msl);

        // Bug A fixed.
        assert!(
            out.contains("EntryPointParams_0 constant* epVar0 [[buffer(1)]])"),
            "Bug A: entry-point parameter must be fixed to constant*"
        );
        assert!(
            out.contains("epVar0->_bw0_0"),
            "Bug A: member access must be rewritten to ->"
        );

        // Bug B fixed.
        assert!(
            out.contains("threadgroup array<uint, int(32)>& scratch"),
            "Bug B: thread array copy must become a threadgroup reference"
        );
    }

    /// Neither bug is present: orchestrator must be a no-op.
    #[test]
    fn patch_compute_msl_noop_when_clean() {
        let msl = concat!(
            "[[kernel]] void cs_main(device uint* buf [[buffer(0)]])\n",
            "{\n",
            "    buf[0] = 42;\n",
            "}\n",
        );

        let out = patch_compute_msl(msl);
        assert_eq!(out, msl, "clean MSL must pass through the orchestrator unchanged");
    }
}
