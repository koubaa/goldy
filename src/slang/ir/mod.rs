//! Reader and analysis toolkit for Slang's front-end IR.
//!
//! Slang can serialize a translation unit's IR (`-emit-ir`, or `spSetOutputContainerFormat`
//! plus `spGetContainerCode` through the C API) as a `.slang-module` RIFF container. This
//! module reads that container back into an in-memory instruction tree and offers the
//! rule-agnostic pieces every static check over the IR needs:
//!
//! - `riff` / `fossil`: the container and Slang's "fossil" serialization format.
//! - `Module`: instruction tree, decorations, name hints, literals, types (with generic
//!   substitution through `Types`), and linking of a translation unit with the modules it
//!   imports into one instruction space.
//! - `source_loc`: the `Sdeb` debug chunks, so instructions map back to Slang source.
//! - `Cfg`: per-function control-flow graph, block-parameter incomings and dominators.
//!
//! Nothing here knows about a particular check; the checks live in
//! [`crate::slang::shader_validation`]. The IR is what the Slang front end produced after
//! semantic checking and SSA construction, before target lowering: names, struct fields,
//! generics, structured control flow and calls are all intact.

use std::fmt;

mod cfg;
mod fossil;
mod module;
mod riff;
mod source_loc;
mod stable_names;

pub(crate) use cfg::Cfg;
pub(crate) use module::{op, Inst, IntShape, IntTy, Module, Types};

/// Slang source location recovered from a module's debug information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceLocation {
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for SourceLocation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{}", self.file, self.line, self.column)
    }
}

/// Malformed `.slang-module` container. Never raised for shaders a check merely cannot
/// reason about.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum IrError {
    #[error("malformed Slang module container: {0}")]
    Malformed(&'static str),
}

/// Parse a translation unit container and the containers of the modules it imports, and
/// link them into one [`Module`].
///
/// `libraries` may be in any order and may be incomplete: calls into a missing module are
/// left unresolved and checks treat them as unknown.
pub(crate) fn link_containers(container: &[u8], libraries: &[&[u8]]) -> Result<Module, IrError> {
    let mut modules = Module::parse_container(container)?;
    if modules.is_empty() {
        return Err(IrError::Malformed("container without IR modules"));
    }
    for lib in libraries {
        modules.extend(Module::parse_container(lib)?);
    }
    Ok(Module::link(modules))
}

/// Names of the modules a translation unit container imports (`import foo;`), excluding
/// Slang's own `core` and `glsl` modules. Derived from the mangled names on `import`
/// decorations, so only modules that are actually referenced are listed.
pub fn imported_modules(container: &[u8]) -> Result<Vec<String>, IrError> {
    let modules = Module::parse_container(container)?;
    let mut out: Vec<String> = Vec::new();
    for m in &modules {
        for i in 0..m.insts.len() as u32 {
            let Some(mangled) = m.import_name(i) else { continue };
            let Some(name) = module_of_mangled_name(mangled) else {
                continue;
            };
            if name == "core" || name == "glsl" || name == m.name || out.iter().any(|n| *n == name) {
                continue;
            }
            out.push(name.to_string());
        }
    }
    Ok(out)
}

/// Module component of a Slang mangled name: `_S<kind letters><len><module>...`.
fn module_of_mangled_name(mangled: &str) -> Option<&str> {
    let rest = mangled.strip_prefix("_S")?;
    let digits_at = rest.find(|c: char| c.is_ascii_digit())?;
    let rest = &rest[digits_at..];
    let len_end = rest.find(|c: char| !c.is_ascii_digit())?;
    let len: usize = rest[..len_end].parse().ok()?;
    rest.get(len_end..len_end + len)
}

#[cfg(test)]
mod tests;
