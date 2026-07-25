# goldy_derive

Proc-macro helpers for [`goldy`](https://crates.io/crates/goldy):

- `#[compute]` — Rust GPU-dialect compute kernels → canonical `[goldy_compute]` Slang + typed `Kernel::prepare` / `record`
- `LayoutCheckable` — layout introspection for `#[repr(C)]` GPU structs
- `StructuredBufferElement` — marker for typed buffer uploads

This crate is typically pulled in automatically via `goldy` (`#[goldy::compute]`, etc.).
You do not need to depend on it directly unless you are working around a workspace/`path`
dependency edge case.
