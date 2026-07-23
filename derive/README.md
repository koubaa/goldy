# goldy_derive

Proc-macro helpers for [`goldy`](https://crates.io/crates/goldy):

- `LayoutCheckable` — layout introspection for `#[repr(C)]` GPU structs
- `StructuredBufferElement` — marker for typed buffer uploads

This crate is typically pulled in automatically via `goldy`. You do not need to depend on it directly unless you are working around a workspace/`path` dependency edge case.
