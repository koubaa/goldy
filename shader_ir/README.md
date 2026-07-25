# goldy_shader_ir

Language-independent shader IR and structured `KernelAbi` metadata for Goldy's
compile-time Rust→Slang compute frontend.

This crate is intentionally small: frontends (Rust proc-macro today; other
languages later) lower into the IR / ABI, and Goldy's virtual-main emitters /
runtime preparation consume the structured result. Slang remains the runtime
backend compiler.
