## Development

After completing any task, run the CI checks locally:

```bash
cargo fmt --all -- --check
cargo clippy --no-default-features -- -D warnings
cargo clippy -- -D warnings
cargo test
```

## Running examples

To run all examples in a row interactively, use

`run_all_examples.sh`

## Debugging

For debugging tips, see [DEBUGGING.md](DEBUGGING.md).

To run `cargo` with **`GOLDY_VALIDATION=1`** (Vulkan Khronos validation / Metal `MTL_SHADER_VALIDATION`), use `./run_with_validation.sh` (for example `./run_with_validation.sh test --features vulkan`).

For backend selection, see [Backend Architecture](docs/src/architecture/backends.md).

For conditional compilation, see [Conditional Compilation](docs/src/architecture/conditional-compilation.md).
