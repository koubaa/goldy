## Development

After completing any task, run the CI checks locally:

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
```

## Running examples

To run all examples in a row interactively, use

`run_all_examples.sh`

## Debugging

For debugging tips, see [DEBUGGING.md](DEBUGGING.md).

For conditional compilation, see [Conditional Compilation](docs/src/architecture/conditional-compilation.md).
