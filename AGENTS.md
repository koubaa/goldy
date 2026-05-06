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

For backend selection, see [Backend Architecture](docs/src/architecture/backends.md).

For conditional compilation, see [Conditional Compilation](docs/src/architecture/conditional-compilation.md).

## Cursor Cloud specific instructions

See [.cursor/cloud-agent.md](.cursor/cloud-agent.md).
