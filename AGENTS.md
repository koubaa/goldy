## Compatibility

Legacy compatibility is not relevant before the 0.2 release. Make breaking changes as required for clean code and update all clients locally in the workspace.

## Development

useful precommit commands:

`cargo fmt --all -- --check`
`cargo clippy -- -D warnings`
`cargo clippy --no-default-features -- -D warnings`
`cargo check`
`RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`

## Running tests

`GOLDY_VALIDATION=all cargo test`

## Running examples

To run all examples in a row interactively, use

`run_all_examples.sh`

To run a specific example (for instance metaballs), use

`cargo run --features examples --example metaballs`

## Debugging

For debugging tips, see [DEBUGGING.md](DEBUGGING.md).

For backend selection, see [Backend Architecture](docs/src/architecture/backends.md).

For conditional compilation, see [Conditional Compilation](docs/src/architecture/conditional-compilation.md).

## Cursor Cloud specific instructions

See [.cursor/cloud-agent.md](.cursor/cloud-agent.md).
