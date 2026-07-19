## Compatibility

Legacy compatibility is not relevant before the 0.2 release. Make breaking changes as required for clean code and update all clients locally in the workspace.

## Design considerations

There is churn happening now, it is important to keep certain architectural principles in mind.

- A "parcel" is a unit of property, generic enough to span lifecycle (leased vs owned) and property type (buffers, textures, images, ...).
- An "exchange" is the mediated relationship with an external subsystem. A "grant" is a
  typed release handle used by exchange paths such as readback; surface handoff prefers
  `SurfaceExchange` + `Transaction` + linear `Claim`.
- All allocations and schemes are threaded through a "context".
- There are two kinds of pools associated with a device: "retained" and "transient". These are shared across contexts.
- We are refactoring in the direction of removing imperative APIs (like read_to_cpu) in favor of scheme submissions as the only way to affect property.

## Support

- WARP is not officially supported but it's useful for catching issues.

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
