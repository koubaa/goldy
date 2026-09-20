## Compatibility

Within 0.3.x, prefer additive changes and document breakages in CHANGELOG. Major API shifts belong in a new minor until 1.0.

## Design considerations

There is churn happening now, it is important to keep certain architectural principles in mind.

- A "parcel" is a stable identity for data held by the runtime in trust for the program, generic enough to span lifecycle (leased vs owned) and type (buffers, textures, images, ...).
- An "exchange" is the mediated relationship with an external subsystem
  (`SurfaceExchange`, `MemoryExchange`). Bind once into a scheme as a
  `Transaction` / withdraw or deposit transaction; each submission may produce a
  linear claim for externally settled handoffs (present, withdraw). Deposits
  settle inside graph execution and do not publish claims.
- `Runtime` is the public machine agent and warehouse owner (retained parcels,
  shaders, pipelines, capabilities). `Context` is a submission timeline with
  transient/deposit pools; `Context::runtime()` returns the shared root. Backend
  device handles stay private.
- There are two kinds of pools associated with a runtime: "retained" (acquired
  on `Runtime`) and "transient" (context-scoped leases). These are shared across
  contexts of the same runtime.
- We are refactoring in the direction of removing imperative APIs (like read_to_cpu) in favor of scheme submissions as the only way to affect parcels.

## Support

- WARP is not officially supported but it's useful for catching issues.

## Development

useful precommit commands:

`cargo fmt --all -- --check`
`cargo clippy -- -D warnings`
`cargo clippy --no-default-features -- -D warnings`
`cargo check`
`RUSTDOCFLAGS='-D warnings' cargo doc --no-deps`
`scripts/check_docs_book.sh` (mdBook build; catches unresolved `{{#include}}` and broken links)

## Running tests

`GOLDY_VALIDATION=all cargo test`

## Running examples

To run all examples in a row interactively, use

`run_all_examples.sh`

To run a specific example (for instance metaballs), use

`cargo run --features examples --example metaballs`

## Debugging

For debugging tips, see [DEBUGGING.md](DEBUGGING.md).

For backend selection, see [Backend Architecture](docs/src/backends/overview.md).

For conditional compilation, see [Conditional Compilation](docs/src/backends/conditional-compilation.md).

## Cursor Cloud specific instructions

See [.cursor/cloud-agent.md](.cursor/cloud-agent.md).
