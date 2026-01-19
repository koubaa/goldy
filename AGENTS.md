## Development

After completing any task, run the CI checks locally:

```bash
cargo fmt --all -- --check
cargo clippy --no-default-features -- -D warnings
cargo clippy --features vulkan -- -D warnings
cargo test
```

For debugging tips, see DEBUGGING.md