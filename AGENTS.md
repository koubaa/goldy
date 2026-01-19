## Development

After completing any task, run the CI checks locally:

```bash
cargo fmt --all -- --check
cargo clippy -- -D warnings
cargo test
```

For debugging tips, see DEBUGGING.md
For conditional compilation, see [Conditional Compilation](docs/src/architecture/conditional-compilation.md) 
