# Contributing

RAG welcomes contributions! Here's how to get involved.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rag.git
   cd rag
   ```
3. Build and run tests:
   ```bash
   cargo build
   cargo test
   cargo run --example triangle --release
   ```

## Development Setup

### Requirements

- Rust 1.70+
- Vulkan SDK 1.3+
- Git

### Building

```bash
cargo build           # Debug build
cargo build --release # Release build
cargo build --examples # Build all examples
```

### Testing

```bash
cargo test            # Run unit tests
cargo clippy          # Lint check
cargo fmt --check     # Format check
```

## Code Style

- Follow Rust idioms and conventions
- Use `cargo fmt` for formatting
- Use `cargo clippy` for linting
- Document public APIs with doc comments

```rust
/// Creates a new buffer with the given data.
///
/// # Arguments
///
/// * `device` - The GPU device
/// * `data` - Initial data to upload
/// * `usage` - Buffer usage flags
///
/// # Example
///
/// ```
/// let buffer = Buffer::with_data(&device, &vertices, BufferUsage::VERTEX)?;
/// ```
pub fn with_data<T: Pod>(device: &Device, data: &[T], usage: BufferUsage) -> Result<Self> {
    // ...
}
```

## Pull Request Process

1. Create a branch for your feature:
   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes with clear commits

3. Ensure tests pass:
   ```bash
   cargo test
   cargo clippy
   ```

4. Push and create a pull request

5. Wait for review

## What to Contribute

### Good First Issues

- Documentation improvements
- Example applications
- Bug fixes
- Test coverage

### Larger Contributions

- New examples
- Performance improvements
- Backend implementations (Metal, DX12)
- API enhancements

### Not Accepting

- OpenGL backend (out of scope)
- Features requiring old hardware support
- Breaking API changes without discussion

## Reporting Issues

When reporting bugs, include:

1. RAG version
2. OS and GPU
3. Minimal reproduction code
4. Expected vs actual behavior
5. Error messages (full text)

## Communication

- GitHub Issues: Bug reports, feature requests
- Pull Requests: Code contributions
- Discussions: Questions, ideas

## License

Contributions are licensed under MIT, same as the project.

By contributing, you agree to license your contribution under the MIT license.

## Code of Conduct

Be respectful and constructive. We're all here to build something useful.

