# AgentBible Rust

Rust workspace for the AgentBible v1 correctness specification.

## Commands

```bash
cd languages/rust
cargo test
cargo clippy -- -D warnings
cargo fmt --check
```

`agentbible::types` contains strict validating newtypes. `agentbible::check`
contains runtime assertion macros that emit provenance JSON and then panic.
