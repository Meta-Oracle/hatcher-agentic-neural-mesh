# HatcherLabs Neural Mesh

A Rust-first architecture for a Hatcher-aligned neural system with a Scematica-inspired agent playground and a terminal UX shell.

## North star

- Keep Hatcher as the control plane for authentication, permissions, secrets, execution, and policy enforcement.
- Let the Rust service own the neural mesh, agent arena, and experimentation loop.
- Expose a small typed API so the Rust engine can integrate cleanly with the broader Hatcher stack.
- Add Scematica-style touches: signal rails, agent playgrounds, and governance-driven experimentation.

## Workspace layout

- `crates/hatcher-core` - shared contracts and typed API payloads.
- `crates/hatcher-neural` - neural mesh and decision engine.
- `crates/hatcher-playground` - multi-agent simulation and playground loop.
- `crates/hatcher-ux` - terminal UX binary demonstrating the system.

## Quick start

```bash
cargo run -p hatcher-ux
```

## Design principles

1. Hatcher retains authority over execution and secrets.
2. The Rust binary provides fast, deterministic inference and orchestration.
3. The playground supports rehearsal, debate, and policy stress-testing.
4. The system can evolve toward ONNX-backed models without forcing a rewrite of the control plane.
