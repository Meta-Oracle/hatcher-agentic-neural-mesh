# HatcherLabs Neural Mesh

A Rust-native, publishable agentic mesh framework for modeling node adaptation, inter-agent coupling, and policy pressure through deterministic equations and digest-backed serialization envelopes.

## The mesh thesis

This project frames agentic behavior as a controllable mesh of stateful nodes and weighted relations. Each node evolves through local adaptation, each edge captures coupling pressure, and the mesh emits a decision signal that can guide orchestration, escalation, or stabilization.

## Core equations embodied in the runtime

The runtime now encodes the following conceptual dynamics as a live simulation loop:

- State evolution:
  $\Omega_{t+1} = \Omega_t + \alpha(L + E + C) - \beta(F + D)$
- Node influence:
  $A_i = I_i \times S_i \times P_i \times C_i \times M_i$
- Aggregate influence:
  $A = \sum_i A_i + \gamma \sum_{i \ne j}(A_i A_j W_{ij})$
- Link adaptation:
  $T_{ij}(t+1) = T_{ij}(t) + \lambda S_{ij} - \mu E_{ij}$
- Pressure / capacity ratio:
  $P = \frac{C + \tau}{U \times B \times I}$

These become a practical lifecycle:
1. Initialize node state from request features.
2. Evolve each node through local adaptation and coupling.
3. Recalculate edge signals and aggregate influence.
4. Serialize the resulting state into a zk-style digest envelope.
5. Emit a decision signal and pressure score for the next control action.

## Architecture

- Core crate: shared contracts, serialization types, and mesh state payloads.
- Neural crate: the engine that evolves node state and edge coupling.
- Playground crate: rehearsal and battle-style simulation loops.
- UX crate: interactive terminal demonstration of the mesh runtime.

## What is included

- A typed mesh contract layer in the core crate.
- Deterministic agentic node serialization with a digest-backed envelope.
- A neural engine that evolves agent nodes through multi-step mesh simulation.
- A playground for rehearsal and battle-style simulations.
- A terminal UX for demonstrating the mesh interactively.

## Quick start

```bash
cargo test
cargo run -p hatcher-ux
```

## Publishability notes

The workspace is structured around crate-level metadata, semantic versioning, and a clear API surface so it can be published to crates.io with documentation and examples.
