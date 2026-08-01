# Architecture overview

## System posture

The HatcherLabs neural mesh is composed of two layers:

- A Hatcher control plane layer for identity, permissions, secrets, and execution.
- A Rust-native intelligence layer for model inference, agent replay, and simulation.

This preserves the Hatcher ethos while letting the intelligence layer become a first-class runtime.

## Core components

### 1. Hatcher bridge
A typed API bridge that carries intent, features, and policy requests between Hatcher and the Rust runtime.

### 2. Neural mesh
A lightweight neural decision engine that consumes feature vectors and emits signals such as `observe`, `escalate`, `stabilize`, or `delegate`.

### 3. Agent playground
A Scematica-style arena where multiple agents can rehearse, debate, and explore strategies under a shared governance policy.

### 4. UX shell
A terminal-first interface that showcases the system and allows operators to inspect decisions, logs, and scenario outcomes.

## Runtime flow

1. The operator or Hatcher workflow submits a request through the typed API.
2. The Rust runtime evaluates the request with the neural mesh.
3. The playground may run scenario rehearsal or multi-agent debate.
4. The result is returned to Hatcher for execution or review.

## Why this fits the brief

- It is Rust-first without replacing the existing Hatcher stack.
- It introduces a Scematica-inspired agent arena and signal-driven policy loop.
- It is modular enough to grow into an inference runtime with optional ONNX support later.
