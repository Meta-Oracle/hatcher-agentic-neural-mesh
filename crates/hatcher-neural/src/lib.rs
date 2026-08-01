use hatcher_core::{
    AgenticNodeData, ExecutionMode, HatcherRequest, HatcherResponse, MeshEdge, MeshSimulation,
    MeshState, MeshStepResult, NeuralSignal,
};

#[derive(Debug, Clone)]
pub enum InferenceBackend {
    Native,
    Onnx { model_path: String },
}

#[derive(Debug, Clone)]
pub struct MeshConfig {
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub mu: f64,
    pub tau: f64,
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            alpha: 0.35,
            beta: 0.2,
            gamma: 0.08,
            lambda: 0.25,
            mu: 0.1,
            tau: 0.15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NeuralMesh {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub backend: InferenceBackend,
    pub config: MeshConfig,
}

impl NeuralMesh {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            backend: InferenceBackend::Native,
            config: MeshConfig::default(),
        }
    }

    pub fn with_onnx(model_path: impl Into<String>) -> Self {
        Self {
            input_dim: 4,
            hidden_dim: 6,
            output_dim: 4,
            backend: InferenceBackend::Onnx { model_path: model_path.into() },
            config: MeshConfig::default(),
        }
    }

    pub fn initial_state(&self, request: &HatcherRequest) -> MeshState {
        let mut nodes = Vec::new();
        let seed_feature = request.features.first().copied().unwrap_or(0.0);

        for (idx, feature) in request.features.iter().enumerate().take(self.output_dim) {
            let mut node = AgenticNodeData::new(format!("{}-{}", request.agent_id, idx), format!("{}-{}", request.prompt, idx));
            node.omega += seed_feature * 0.2 + (idx as f64) * 0.05;
            node.learning += feature * 0.2;
            node.exploration += feature * 0.1;
            node.policy += 0.05;
            node.capacity += 0.2;
            node.utilization += 0.1;
            nodes.push(node);
        }

        if nodes.is_empty() {
            let mut node = AgenticNodeData::new(&request.agent_id, &request.prompt);
            node.omega += seed_feature * 0.2;
            node.learning += 0.15;
            nodes.push(node);
        }

        MeshState { nodes, edges: vec![] }
    }

    pub fn step_state(&self, state: &MeshState, request: &HatcherRequest) -> MeshStepResult {
        let mut updated_nodes = Vec::with_capacity(state.nodes.len());
        let mut edge_updates = Vec::new();
        let mut aggregate_influence = 0.0;

        for (idx, node) in state.nodes.iter().enumerate() {
            let feature = request.features.get(idx).copied().unwrap_or(0.0);
            let activity = node.learning + node.exploration + node.cohesion + feature;
            let omega_next = (node.omega + self.config.alpha * activity - self.config.beta * (node.friction + node.disruption)).clamp(0.0, 4.0);
            let ai = (node.intention * node.stability * node.policy * node.coherence * node.memory).max(0.0);
            let influence = ai + feature * 0.15;
            aggregate_influence += influence;

            let mut next = node.clone();
            next.omega = omega_next;
            next.learning = (node.learning + self.config.lambda * feature * 0.1).clamp(0.0, 1.0);
            next.exploration = (node.exploration + self.config.mu * 0.08).clamp(0.0, 1.0);
            next.cohesion = (node.cohesion + 0.03).clamp(0.0, 1.0);
            next.friction = (node.friction + 0.01).clamp(0.0, 1.0);
            next.disruption = (node.disruption + 0.005).clamp(0.0, 1.0);
            next.intention = (node.intention + 0.02).clamp(0.0, 1.0);
            next.stability = (node.stability + 0.03).clamp(0.0, 1.0);
            next.policy = (node.policy + 0.01).clamp(0.0, 1.0);
            next.coherence = (node.coherence + 0.02).clamp(0.0, 1.0);
            next.memory = (node.memory + 0.01).clamp(0.0, 1.0);
            next.capacity = (node.capacity + feature * 0.01).clamp(0.0, 2.0);
            next.utilization = (node.utilization + 0.01).clamp(0.0, 1.0);
            next.bandwidth = (node.bandwidth + 0.01).clamp(0.0, 2.0);
            next.intensity = (node.intensity + 0.01).clamp(0.0, 1.0);
            next.tau = (node.tau + self.config.tau * 0.01).clamp(0.0, 1.0);
            updated_nodes.push(next);
        }

        for (left_idx, left) in updated_nodes.iter().enumerate() {
            for (right_idx, right) in updated_nodes.iter().enumerate() {
                if left_idx == right_idx {
                    continue;
                }

                let left_ai = (left.intention * left.stability * left.policy * left.coherence * left.memory).max(0.0);
                let right_ai = (right.intention * right.stability * right.policy * right.coherence * right.memory).max(0.0);
                let coupling = (left_ai * right_ai * ((left.omega + right.omega) / 2.0)).clamp(0.0, 1.0);
                let signal = (coupling * self.config.gamma).clamp(0.0, 1.0);
                let energy = (left.memory + right.memory) / 2.0;
                edge_updates.push(MeshEdge {
                    from: left.id.clone(),
                    to: right.id.clone(),
                    weight: coupling,
                    signal,
                    energy,
                });
                aggregate_influence += coupling * self.config.gamma;
            }
        }

        let avg_coherence = updated_nodes.iter().map(|node| node.coherence).sum::<f64>() / updated_nodes.len().max(1) as f64;
        let avg_utilization = updated_nodes.iter().map(|node| node.utilization).sum::<f64>() / updated_nodes.len().max(1) as f64;
        let avg_bandwidth = updated_nodes.iter().map(|node| node.bandwidth).sum::<f64>() / updated_nodes.len().max(1) as f64;
        let avg_intensity = updated_nodes.iter().map(|node| node.intensity).sum::<f64>() / updated_nodes.len().max(1) as f64;
        let denominator = (avg_utilization * avg_bandwidth * avg_intensity).max(1e-6);
        let pressure = ((avg_coherence + self.config.tau) / denominator).clamp(0.0, 10.0);

        let digest = updated_nodes
            .iter()
            .map(|node| node.to_zk_envelope().unwrap().digest.clone())
            .collect::<Vec<_>>()
            .join("");

        MeshStepResult {
            step_index: 0,
            nodes: updated_nodes,
            edges: edge_updates,
            aggregate_influence,
            pressure,
            digest,
        }
    }

    pub fn simulate(&self, request: &HatcherRequest, steps: usize) -> MeshSimulation {
        let mut current_state = self.initial_state(request);
        let mut history = Vec::new();
        let mut last_digest = String::new();
        let mut final_pressure = 0.0;
        let mut final_aggregate_influence = 0.0;

        for step_index in 1..=steps.max(1) {
            let step = self.step_state(&current_state, request);
            let digest = step.digest.clone();
            let state_nodes = step.nodes.clone();
            let state_edges = step.edges.clone();
            current_state = MeshState { nodes: state_nodes, edges: state_edges };
            final_pressure = step.pressure;
            final_aggregate_influence = step.aggregate_influence;
            last_digest = digest;
            history.push(MeshStepResult {
                step_index,
                nodes: step.nodes,
                edges: step.edges,
                aggregate_influence: step.aggregate_influence,
                pressure: step.pressure,
                digest: step.digest,
            });
        }

        MeshSimulation {
            request_id: request.agent_id.clone(),
            steps: history,
            final_pressure,
            final_aggregate_influence,
            final_digest: last_digest,
        }
    }

    pub fn evaluate(&self, request: &HatcherRequest) -> HatcherResponse {
        let state = self.initial_state(request);
        let result = self.step_state(&state, request);
        let first_node = result.nodes.first().cloned().unwrap_or_else(|| AgenticNodeData::new(&request.agent_id, &request.prompt));
        let envelope = first_node.to_zk_envelope().unwrap();
        let confidence = (result.aggregate_influence / 2.0 + result.pressure * 0.02).clamp(0.0, 1.0);

        let action = if result.pressure > 1.2 {
            "escalate"
        } else if confidence > 0.6 {
            "stabilize"
        } else {
            "observe"
        };

        let decision = NeuralSignal {
            agent: request.agent_id.clone(),
            intent: request.prompt.clone(),
            confidence,
            action: action.to_string(),
            rationale: format!(
                "mesh step {} with aggregate influence {:.4} and pressure {:.4}; zk digest {}",
                execution_mode_name(&request.execution_mode),
                result.aggregate_influence,
                result.pressure,
                &envelope.digest[..12]
            ),
        };

        HatcherResponse {
            accepted: matches!(request.execution_mode, ExecutionMode::Sandbox | ExecutionMode::Controlled | ExecutionMode::Production),
            decision,
            trace_id: format!("trace-{}", request.agent_id.len()),
        }
    }
}

fn execution_mode_name(mode: &ExecutionMode) -> &'static str {
    match mode {
        ExecutionMode::Sandbox => "sandbox",
        ExecutionMode::Controlled => "controlled",
        ExecutionMode::Production => "production",
    }
}

/// Example usage:
///
/// ```rust
/// use hatcher_core::{AgentRole, ExecutionMode, HatcherRequest};
/// use hatcher_neural::NeuralMesh;
///
/// let mesh = NeuralMesh::new(4, 6, 4);
/// let request = HatcherRequest {
///     agent_id: "demo-agent".to_string(),
///     role: AgentRole::Explorer,
///     execution_mode: ExecutionMode::Controlled,
///     prompt: "mesh demo".to_string(),
///     features: vec![0.2, 0.4, 0.6, 0.8],
/// };
/// let simulation = mesh.simulate(&request, 3);
/// assert_eq!(simulation.steps.len(), 3);
/// ```
#[cfg(test)]
mod tests {
    use super::*;
    use hatcher_core::{AgentRole, ExecutionMode, HatcherRequest};

    #[test]
    fn evaluates_to_a_decision() {
        let mesh = NeuralMesh::new(4, 6, 4);
        let request = HatcherRequest {
            agent_id: "agent-1".into(),
            role: AgentRole::Orchestrator,
            execution_mode: ExecutionMode::Controlled,
            prompt: "stress test".into(),
            features: vec![0.2, 0.4, 0.6, 0.8],
        };

        let response = mesh.evaluate(&request);
        assert!(response.accepted);
        assert_eq!(response.decision.agent, "agent-1");
        assert!(response.decision.confidence >= 0.0);
    }

    #[test]
    fn step_state_produces_mesh_outputs() {
        let mesh = NeuralMesh::new(4, 6, 4);
        let request = HatcherRequest {
            agent_id: "agent-2".into(),
            role: AgentRole::Guardian,
            execution_mode: ExecutionMode::Sandbox,
            prompt: "mesh rehearsal".into(),
            features: vec![0.3, 0.7, 0.2, 0.9],
        };

        let state = mesh.initial_state(&request);
        let result = mesh.step_state(&state, &request);
        assert!(!result.nodes.is_empty());
        assert!(!result.edges.is_empty());
        assert!(result.aggregate_influence >= 0.0);
        assert!(result.pressure >= 0.0);
    }

    #[test]
    fn simulate_runs_multiple_steps() {
        let mesh = NeuralMesh::new(4, 6, 4);
        let request = HatcherRequest {
            agent_id: "agent-3".into(),
            role: AgentRole::Explorer,
            execution_mode: ExecutionMode::Controlled,
            prompt: "multi-step rehearsal".into(),
            features: vec![0.4, 0.6, 0.2, 0.8],
        };

        let simulation = mesh.simulate(&request, 4);
        assert_eq!(simulation.steps.len(), 4);
        assert_eq!(simulation.steps.first().unwrap().step_index, 1);
        assert!(simulation.final_pressure >= 0.0);
        assert!(simulation.final_aggregate_influence >= 0.0);
        assert!(!simulation.final_digest.is_empty());
    }
}
