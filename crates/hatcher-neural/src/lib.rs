use hatcher_core::{ExecutionMode, HatcherRequest, HatcherResponse, NeuralSignal};

#[derive(Debug, Clone)]
pub enum InferenceBackend {
    Native,
    Onnx { model_path: String },
}

#[derive(Debug, Clone)]
pub struct NeuralMesh {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub backend: InferenceBackend,
}

impl NeuralMesh {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim,
            backend: InferenceBackend::Native,
        }
    }

    pub fn with_onnx(model_path: impl Into<String>) -> Self {
        Self {
            input_dim: 4,
            hidden_dim: 6,
            output_dim: 4,
            backend: InferenceBackend::Onnx { model_path: model_path.into() },
        }
    }

    pub fn evaluate(&self, request: &HatcherRequest) -> HatcherResponse {
        let mut scores = vec![0.0_f64; self.output_dim];
        let features = &request.features;

        for (idx, score) in scores.iter_mut().enumerate() {
            let base = features.get(idx).copied().unwrap_or(0.0);
            *score = match idx {
                0 => 0.25 + base * 0.6,
                1 => 0.2 + base * 0.3,
                2 => 0.18 + base * 0.5,
                3 => 0.15 + base * 0.4,
                _ => 0.1 + base * 0.2,
            };
        }

        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let action = match best_idx {
            0 => "observe",
            1 => "stabilize",
            2 => "escalate",
            3 => "delegate",
            _ => "review",
        };

        let confidence = (scores[best_idx] / 2.0).clamp(0.0, 1.0);
        let decision = NeuralSignal {
            agent: request.agent_id.clone(),
            intent: request.prompt.clone(),
            confidence,
            action: action.to_string(),
            rationale: format!("Rust neural mesh evaluated {:?} under {:?}", request.role, request.execution_mode),
        };

        HatcherResponse {
            accepted: matches!(request.execution_mode, ExecutionMode::Sandbox | ExecutionMode::Controlled | ExecutionMode::Production),
            decision,
            trace_id: format!("trace-{}", request.agent_id.len()),
        }
    }
}

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
}
