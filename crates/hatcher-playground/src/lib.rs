use hatcher_core::{AgentRole, ExecutionMode, HatcherRequest};
use hatcher_neural::NeuralMesh;

#[derive(Debug, Clone)]
pub struct AgentArena {
    pub mesh: NeuralMesh,
}

impl AgentArena {
    pub fn new() -> Self {
        Self {
            mesh: NeuralMesh::new(4, 6, 4),
        }
    }

    pub fn run_round(&self, agent_id: &str, prompt: &str, features: Vec<f64>) -> String {
        let request = HatcherRequest {
            agent_id: agent_id.to_string(),
            role: AgentRole::Explorer,
            execution_mode: ExecutionMode::Sandbox,
            prompt: prompt.to_string(),
            features,
        };

        let response = self.mesh.evaluate(&request);
        response.to_json().unwrap()
    }

    pub fn run_simulation(&self, agent_id: &str, prompt: &str, features: Vec<f64>, steps: usize) -> serde_json::Value {
        let request = HatcherRequest {
            agent_id: agent_id.to_string(),
            role: AgentRole::Explorer,
            execution_mode: ExecutionMode::Controlled,
            prompt: prompt.to_string(),
            features,
        };

        serde_json::to_value(self.mesh.simulate(&request, steps)).unwrap()
    }
}

#[derive(Debug, Clone)]
pub struct BattleReport {
    pub winner: String,
    pub rounds: usize,
    pub summary: String,
}

#[derive(Debug, Clone)]
pub struct AgentBattle {
    pub arena: AgentArena,
}

impl AgentBattle {
    pub fn new() -> Self {
        Self {
            arena: AgentArena::new(),
        }
    }

    pub fn run(&self) -> BattleReport {
        let mut rounds = 0usize;
        let mut scores = std::collections::HashMap::from([
            ("guardian".to_string(), 0.0f64),
            ("critic".to_string(), 0.0f64),
            ("explorer".to_string(), 0.0f64),
        ]);

        for (agent_id, features) in [
            ("guardian", vec![0.2, 0.4, 0.6, 0.8]),
            ("critic", vec![0.5, 0.3, 0.7, 0.1]),
            ("explorer", vec![0.8, 0.6, 0.2, 0.4]),
        ] {
            let response = self.arena.run_round(agent_id, "battle rehearsal", features);
            let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
            let confidence = parsed["decision"]["confidence"].as_f64().unwrap_or(0.0);
            *scores.entry(agent_id.to_string()).or_insert(0.0) += confidence;
            rounds += 1;
        }

        let winner = scores.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0.clone();

        BattleReport {
            winner,
            rounds,
            summary: format!("battle complete with {} rounds", rounds),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn battle_simulation_selects_a_winner() {
        let battle = AgentBattle::new();
        let report = battle.run();
        assert!(!report.winner.is_empty());
        assert_eq!(report.rounds, 3);
    }
}
