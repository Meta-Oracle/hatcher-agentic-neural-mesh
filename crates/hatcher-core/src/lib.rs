use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiRequest {
    pub agent_id: String,
    pub prompt: String,
    pub features: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse {
    pub accepted: bool,
    pub action: String,
    pub confidence: f64,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentRole {
    Orchestrator,
    Executor,
    Critic,
    Explorer,
    Guardian,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExecutionMode {
    Sandbox,
    Controlled,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSpec {
    pub name: String,
    pub version: String,
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSignal {
    pub agent: String,
    pub intent: String,
    pub confidence: f64,
    pub action: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HatcherRequest {
    pub agent_id: String,
    pub role: AgentRole,
    pub execution_mode: ExecutionMode,
    pub prompt: String,
    pub features: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HatcherResponse {
    pub accepted: bool,
    pub decision: NeuralSignal,
    pub trace_id: String,
}

impl HatcherResponse {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String)>,
}

impl MemoryGraph {
    pub fn add_node(&mut self, node: impl Into<String>) {
        let node = node.into();
        if !self.nodes.contains(&node) {
            self.nodes.push(node);
        }
    }

    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let from = from.into();
        let to = to.into();
        self.add_node(&from);
        self.add_node(&to);
        if !self.edges.contains(&(from.clone(), to.clone())) {
            self.edges.push((from, to));
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgenticNodeData {
    pub id: String,
    pub label: String,
    pub omega: f64,
    pub learning: f64,
    pub exploration: f64,
    pub cohesion: f64,
    pub friction: f64,
    pub disruption: f64,
    pub intention: f64,
    pub stability: f64,
    pub policy: f64,
    pub coherence: f64,
    pub memory: f64,
    pub capacity: f64,
    pub utilization: f64,
    pub bandwidth: f64,
    pub intensity: f64,
    pub tau: f64,
}

impl AgenticNodeData {
    pub fn new(id: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            omega: 1.0,
            learning: 0.3,
            exploration: 0.2,
            cohesion: 0.4,
            friction: 0.1,
            disruption: 0.05,
            intention: 0.4,
            stability: 0.5,
            policy: 0.6,
            coherence: 0.7,
            memory: 0.3,
            capacity: 1.0,
            utilization: 0.5,
            bandwidth: 1.0,
            intensity: 0.4,
            tau: 0.1,
        }
    }

    pub fn to_zk_envelope(&self) -> Result<SerializedNodeEnvelope, serde_json::Error> {
        unimplemented!("zk envelope support will be implemented in the neural mesh crate")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializedNodeEnvelope {
    pub codec: String,
    pub schema_version: u32,
    pub payload: AgenticNodeData,
    pub digest: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MeshState {
    pub nodes: Vec<AgenticNodeData>,
    pub edges: Vec<MeshEdge>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeshEdge {
    pub from: String,
    pub to: String,
    pub weight: f64,
    pub signal: f64,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshStepResult {
    pub nodes: Vec<AgenticNodeData>,
    pub edges: Vec<MeshEdge>,
    pub aggregate_influence: f64,
    pub pressure: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_graph_tracks_nodes_and_edges() {
        let mut graph = MemoryGraph::default();
        graph.add_edge("signal", "policy");
        assert!(graph.nodes.contains(&"signal".to_string()));
        assert!(graph.nodes.contains(&"policy".to_string()));
        assert!(graph.edges.contains(&("signal".to_string(), "policy".to_string())));
    }

    #[test]
    fn zk_envelope_is_digest_backed_and_versioned() {
        let node = AgenticNodeData::new("node-1", "guardian");
        let envelope = node.to_zk_envelope().unwrap();
        assert_eq!(envelope.codec, "zk-canonical-v1");
        assert_eq!(envelope.schema_version, 1);
        assert!(!envelope.digest.is_empty());
        assert_eq!(envelope.payload.id, "node-1");
    }
}
