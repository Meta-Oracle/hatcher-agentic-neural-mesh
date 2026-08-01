use hatcher_core::{ApiRequest, ApiResponse, MemoryGraph};
use hatcher_playground::{AgentArena, AgentBattle};
use serde_json::json;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tokio::runtime::Runtime;
use warp::Filter;

#[derive(Clone)]
struct AppState {
    arena: Arc<Mutex<AgentArena>>,
    graph: Arc<Mutex<MemoryGraph>>,
}

impl AppState {
    fn new() -> Self {
        Self {
            arena: Arc::new(Mutex::new(AgentArena::new())),
            graph: Arc::new(Mutex::new(MemoryGraph::default())),
        }
    }
}

async fn serve_http(state: AppState) {
    let state_filter = warp::any().map(move || state.clone());

    let health = warp::path!("health").map(|| warp::reply::json(&json!({"status": "ok"})));

    let infer = warp::path!("api" / "infer")
        .and(warp::post())
        .and(warp::body::json())
        .and(state_filter.clone())
        .map(|req: ApiRequest, state: AppState| {
            let arena = state.arena.lock().unwrap();
            let response = arena.run_round(&req.agent_id, &req.prompt, req.features.clone());
            let parsed: serde_json::Value = serde_json::from_str(&response).unwrap();
            let decision = &parsed["decision"];
            let api_response = ApiResponse {
                accepted: parsed["accepted"].as_bool().unwrap_or(false),
                action: decision["action"].as_str().unwrap_or("review").to_string(),
                confidence: decision["confidence"].as_f64().unwrap_or(0.0),
                trace_id: parsed["trace_id"].as_str().unwrap_or("trace").to_string(),
            };
            warp::reply::json(&api_response)
        });

    let memory = warp::path!("api" / "memory")
        .and(state_filter.clone())
        .map(|state: AppState| {
            let mut graph = state.graph.lock().unwrap();
            graph.add_edge("intent", "policy");
            graph.add_edge("signal", "intent");
            let payload = graph.clone();
            warp::reply::json(&payload)
        });

    let battle = warp::path!("api" / "battle")
        .and(state_filter.clone())
        .map(|state: AppState| {
            let battle = AgentBattle::new();
            let report = battle.run();
            warp::reply::json(&json!({"winner": report.winner, "rounds": report.rounds, "summary": report.summary}))
        });

    let routes = health.or(infer).or(memory).or(battle);
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}

#[derive(Debug, Clone)]
struct TelemetrySnapshot {
    tick: usize,
    active_agents: usize,
    confidence: f64,
    battle_winner: String,
    memory_nodes: usize,
    memory_edges: usize,
    status: String,
}

impl TelemetrySnapshot {
    fn render(&self) -> String {
        format!(
            "\x1b[2J\x1b[H
╔════════════════ HatcherLabs Live Telemetry ═══════════════╗
║ Tick        : {:>3}                                         ║
║ Agents      : {:>2}                                         ║
║ Confidence  : {:>5.2}                                      ║
║ Winner      : {:<12}                                     ║
║ Memory      : {:>2} nodes / {:>2} edges                      ║
║ Status      : {:<18}                                     ║
╚════════════════════════════════════════════════════════════╝
",
            self.tick,
            self.active_agents,
            self.confidence,
            self.battle_winner,
            self.memory_nodes,
            self.memory_edges,
            self.status
        )
    }
}

fn build_telemetry_snapshot(tick: usize) -> TelemetrySnapshot {
    let battle = AgentBattle::new();
    let report = battle.run();
    let mut graph = MemoryGraph::default();
    graph.add_edge("intent", "policy");
    graph.add_edge("signal", "intent");
    TelemetrySnapshot {
        tick,
        active_agents: 3,
        confidence: 0.7 + (tick as f64 * 0.03),
        battle_winner: report.winner,
        memory_nodes: graph.nodes.len(),
        memory_edges: graph.edges.len(),
        status: if tick % 2 == 0 { "stabilizing".into() } else { "rehearsing".into() },
    }
}

fn run_live_telemetry() {
    println!("Streaming telemetry. Press Ctrl+C to stop.");
    for tick in 0..20 {
        let snapshot = build_telemetry_snapshot(tick);
        print!("{}", snapshot.render());
        io::stdout().flush().unwrap();
        thread::sleep(Duration::from_millis(400));
    }
    println!("\nTelemetry stream complete.");
}

fn interactive_cli() {
    println!("HatcherLabs neural mesh playground");
    println!("=================================");
    println!("Interactive console");
    println!("  run / r      - execute a sample neural inference");
    println!("  battle / b   - run an agent battle simulation");
    println!("  memory / m   - inspect the memory graph");
    println!("  api / a      - start the HTTP API server");
    println!("  telemetry / t - stream live animated telemetry");
    println!("  help / h     - show this menu");
    println!("  quit / q     - exit");
    println!("Tip: pressing Enter defaults to a quick demo run.");

    loop {
        print!("\n> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let command = input.trim().to_ascii_lowercase();
        let command = if command.is_empty() { "run".to_string() } else { command };

        match command.as_str() {
            "run" | "r" | "demo" | "d" => {
                let arena = AgentArena::new();
                let simulation = arena.run_round(
                    "guardian-01",
                    "scematica-style rehearsal: protect the policy envelope",
                    vec![0.2, 0.5, 0.8, 0.3],
                );
                println!("{}", simulation);
            }
            "battle" | "b" => {
                let battle = AgentBattle::new();
                let report = battle.run();
                println!("winner: {} | rounds: {} | {}", report.winner, report.rounds, report.summary);
            }
            "memory" | "m" => {
                let mut graph = MemoryGraph::default();
                graph.add_edge("intent", "policy");
                graph.add_edge("signal", "intent");
                println!("nodes: {:?}", graph.nodes);
                println!("edges: {:?}", graph.edges);
            }
            "api" | "a" => {
                let state = AppState::new();
                let rt = Runtime::new().unwrap();
                rt.block_on(async { serve_http(state).await });
            }
            "telemetry" | "t" => {
                run_live_telemetry();
            }
            "help" | "h" => {
                println!("Commands: run, battle, memory, api, telemetry, help, quit");
            }
            "quit" | "q" | "exit" | "e" => break,
            _ => println!("unknown command. try help"),
        }
    }
}

fn main() {
    interactive_cli();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn telemetry_snapshot_render_contains_key_fields() {
        let snapshot = TelemetrySnapshot {
            tick: 4,
            active_agents: 3,
            confidence: 0.91,
            battle_winner: "guardian".into(),
            memory_nodes: 2,
            memory_edges: 2,
            status: "rehearsing".into(),
        };

        let rendered = snapshot.render();
        assert!(rendered.contains("HatcherLabs Live Telemetry"));
        assert!(rendered.contains("guardian"));
        assert!(rendered.contains("rehearsing"));
    }
}
