"""Orchestrates multiple research agents running in parallel."""
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .agents.research_agent import ResearchAgent
from .bridges.agenthub_bridge import AutoresearchAgenthubBridge, ExperimentResult

logger = logging.getLogger(__name__)

STRATEGIES = ["explore_depth", "explore_width", "explore_random"]


@dataclass
class OrchestratorConfig:
    agenthub_url: str = "http://localhost:8080"
    agenthub_api_key: str = "test-key"
    autoresearch_path: str = "."
    num_agents: int = 3
    max_experiments_per_agent: int = 10
    poll_interval_seconds: int = 60
    improvement_threshold: float = 0.02  # 2% BPB improvement triggers self-learning


class ExperimentOrchestrator:
    """Spawns multiple research agents and monitors the experiment DAG."""

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self.bridge = AutoresearchAgenthubBridge(
            server_url=config.agenthub_url,
            api_key=config.agenthub_api_key,
        )
        self.autoresearch_path = Path(config.autoresearch_path)
        self._results: list[ExperimentResult] = []

    def spawn_agents(self) -> list[ResearchAgent]:
        """Create agents with different exploration strategies."""
        return [
            ResearchAgent(
                agent_id=f"agent-{i}",
                bridge=self.bridge,
                autoresearch_path=self.autoresearch_path,
                strategy=STRATEGIES[i % len(STRATEGIES)],
            )
            for i in range(self.config.num_agents)
        ]

    def start(self, max_experiments: int | None = None) -> list[ExperimentResult]:
        """Start the orchestration loop."""
        agents = self.spawn_agents()
        total = max_experiments or (self.config.max_experiments_per_agent * self.config.num_agents)

        logger.info("Starting orchestrator: %d agents, %d total experiments", len(agents), total)

        with ThreadPoolExecutor(max_workers=len(agents)) as executor:
            futures = {
                executor.submit(self._run_agent, agent, total // len(agents)): agent
                for agent in agents
            }
            for future in as_completed(futures):
                agent = futures[future]
                try:
                    results = future.result()
                    self._results.extend(results)
                    logger.info("Agent %s completed %d experiments", agent.agent_id, len(results))
                except Exception as e:
                    logger.error("Agent %s failed: %s", agent.agent_id, e)

        return self._results

    def _run_agent(self, agent: ResearchAgent, n_experiments: int) -> list[ExperimentResult]:
        """Run a single agent for n_experiments."""
        results = []
        for _ in range(n_experiments):
            hyperparams = agent.propose_hyperparams()
            result = agent.run_experiment(hyperparams)
            if result:
                results.append(result)
                self.bridge.post_result(result)
                self.bridge.push_experiment(result)
        return results

    def get_pareto_frontier(self) -> list[ExperimentResult]:
        """Return experiments on the Pareto frontier (val_bpb vs memory_gb)."""
        if not self._results:
            return []
        sorted_by_bpb = sorted(
            [r for r in self._results if r.status == "success"],
            key=lambda r: r.val_bpb,
        )
        frontier: list[ExperimentResult] = []
        min_memory = float("inf")
        for result in sorted_by_bpb:
            if result.memory_gb < min_memory:
                frontier.append(result)
                min_memory = result.memory_gb
        return frontier

    def should_trigger_self_learning(self, baseline_bpb: float = 1.0) -> ExperimentResult | None:
        """Check if best experiment exceeds improvement threshold."""
        successes = [r for r in self._results if r.status == "success"]
        if not successes:
            return None
        best = min(successes, key=lambda r: r.val_bpb)
        improvement = (baseline_bpb - best.val_bpb) / baseline_bpb
        if improvement >= self.config.improvement_threshold:
            logger.info("Improvement %.1f%% exceeds threshold, triggering self-learning", improvement * 100)
            return best
        return None
