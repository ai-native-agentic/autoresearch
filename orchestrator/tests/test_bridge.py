"""Tests for autoresearch orchestrator bridges and agents."""
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.bridges.agenthub_bridge import AutoresearchAgenthubBridge, ExperimentResult
from orchestrator.bridges.economic_bridge import EconomicValidationBridge
from orchestrator.bridges.ondevice_bridge import OndeviceBenchmarkBridge
from orchestrator.agents.research_agent import ResearchAgent, HyperparamSet
from orchestrator.experiment_orchestrator import ExperimentOrchestrator, OrchestratorConfig


def make_result(**kwargs) -> ExperimentResult:
    defaults = dict(
        commit_hash="abc123",
        val_bpb=0.95,
        memory_gb=40.0,
        training_seconds=300.0,
        status="success",
        description="test experiment",
    )
    defaults.update(kwargs)
    return ExperimentResult(**defaults)


class TestAgenthubBridge:
    def setup_method(self) -> None:
        self.bridge = AutoresearchAgenthubBridge("http://localhost:8080", "test-key")

    @patch("requests.get")
    def test_get_frontier_experiments_success(self, mock_get: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"leaves": [{"hash": "abc"}]})
        leaves = self.bridge.get_frontier_experiments()
        assert len(leaves) == 1
        assert leaves[0]["hash"] == "abc"

    @patch("requests.get")
    def test_get_frontier_experiments_failure(self, mock_get: MagicMock) -> None:
        mock_get.side_effect = Exception("Connection refused")
        leaves = self.bridge.get_frontier_experiments()
        assert leaves == []

    @patch("requests.post")
    def test_post_result(self, mock_post: MagicMock) -> None:
        mock_post.return_value = MagicMock(status_code=200)
        result = make_result()
        ok = self.bridge.post_result(result)
        assert ok is True


class TestResearchAgent:
    def test_propose_hyperparams_depth_strategy(self) -> None:
        bridge = MagicMock()
        agent = ResearchAgent("a1", bridge, autoresearch_path=MagicMock(), strategy="explore_depth")
        for i in range(5):
            hp = agent.propose_hyperparams()
            assert hp.depth in [4, 6, 8, 10, 12]
            agent._experiments_run += 1

    def test_propose_hyperparams_width_strategy(self) -> None:
        bridge = MagicMock()
        agent = ResearchAgent("a2", bridge, autoresearch_path=MagicMock(), strategy="explore_width")
        hp = agent.propose_hyperparams()
        assert hp.aspect_ratio in [32, 48, 64, 80, 96]

    def test_propose_hyperparams_random_strategy(self) -> None:
        bridge = MagicMock()
        agent = ResearchAgent("a3", bridge, autoresearch_path=MagicMock(), strategy="explore_random")
        hp = agent.propose_hyperparams()
        assert hp.strategy == "explore_random"
        assert hp.depth in [4, 6, 8, 10, 12]


class TestEconomicBridge:
    def test_calculate_roi_positive(self) -> None:
        bridge = EconomicValidationBridge()
        roi = bridge.calculate_roi(10.0, 25.0)
        assert abs(roi - 1.5) < 0.001

    def test_calculate_roi_zero_cost(self) -> None:
        bridge = EconomicValidationBridge()
        roi = bridge.calculate_roi(0.0, 25.0)
        assert roi == 0.0


class TestOrchestrator:
    def test_pareto_frontier_empty(self) -> None:
        config = OrchestratorConfig(agenthub_api_key="key")
        orch = ExperimentOrchestrator(config)
        frontier = orch.get_pareto_frontier()
        assert frontier == []

    def test_pareto_frontier_selects_best(self) -> None:
        config = OrchestratorConfig(agenthub_api_key="key")
        orch = ExperimentOrchestrator(config)
        orch._results = [
            make_result(val_bpb=0.90, memory_gb=40.0),
            make_result(val_bpb=0.95, memory_gb=30.0),
            make_result(val_bpb=1.00, memory_gb=20.0),
        ]
        frontier = orch.get_pareto_frontier()
        assert len(frontier) >= 1
        assert frontier[0].val_bpb == 0.90
