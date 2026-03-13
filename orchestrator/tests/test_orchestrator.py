from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from ..agents.research_agent import ResearchAgent
from ..bridges.agenthub_bridge import ExperimentResult
from ..cli import cli
from ..experiment_orchestrator import (
    ExperimentOrchestrator,
    OrchestratorConfig,
)


def make_result(**kwargs):
    result = ExperimentResult(
        commit_hash="hash",
        val_bpb=1.0,
        memory_gb=10.0,
        training_seconds=300.0,
        status="success",
        description="exp",
    )
    for key, value in kwargs.items():
        setattr(result, key, value)
    return result


@patch("orchestrator.experiment_orchestrator.AutoresearchAgenthubBridge")
def test_orchestrator_initialization_sets_config_and_path(mock_bridge):
    config = OrchestratorConfig(
        agenthub_url="http://agenthub.local",
        agenthub_api_key="secret",
        autoresearch_path="/tmp/autoresearch",
        num_agents=4,
    )

    orchestrator = ExperimentOrchestrator(config)

    mock_bridge.assert_called_once_with(
        server_url="http://agenthub.local", api_key="secret"
    )
    assert orchestrator.autoresearch_path.as_posix() == "/tmp/autoresearch"
    assert orchestrator.config.num_agents == 4
    assert orchestrator._results == []


@patch("orchestrator.experiment_orchestrator.AutoresearchAgenthubBridge")
def test_spawn_agents_assigns_round_robin_strategies(mock_bridge):
    config = OrchestratorConfig(agenthub_api_key="k", num_agents=5)
    orchestrator = ExperimentOrchestrator(config)

    agents = orchestrator.spawn_agents()

    assert len(agents) == 5
    assert [agent.strategy for agent in agents] == [
        "explore_depth",
        "explore_width",
        "explore_random",
        "explore_depth",
        "explore_width",
    ]


@patch("orchestrator.experiment_orchestrator.AutoresearchAgenthubBridge")
def test_get_pareto_frontier_filters_failures_and_dominated_results(mock_bridge):
    orchestrator = ExperimentOrchestrator(OrchestratorConfig(agenthub_api_key="k"))
    orchestrator._results = [
        make_result(
            val_bpb=0.9, memory_gb=16.0, status="success", description="dominated"
        ),
        make_result(
            val_bpb=0.95, memory_gb=8.0, status="success", description="frontier-1"
        ),
        make_result(
            val_bpb=0.85, memory_gb=14.0, status="success", description="frontier-0"
        ),
        make_result(
            val_bpb=0.8, memory_gb=15.0, status="failed", description="ignored"
        ),
    ]

    frontier = orchestrator.get_pareto_frontier()

    assert [result.description for result in frontier] == ["frontier-0", "frontier-1"]


def test_research_agent_depth_and_width_strategies_cycle_deterministically():
    bridge = MagicMock()
    depth_agent = ResearchAgent(
        "depth", bridge, autoresearch_path=MagicMock(), strategy="explore_depth"
    )
    width_agent = ResearchAgent(
        "width", bridge, autoresearch_path=MagicMock(), strategy="explore_width"
    )

    first_depth = depth_agent.propose_hyperparams()
    first_width = width_agent.propose_hyperparams()
    depth_agent._experiments_run = 1
    width_agent._experiments_run = 1
    second_depth = depth_agent.propose_hyperparams()
    second_width = width_agent.propose_hyperparams()

    assert (first_depth.depth, second_depth.depth) == (4, 6)
    assert (first_width.aspect_ratio, second_width.aspect_ratio) == (32, 48)


@patch(
    "orchestrator.agents.research_agent.random.uniform", side_effect=[0.02, 0.55, 0.21]
)
@patch("orchestrator.agents.research_agent.random.choice", side_effect=[10, 80, "LLLL"])
def test_research_agent_random_strategy_uses_randomized_values(
    mock_choice, mock_uniform
):
    agent = ResearchAgent(
        "random", MagicMock(), autoresearch_path=MagicMock(), strategy="explore_random"
    )

    hp = agent.propose_hyperparams()

    assert hp.strategy == "explore_random"
    assert hp.depth == 10
    assert hp.aspect_ratio == 80
    assert hp.window_pattern == "LLLL"
    assert hp.matrix_lr == 0.02
    assert hp.embedding_lr == 0.55
    assert hp.weight_decay == 0.21
    assert mock_choice.call_count == 3
    assert mock_uniform.call_count == 3


@patch("orchestrator.cli.ExperimentOrchestrator")
def test_cli_start_parses_options_and_invokes_orchestrator(mock_orchestrator):
    runner = CliRunner()
    instance = mock_orchestrator.return_value
    instance.start.return_value = [
        make_result(status="success", val_bpb=0.87, description="winner"),
        make_result(status="failed", val_bpb=999.0, description="failed"),
    ]

    result = runner.invoke(
        cli,
        [
            "start",
            "--agenthub-url",
            "http://localhost:9000",
            "--api-key",
            "abc",
            "--autoresearch-path",
            "/tmp/ar",
            "--num-agents",
            "2",
            "--max-experiments",
            "4",
        ],
    )

    assert result.exit_code == 0
    mock_orchestrator.assert_called_once()
    config = mock_orchestrator.call_args.args[0]
    assert config.agenthub_url == "http://localhost:9000"
    assert config.agenthub_api_key == "abc"
    assert config.autoresearch_path == "/tmp/ar"
    assert config.num_agents == 2
    instance.start.assert_called_once_with(max_experiments=4)
    assert "Completed 2 experiments" in result.output
    assert "Best val_bpb: 0.8700 (winner)" in result.output
