"""CLI for the autoresearch orchestrator."""
import logging

import click

from .experiment_orchestrator import ExperimentOrchestrator, OrchestratorConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")


@click.group()
def cli() -> None:
    """Autoresearch Experiment Orchestrator CLI."""


@cli.command()
@click.option("--agenthub-url", default="http://localhost:8080")
@click.option("--api-key", required=True, envvar="AGENTHUB_API_KEY")
@click.option("--autoresearch-path", default=".", type=click.Path())
@click.option("--num-agents", default=3)
@click.option("--max-experiments", default=30)
def start(agenthub_url: str, api_key: str, autoresearch_path: str, num_agents: int, max_experiments: int) -> None:
    """Start the experiment orchestrator."""
    config = OrchestratorConfig(
        agenthub_url=agenthub_url,
        agenthub_api_key=api_key,
        autoresearch_path=autoresearch_path,
        num_agents=num_agents,
    )
    orchestrator = ExperimentOrchestrator(config)
    results = orchestrator.start(max_experiments=max_experiments)
    click.echo(f"Completed {len(results)} experiments")
    successes = [r for r in results if r.status == "success"]
    if successes:
        best = min(successes, key=lambda r: r.val_bpb)
        click.echo(f"Best val_bpb: {best.val_bpb:.4f} ({best.description})")


@cli.command()
@click.option("--agenthub-url", default="http://localhost:8080")
@click.option("--api-key", required=True, envvar="AGENTHUB_API_KEY")
def frontier(agenthub_url: str, api_key: str) -> None:
    """Show Pareto frontier from agenthub."""
    from .bridges.agenthub_bridge import AutoresearchAgenthubBridge
    bridge = AutoresearchAgenthubBridge(agenthub_url, api_key)
    leaves = bridge.get_frontier_experiments()
    click.echo(f"Frontier: {len(leaves)} experiments")
    for leaf in leaves[:10]:
        click.echo(f"  {leaf}")


if __name__ == "__main__":
    cli()
