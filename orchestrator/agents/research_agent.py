"""Research agent that explores hyperparameter space autonomously."""
import logging
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..bridges.agenthub_bridge import AutoresearchAgenthubBridge, ExperimentResult

logger = logging.getLogger(__name__)

# Hyperparameter search space (only train.py-modifiable params)
DEPTH_OPTIONS = [4, 6, 8, 10, 12]
ASPECT_RATIO_OPTIONS = [32, 48, 64, 80, 96]
WINDOW_PATTERNS = ["SSSL", "SSLL", "SLSL", "LLLL"]
LR_RANGES = {"matrix": (0.01, 0.08), "embedding": (0.3, 0.8)}


@dataclass
class HyperparamSet:
    depth: int = 8
    aspect_ratio: int = 64
    window_pattern: str = "SSSL"
    matrix_lr: float = 0.04
    embedding_lr: float = 0.6
    weight_decay: float = 0.2
    strategy: str = "random"

    def to_patch(self) -> dict[str, str]:
        """Return {variable_name: new_value} for patching train.py."""
        return {
            "DEPTH": str(self.depth),
            "ASPECT_RATIO": str(self.aspect_ratio),
            "WINDOW_PATTERN": f'"{self.window_pattern}"',
            "MATRIX_LR": str(self.matrix_lr),
            "EMBEDDING_LR": str(self.embedding_lr),
            "WEIGHT_DECAY": str(self.weight_decay),
        }


class ResearchAgent:
    """Autonomous research agent exploring hyperparameter space."""

    def __init__(
        self,
        agent_id: str,
        bridge: AutoresearchAgenthubBridge,
        autoresearch_path: Path,
        strategy: str = "random",
    ) -> None:
        self.agent_id = agent_id
        self.bridge = bridge
        self.autoresearch_path = autoresearch_path
        self.strategy = strategy
        self._experiments_run = 0

    def propose_hyperparams(self) -> HyperparamSet:
        """Propose next hyperparameter set based on strategy."""
        if self.strategy == "explore_depth":
            return self._explore_depth()
        elif self.strategy == "explore_width":
            return self._explore_width()
        else:
            return self._explore_random()

    def _explore_depth(self) -> HyperparamSet:
        """Systematically vary DEPTH while keeping other params fixed."""
        depth = DEPTH_OPTIONS[self._experiments_run % len(DEPTH_OPTIONS)]
        return HyperparamSet(depth=depth, strategy="explore_depth")

    def _explore_width(self) -> HyperparamSet:
        """Systematically vary ASPECT_RATIO (model width)."""
        ar = ASPECT_RATIO_OPTIONS[self._experiments_run % len(ASPECT_RATIO_OPTIONS)]
        return HyperparamSet(aspect_ratio=ar, strategy="explore_width")

    def _explore_random(self) -> HyperparamSet:
        """Random exploration of the full hyperparameter space."""
        return HyperparamSet(
            depth=random.choice(DEPTH_OPTIONS),
            aspect_ratio=random.choice(ASPECT_RATIO_OPTIONS),
            window_pattern=random.choice(WINDOW_PATTERNS),
            matrix_lr=round(random.uniform(*LR_RANGES["matrix"]), 4),
            embedding_lr=round(random.uniform(*LR_RANGES["embedding"]), 4),
            weight_decay=round(random.uniform(0.1, 0.4), 2),
            strategy="explore_random",
        )

    def run_experiment(self, hyperparams: HyperparamSet) -> ExperimentResult | None:
        """Run a single autoresearch experiment with given hyperparams."""
        train_py = self.autoresearch_path / "train.py"
        if not train_py.exists():
            logger.error("train.py not found at %s", train_py)
            return None

        # Patch train.py with new hyperparams
        original = train_py.read_text()
        patched = original
        for var, val in hyperparams.to_patch().items():
            patched = re.sub(
                rf"^({var}\s*=\s*).*$",
                rf"\g<1>{val}",
                patched,
                flags=re.MULTILINE,
            )
        train_py.write_text(patched)

        try:
            logger.info("Running experiment: depth=%d ar=%d strategy=%s",
                        hyperparams.depth, hyperparams.aspect_ratio, hyperparams.strategy)
            result = subprocess.run(
                ["python", "train.py"],
                capture_output=True, text=True,
                timeout=360,  # 5min + buffer
                cwd=str(self.autoresearch_path),
            )
            self._experiments_run += 1

            if result.returncode == 0:
                val_bpb = self._parse_val_bpb(result.stdout)
                return ExperimentResult(
                    commit_hash=self._get_commit_hash(),
                    val_bpb=val_bpb,
                    memory_gb=0.0,
                    training_seconds=300.0,
                    status="success",
                    description=f"depth={hyperparams.depth} ar={hyperparams.aspect_ratio} strategy={hyperparams.strategy}",
                    hyperparams=hyperparams.to_patch(),
                )
            else:
                logger.error("Experiment failed: %s", result.stderr[:500])
                return ExperimentResult(
                    commit_hash=self._get_commit_hash(),
                    val_bpb=999.0,
                    memory_gb=0.0,
                    training_seconds=0.0,
                    status="failed",
                    description=result.stderr[:200],
                )
        except subprocess.TimeoutExpired:
            return ExperimentResult(
                commit_hash=self._get_commit_hash(),
                val_bpb=999.0, memory_gb=0.0, training_seconds=360.0,
                status="timeout", description="Training timed out",
            )
        finally:
            # Restore original train.py
            train_py.write_text(original)

    def _parse_val_bpb(self, output: str) -> float:
        """Extract val_bpb from train.py stdout."""
        match = re.search(r"val_bpb:\s+([\d.]+)", output)
        return float(match.group(1)) if match else 999.0

    def _get_commit_hash(self) -> str:
        """Get current git HEAD hash."""
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(self.autoresearch_path),
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
