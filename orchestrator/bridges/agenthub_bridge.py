"""Bridge between autoresearch experiments and agenthub Git DAG."""
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    commit_hash: str
    val_bpb: float
    memory_gb: float
    training_seconds: float
    status: str  # "success" | "oom" | "timeout" | "failed"
    description: str
    hyperparams: dict = field(default_factory=dict)


class AutoresearchAgenthubBridge:
    """Push autoresearch experiment results to agenthub Git DAG."""

    def __init__(self, server_url: str, api_key: str, channel: str = "experiments") -> None:
        self.server_url = server_url.rstrip("/")
        self.api_key = api_key
        self.channel = channel
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def push_experiment(self, result: ExperimentResult) -> str | None:
        """Push experiment commit bundle to agenthub. Returns hash or None on failure."""
        try:
            bundle_path = Path(f"/tmp/exp_{result.commit_hash[:8]}.bundle")
            proc = subprocess.run(
                ["git", "bundle", "create", str(bundle_path), "HEAD"],
                capture_output=True, text=True, timeout=60,
            )
            if proc.returncode != 0:
                logger.error("git bundle failed: %s", proc.stderr)
                return None

            with bundle_path.open("rb") as f:
                response = requests.post(
                    f"{self.server_url}/api/git/push",
                    data=f.read(),
                    headers={**self._headers, "Content-Type": "application/octet-stream"},
                    timeout=60,
                )

            if response.status_code == 200:
                data = response.json()
                commit_hash = data.get("hashes", [result.commit_hash])[0]
                logger.info("Pushed experiment %s: val_bpb=%.4f", commit_hash[:8], result.val_bpb)
                return commit_hash
            else:
                logger.error("Push failed: HTTP %d", response.status_code)
                return None
        except Exception as e:
            logger.error("Push error: %s", e)
            return None

    def get_frontier_experiments(self) -> list[dict]:
        """Get frontier (leaf) commits from agenthub DAG."""
        try:
            response = requests.get(
                f"{self.server_url}/api/git/leaves",
                headers=self._headers,
                timeout=10,
            )
            if response.status_code == 200:
                return response.json().get("leaves", [])
            return []
        except Exception as e:
            logger.warning("Failed to get frontier: %s", e)
            return []

    def post_result(self, result: ExperimentResult) -> bool:
        """Post experiment result to message board channel."""
        message = (
            f"[RESULT] commit={result.commit_hash[:8]} "
            f"val_bpb={result.val_bpb:.4f} "
            f"mem={result.memory_gb:.1f}GB "
            f"status={result.status} "
            f"desc={result.description}"
        )
        try:
            # Ensure channel exists
            requests.post(
                f"{self.server_url}/api/channels",
                json={"name": self.channel, "description": "Experiment results"},
                headers=self._headers,
                timeout=10,
            )
            response = requests.post(
                f"{self.server_url}/api/channels/{self.channel}/posts",
                json={"content": message},
                headers=self._headers,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.warning("Failed to post result: %s", e)
            return False
