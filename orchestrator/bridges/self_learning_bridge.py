"""Bridge to ai-native-self-learning-agents for model optimization."""
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    version_id: str
    model_path: str
    val_bpb: float
    training_accuracy: float = 0.0


class SelfLearningBridge:
    """Trigger DSPy/PEFT optimization from best experiments."""

    def __init__(self, base_url: str = "http://localhost:8001") -> None:
        self.base_url = base_url
        self._mgr = None
        self._engine = None
        try:
            from ai_native_self_learning_agents.orchestration.learning.model_version_manager import ModelVersionManager  # type: ignore[import-not-found]
            from ai_native_self_learning_agents.orchestration.learning.self_learning_engine import SelfLearningEngine  # type: ignore[import-not-found]
            self._mgr = ModelVersionManager()
            self._engine = SelfLearningEngine()
            logger.info("self-learning-agents connected")
        except ImportError:
            logger.info("self-learning-agents not importable, using HTTP stub")

    def trigger_dspy_optimization(self, version_id: str, model_path: str) -> bool:
        """Trigger DSPy BootstrapFewShot recompilation."""
        logger.info("DSPy optimization triggered for %s", version_id)
        if self._engine:
            try:
                self._engine.trigger_optimization(version_id)  # type: ignore[union-attr]
                return True
            except Exception as e:
                logger.error("DSPy optimization failed: %s", e)
        return False

    def trigger_peft_finetuning(self, version_id: str, training_data_path: str) -> bool:
        """Trigger PEFT/LoRA fine-tuning."""
        logger.info("PEFT fine-tuning triggered for %s on %s", version_id, training_data_path)
        return False  # Stub: requires GPU setup

    def register_model_version(self, version: ModelVersion) -> bool:
        """Register a new model version in the version manager."""
        if self._mgr:
            try:
                self._mgr.register_model(  # type: ignore[union-attr]
                    version_id=version.version_id,
                    model_path=version.model_path,
                    vllm_path=version.model_path,
                    base_model="gpt-custom",
                    training_samples=1000,
                    training_accuracy=version.training_accuracy,
                )
                logger.info("Registered model version: %s", version.version_id)
                return True
            except Exception as e:
                logger.error("Registration failed: %s", e)
        return False
