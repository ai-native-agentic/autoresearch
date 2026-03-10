"""Bridge to economic-sdk for ROI tracking."""
import logging

logger = logging.getLogger(__name__)


class EconomicValidationBridge:
    """Track research costs and income via economic-sdk."""

    def __init__(self) -> None:
        self._tracker = None
        try:
            from economic_sdk import EconomicTracker  # type: ignore[import-not-found]
            self._tracker = EconomicTracker(
                signature="autoresearch-orchestrator",
                initial_balance=100.0,
                input_price_per_1m=3.0,
                output_price_per_1m=15.0,
            )
            self._tracker.initialize()
            logger.info("economic-sdk connected")
        except ImportError:
            logger.info("economic-sdk not available, using stub")

    def calculate_roi(self, experiment_cost: float, experiment_value: float) -> float:
        """Compute ROI: (value - cost) / cost."""
        if experiment_cost <= 0:
            return 0.0
        return (experiment_value - experiment_cost) / experiment_cost

    def log_experiment_cost(self, task_id: str, gpu_hours: float, cost_per_hour: float = 2.0) -> float:
        """Log GPU cost for an experiment."""
        cost = gpu_hours * cost_per_hour
        if self._tracker:
            try:
                self._tracker.start_task(task_id)
                self._tracker.track_api_call("gpu_compute", cost=cost)
                return cost
            except Exception as e:
                logger.error("Cost tracking failed: %s", e)
        logger.info("Experiment cost: task=%s cost=$%.4f", task_id, cost)
        return cost

    def log_experiment_value(self, task_id: str, val_bpb: float, baseline_bpb: float = 1.0) -> float:
        """Log experiment value based on BPB improvement."""
        improvement = max(0, baseline_bpb - val_bpb)
        # $10 per 0.01 BPB improvement (heuristic)
        value = improvement * 1000
        if self._tracker:
            try:
                self._tracker.add_work_income(amount=value, task_id=task_id, evaluation_score=0.8)
                self._tracker.end_task()
            except Exception as e:
                logger.error("Value tracking failed: %s", e)
        logger.info("Experiment value: task=%s val_bpb=%.4f value=$%.2f", task_id, val_bpb, value)
        return value
