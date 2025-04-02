import logging
from transformers import EarlyStoppingCallback

logger = logging.getLogger("sentence_transformers")
hf_logger = logging.getLogger("transformers")
st_logger = logging.getLogger('sentence_transformers')


class EarlyStoppingWithLoggingCallback(EarlyStoppingCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"Early stopping required `metric_for_best_model`, but `{metric_to_check}` was not found in metrics. "
                "Early stopping is disabled."
            )
            return

        previous_counter = self.early_stopping_patience_counter
        self.check_metric_value(args, state, control, metric_value)

        if self.early_stopping_patience_counter != previous_counter:
            logger.info(
                f"No improvement in {metric_to_check}: "
                f"current = {metric_value:.6f}, best = {state.best_metric:.6f}, "
                f"threshold = {self.early_stopping_threshold:.6f} "
                f"({self.early_stopping_patience_counter}/{self.early_stopping_patience})"
            )

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            logger.info(
                f"Early stopping triggered. `{metric_to_check}` did not improve for "
                f"{self.early_stopping_patience} evaluation steps."
            )
            control.should_training_stop = True
