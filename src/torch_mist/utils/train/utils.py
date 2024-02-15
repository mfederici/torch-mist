from typing import Optional

from torch import nn


class RunTerminationManager:
    def __init__(
        self,
        early_stopping: bool,
        tolerance: float,
        patience: int,
        verbose: bool,
        warmup_iterations: Optional[int],
        max_iterations: Optional[int],
        maximize: bool = False,
        minimize: bool = False,
    ):
        if early_stopping:
            if not maximize and not minimize:
                print(
                    "[Warning]: early_stopping can be used only when maximizing or minimizing, the parameter will be ignored."
                )
                early_stopping = False
        self.early_stopping = early_stopping
        self.maximize = maximize
        self.minimize = minimize
        self.best_value = 0
        self.tolerance = tolerance
        self.patience = patience
        self.current_patience = patience
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.best_state_dict = None

    def should_stop(
        self, iteration: int, valid_mi: Optional[float], model: nn.Module
    ) -> bool:
        stop = False

        if iteration <= self.warmup_iterations:
            return False

        if self.early_stopping:
            improvement = (
                (valid_mi - self.best_value)
                if self.maximize
                else (self.best_value - valid_mi)
            )
            if improvement >= self.tolerance:
                # Improvement, update best and reset the patience
                self.best_value = valid_mi
                self.current_patience = self.patience
                self.best_state_dict = model.state_dict()
            else:
                self.current_patience -= 1
                if self.verbose:
                    print(f"Loosing patience: {self.current_patience}")

            if self.current_patience <= 0:
                if self.verbose:
                    print("No improvements on validation, stopping.")
                stop = True

        if self.max_iterations:
            if iteration >= self.max_iterations:
                stop = True
        return stop
