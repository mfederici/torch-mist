from typing import Optional


class RunTerminationManager:
    def __init__(
        self,
        early_stopping: bool,
        delta: float,
        patience: int,
        verbose: bool,
        max_iterations: Optional[int],
        maximize: bool = False,
        minimize: bool = False,
    ):
        if early_stopping:
            if not maximize and not minimize:
                raise ValueError(
                    "early_stopping can be used only when maximizing or minimizing"
                )
        self.early_stopping = early_stopping
        self.maximize = maximize
        self.minimize = minimize
        self.best_value = 0
        self.delta = delta
        self.patience = patience
        self.current_patience = patience
        self.verbose = verbose
        self.max_iterations = max_iterations

    def should_stop(self, iteration: int, valid_mi: Optional[float]) -> bool:
        stop = False
        if self.early_stopping:
            improvement = (
                (valid_mi - self.best_value)
                if self.maximize
                else (self.best_value - valid_mi)
            )
            if improvement >= self.delta:
                # Improvement, update best and reset the patience
                self.best_value = valid_mi
                self.current_patience = self.patience
            else:
                self.patience -= 1

            if self.patience < 0:
                if self.verbose:
                    print("No improvements on validation, stopping.")
                stop = True

        if self.max_iterations:
            if iteration >= self.max_iterations:
                stop = True
        return stop
