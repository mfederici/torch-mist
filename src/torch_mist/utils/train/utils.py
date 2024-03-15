import os.path
import tempfile
from copy import deepcopy
from typing import Optional
import time

from torch import nn
import torch


class RunTerminationManager:
    def __init__(
        self,
        early_stopping: bool,
        tolerance: float,
        patience: int,
        warmup_iterations: Optional[int],
        max_iterations: Optional[int],
        verbose: bool = False,
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
        self.max_iterations = max_iterations
        self.warmup_iterations = warmup_iterations
        self.verbose = verbose
        self.best_state_dict = None
        self.__best_model_path = None

    def save_weights(self, model: nn.Module, iteration: int):
        self.delete_best_weights()
        self.__best_model_path = os.path.join(
            tempfile.gettempdir(), f"model_{iteration}_{time.time}.pyt"
        )
        torch.save(model.state_dict(), self.__best_model_path)

    def load_best_weights(self, model: nn.Module):
        if not (self.__best_model_path is None):
            if self.verbose:
                iteration = self.__best_model_path.split("_")[1].split(".")[0]
                print(f"Loading the weights saved at iteration {iteration}")
            model.load_state_dict(torch.load(self.__best_model_path))
        else:
            if self.verbose:
                print(f"Using the weights from the last iteration")

    def delete_best_weights(self):
        if not (self.__best_model_path is None):
            os.remove(self.__best_model_path)

    def should_stop(
        self, iteration: int, score: Optional[float], model: nn.Module
    ) -> bool:
        stop = False

        if iteration <= self.warmup_iterations:
            return False

        if self.early_stopping:
            improvement = (
                (score - self.best_value)
                if self.maximize
                else (self.best_value - score)
            )
            if improvement > 0:
                # Improvement, update best and reset the patience
                self.best_value = score
                self.current_patience = self.patience

                # Write the best state_dict to disk
                self.save_weights(model, iteration)

                self.best_state_dict = deepcopy(model.state_dict())
            elif -improvement < self.tolerance:
                pass
            else:
                self.current_patience -= 1

            if self.current_patience <= 0:
                stop = True

        if self.max_iterations:
            if iteration >= self.max_iterations:
                stop = True
        return stop

    def __del__(self):
        self.delete_best_weights()
