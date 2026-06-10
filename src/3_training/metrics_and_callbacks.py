from composer import Callback
from pathlib import Path
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Metric
from typing import Optional


class CosineDistanceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
        target: torch.Tensor,
        reduction: Optional[str] = "sum",
        return_dist=False,
    ) -> torch.Tensor | tuple:
        """
        Computed loss of distance between paired samples

        Args:
            input1: embeddings of anchor samples
            input2: embeddings of complement samples
            target: target cosine distance
            reduction: reduction to be applied to loss
            return_dist: whether to return predicted distance alongside loss

        Returns:
            either only the loss or the loss plus the predicted cosine distance
        """
        cos_sim = F.cosine_similarity(input1, input2)
        cos_dist = 1 - cos_sim

        elementwise_loss = F.smooth_l1_loss(cos_dist, target, reduction="none")

        if reduction == "mean":
            loss = elementwise_loss.mean()
        elif reduction == "sum":
            loss = elementwise_loss.sum()
        else:
            loss = elementwise_loss

        return (loss, cos_dist) if return_dist else loss


class BiologicalDistance(Metric):
    full_state_update = False

    def __init__(self, loss_fn: nn.Module, dist_sync_on_step: bool = False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss_function = loss_fn
        self.add_state("total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total_batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, anchors: Tensor, complements: Tensor, distances: Tensor) -> None:
        """
        Update the state with new predictions and targets.
        Loss calculated over samples/batch, accumulate loss over all batches.

        Args:
            anchors: anchor embeddings
            complements: complement embeddings
            distances: target cosine distances
        """
        self.total_loss += self.loss_function(anchors, complements, distances).detach()
        self.total_batches += 1

    def compute(self) -> float:
        """
        Aggregate state over all processes and compute the metric.
        Return average loss over entire validation dataset.

        Returns:
            Average loss across batches
        """
        assert isinstance(self.total_batches, Tensor)
        assert isinstance(self.total_loss, Tensor)
        return self.total_loss / self.total_batches


class SaveAfterEpochCallback(Callback):
    """
    Explicitly saves the model after the every epoch
    """

    def __init__(self, save_path):
        Path(save_path).mkdir(exist_ok=True, parents=True)
        self.save_path = f"{save_path}/after_epoch"

    def epoch_end(self, state, _):
        torch.save(
            state.model.state_dict(),
            f"{self.save_path}_{state.timestamp.epoch}.pt",
        )
        torch.save(
            {"state": {"model": state.model.state_dict()}},
            f"{self.save_path}_{state.timestamp.epoch}_composer.pt",
        )


class SaveBestContrastiveModelCallback(Callback):
    """
    Callback class that saves the best model so far (after validation) and
    prints the best loss achieved so far during validation.
    """

    def __init__(self, save_path):
        Path(save_path).mkdir(exist_ok=True, parents=True)

        self.best_dist = float("inf")
        self.save_path = f"{save_path}/best_dist"

    def eval_end(self, state, logger):
        step = state.timestamp.batch.value
        dist = state.eval_metrics["eval"]["BiologicalDistance"].compute()
        print(f"==============={dist} - {self.best_dist}==========")
        if dist < self.best_dist:
            self.best_dist = dist
            torch.save(state.model.state_dict(), f"{self.save_path}.pt")
            torch.save(
                {"state": {"model": state.model.state_dict()}},
                f"{self.save_path}_composer.pt",
            )
            logger.log_metrics({"best_dist": self.best_dist, "step": step})


class EmptyCacheBeforeEval(Callback):
    """
    Callback to empty cache before validation
    """

    def eval_start(self, state, _):
        torch.cuda.empty_cache()
