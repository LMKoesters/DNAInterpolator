from composer.models import ComposerModel
import logging
from torchmetrics import Metric
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

from metrics_and_callbacks import BiologicalDistance


class GeneticDistanceModel(ComposerModel):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        tokenizer: PreTrainedTokenizer,
        logger: logging.Logger,
        feat_dim: int = 128,
    ):
        super().__init__()
        logger.info("-----Initializing GeneticDistanceModel-----")
        self.custom_logger = logger
        self.tokenizer = tokenizer
        self.dnabert = model
        self.loss_fn = loss_fn

        self.emb_size = 768
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False),
        )

    def forward(self, batch: dict):
        """
        Usual forward method of a model

        Args:
            batch: The current batch to be processed

        Returns:
            Features extracted from the model
        """

        out_anchor = self.dnabert.forward(
            input_ids=batch["anchor"], attention_mask=batch["anchor_att"]
        )
        out_compl = self.dnabert.forward(
            input_ids=batch["complement"], attention_mask=batch["compl_att"]
        )

        att_anchor = batch["anchor_att"].unsqueeze(-1)
        att_compl = batch["compl_att"].unsqueeze(-1)

        # get mean embeddings
        mean_output_anchor = torch.sum(out_anchor[0] * att_anchor, dim=1) / torch.clamp(
            torch.sum(att_anchor, dim=1), min=1e-9
        )
        mean_output_complement = torch.sum(
            out_compl[0] * att_compl, dim=1
        ) / torch.clamp(torch.sum(att_compl, dim=1), min=1e-9)

        del out_anchor
        del out_compl
        del att_anchor
        del att_compl

        # put through linear layers
        feat_anchor = F.normalize(self.contrast_head(mean_output_anchor), dim=1)
        feat_complement = F.normalize(self.contrast_head(mean_output_complement), dim=1)

        return feat_anchor, feat_complement

    def loss(self, outputs: torch.Tensor, batch: dict) -> torch.Tensor:
        """
        Computes loss per batch

        Args:
            outputs: Embeddings of batch
            batch: Samples of batch

        Returns:
            Loss
        """
        anchors, complements = outputs
        distances = batch["distance"]

        loss_fn = self.loss_fn
        loss = loss_fn(anchors, complements, distances)

        assert loss.requires_grad
        return loss

    def update_metric(self, batch: dict, outputs: torch.Tensor, metric: Metric) -> None:
        """
        Updates the metric for evaluation of model performance

        Args:
            batch: The current batch
            outputs: Embeddings of current batch
            metric: The metric for evaluation of model performance
        """
        anchors, complements = outputs
        distances = batch["distance"]

        metric.update(anchors, complements, distances)

    def get_metrics(self, is_train) -> dict:
        """
        Gets the metrics

        Returns:
            The computed metric
        """
        return {"BiologicalDistance": BiologicalDistance(self.loss_fn)}
