from composer.models import ComposerModel
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


class GeneticDistanceModel(ComposerModel):
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        logger: logging.Logger,
        feat_dim: int = 64,
    ):
        super().__init__()
        logger.info("-----Initializing GeneticDistanceModel-----")
        self.custom_logger = logger
        self.tokenizer = tokenizer
        self.dnabert = model

        self.emb_size = 768
        self.feat_dim = feat_dim

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False),
        )

    def forward(self, batch: dict) -> torch.tensor:
        """
        Usual forward method of a model

        Args:
            batch: The current batch to be processed

        Returns:
            Features extracted from the model
        """
        out = self.dnabert.forward(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        att = batch["attention_mask"].unsqueeze(-1)

        # get mean embeddings
        mean_output = torch.sum(out[0] * att, dim=1) / torch.clamp(
            torch.sum(att, dim=1), min=1e-9
        )

        # put through linear layers
        feat = F.normalize(self.contrast_head(mean_output), dim=1)

        return feat
