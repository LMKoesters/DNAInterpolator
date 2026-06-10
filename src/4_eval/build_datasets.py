from functools import partial
import composer.utils.dist as comp_dist
from omegaconf import DictConfig
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer, BatchEncoding
from typing import Dict


def build_tokenizer(cfg: DictConfig) -> PreTrainedTokenizer:
    """
    Helper method that creates a tokenizer object

    Args:
        cfg: Configuration from user-defined YAML

    Returns:
        A pretrained tokenizer
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.tokenizer_name,
        model_max_length=cfg.get("max_seq_len"),
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )

    return tokenizer


class GeneDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        ds_cfg: DictConfig,
        genus_to_test: str,
        logger=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.custom_logger = logger

        self.max_seq_len = ds_cfg.dataset.get("max_seq_len")
        self.samples = self._collect_samples(
            f"{ds_cfg.dataset.get('local')}/test_samples.csv", genus_to_test
        )

    def _collect_samples(self, ds_path: str, genus_to_test: str) -> pd.DataFrame:
        """
        Mini method that reads samples from a given file path

        Args:
            ds_path: Path to file with dataset
            genus_to_test: The genus the samples should belong to

        Returns:
            A dataframe containing samples belonging to genus_to_test
        """
        samples = pd.read_csv(
            ds_path, header=0, engine="pyarrow", dtype={"gene_id": str}
        )
        samples = samples[samples["genus"] == genus_to_test]
        return samples

    def __len__(self) -> int:
        """
        Common dataset method that returns the length of the dataset/the number of samples

        Returns:
            Length of dataset
        """
        return len(self.samples.index)

    def __getitem__(self, i) -> Dict:
        """
        Common dataset method that returns an item of the dataset by index

        Args:
            i: The index of the item to be retrieved

        Returns:
            A dataset item defined by a DNA sequence and its name (i.e., the index within the original dataframe)
        """
        sample = self.samples.iloc[i]
        return {"seq": sample["seq"], "sample_idx": sample.name}


def tokenize_text(
    text_samples: list, tokenizer: PreTrainedTokenizer, max_seq_len: int
) -> BatchEncoding:
    """
    Tokenizes batches of text

    Args:
        text_samples: List of DNA sequences
        tokenizer: A pretrained tokenizer
        max_seq_len: The maximum sequence length within the dataset

    Returns:
        Encoded/tokenized DNA sequences
    """
    token_samples = tokenizer(
        text_samples,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
        max_length=max_seq_len,
    )

    return token_samples


def collate_pairs_test(
    batch: list, tokenizer: PreTrainedTokenizer, max_seq_len: int
) -> dict:
    """
    Formats model input

    Args:
        batch: A list of samples
        tokenizer: A pretrained tokenizer
        max_seq_len: The maximum sequence length within the dataset

    Returns:
        A dictionary with input ids, attention masks and sample indices
    """
    tokenized_anchors = tokenize_text(
        [sample["seq"] for sample in batch], tokenizer, max_seq_len
    )

    # sample indices
    sample_idx = torch.tensor(
        [sample["sample_idx"] for sample in batch], dtype=torch.float
    )

    return {
        "input_ids": tokenized_anchors["input_ids"],
        "attention_mask": tokenized_anchors["attention_mask"],
        "sample_idx": sample_idx.detach(),
    }


def build_dataset(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    genus_to_test: str,
    logger=None,
) -> Dataset:
    """
    Builds a dataset using the configuration provided within a user-defined YAML and based on the current genus of interest

    Args:
        cfg: The configuration provided within a user-defined YAML
        tokenizer: A pretrained tokenizer
        genus_to_test: The current genus of interest that should be evaluated
        logger: A logger object

    Returns:
        A dataset for training
    """
    # build dataset
    ds_cfg = cfg.test_loader
    dataset = GeneDataset(
        tokenizer,
        ds_cfg,
        cfg.model,
        genus_to_test=genus_to_test,
        logger=logger,
    )

    return dataset


def build_dataloader(
    dataset: Dataset,
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizer,
    device_batch_size: int,
) -> DataLoader:
    """
    Builds a dataloader for training given a user-defined configuration, a dataset, a pretrained tokenizer and the preferred batch size

    Args:
        dataset: A dataset object for training
        cfg: A user-defined configuration
        tokenizer: A pretrained tokenizer
        device_batch_size: The preferred batch size

    Returns:
        A dataloader
    """
    ds_cfg = cfg.test_loader

    collate_fn = partial(
        collate_pairs_test,
        tokenizer=tokenizer,
        max_seq_len=ds_cfg.dataset.get("max_seq_len"),
    )
    dataloader = DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=device_batch_size,
        num_workers=ds_cfg.num_workers,
        pin_memory=cfg.get("pin_memory", True),
        prefetch_factor=cfg.get("prefetch_factor", 2),
        persistent_workers=cfg.get("persistent_workers", True),
        timeout=cfg.get("timeout", 0),
        sampler=comp_dist.get_sampler(
            dataset=dataset,
            drop_last=ds_cfg.get("drop_last", True),
            shuffle=ds_cfg.dataset.get("shuffle", False),
        ),  # needed for multi-GPU training
    )
    return dataloader
