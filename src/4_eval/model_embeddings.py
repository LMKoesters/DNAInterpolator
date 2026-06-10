from composer import Trainer
from composer.models import ComposerModel
from composer.utils import dist, reproducibility
from glob import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer
import torch
from tqdm import tqdm

from train_modules import TrainCommons, update_batch_size_info, cleanup_cuda


def get_sign_level(p_value: float) -> str:
    """
    Simple helper method that turns float values into significance strings

    Args:
        p_value: The p-value to be turned into a significance string

    Returns:
        Significance string corresponding to input p-value
    """
    if p_value < 0.0001:
        return "****"
    elif p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return "ns"


class EmbeddingWrapper:
    """
    Wrapper class for generating embeddings for test samples
    """

    def __init__(self, cfg, logger):
        reproducibility.seed_all(cfg.seed)
        cfg = update_batch_size_info(cfg)

        self.cfg = cfg
        self.logger = logger

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name,
            model_max_length=cfg.max_seq_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        self.train_commons = TrainCommons(cfg, self.tokenizer, logger)
        self.test_data = pd.read_csv(
            f"{self.cfg.test_local}/test_samples.csv",
            header=0,
            engine="pyarrow",
            dtype={"gene_id": str},
        )

    def setup_trainer(self, loo_genus: str) -> tuple[ComposerModel, Trainer]:
        """
        Sets up the Trainer object that will be used for device retrieval

        Args:
            loo_genus: Defines the LOO model

        Returns:
            A tuple containing the base model and the Trainer object
        """
        base_model = self.train_commons.get_model()

        load_path = (
            f"{self.cfg.callbacks.checkpoint_saver_contrastive.save_path}/best_dist.pt"
        )
        self.logger.info(f"{loo_genus}, {load_path}")
        base_model = self.train_commons.load_model_weights(base_model, load_path)

        trainer = Trainer(
            run_name=self.cfg.run_name,
            seed=self.cfg.seed,
            model=base_model,
            train_dataloader=[],
            device_train_microbatch_size=20,
            log_to_console=self.cfg.log_to_console,
            device=self.cfg.get("device", None),
        )

        return (base_model, trainer)

    def eval_test(self) -> None:
        """
        Main method for evaluation of test set. Iterates over
        """
        run_name = self.cfg.run_name

        for loo_genus in self.cfg.loo_runs:
            # change run name throughout config
            self.cfg.run_name = f"{run_name}_test_{loo_genus}"
            self.cfg.results_name = f"{run_name}_{loo_genus}"

            if (
                os.path.isdir(self.cfg.eval_results_path)
                and len(
                    glob(f"{self.cfg.eval_results_path}/ftrs_{run_name}_{loo_genus}.*")
                )
                > 0
            ):
                self.logger.info(
                    f"Skipping testing -{loo_genus} model. Already evaluated."
                )
                continue

            Path(self.cfg.eval_results_path).mkdir(parents=True, exist_ok=True)

            try:
                model, trainer = self.setup_trainer(loo_genus)
            except (FileNotFoundError, IndexError):
                self.logger.info(
                    f"Skipping testing -{loo_genus} model. Model does not exist. "
                )
                continue

            self.logger.info(f"Testing -{loo_genus} model")
            test_recs = self.test_data[self.test_data["genus"] == loo_genus]
            self.get_embeddings(
                model,
                trainer,
                test_recs,
                self.cfg.eval_results_path,
                f"{run_name}_{loo_genus}",
                genus_to_test=loo_genus,
            )

    def get_embeddings(
        self,
        model: ComposerModel,
        trainer: Trainer,
        test_recs: pd.DataFrame,
        out_d: str,
        f_addon: str,
        genus_to_test: str,
    ) -> None:
        """
        Extracts embeddings from trained model a given genus_to_test.

        Args:
            model: The trained model
            trainer: A Trainer object for device extraction
            test_recs: A dataframe with the test samples to evaluate
            out_d: The directory where the embeddings should be stored
            f_addon: A string describing this evaluation that will be added to the output file name
            genus_to_test: The name of the genus to be evaluated
        """
        device = trainer.state.device._device
        model.to(device)
        model.eval()

        test_dataloader = self.train_commons.get_dataloader(genus_to_test)
        feature_df = []
        test_recs = test_recs[["genus", "gene_id", "individual_id"]]

        cleanup_cuda()

        with torch.no_grad():
            self.logger.info(
                f"Rank {dist.get_global_rank()} gets {len(test_dataloader)} batches"
            )

            # Pre-allocate dicts outside the loop
            local_ftrs = {k: [] for k in range(dist.get_world_size())}
            for batch in tqdm(test_dataloader):
                # move to GPU
                batch = {
                    k: v.to(device, non_blocking=True) if k != "sample_idx" else v
                    for k, v in batch.items()
                }

                # get embeddings
                feat = model(batch)

                # move tensors to CPU once
                sample_indices = batch["sample_idx"].cpu().numpy()
                feat = feat.cpu().numpy()

                # Pandas slice for all relevant rows
                samples = test_recs.loc[sample_indices,].reset_index(drop=True)

                # features with genus, gene_id, anchor_id + embedding
                meta = samples[["genus", "gene_id", "individual_id"]].values
                local_ftrs[dist.get_global_rank()].extend(
                    np.concatenate([meta, feat], axis=1).tolist()
                )

            gathered_ftrs = [None] * dist.get_world_size()
            torch.distributed.all_gather_object(
                gathered_ftrs, local_ftrs[dist.get_global_rank()]
            )

            # store gathered results
            if dist.get_global_rank() == 0:
                for sublist in gathered_ftrs:
                    feature_df.extend(sublist)

        # write results
        if dist.get_global_rank() == 0:
            self.logger.info("Writing results...")
            self.write_ftrs(feature_df, out_d, f_addon)

    def write_ftrs(self, feature_df: pd.DataFrame, out_d: str, f_addon: str) -> None:
        """
        A helper method that writes calculcated embeddings to a csv file.

        Args:
            feature_df: A dataframe with the extracted features
            out_d: The directory where the features should
            f_addon: A string describing this evaluation that will be added to the output file name
        """
        self.logger.info("Writing features...")

        # columns (=feature dimension)
        cols = ["taxon", "gene_id", "individual_id"]
        cols_ftrs = [f"ftr_{i}" for i in range(len(feature_df[0]) - len(cols))]
        cols.extend(cols_ftrs)

        ftrs = pd.DataFrame.from_records(feature_df, columns=cols)
        ftrs.to_csv(f"{out_d}/ftrs_{f_addon}.csv", header=True, index=False)
