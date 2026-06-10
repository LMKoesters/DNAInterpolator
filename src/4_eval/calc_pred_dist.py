import numpy as np
import os
import pandas as pd
from pandarallel import pandarallel
import torch
import torch.nn.functional as F
from tqdm import tqdm


class DistWrapper:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def calc_dists(self):
        """
        Gets embeddings and paired samples for distane calculation and runs the calculation
        """
        pandarallel.initialize(progress_bar=True, nb_workers=1)

        run_name = self.cfg.run_name

        for loo_genus in self.cfg.genera_to_test:
            # change run name throughout config
            self.cfg.run_name = f"{run_name}_{loo_genus}"

            print(
                f"{self.cfg.interpolation} | Calculating distances -{loo_genus} model"
            )
            if os.path.isfile(
                f"{self.cfg.eval_results_path}/dist_pred_{self.cfg.run_name}_{loo_genus}.csv"
            ):
                self.logger.info(
                    f"{self.cfg.interpolation} | {loo_genus} | Already evaluated. Skipping testing -{loo_genus} model"
                )
                continue

            self.logger.info(
                f"{self.cfg.interpolation} | {loo_genus} | Calculating distances -{loo_genus} model"
            )
            # get comparisons
            these_test_recs = pd.read_csv(
                self.cfg.data_local_dist.replace(".csv", f"_{loo_genus}.csv"),
                header=0,
                engine="pyarrow",
                dtype={"gene_id": str},
            ).drop(columns=["seq_anchor", "seq_complement"])

            # get embeddings
            embeds = pd.read_csv(
                f"{self.cfg.eval_results_path}/ftrs_{self.cfg.run_name}.csv",
                header=0,
                engine="pyarrow",
                dtype={"gene_id": str},
            )
            embeds = embeds[embeds["taxon"] == loo_genus]

            # get individual_id as 2-col dataframe and the feature columns as tensor
            embeddings = {}
            embed_cols = [c for c in embeds.columns if c.startswith("ftr_")]
            for grouped_ftr, group in embeds.groupby(["gene_id"]):
                mapping = {
                    individual_id: i
                    for i, individual_id in enumerate(group["individual_id"].values)
                }
                embeddings[grouped_ftr[0]] = [
                    mapping,
                    torch.tensor(group[embed_cols].values, dtype=torch.float32),
                ]

            # group based on gene_id
            tqdm.pandas()
            these_test_recs = these_test_recs.groupby(
                ["gene_id"], group_keys=False
            ).progress_apply(
                lambda group_recs: self.calc_pred_dist(
                    group_recs, embeddings[group_recs["gene_id"].iloc[0]]
                )
            )

            these_test_recs[["loo_genus"]] = [loo_genus]
            these_test_recs.to_csv(
                f"{self.cfg.eval_results_path}/dist_pred_{self.cfg.run_name}_{loo_genus}.csv",
                header=True,
                index=False,
            )

    def balance_dists(self, dist_recs: pd.DataFrame) -> pd.DataFrame:
        """
        Adds column for balancing of test data based on 30 distance sectors

        Args:
            dist_recs: Dataframe with paired samples and the corresponding cosine distance

        Returns:
            Dataframe with added column for balancing
        """
        distance_sectors = np.linspace(0, 2, num=31)
        for dist_sector in range(30):
            dist_recs.loc[
                (dist_recs["cosine_distance"] > distance_sectors[dist_sector])
                & (dist_recs["cosine_distance"] <= distance_sectors[dist_sector + 1]),
                "sector",
            ] = dist_sector

        random_sample_size = 200
        dist_recs_balanced = dist_recs.sample(frac=1)
        dist_recs_balanced = dist_recs_balanced.groupby("sector").head(
            random_sample_size
        )
        dist_recs["balanced"] = False
        dist_recs.loc[dist_recs_balanced.index, "balanced"] = True
        return dist_recs

    def calc_pred_dist(
        self, recs: pd.DataFrame, embeddings: list[dict[str, int], torch.Tensor]
    ) -> pd.DataFrame:
        """
        Calculates distances based on generated embeddings

        Args:
            recs: Subsetted dataframe with samples of a specific locus
            embeddings: Generated embeddings of samples

        Returns:
            DataFrame with added predicted cosine distance
        """
        # anchor
        anchor_ids = recs["anchor_id"].values
        order = [embeddings[0][id_] for id_ in anchor_ids]
        anchor_embed = torch.stack([embeddings[1][i] for i in order])

        # complement
        complement_ids = recs["complement_id"].values
        order = np.array([embeddings[0][id_] for id_ in complement_ids])
        compl_embed = torch.stack([embeddings[1][i] for i in order])

        num_samples = len(recs.index)
        if anchor_embed.shape[0] != num_samples or compl_embed.shape[0] != num_samples:
            raise Exception(f"{recs['genus'].iloc[0]}-{recs['gene_id'].iloc[0]}")

        anchor_embed.to("cuda")
        compl_embed.to("cuda")

        batch_cos_dist = F.cosine_similarity(anchor_embed, compl_embed).numpy()
        batch_cos_dist = 1 - batch_cos_dist

        recs["cos_dist"] = batch_cos_dist

        return recs
