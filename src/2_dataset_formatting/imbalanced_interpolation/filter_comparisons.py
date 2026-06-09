from functools import partial
import logging
import multiprocessing as mp
import os
import pandas as pd
from pathlib import Path
import re
from tqdm import tqdm
from typing import Optional


logger = logging.getLogger("imbalanced_interpolation_ds")


def filter_seqs(
    recs: pd.DataFrame,
    min_len: Optional[float] = 0.5,
    max_len: Optional[float] = None,
    min_overlap: float = 0.95,
) -> Optional[pd.DataFrame]:
    """
    Applies more advanced filtering to pairs of sequences

    Args:
        recs: Dataframe with pairs of DNA samples and additional information on sequences
        min_len: Minimum sequence length for filtering
        max_len: Maximum sequence length for filtering
        min_overlap: Minimum overlap between paired DNA sequences

    Returns:
        Filtered dataframe with pairs of DNA samples and additional information on sequences
    """
    recs = filter_length(recs, min_len, max_len)
    if recs.empty:
        return
    recs = filter_overlap(recs, min_overlap)
    return recs if not recs.empty else None


def filter_length(
    recs: pd.DataFrame, min_len: Optional[float] = 0.5, max_len: Optional[float] = None
) -> pd.DataFrame:
    """
    Filters sequence by length

    Args:
        recs: Dataframe with pairs of DNA samples and additional information on sequences
        min_len: Minimum sequence length for filtering
        max_len: Maximum sequence length for filtering

    Returns:
        Filtered dataframe with pairs of DNA samples and additional information on sequences
    """
    recs.loc[:, "seq_len_anchor"] = (
        recs["seq_anchor"].str.replace(r"-|\?|N|n", "", regex=True).str.len()
    )
    recs.loc[:, "seq_len_complement"] = (
        recs["seq_complement"].str.replace(r"-|\?|N|n", "", regex=True).str.len()
    )

    unique_recs = pd.concat(
        [
            recs[["anchor_id", "seq_len_anchor"]].rename(
                columns={"anchor_id": "individual_id", "seq_len_anchor": "seq_len"}
            ),
            recs[["complement_id", "seq_len_complement"]].rename(
                columns={
                    "complement_id": "individual_id",
                    "seq_len_complement": "seq_len",
                }
            ),
        ]
    ).drop_duplicates(subset=["individual_id"])
    median_seq_len = unique_recs["seq_len"].median()

    if min_len:
        min_len = median_seq_len * min_len
        recs = recs.loc[
            (recs["seq_len_anchor"] >= min_len)
            & (recs["seq_len_complement"] >= min_len),
            :,
        ].copy()
    if max_len:
        max_len = median_seq_len * max_len
        recs = recs.loc[
            (recs["seq_len_anchor"] <= max_len)
            & (recs["seq_len_complement"] <= max_len),
            :,
        ].copy()

    recs.drop(columns=["seq_len_anchor", "seq_len_complement"], inplace=True)
    return recs


def check_overlap(row: pd.Series, min_overlap: float) -> bool:
    """
    Checks whether overlap between DNA sequences is at least min_overlap

    Args:
        row: A dataframe row with a pair of DNA sequences
        min_overlap: Minimum overlap between paired DNA sequences

    Returns:
        A boolean determining whether the overlap between the pair of DNA sequences was at least min_overlap
    """
    seq1 = re.sub(r"\?|N", "-", row["seq_anchor"].upper())
    seq2 = re.sub(r"\?|N", "-", row["seq_complement"].upper())

    # determine sequence boundaries
    pairs = zip(seq1, seq2)
    starting = 0
    started = False

    ending = float("inf")

    for i, pair in enumerate(pairs):
        if (pair[0] == "-" and pair[1] == "-") and not started:
            starting += 1
        else:
            started = True
            ending = i

    relevant_pairs = list(pairs)[starting:ending]
    matches = sum(
        [1 if pair[0] != "-" and pair[1] != "-" else 0 for pair in relevant_pairs]
    )
    if matches >= len(relevant_pairs) * min_overlap:
        return True
    else:
        return False


def filter_overlap(recs: pd.DataFrame, min_overlap: float = 0.95) -> pd.DataFrame:
    """
    Filters paired DNA samples based on whether they overlap by at least min_overlap

    Args:
        recs: Dataframe with pairs of DNA samples and additional information on sequences
        min_overlap: Minimum overlap between paired DNA sequences

    Returns:
        Filtered dataframe with pairs of DNA samples and additional information on sequences
    """
    recs["passed_test"] = recs.apply(
        lambda row: check_overlap(row, min_overlap), axis=1
    )
    recs = recs[recs["passed_test"]].copy()
    recs.drop(columns=["passed_test"], inplace=True)
    return recs


def filter_comparisons(
    genus: str,
    recs: pd.DataFrame,
    data_dir: str,
    min_len: float = 0.5,
    max_len: Optional[float] = None,
    min_overlap: float = 0.95,
) -> pd.DataFrame:
    """
    Filters paired DNA samples within recs by sequence length and overlap between paired sequences

    Args:
        genus: The genus samples from recs belong to
        recs: A dataframe with paired DNA samples and additional information
        data_dir: The parent directory where training/validation data for all genera can be found
            (data_dir > genus > val.csv)
        min_len: Minimum sequence length for filtering
        max_len: Maximum sequence length for filtering
        min_overlap: Minimum overlap between paired DNA sequences

    Returns:
        Filtered dataframe with pairs of DNA samples and additional information on sequences
    """
    if os.path.isfile(f"{data_dir}/{genus}/train.csv"):
        recs = pd.read_csv(
            f"{data_dir}/{genus}/train.csv",
            header=0,
            engine="pyarrow",
            dtype={"gene_id": str},
        )
        return recs

    ############
    # filter
    logger.info(f"Filtering {genus} sequences...")

    initial_recs = len(recs.index)
    func = partial(
        filter_seqs, min_len=min_len, max_len=max_len, min_overlap=min_overlap
    )
    groups = (g for _, g in recs.groupby(["gene_id"], group_keys=False))
    new_recs = []
    with mp.Pool(processes=32) as pool:
        for result in tqdm(
            pool.imap_unordered(func, groups, chunksize=73), total=len(recs["gene_id"].unique())
        ):
            if result is not None:
                new_recs.append(result)
            else:
                pass
    recs = pd.concat(new_recs)
    logger.info(
        f"From {genus} removed {initial_recs - len(recs.index)} entries. {len(recs.index)} entries remaining..."
    )

    ############
    # stats
    logger.info(f"Max distance {genus}: {recs['distance'].max()}")
    logger.info(f"Min distance {genus}: {recs['distance'].min()}")

    ############
    logger.info("Adding cosine distance...")
    with open(f"{Path(data_dir).parent}/no_interpolation/cosine_dist_max.txt") as f:
        quantile = float(f.read())
    logger.info(f"Quantil: {quantile}")
    recs["cosine_distance"] = (recs["distance"] / quantile) * 2
    recs["cosine_similarity"] = 1 - recs["cosine_distance"]
    recs = recs[recs["cosine_distance"] <= 2].copy()

    ############
    max_records = 0
    for g in ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]:
        num_recs = len(
            pd.read_csv(
                f"{Path(data_dir).parent}/no_interpolation/{g}/train.csv",
                header=0,
                engine="pyarrow",
            ).index
        )
        if num_recs > max_records:
            max_records = num_recs

    logger.info(f"Randomly choosing {max_records} records from {genus}...")
    recs_original = recs[~recs["aug"]]
    recs_aug = recs[recs["aug"]]
    max_recs = max_records - len(recs_original.index)
    if max_records > 0:
        recs_aug = recs_aug.sample(n=max_recs, random_state=42)
        recs = pd.concat([recs_original, recs_aug])
    else:
        recs = recs_original
    del recs_original
    del recs_aug
    recs.loc[:, "seq_anchor"] = (
        recs["seq_anchor"].str.replace(r"-|\?|N|n", "", regex=True).str.upper()
    )
    recs.loc[:, "seq_complement"] = (
        recs["seq_complement"].str.replace(r"-|\?|N|n", "", regex=True).str.upper()
    )
    recs.to_csv(f"{data_dir}/{genus}/train.csv", header=True, index=False)
    return recs
