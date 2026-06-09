from Bio import AlignIO
from glob import glob
from functools import partial
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import traceback
from tqdm import tqdm


logger = logging.getLogger("imbalanced_interpolation_ds")


def read_genus_alignments(raw_files_dir: str, genus: str) -> pd.DataFrame:
    """
    Reads samples from alignment files and stores them inside a dataframe

    Args:
        raw_files_dir: Directory where raw alignment files are stored
        genus: The current genus to be processed

    Returns:
        A dataframe with samples read from the alignment files
    """
    recs = []

    reduced_alns = [
        Path(aln_f).stem.split(".")[0]
        for aln_f in glob(f"{raw_files_dir}/alignments/*")
        if ".reduced" in aln_f
    ]

    logger.info(
        f"Dataset size info: {genus}, {len(glob(f'{raw_files_dir}/alignments/*.fas*'))}"
    )
    for _, aln_f in enumerate(glob(f"{raw_files_dir}/alignments/*")):
        if ".reduced" not in aln_f:
            gene_id = Path(aln_f).stem
            if gene_id in reduced_alns:
                continue
            alignment_format = "fasta"
        else:
            gene_id = Path(aln_f).stem.split(".")[0]
            alignment_format = "phylip-relaxed"

        try:
            msa = AlignIO.read(aln_f, alignment_format)
            if len(msa) <= 1:
                continue
            for record in msa:
                seq = str(record.seq)
                individual_id = record.description

                recs.append([genus, gene_id, individual_id, seq])
        except ValueError:
            print(f"No alignment found for {aln_f}")
            print(traceback.format_exc())
            logger.info(f"No alignment found for {aln_f}")

    # create dataframe with sample info - no combinations & no distances
    recs = pd.DataFrame.from_records(
        recs, columns=["genus", "gene_id", "individual_id", "seq"]
    )
    return recs


def read_distances(alignment_dir: str, genus: str, gene_id: str) -> pd.DataFrame:
    """
    Reads distance file of a given alignment

    Args:
        alignment_dir: Parent directory of subset (alignment_dir > subset > RAxML > distance files)
        genus: The current genus to process
        gene_id: The ID of the gene to process

    Returns:
        A dataframe with combined samples and distances
    """
    distance_f = f"{alignment_dir}/RAxML/RAxML_distances.{gene_id}"
    distances = pd.read_csv(distance_f, header=None, sep="\t")

    # split first column (formatting is weird with RAxML)
    distances[[0, "B"]] = distances[0].str.split(" ", expand=True)[[0, 1]]
    distances.columns = ["A", "val", "B"]

    if genus == "Pterocarpus":
        # renaming Pterocarpus after distance merging so we get our number of individuals straight :)
        distances["A"] = distances["A"].str.replace(r"^_R_", "", regex=True)
        distances["B"] = distances["B"].str.replace(r"^_R_", "", regex=True)

    return distances


def merge_raxml(sub_recs: pd.DataFrame, alignment_dir: str, genus: str) -> pd.DataFrame:
    """
    Merges distances read from RAxML file with combined sample information

    Args:
        sub_recs: A subset of the sample dataframe containing samples from one locus
        alignment_dir: Parent directory of subset (alignment_dir > subset > RAxML > distance files)
        genus: The current genus to process

    Returns:
        A dataframe with combined sequences information and the corresponding genetic distances
    """
    # read distances from RAxML output
    gene_id = sub_recs["gene_id"].values[0]

    try:
        distances = read_distances(alignment_dir, genus, gene_id)
    except (FileNotFoundError, pd.errors.EmptyDataError, IndexError):
        logger.info(f"RAxML file error with {genus} {gene_id}")
        return sub_recs

    distances = distances[
        (distances["A"].isin(sub_recs["individual_id"]))
        & (distances["B"].isin(sub_recs["individual_id"]))
    ].copy()
    distances["aug"] = (distances["A"].str.contains("_aug")) | (
        distances["B"].str.contains("_aug")
    )
    distances_original = distances[~distances["aug"]]
    distances_aug = distances[distances["aug"]]
    max_recs = 40_000 - len(distances_original.index)
    distances_aug = distances_aug.sample(
        n=min(max_recs, len(distances_aug.index)), random_state=42
    )
    distances = pd.concat([distances_original, distances_aug])
    del distances_original
    del distances_aug

    # merge anchor_id
    pairs = pd.merge(
        distances, sub_recs, how="inner", left_on="A", right_on="individual_id"
    )
    pairs = pairs.drop(columns=["A"])
    sub_recs = sub_recs.drop(columns=["gene_id", "genus"])
    # merge complement_id
    pairs = pd.merge(
        pairs,
        sub_recs,
        how="inner",
        left_on="B",
        right_on="individual_id",
        suffixes=("_anchor", "_complement"),
    )
    # pruning
    pairs = pairs.drop(columns=["B"])
    pairs.rename(
        columns={
            "val": "distance",
            "individual_id_anchor": "anchor_id",
            "individual_id_complement": "complement_id",
        },
        inplace=True,
    )
    pairs = filter_basics(pairs)
    return pairs


def filter_basics(recs: pd.DataFrame) -> pd.DataFrame:
    """
    Applies filtering to the combinations of sequences, specifically: removal of duplicate combinations and same-sequence combinations

    Args:
        recs: Dataframe with samples to be processed

    Returns:
        A filtered dataframe
    """

    """
    same-sequence combinations (after gap removal) -- 
    nothing to be done for model
    """
    recs["seq_anchor_short"] = recs["seq_anchor"].str.replace(
        r"-|\?|N|n", "", regex=True
    )
    recs["seq_complement_short"] = recs["seq_complement"].str.replace(
        r"-|\?|N|n", "", regex=True
    )
    recs = recs[recs["seq_anchor_short"] != recs["seq_complement_short"]].copy()
    recs.drop(columns=["seq_anchor_short", "seq_complement_short"], inplace=True)

    """
    duplicate combinations (before gap removal) -- 
    we don't want to unknowingly train or eval some combinations more often than others
    """
    recs["combined_seq"] = list(
        map("".join, np.sort(recs[["seq_anchor", "seq_complement"]]))
    )
    recs = recs.drop_duplicates(subset=["combined_seq"])
    recs.drop(columns=["combined_seq"], inplace=True)
    return recs


def create_genus_df(genus: str, data_dir: str, raw_files_dir: str) -> pd.DataFrame:
    """
    Creates a dataframe with combined sequences with genetic distances and sequence information (locus, individual ID, etc.)

    Args:
        genus: The current genus to process
        data_dir: Parent folder where data is stored
        raw_files_dir: Directory where raw alignment files are stored

    Returns:
        A dataframe with combined sequences, genetic distances and identifying information
    """
    if os.path.isfile(f"{data_dir}/{genus}/{genus}_unfiltered_raw_distances.csv"):
        return pd.read_csv(
            f"{data_dir}/{genus}/{genus}_unfiltered_raw_distances.csv",
            header=0,
            engine="pyarrow",
            dtype={"gene_id": str},
        )

    logger.info(f"Reading {genus} sequences...")
    genus_recs = read_genus_alignments(raw_files_dir, genus)

    logger.info(f"Merging {genus} distances...")
    func = partial(merge_raxml, alignment_dir=raw_files_dir, genus=genus)
    groups = (g for _, g in genus_recs.groupby(["gene_id"], group_keys=False))
    
    with mp.Pool(processes=4) as pool:
        genus_recs = list(
            tqdm(pool.imap_unordered(func, groups, chunksize=10), total=len(genus_recs["gene_id"].unique()))
        )
    genus_recs = pd.concat(genus_recs)

    Path(f"{data_dir}/{genus}").mkdir(parents=True, exist_ok=True)
    logger.info(
        f"{genus}_unfiltered_raw_distances.csv has {len(genus_recs.index)} samples..."
    )

    logger.info(f"Randomly choosing 1.5M records from {genus}")
    genus_recs_original = genus_recs[~genus_recs["aug"]]
    genus_recs_aug = genus_recs[genus_recs["aug"]]
    max_recs = 1_500_000 - len(genus_recs_original.index)
    genus_recs_aug = genus_recs_aug.sample(
        n=min(max_recs, len(genus_recs_aug.index)), random_state=42
    )
    genus_recs = pd.concat([genus_recs_original, genus_recs_aug])
    del genus_recs_original
    del genus_recs_aug
    genus_recs.to_csv(
        f"{data_dir}/{genus}/{genus}_unfiltered_raw_distances.csv",
        header=True,
        index=False,
    )
    return genus_recs
