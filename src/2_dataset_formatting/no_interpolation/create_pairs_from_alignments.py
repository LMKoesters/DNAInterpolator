from Bio import AlignIO
from glob import glob
from functools import partial
import logging
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import traceback
from tqdm import tqdm


logger = logging.getLogger("no_interpolation_ds")


def read_genus_alignments(
    raw_files_dir: str, species_info_dir: str, genus: str
) -> pd.DataFrame:
    """
    Reads samples from alignment files and stores them inside a dataframe

    Args:
        raw_files_dir: Directory where raw alignment files are stored
        species_info_dir: Directory where files with species information are stored
        genus: The current genus to be processed

    Returns:
    A dataframe with samples read from the alignment files
    """
    recs = []

    logger.info(
        f"Dataset size info: {genus}, {len(glob(f'{raw_files_dir}/alignments/*.fas*'))}"
    )
    for aln_f in glob(f"{raw_files_dir}/alignments/*"):
        if ".reduced" in aln_f or "raxml.log" in aln_f:
            continue

        if "reduced" in aln_f and genus != "Dactylorhiza":
            logger.info(f"reduced in {genus} {aln_f}")

        gene_id = Path(aln_f).stem
        if genus in ["Dactylorhiza", "Lomatium", "Palaquium"]:
            pass
        elif genus == "Pterocarpus":
            gene_id = gene_id.replace("_supercontig", "")
        elif genus == "Ranunculus" or genus == "Xanthium":
            gene_id = gene_id.split("_")[1]
        else:
            logger.info(f"Do not know {genus}")
            raise Exception(f"Do not know {genus}")

        try:
            msa = AlignIO.read(aln_f, "fasta")
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

    # merge with species + country info
    species_info = pd.read_csv(
        f"{species_info_dir}/{genus}.csv", header=0, engine="pyarrow"
    )
    if genus == "Pterocarpus":
        # correct Pterocarpus fasta names for merging with species/country info
        recs["individual_id"] = recs["individual_id"].str.replace(
            r"^_R_", "", regex=True
        )

    recs = pd.merge(recs, species_info, how="left", on="individual_id")
    recs.loc[recs["species"].isna(), "species"] = "outgroup"
    return recs


def read_distances(
    alignment_dir: str, genus: str, gene_id: str
) -> pd.DataFrame:
    """
    Reads distance file of a given alignment

    Args:
        alignment_dir: Parent directory of RAxML folder (alignment_dir > RAxML > distance files)
        genus: The current genus to process
        gene_id: The ID of the gene to process

    Returns:
    :return: A dataframe with combined samples and dsitances
    :rtype: DataFrame
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
        sub_recs: A subset of the sample dataframe containing samples from one locus (grouped, i.e., with name and dataframe)
        alignment_dir: Parent directory of RAxML folder (alignment_dir > RAxML > distance files)
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

    try:
        # merge anchor_id
        pairs = pd.merge(
            distances, sub_recs, how="inner", left_on="A", right_on="individual_id"
        )
        # merge complement_id
        pairs = pd.merge(
            pairs,
            sub_recs.drop(columns=["country", "gene_id", "genus"]),
            how="inner",
            left_on="B",
            right_on="individual_id",
            suffixes=("_anchor", "_complement"),
        )
        # pruning
        pairs.drop(columns=["A", "B"], inplace=True)
        pairs.rename(
            columns={
                "val": "distance",
                "individual_id_anchor": "anchor_id",
                "individual_id_complement": "complement_id",
            },
            inplace=True,
        )
        return pairs
    except Exception:
        logger.info(f"RAxML file error with {genus}, {gene_id}")
        raise Exception(f"RAxML file error with {genus}, {gene_id}")


def filter_basics(recs: pd.DataFrame) -> pd.DataFrame:
    """
    Applies filtering to the combinations of sequences, specifically: removal of duplicate combinations and same-sequence combinations

    Args:
        recs: Dataframe with samples to be processed

    Returns:
    :return: A filtered dataframe
    """
    # duplicate combinations (before gap removal) -- we don't want to unknowingly train or eval some combinations more often than others
    recs["combined_seq"] = list(
        map("".join, np.sort(recs[["seq_anchor", "seq_complement"]]))
    )
    recs.drop_duplicates(subset=["combined_seq"], inplace=True)

    # same-sequence combinations (after gap removal) -- nothing to be done for model
    recs["seq_anchor_short"] = recs["seq_anchor"].str.replace(
        r"-|\?|N|n", "", regex=True
    )
    recs["seq_complement_short"] = recs["seq_complement"].str.replace(
        r"-|\?|N|n", "", regex=True
    )
    recs = recs[recs["seq_anchor_short"] != recs["seq_complement_short"]].copy()
    recs.drop(
        columns=["combined_seq", "seq_anchor_short", "seq_complement_short"],
        inplace=True,
    )
    return recs


def create_genus_df(
    genus: str, data_dir: str, raw_files_dir: str, species_info_dir: str
) -> pd.DataFrame:
    """
    Creates a dataframe with combined sequences with genetic distances and sequence information (locus, individual ID, etc.)

    Args:
        genus: The current genus to process
        data_dir: Parent folder where data is stored
        raw_files_dir: Directory where raw alignment files are stored
        species_info_dir: Directory where files with species information are stored

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
    genus_recs = read_genus_alignments(raw_files_dir, species_info_dir, genus)

    logger.info(f"Merging {genus} distances...")
    func = partial(merge_raxml, alignment_dir=raw_files_dir, genus=genus)
    groups = (g for _, g in genus_recs.groupby(["gene_id"], group_keys=False))

    with mp.Pool(processes=4) as pool:
        genus_recs = list(
            tqdm(pool.imap_unordered(func, groups, chunksize=10), total=len(genus_recs["gene_id"].unique()))
        )
    genus_recs = pd.concat(genus_recs)

    logger.info(
        f"Basic filtering (i.e. removal of same-sequence combinations and duplicate sequence combinations) for {genus}..."
    )
    genus_recs = filter_basics(genus_recs)

    Path(f"{data_dir}/{genus}").mkdir(parents=True, exist_ok=True)
    logger.info(
        f"{genus}_unfiltered_raw_distances.csv has {len(genus_recs.index)} samples..."
    )
    genus_recs.to_csv(
        f"{data_dir}/{genus}/{genus}_unfiltered_raw_distances.csv",
        header=True,
        index=False,
    )
    return genus_recs
