from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from glob import glob
import logging
import math
import multiprocessing as mp
import numpy as np
import os
import random
import re
import subprocess
from scipy.stats import skewnorm
import pandas as pd
from pathlib import Path
import sys
import traceback
from typing import Optional
from tqdm import tqdm


class InterpolationError(Exception):
    pass


def setup_logger(logfile_path: str | Path) -> logging.Logger:
    """
    Sets up a logger to be used throughout
    the main pipeline, i.e., data formatting.

    Args:
        logfile_path: Path to where the logging file can be found

    Returns:
        A logging object
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(filename=logfile_path, mode="a")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def apply_majority_rule(base_arr: np.array) -> str:
    """
    Applies the majority rule to a given sequence alignment at a given position

    Args:
        base_arr: Array with DNA bases at a given position

    Returns:
        Most frequent DNA base
    """
    base_arr = base_arr[(base_arr != "-") & (base_arr != "N") & (base_arr != "?")]
    els, cnts = np.unique(base_arr, return_counts=True)
    try:
        most_frequent_base = els[np.argmax(cnts)]
        return str(most_frequent_base)
    except ValueError:
        return "-"


def skewed_random(max_value: int, min_value: int, skewness: int = 6) -> int:
    """
    Returns a number from a skewed random distribution

    Args:
        max_value: Max value to be used for distribution
        min_value: Min value to be used for distribution
        skewness: Param to be given to skewnorm.rsv to determine skewness (default: 6)

    Returns:
        Random number
    """
    random_dist = skewnorm.rvs(
        a=skewness, loc=max_value, size=10_000
    )

    random_dist = (random_dist - min(random_dist)) / (
        max(random_dist) - min(random_dist)
    )
    random_dist = random_dist * (max_value - min_value) + min_value
    return int(random.choice(random_dist))


def calculate_p_distance(
    aln_file: str,
    out_file: str,
    logger: logging.Logger,
    aln_type: str = "fasta",
) -> None:
    """
    Run R script to calculate p-distances

    Args:
        aln_file: Path to the input alignment with samples for which to calculate p-distances
        out_file: Path where the output distance file should be created
        logger: A logger
        aln_type: The type of alignment (default: fasta)
    """
    logger.info(f"Calculating p-distances based on {aln_file}...")
    r_cmd = f"Rscript {Path(__file__).parent}/calculate_p_distances.R --alignment {aln_file}" \
            f" --out {out_file} --type {aln_type}"
    logger.info(r_cmd)
    subprocess.call(r_cmd, shell=True)


class DNAInterpolatorWrapper:
    """
    This class is a wrapper for the DNAInterpolator class, which preprends
    some crucial steps such as writing of fasta files, p-distance calculation
    and consensus generation. It then splits the individuals within a given dataset
    into batches and runs the interpolation.
    """

    def __init__(
        self,
        in_dir: str,
        out_dir: str,
        records_file: str,
        taxonomic_group: str,
        num_workers: int = 4,
        seqs_to_generate: int = 50,
        interpolation_radius: float = 0.4,
        consensus_interpolation: bool = False,
        limit_interpolation: bool = False,
        species_aware: bool = False,
        patience: int = 10
    ):
        self.patience = patience

        self.out_dir = out_dir
        self.in_dir = in_dir
        self.num_workers = num_workers
        self.seqs_to_generate = seqs_to_generate
        self.limit_interpolation = limit_interpolation
        self.interpolation_radius = interpolation_radius
        self.consensus_interpolation = consensus_interpolation
        self.species_aware = species_aware
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        self.taxonomic_group = taxonomic_group
        logger = setup_logger(f"dna_interpolation_{self.taxonomic_group}.log")
        self.samples = self.read_records(records_file)
        self.locus_lengths = {}
        for locus in self.samples["gene_id"].unique():
            self.locus_lengths[locus] = len(
                self.samples.loc[self.samples["gene_id"] == locus, "seq"].values[0]
            )
            Path(f"{self.out_dir}/interpolated_seqs/{locus}").mkdir(
                parents=True, exist_ok=True
            )

        if self.consensus_interpolation and self.species_aware:
            self.p_distance_dfs = self.calculate_distances_with_snps_per_group(
                ["species", "gene_id"], logger
            )
            self.snp_positions = self.run_snp_sites(
                f"{self.out_dir}/alignments_species_gene_id",
                f"{self.out_dir}/snp_sites_out_species_gene_id",
                logger,
            )
            self.max_consensus_divergences = self.calc_max_consensus_divergences(logger)
            self.max_consensus_divergences_across_loci = (
                self.calc_max_consensus_divergences(logger, across_loci=True)
            )
            max_cons = pd.DataFrame.from_dict(
                self.max_consensus_divergences_across_loci, orient="index"
            )
            max_cons = max_cons.reset_index()
            max_cons.columns = ["species", "divergence"]
            max_cons.to_csv(
                f"{self.out_dir}/{self.taxonomic_group}_consensus_divergence_across_loci.csv",
                header=True,
                index=False,
            )
        elif self.species_aware and self.limit_interpolation:
            self.p_distance_dfs = self.calculate_distances_with_snps_per_group(
                ["individual_id"], logger
            )
            self.snp_positions = self.run_snp_sites(
                f"{self.out_dir}/alignments_individual_id",
                f"{self.out_dir}/snp_sites_out_individual_id",
                logger,
            )
        else:
            self.p_distance_dfs = self.calculate_distances_with_snps_per_group(
                ["gene_id"], logger
            )

        self.snp_positions = self.run_snp_sites(
            self.in_dir, f"{self.out_dir}/snp_sites_out_gene_id", logger
        )

    def run(self) -> dict[str, int] | None:
        if os.path.isfile(
            f"{self.out_dir}/records_interpolated_{self.taxonomic_group}.csv"
        ):
            return

        logger = setup_logger(f"dna_interpolation_{self.taxonomic_group}.log")
        individual_ids = self.samples["individual_id"].unique()
        self.write_snps_of_individuals(individual_ids)

        with mp.Pool(
            self.num_workers,
            initializer=setup_logger,
            initargs=(f"dna_interpolation_{self.taxonomic_group}.log",),
        ) as pool:
            new_recs = list(
                tqdm(
                    pool.imap(
                        self.interpolate_sequences,
                        zip(range(0, len(individual_ids)), individual_ids),
                    ),
                    total=len(individual_ids),
                )
            )

        new_recs = pd.concat(new_recs)
        logger.info(f"Created {len(new_recs)} new records for {self.taxonomic_group}")
        logger.info(
            f"Of a total of {len(new_recs)} sequences of genus {self.taxonomic_group},"
            f" {len(new_recs['seq'].unique())} are unique"
        )
        new_recs.to_csv(
            f"{self.out_dir}/records_interpolated_{self.taxonomic_group}.csv",
            header=True,
            index=False,
        )

    def write_snps_of_individuals(self, individual_ids: list) -> None:
        """
        Gathers and writes original SNP matrices. Important for eliminating duplicates when interpolating

        Args:
            individual_ids: IDs of individuals within dataset
        """
        snp_dir = f"{self.out_dir}/snps_intermediates"
        Path(snp_dir).mkdir(parents=True, exist_ok=True)
        if os.path.isfile(f"{snp_dir}/snps_originals.csv"):
            return

        snps_df = pd.DataFrame(columns=["snp"])

        for individual_id in individual_ids:
            complete_snps = ""
            for locus in self.locus_lengths.keys():
                try:
                    locus_snp_positions = self.snp_positions[locus]
                except KeyError:  # no SNPs for this locus
                    continue

                try:
                    seq = self.samples.loc[
                        (self.samples["individual_id"] == individual_id)
                        & (self.samples["gene_id"] == locus),
                        "seq",
                    ].values[0]
                    complete_snps += "".join(
                        [seq[snp_position] for snp_position in locus_snp_positions]
                    )
                except IndexError:
                    complete_snps += "-" * len(locus_snp_positions)

            snps_df.loc[len(snps_df)] = [complete_snps]

        snps_df.to_csv(f"{snp_dir}/snps_originals.csv", header=True, index=False)

    def read_genus_alignments(self) -> pd.DataFrame:
        """
        Read alignment files; gather sample dataframe

        Returns:
            Dataframe containing all samples within dataset
        """
        samples = []

        for aln_f in glob(f"{self.in_dir}/*"):
            if ".reduced" in aln_f or "raxml.log" in aln_f:
                continue

            gene_id = Path(aln_f).stem
            if self.taxonomic_group in ["Dactylorhiza", "Lomatium", "Palaquium"]:
                pass
            elif self.taxonomic_group == "Pterocarpus":
                gene_id = gene_id.replace("_supercontig", "")
            else:
                print(f"Unknown taxonomic group: {self.taxonomic_group}")
                sys.exit()

            alignment_format = "fasta"
            try:
                msa = AlignIO.read(aln_f, alignment_format)
                if len(msa) <= 1:
                    continue
                for record in msa:
                    seq = str(record.seq).upper()
                    individual_id = record.description

                    samples.append([self.taxonomic_group, gene_id, individual_id, seq])
            except ValueError:
                print(f"No alignment found for {aln_f}")
                print(traceback.format_exc())
                print(f"No alignment found for {aln_f}")

        # create dataframe with sample info - no combinations & no distances
        samples = pd.DataFrame.from_records(
            samples, columns=["genus", "gene_id", "individual_id", "seq"]
        )

        # merge with species + country info
        species_info = pd.read_csv(
            f"{Path(self.in_dir).parent.parent.parent.parent}/species_info/{self.taxonomic_group}.csv",
            header=0,
        )
        if self.taxonomic_group == "Pterocarpus":
            # correct Pterocarpus fasta names for merging with species/country info
            samples["individual_id"] = samples["individual_id"].str.replace(
                r"^_R_", "", regex=True
            )
        samples = pd.merge(samples, species_info, how="left", on="individual_id")
        samples.loc[samples["species"].isna(), "species"] = "outgroup"
        samples["species"] = samples["species"].str.replace(
            r"[\:\s.]+", "_", regex=True
        )
        return samples

    def read_records(
            self,
            recs_f: str) -> pd.DataFrame:
        """
        Reads specified master table with the following required columns and sorts records based on locus:
            genus
            species (optional, only needed if 'species-aware' is turned on)
            gene_id
            anchor_id
            complement_id
            anchor_seq
            complement_seq
            distance

        Args:
            recs_f: Path to file with original sample combinations and their distances

        Returns:
            Dataframe with unique samples (de-combined)
        """
        train_ds = pd.read_csv(recs_f, header=0, dtype={"gene_id": str})
        train_ds = train_ds[["gene_id", "anchor_id", "complement_id"]]

        samples = self.read_genus_alignments()
        samples.drop(columns="country", inplace=True)
        samples.sort_values(by="gene_id", inplace=True)  # for iteration over loci
        # remove samples not belonging to train set
        samples = samples[
            (samples["individual_id"].isin(train_ds["anchor_id"]))
            | (samples["individual_id"].isin(train_ds["complement_id"]))
        ].copy()
        samples[["base", "donor", "gen_dist_donor", "nucleotides_changed"]] = None
        samples["gene_id"] = samples["gene_id"].str.replace(r"\D+", "", regex=True)
        samples.to_csv(
            f"{self.out_dir}/records_original.csv", header=True, index=False
        )
        return samples

    def run_snp_sites(
        self, in_dir: str | Path, snp_out_dir: str | Path, logger: logging.Logger
    ) -> dict[str, list]:
        """
        Runs snp-sites to get location of SNPs within alignments of a given directory

        Args:
            in_dir: Path to directory with alignment files
            snp_out_dir: Path to directory where SNP output files should be stored
            logger: A logger object

        Returns:
            A dictionary with the SNP file as key and the SNP positions as value
        """
        logger.info("Running SNP-sites...")
        Path(snp_out_dir).mkdir(exist_ok=True)
        snp_calc_cmd = f"./run_snp_sites.sh {in_dir} {snp_out_dir}"
        subprocess.call(snp_calc_cmd, shell=True)

        snp_positions = {}
        overall_snps = 0
        for snp_file in glob(f"{snp_out_dir}/*"):
            snp_file_basename = Path(snp_file).stem

            # retrieve SNP positions
            try:
                vcf = pd.read_csv(snp_file, header=3, sep="\t")
                snp_positions[snp_file_basename] = [
                    int(x) - 1 for x in vcf["POS"].values
                ]  # 1-based hence -1
            except FileNotFoundError:
                continue
            overall_snps += len(snp_positions[snp_file_basename])
        try:
            logger.info(
                f"Number of SNPs across all {len(snp_positions)} SNP files is {overall_snps} with a mean of"
                f"  {round(overall_snps / len(snp_positions), 2)} at an average locus length of"
                f"  {round(float(np.mean(list(self.locus_lengths.values()))), 2)}"
            )
        except ZeroDivisionError:
            print(len(snp_positions))
            print(overall_snps)
            sys.exit()
        return snp_positions

    def generate_group_alignment(
        self,
        grouped_recs: pd.DataFrame,
        alignment_file: str | Path,
        logger: logging.Logger,
        concatenate: bool = False,
        return_aln: bool = False,
    ) -> Optional[list[str]]:
        """
        Generate alignment file for group

        Args:
            grouped_recs: A dataframe object from a grouped parent dataframe
            alignment_file: Path to the alignment file to be generated from the records within the dataframe
            logger: A logger
            concatenate: A boolean switch to decide whether or not records within the dataframe should
                be concatenated by locus
            return_aln: Whether to return the alignment as a list

        Returns:
            The generated alignment as a list
        """
        logger.info(f"Generating group alignment file {alignment_file}...")
        alignment = []
        total_seq = []

        # either grouped by a) ['species', 'gene_id'] or b) ['individual_id']
        if concatenate:
            loci = list(self.locus_lengths.keys())
            for locus in loci:
                if locus in grouped_recs["gene_id"].values:
                    total_seq.append(
                        grouped_recs.loc[
                            grouped_recs["gene_id"] == locus, "seq"
                        ].values[0]
                    )
                else:
                    total_seq.append("N" * self.locus_lengths[locus])
        else:
            for _, rec in grouped_recs.iterrows():
                total_record = SeqRecord(
                    Seq(rec["seq"].replace("?", "N")),
                    id=rec["individual_id"],
                    description="",
                )

                alignment.append(total_record)

        if concatenate:
            total_record = SeqRecord(
                Seq("".join(total_seq).replace("?", "N")),
                id=grouped_recs["individual_id"].values[0],
                description="",
            )

            alignment.append(total_record)

        if return_aln:
            return [str(seq.seq) for seq in alignment]
        else:
            # write fasta entries
            with open(alignment_file, "a+") as conc_aln_f:
                for seq in alignment:
                    SeqIO.write(seq, conc_aln_f, "fasta")

    def calculate_distances_with_snps_per_group(
            self,
            group_els: list[str],
            logger: logging.Logger) -> dict[str, pd.DataFrame]:
        """
        Generates group-specific (unaligned) fasta files, aligns them with mafft, and creates VCF files

        Args:
            group_els: Elements to group by
            logger: A logger

        Returns:
            A dictionary with group name -> dataframe with p_distances
        """
        logger.info(
            f"Calculating distances with SNP per group based on {' and '.join(group_els)}..."
        )
        Path(f"{self.out_dir}/alignments_{'_'.join(group_els)}").mkdir(
            exist_ok=True, parents=True
        )
        Path(f"{self.out_dir}/distances_{'_'.join(group_els)}").mkdir(
            exist_ok=True, parents=True
        )
        p_dist_dfs = {}

        for name, grouped_recs in self.samples.groupby(group_els):
            name_tup = name if isinstance(name, tuple) else (name,)

            if group_els == tuple("gene_id"):
                # records would already be within locus-specific fasta files
                #   --> no need to create fasta files/alignments;
                #   just calculate p-distances and create VCF files
                locus = name_tup[group_els.index("gene_id")]
                alignment_file = [
                    f
                    for f in glob(f"{self.in_dir}/*{locus}*")
                    if not re.match(rf"{locus}[A-Z\d]", f, flags=re.IGNORECASE)
                    and ".reduced" not in f
                ][0]
            else:
                alignment_file = f"{self.out_dir}/alignments_{'_'.join(group_els)}/{'_'.join(name_tup)}.aln"
                if not os.path.isfile(alignment_file):
                    self.generate_group_alignment(
                        grouped_recs,
                        alignment_file,
                        logger,
                        concatenate=group_els == ["individual_id"],
                    )

            distance_file = f"{self.out_dir}/distances_{'_'.join(group_els)}/{'_'.join(name_tup)}_distances.csv"
            if not os.path.isfile(distance_file):
                calculate_p_distance(alignment_file, distance_file, logger)

            # read distances and change Pterocarpus individual names so they match those in master table
            try:
                p_dist_df = pd.read_csv(distance_file, header=0)
            except Exception as e:
                print(e)
                print(distance_file)
                sys.exit()
            if self.taxonomic_group == "Pterocarpus":
                p_dist_df.loc[:, "sampleA"] = p_dist_df["sampleA"].str.replace(
                    r"^_R_", "", regex=True
                )
                p_dist_df.loc[:, "sampleB"] = p_dist_df["sampleB"].str.replace(
                    r"^_R_", "", regex=True
                )
            p_dist_dfs["_".join(name_tup)] = p_dist_df

        return p_dist_dfs

    def calc_max_consensus_divergences(
        self, logger: logging.Logger, across_loci: bool = False
    ) -> dict[str, int]:
        """
        Calculates the maximum divergence of any species-and-locus-specific DNA sequence to the consensus sequence of
            the respective sequence group

        Args:
            logger: A logger
            across_loci: Boolean switch to determine whether consensus divergences should be calculated on a
                per-locus basis or across loci

        Returns:
            Dictionary with either the species or the species+locus as key and a number specifying the divergence
                from the consensus sequence as value
        """
        logger.info("Calculating maximum consensus divergence...")
        self.get_consensus(logger, True)  # needed for final check downstream
        consensus_seqs = self.get_consensus(logger, False)

        max_consensus_divergences = {}

        # we replace 'N's and '?'s with '-', because '-' is the only element that can be found in our consensus
        #   sequences and we do not want to count 'N's and '?'s as deviations from that

        if across_loci:
            samples = self.samples.copy()
            for group_labels, species_recs in samples.groupby(["species"]):
                group_labs_tup = group_labels if isinstance(group_labels, tuple) else (group_labels,)

                species = group_labs_tup[0]
                for locus in species_recs["gene_id"].unique():
                    consensus_seq = consensus_seqs.loc[
                        (consensus_seqs["species"] == species)
                        & (consensus_seqs["locus"] == locus),
                        "consensus"
                    ].values[0]

                    species_locus_recs_idx = species_recs["gene_id"] == locus
                    species_recs.loc[species_locus_recs_idx, "seq"] = species_recs["seq"].str\
                        .replace("N", "-")\
                        .replace("?", "-")
                    species_recs.loc[species_locus_recs_idx, "cons_div"] = species_recs.loc[
                        species_locus_recs_idx, "seq"].apply(
                        lambda x: len(
                                [
                                    base
                                    for idx, base in enumerate(x)
                                    if base != consensus_seq[idx]
                                ]
                            )
                    )

                species_recs["cat_cons_div"] = species_recs.groupby('individual_id')["cons_div"]\
                    .transform("sum")

                max_consensus_divergences[species] = max(species_recs["cat_cons_div"])
        else:
            for group_labels, species_locus_recs in self.samples.groupby(
                ["species", "gene_id"]
            ):
                group_labs_tup = group_labels if isinstance(group_labels, tuple) else (group_labels,)

                species = group_labs_tup[0]
                locus = group_labs_tup[1]
                consensus_seq = consensus_seqs.loc[
                    (consensus_seqs["species"] == species)
                    & (consensus_seqs["locus"] == locus),
                    "consensus",
                ].values[0]
                max_consensus_divergences[f"{species}_{locus}"] = max(
                    [
                        len(
                            [
                                base
                                for idx, base in enumerate(seq)
                                if base != consensus_seq[idx]
                            ]
                        )
                        for seq in species_locus_recs["seq"].str
                        .replace("N", "-")
                        .replace("?", "-")
                    ]
                )
        return max_consensus_divergences

    def get_consensus(
        self, logger: logging.Logger, across_loci: bool = False
    ) -> pd.DataFrame:
        """
        Gathers either DNA sequences per species and locus,
            or concatenated DNA sequences per species across loci
            and generates consensus

        Args:
            logger: A logger
            across_loci: A boolean switch determining whether to generate consensus sequences on a per-locus basis
                or across loci

        Returns:
            A dataframe with consensus sequences and information on the respective genus/species/locus
        """
        logger.info(f"Getting consensus sequences. Across loci: {across_loci}...")
        csv_file = (
            f"{self.out_dir}/consensus_sequences_across.csv"
            if across_loci
            else f"{self.out_dir}/consensus_sequences_loci.csv"
        )

        if os.path.isfile(csv_file):
            return pd.read_csv(csv_file, header=0, dtype={"locus": str})

        consensus_seqs = []
        group_els = ["species"] if across_loci else ["species", "gene_id"]
        for name, grouped_recs in self.samples.groupby(group_els):
            name_tup = name if (isinstance(name, tuple)) else (name,)

            species = name_tup[0]
            genus = grouped_recs["genus"].values[0]

            if across_loci:
                species_concat_seqs = []
                for _, ind_grouped_recs in grouped_recs.groupby(["individual_id"]):
                    species_concat_seqs.extend(
                        self.generate_group_alignment(
                            ind_grouped_recs,
                            f"{self.out_dir}/intermediate_file.fa",
                            logger,
                            concatenate=True,
                            return_aln=True,
                        )
                    )
                locus = "concatenated"
                alignment = np.array([list(seq) for seq in species_concat_seqs])
            else:
                locus = name_tup[1]
                alignment = np.array([list(seq) for seq in grouped_recs["seq"].values])

            # calculate consensus
            consensus = "".join(
                [
                    apply_majority_rule(alignment[:, pos])
                    for pos in range(len(alignment[0]))
                ]
            )
            consensus_seqs.append((genus, species, locus, consensus))

        consensus_seqs = pd.DataFrame.from_records(
            consensus_seqs, columns=["genus", "species", "locus", "consensus"]
        )
        consensus_seqs.to_csv(csv_file, header=True, index=False)
        return consensus_seqs

    def interpolate_sequences(self, args: tuple[int, str]) -> pd.DataFrame:
        """
        Generate interpolated sequences of batch with multiple individuals

        Args:
            args: The index of this process and the individual to process

        Returns:
            A dataframe with newly generated interpolated samples
        """
        logger = logging.getLogger()
        logger.info("Generating interpolated sequences...")

        idx = args[0]
        if self.species_aware:
            species = self.samples.loc[
                self.samples["individual_id"] == args[1], "species"
            ].values[0]
            new_recs = DNAInterpolator(
                args[1],
                self.max_consensus_divergences,
                self.max_consensus_divergences_across_loci,
                self.samples,
                {k: v for k, v in self.p_distance_dfs.items() if species in k},
                self.snp_positions,
                self.locus_lengths,
                self.out_dir,
                self.species_aware,
                self.limit_interpolation,
                self.interpolation_radius,
                self.consensus_interpolation,
                idx,
                logger,
                self.patience,
            ).interpolate(seqs_to_generate=self.seqs_to_generate)
        else:
            new_recs = DNAInterpolator(
                args[1],
                self.max_consensus_divergences,
                self.max_consensus_divergences_across_loci,
                self.samples,
                self.p_distance_dfs,
                self.snp_positions,
                self.locus_lengths,
                self.out_dir,
                self.species_aware,
                self.limit_interpolation,
                self.interpolation_radius,
                self.consensus_interpolation,
                idx,
                logger,
                self.patience,
            ).interpolate(seqs_to_generate=self.seqs_to_generate)

        # save new records
        Path(f"{self.out_dir}/interpolated_csvs").mkdir(exist_ok=True, parents=True)
        batch_records_f = f"{self.out_dir}/interpolated_csvs/batch_{idx}.csv"
        new_recs.to_csv(batch_records_f, header=True, index=False)
        return new_recs


class DNAInterpolator:
    """
    This class generates new individuals by interpolating between
    individuals. Pairs of sequences that serve as a basis for interpolation
    can be selected either with knowledge of the species label or without.
    Using the species label, a consensus-aware method ensures that new individuals
    stay within the species-defined boundaries by limiting the number of bases
    to be altered. When not opting for the consensus-aware approach,
    the minimal p-distance to a sample belonging to a different species and
    the maximum p-distance to a sample belonging to the same class are used for
    a guided threshold-based method. Without the species label,
    any number of bases between the number of SNPs of that locus and 0
    can be altered.
    """

    def __init__(
        self,
        individual_id: str,
        max_consensus_divergences: dict[str, int],
        max_consensus_divergences_across_loci: dict[str, int],
        samples: pd.DataFrame,
        p_distance_dfs: dict[str, pd.DataFrame],
        snp_positions: dict[str, list],
        locus_lengths: dict[str, int],
        out_dir: str | Path,
        species_aware: bool,
        limit_interpolation: bool,
        interpolation_radius: float,
        consensus_interpolation: bool,
        idx: int,
        logger: logging.Logger,
        patience: int = 10,
    ):
        self.patience = patience
        self.individual_id = individual_id
        self.max_consensus_divergences = max_consensus_divergences
        self.max_consensus_divergences_across_loci = max_consensus_divergences_across_loci

        self.samples = samples
        self.p_distance_dfs = p_distance_dfs

        self.snp_positions = snp_positions
        self.locus_lengths = locus_lengths

        self.out_dir = out_dir
        self.species_aware = species_aware
        self.limit_interpolation = limit_interpolation
        self.interpolation_radius = interpolation_radius
        self.consensus_interpolation = consensus_interpolation
        self.idx = idx
        self.logger = logger

        self.species = self.samples.loc[self.samples["individual_id"] == self.individual_id, "species"].values[0]
        self.consensus_seqs = pd.read_csv(
            f"{self.out_dir}/consensus_sequences_loci.csv",
            header=0,
            dtype={"locus": str},
        )
        self.consensus_seqs = self.consensus_seqs.loc[self.consensus_seqs["species"] == self.species, :]

    def interpolate(self, seqs_to_generate: int = 50) -> pd.DataFrame:
        """
        Run interpolation for all loci of individual

        Args:
            seqs_to_generate: The number of samples to generate

        Returns:
            A dataframe with newly generated interpolated samples
        """
        new_recs = pd.DataFrame(columns=self.samples.columns)
        snps_df = pd.DataFrame(columns=["snp"])
        interpolated_seqs_num = 0
        tries = 0  # reset whenever valid new sequence was found
        num_loci = len(self.samples["gene_id"].unique())
        sample_not_found = 0

        snp_dir = f"{self.out_dir}/snps_intermediates"

        individual_nucleotide_div = self.samples[self.samples['individual_id'] == self.individual_id].apply(
            lambda x: len([
                idx
                for idx, y in enumerate(x["seq"].replace("N", "-").replace("?", "-"))
                if y != self.consensus_seqs.loc[
                    self.consensus_seqs["locus"] == x["gene_id"], "consensus"
                ].values[0][idx]
            ]),
            axis=1
        ).sum()

        while (
            interpolated_seqs_num < seqs_to_generate
            and tries < self.patience
            and sample_not_found < num_loci
        ):
            snp_matrix = {}
            overall_nucleotides_changed = individual_nucleotide_div

            sample_not_found = 0
            locus_i = 0

            for name, locus_group in self.samples.sample(frac=1).groupby(["gene_id"]):
                locus = name[0]

                if self.species_aware and self.consensus_interpolation:
                    accessor = f"{self.species}_{locus}"
                elif self.species_aware:
                    accessor = self.individual_id
                else:
                    accessor = locus

                try:
                    rec = (
                        locus_group[locus_group["individual_id"] == self.individual_id]
                        .head(1)
                        .to_dict(orient="records")[0]
                    )
                    if (
                        self.species_aware
                    ):  # species-aware -> filter locus group for same-species records
                        interpolation_group = locus_group.loc[
                            locus_group["species"] == self.species, :
                        ].copy()
                    else:
                        interpolation_group = locus_group
                except IndexError:  # no sequence for this gene/individual combo - skip
                    try:
                        snp_matrix[locus] = "-" * len(self.snp_positions[locus])
                    except KeyError:  # no SNPs for this locus
                        pass
                    sample_not_found += 1
                    continue

                try:
                    (
                        interpolated_seq,
                        div_nucleotides,
                        nucleotides_changed,
                        donor_id,
                        p_distance,
                        snps,
                    ) = self.interpolate_sequence(
                        rec,
                        interpolation_group,
                        overall_nucleotides_changed,
                        accessor,
                    )
                    snp_matrix[locus] = snps
                    p_distance = round(float(p_distance), 4)
                except (KeyError, InterpolationError):
                    """
                    if there are no SNPs to be taken for this locus, just take sequence as is since
                    we're interested in differing overall sequences, not in locus-specific differences
                    """
                    interpolated_seq = rec["seq"]

                    try:
                        snp_matrix[locus] += "".join(
                            [interpolated_seq[snp_position] for snp_position in self.snp_positions[locus]]
                        )
                    except KeyError:  # no SNPs for this locus
                        pass

                    div_nucleotides = 0
                    nucleotides_changed = 0
                    donor_id = None
                    p_distance = None
                    sample_not_found += 1

                overall_nucleotides_changed += div_nucleotides

                new_recs.loc[len(new_recs), :] = [
                    rec["genus"],
                    rec["gene_id"],
                    f"{self.individual_id}_aug{interpolated_seqs_num}",
                    interpolated_seq,
                    self.samples.loc[
                        self.samples["individual_id"] == self.individual_id, "species"
                    ].values[0],
                    self.individual_id,
                    donor_id,
                    p_distance,
                    nucleotides_changed,
                ]

                locus_i += 1

            # Shuffling our master table means that both the snp_matrix will be shuffled as well
            #   -> so we need to re-sort it
            complete_snps = ""
            for locus in self.locus_lengths.keys():
                try:
                    complete_snps += snp_matrix[locus]
                except KeyError:  # no SNPs for this locus
                    pass

            # final check
            if (
                not self.interpolated_sequence_ok(complete_snps)
                or sample_not_found >= num_loci
            ):
                new_recs = new_recs[
                    new_recs["individual_id"]
                    != f"{self.individual_id}_aug{interpolated_seqs_num}"
                ]
                tries += 1
                if tries % 10 == 0:
                    self.logger.info(f"Tries +1 --> {tries}")
            else:  # successful interpolation
                tries = 0
                interpolated_seqs_num += 1
                snps_df.loc[len(snps_df)] = [complete_snps]
                snps_df.to_csv(
                    f"{snp_dir}/snps_{self.idx}.csv", header=True, index=False
                )
                if interpolated_seqs_num % 10 == 0:
                    self.logger.info(f"{self.individual_id}: {interpolated_seqs_num}")

        self.write_records(new_recs)
        return new_recs

    def write_records(self, new_recs: pd.DataFrame) -> None:
        """
        Writes single fastas to locus-specific directories based on new records

        Args:
            new_recs: A dataframe with newly generated interpolated samples
        """
        for _, rec in new_recs.iterrows():
            fa_file_dir = f"{self.out_dir}/interpolated_seqs/{rec['gene_id']}"

            aug_record = SeqRecord(
                Seq(str(rec["seq"]).replace("?", "N")),
                id=rec["individual_id"],
                description="",
            )

            with open(
                f"{fa_file_dir}/{self.individual_id}_aug.fasta", "a+"
            ) as single_fa:
                SeqIO.write(aug_record, single_fa, "fasta")

    def interpolated_sequence_ok(self, snp_matrix: str) -> bool:
        """
        Checks if interpolated sequence is a duplicate of original/interpolated sequence

        Args:
            snp_matrix: A string with DNA bases at SNP positions across loci

        Returns:
            A boolean determining whether interpolation was successful
        """
        # check that no duplicate sequences were created between individuals based on concatenated sequences,
        #   i.e., across all loci
        if snp_matrix in self.snps():
            # duplicate found
            return False
        else:  # successful interpolation
            return True

    def snps(self) -> list | pd.DataFrame:
        """
        Reads all SNP-reduced concatenated/whole-individual sequences generated so far

        Returns:
            A list of SNP-reduced DNA sequences
        """
        # reads temporary files with snp information to compare new sequences with (to prevent duplicates)
        try:
            return pd.concat(
                [
                    pd.read_csv(f, header=0)
                    for f in glob(f"{self.out_dir}/snps_intermediates/snps_*.csv")
                ]
            )["snp"].values
        except ValueError:
            return []

    def interpolate_sequence(
        self,
        rec: pd.Series,
        locus_group: pd.DataFrame,
        overall_nucleotides_changed: int,
        accessor: str,
    ) -> tuple:
        """
        Interpolates a given DNA (locus-specific) sequence

        Args:
            rec: A single sample from a dataframe
            locus_group: A dataframe with samples belonging to the same locus
            overall_nucleotides_changed: The number of nucleotides altered so far for this individual
            accessor: The key to the required p-distance dataframe

        Returns:
            Interpolated sequence with additional info on changed nucleotides, ID of donor sequence,
                p-distance, and the complete SNP matrix
        """
        p_dist_df = self.p_distance_dfs[accessor]

        # choose individual to copy from aka donor
        potential_donor_ids = locus_group.loc[
            locus_group["individual_id"] != self.individual_id, "individual_id"
        ].values

        try:
            # we don't need to check the other way round (i.e., switch sampleB and sampleA),
            #   because distances are mirrored
            donor_id = (
                p_dist_df.loc[
                    (p_dist_df["sampleB"].isin(potential_donor_ids))
                    & (p_dist_df["sampleA"] == self.individual_id)
                    & (p_dist_df["p_distance"] > 0),
                    "sampleB"
                ]
                .sample(1)
                .values[0]
            )
        except ValueError:  # no valid donor - usually due to all-zero distances
            raise InterpolationError(
                "No valid donor sequence found. Usually, this is caused by all-zero distances."
            )

        locus_snp_positions = self.snp_positions[rec["gene_id"]]
        original_seq_consensus_divergence = max_locus_consensus_divergence = 1
        used_snp_positions = []

        # cluster divergence-based max_nucleotides_to_change
        consensus_seq = self.consensus_seqs.loc[
            self.consensus_seqs["locus"] == rec["gene_id"],
            "consensus",
        ].values[0]

        used_snp_positions = [
            idx
            for idx, base in enumerate(
                rec["seq"].replace("N", "-").replace("?", "-")
            )
            if base != consensus_seq[idx]
        ]
        original_seq_consensus_divergence = len(used_snp_positions)

        if self.species_aware and self.consensus_interpolation:
            max_locus_consensus_divergence = self.max_consensus_divergences[
                f"{self.species}_{rec['gene_id']}"
            ]
            max_nucleotides_to_change = max(min(
                max_locus_consensus_divergence,
                self.max_consensus_divergences_across_loci[self.species]
                - overall_nucleotides_changed
            ), original_seq_consensus_divergence)

            min_nucleotides_to_change = 1
            # if we want all datasets to have roughly the same size, we need to make sure that there are as
            #   few duplicates (locus-wise) as possible
        elif self.species_aware and self.limit_interpolation:
            # threshold-based max_nucleotides_to_change
            # collect individual_id of samples belonging to another species
            non_species_ids = self.samples.loc[
                (self.samples["gene_id"] == rec["gene_id"])
                & (self.samples["species"] != self.species),
                "individual_id",
            ].values
            # get the minimal distance to a sample that belongs to a different species
            smallest_genetic_distance = p_dist_df.loc[
                (p_dist_df["sampleA"] == self.individual_id)
                & (p_dist_df["sampleB"].isin(non_species_ids))
                & (p_dist_df["p_distance"] > 0),
                "p_distance",
            ].min()
            if smallest_genetic_distance * len(rec["seq"]) <= 2:
                max_nucleotides_to_change = (
                    1  # 100% of nucleotides in case of just one nuc differences
                )
            else:
                max_nucleotides_to_change = max(
                    1,
                    int(
                        math.floor(
                            smallest_genetic_distance
                            * len(rec["seq"])
                            * self.interpolation_radius
                        )
                    ),
                )
                # number of nucleotides to change, i.e., <=interpolation_radius% of number of nucleotides in which
                #   sequence differs from closest sequence
            min_nucleotides_to_change = 1
        else:
            max_nucleotides_to_change = len(locus_snp_positions)
            min_nucleotides_to_change = 0

        original_seq = list(rec["seq"])
        interpolated_seq = original_seq

        donor_seq = locus_group.loc[
            locus_group["individual_id"] == donor_id, "seq"
        ].values[0]

        # determine random SNPs
        if max_nucleotides_to_change > 0:
            # remove identical positions from snp positions to prevent silent mutations
            relevant_snps = [
                snp_pos
                for snp_pos in locus_snp_positions
                if original_seq[snp_pos] != donor_seq[snp_pos]
            ]

            if (
                self.species_aware
                and self.consensus_interpolation
                and (original_seq_consensus_divergence == max_locus_consensus_divergence
                     or overall_nucleotides_changed == self.max_consensus_divergences_across_loci[self.species])
            ):
                relevant_snps = [
                    snp_pos
                    for snp_pos in relevant_snps
                    if snp_pos in used_snp_positions
                ]  # this results in reduced or equal divergence and prevents even further divergence from consensus

            if len(relevant_snps) == 0:
                raise InterpolationError(
                    "No differing positions between receiver and donor sequences"
                )

            # get positions to retrieve from donor
            try:
                max_val = min(max_nucleotides_to_change, len(relevant_snps))
                nucleotides_to_change = skewed_random(
                    max_value=max_val,
                    min_value=min_nucleotides_to_change,
                    skewness=max_val,
                )

                snp_positions_to_change = random.sample(
                    relevant_snps, nucleotides_to_change
                )[0]
            except IndexError:
                raise InterpolationError("Nucleotides to change was 0")

            # get nucleotides from donor sequence
            interpolated_seq[snp_positions_to_change] = donor_seq[
                snp_positions_to_change
            ]
            interpolated_seq = "".join(interpolated_seq)

            p_distance = p_dist_df.loc[
                (p_dist_df["sampleA"] == self.individual_id)
                & (p_dist_df["sampleB"] == donor_id),
                "p_distance",
            ].values[0]
            snp_matrix = "".join(
                [interpolated_seq[snp_position] for snp_position in locus_snp_positions]
            )

            consensus_seq = self.consensus_seqs.loc[
                self.consensus_seqs["locus"] == rec["gene_id"],
                "consensus",
            ].values[0]
            interpolated_seq_consensus_divergence = len([
                idx
                for idx, base in enumerate(
                    interpolated_seq.replace("N", "-").replace("?", "-")
                )
                if base != consensus_seq[idx]
            ])

            div_nucleotides = interpolated_seq_consensus_divergence - original_seq_consensus_divergence

            return (
                interpolated_seq,
                div_nucleotides,
                nucleotides_to_change,
                donor_id,
                p_distance,
                snp_matrix,
            )
        else:
            raise InterpolationError("Max nucleotides to change was 0")
