import argparse
import pandas as pd

import logging

from create_pairs_from_alignments import create_genus_df
from filter_comparisons import filter_comparisons
from assign_splits import assign_splits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        "-d",
        type=str,
        required=True,
        help="Data dir for non-interpolated data that contains raw_sorted_files folder (required)."
    )
    parser.add_argument(
        "--species-info-dir",
        "-s",
        type=str,
        required=True,
        help="Directory with csvs containing species information for each genus."
    )
    parser.add_argument(
        "--val-frac",
        "-v",
        default=0.2,
        type=float,
        help="Fraction of data to use for validation. Default: 0.2",
    )
    parser.add_argument(
        "--min-len",
        "-i",
        default=0.5,
        type=float,
        help="Minimum length of sequences compared to median per gene_id/genus. Default: 0.5",
    )
    parser.add_argument(
        "--max-len",
        "-a",
        default=None,
        type=float,
        help="Maximum length of sequences compared to median per gene_id/genus. Default: None",
    )
    parser.add_argument(
        "--min-overlap",
        "-o",
        default=0.95,
        type=float,
        help="Minimum overlap between sequence pairs. Default: 0.95",
    )
    return parser.parse_args(args=None)


if __name__ == "__main__":
    # get user arguments
    args = get_args()

    # create a logger
    logger = logging.getLogger("no_interpolation_ds")
    fh = logging.FileHandler(filename="no_interpolation_ds.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[fh],
    )

    genera = [
        "Dactylorhiza",
        "Lomatium",
        "Palaquium",
        "Pterocarpus",
    ]
    all_recs = []
    for genus in genera:
        recs = create_genus_df(
            genus,
            args.data_dir,
            f"{args.data_dir}/raw_sorted_files/{genus}",
            args.species_info_dir,
        )
        all_recs.append(
            filter_comparisons(
                genus,
                recs,
                args.data_dir,
                min_len=args.min_len,
                max_len=args.max_len,
                min_overlap=args.min_overlap,
            )
        )

    all_recs = pd.concat(all_recs)

    ############
    logger.info("Adding cosine distance...")
    cosine_dist_max = 0.6156182520000022
    all_recs["cosine_distance"] = (all_recs["distance"] / cosine_dist_max) * 2
    all_recs["cosine_similarity"] = 1 - all_recs["cosine_distance"]
    all_recs = all_recs[all_recs["cosine_distance"] <= 2].copy()

    all_recs.to_csv(f"{args.data_dir}/test_imbalanced.csv", header=True, index=False)
    with open(f"{args.data_dir}/cosine_dist_max.txt", "w+") as f:
        f.write(str(cosine_dist_max))

    ############
    train_genera = ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]
    for genus in train_genera:
        logger.info(f"Splitting subsets for {genus}...")
        assign_splits(
            genus,
            all_recs[all_recs["genus"] == genus],
            args.data_dir,
            args.val_frac,
        )
        
    for genus in train_genera:
        logger.info(f"Writing LOO train/val for {genus}...")
        # train
        pd.concat(
            [
                pd.read_csv(
                    f"{args.data_dir}/{g}/train.csv", header=0, engine="pyarrow"
                )
                for g in train_genera
                if g != genus
            ]
        ).to_csv(f"{args.data_dir}/{genus}/loo_train.csv", header=True, index=False)

        # val
        pd.concat(
            [
                pd.read_csv(f"{args.data_dir}/{g}/val.csv", header=0, engine="pyarrow")
                for g in train_genera
                if g != genus
            ]
        ).to_csv(f"{args.data_dir}/{genus}/loo_val.csv", header=True, index=False)
