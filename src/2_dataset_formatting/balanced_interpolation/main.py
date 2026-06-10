import argparse
import pandas as pd
from pathlib import Path

import logging
import os

from create_pairs_from_alignments import create_genus_df
from filter_comparisons import filter_comparisons


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
    logger = logging.getLogger("balanced_interpolation_ds")
    fh = logging.FileHandler(filename="balanced_interpolation_ds.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[fh],
    )

    # read and filter datasets
    genera = ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]
    all_recs = []
    for genus in genera:
        recs = create_genus_df(
            genus,
            args.data_dir,
            f"{Path(args.data_dir).parent}/imbalanced_interpolation/{genus}",
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

    # find maximum number of records
    max_records = 0
    train_genera = ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]
    for g in train_genera:
        num_recs = len(
            pd.read_csv(
                f"{Path(args.data_dir).parent}/no_interpolation/{g}/train.csv",
                header=0,
                engine="pyarrow",
            ).index
        )
        if num_recs > max_records:
            max_records = num_recs

    # sample train data
    for genus in train_genera:
        if os.path.isfile(f"{args.data_dir}/{genus}/loo_train.csv"):
            continue

        logger.info(f"Writing LOO train/val for {genus}...")
        # train
        loo_recs = all_recs[all_recs["genus"] != genus]
        print(genus, len(loo_recs))

        logger.info(f"Choosing {max_records * 3} records for {genus} LOO...")
        loo_recs = loo_recs.sample(frac=1)
        loo_recs = (
            loo_recs.groupby("sector").head(max_records * 3 // 5).reset_index(drop=True)
        )
        print(genus, len(loo_recs.index))
        loo_recs.to_csv(
            f"{args.data_dir}/{genus}/loo_train.csv", header=True, index=False
        )
