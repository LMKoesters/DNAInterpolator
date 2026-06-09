import argparse
import pandas as pd
from pathlib import Path
import shutil

import logging

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
    logger = logging.getLogger("imbalanced_interpolation_ds")
    fh = logging.FileHandler(filename="imbalanced_interpolation_ds.log")
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[fh],
    )

    genera = ["Dactylorhiza", "Lomatium", "Palaquium"]
    all_recs = []
    for genus in genera:
        recs = create_genus_df(genus, args.data_dir, f"{args.data_dir}/{genus}")
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

    # Pterocarpus is the largest dataset and can, therefore, be just read from no interpolation
    shutil.copyfile(f"{Path(args.data_dir).parent}/no_interpolation/Pterocarpus/train.csv",
                    f"{args.data_dir}/Pterocarpus/train.csv")
    pter = pd.read_csv(
        f"{args.data_dir}/Pterocarpus/train.csv", header=0, engine="pyarrow"
    )

    all_recs.append(pter)
    all_recs = pd.concat(all_recs)

    ############
    train_genera = ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]
    for genus in train_genera:
        logger.info(f"Writing LOO train for {genus}...")
        # train
        all_recs[all_recs["genus"] != genus].to_csv(
            f"{args.data_dir}/{genus}/loo_train.csv", header=True, index=False
        )
