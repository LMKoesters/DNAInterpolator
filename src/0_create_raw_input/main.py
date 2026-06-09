import argparse
import sys

from aln_from_csv import aln_from_csv
from calc_distances import calc_distances


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-d",
        help="Directory where alignments reside (required).",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Output dir (required).",
        required=True
    )
    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])


if __name__ == "__main__":
    args = get_args()

    for genus in ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"]:
        input_csv = f"{args.input_dir}/{genus}.csv"
        
        aln_from_csv(
            input_csv,
            genus,
            args.output_dir,
        )

        calc_distances(
            genus,
            args.output_dir
        )
