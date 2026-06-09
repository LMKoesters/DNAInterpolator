import argparse
import sys

from dna_interpolation import DNAInterpolatorWrapper
from interpolated_formatter import InterpolatedDataFormatter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records-file",
        "-r",
        help="Master table of records (required).",
        required=True
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        help="Directory where alignments reside (required).",
        required=True,
    )
    parser.add_argument(
        "--out",
        "-o",
        help="Output dir (required).",
        required=True
    )
    parser.add_argument(
        "--taxonomic-group",
        "-t",
        help="Taxonomic group to search for within master table (required).",
        required=True,
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        help="Workers for data loading (default: 4).",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--interpolation-radius",
        "-i",
        help="Takes distance to nearest neighbour and limits interpolation to a certain fraction of possible"
             "  alterations (default: .4).",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--consensus-interpolation",
        "-c",
        help="Refers to consensus sequence per species and locus to determine maximum number of nucleotides to"
             "  change (overwrites interpolation radius, default: True).",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--limit-interpolation",
        "-w",
        help="Use radius to interpolate; otherwise use random number of alterations within all possible"
             "  alterations and none.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--seqs-to-generate",
        "-g",
        help="Max number of sequences to generate through interpolation (default: 50).",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--species-aware",
        "-s",
        help="Only interpolate between sequences of the same species.",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--patience",
        "-p",
        help="A number that defines how many times the script should try to interpolate between sequences"
             "  before moving on the next individual (default: 10).",
        type=int,
        default=10,
    )
    return parser.parse_args(args=None if sys.argv[1:] else ["--help"])


if __name__ == "__main__":
    args = get_args()

    DNAInterpolatorWrapper(
        args.input_dir,
        args.out,
        args.records_file,
        args.taxonomic_group,
        num_workers=args.num_workers,
        seqs_to_generate=args.seqs_to_generate,
        interpolation_radius=args.interpolation_radius,
        consensus_interpolation=args.consensus_interpolation,
        limit_interpolation=args.limit_interpolation,
        species_aware=args.species_aware,
        patience=args.patience,
    ).run()

    InterpolatedDataFormatter(
        args.input_dir,
        args.out,
        args.taxonomic_group,
        num_workers=args.num_workers,
        consensus_interpolation=args.consensus_interpolation,
        species_aware=args.species_aware
    ).run()
