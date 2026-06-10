import argparse
import pandas as pd

"""
This is a small helper script that reads test_imbalanced.csv (i.e.,
is the paired version of our test set), extracts the unique DNA samples,
and concatenates them into a dataframe that lists unpaired samples.
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        "-b",
        type=str,
        required=True,
        help="Data dir with imbalanced test data csv (required)."
    )
    return parser.parse_args(args=None)

if __name__ == "__main__":
    # get user arguments
    args = get_args()
    base_dir = args.base_dir
    test = pd.read_csv(f"{base_dir}/test_imbalanced.csv",
                       header=0,
                       engine="pyarrow",
                       dtype={"gene_id": str})

    anchors = test[
        ["anchor_id", "seq_anchor", "species_anchor", "gene_id", "genus"]
    ].rename(
        columns={
            "anchor_id": "individual_id",
            "seq_anchor": "seq",
            "species_anchor": "species",
        }
    )
    complements = test[
        [
            "complement_id",
            "seq_complement",
            "species_complement",
            "gene_id",
            "genus",
        ]
    ].rename(
        columns={
            "complement_id": "individual_id",
            "seq_complement": "seq",
            "species_complement": "species",
        }
    )
    samples = pd.concat([anchors, complements]).drop_duplicates(
        subset=["individual_id", "gene_id"]
    )
    
    del anchors
    del complements
    
    samples.loc[:, "seq"] = (
        samples["seq"].str.replace(r"-|\?|N|n", "", regex=True).str.upper()
    )
    samples.to_csv(f"{base_dir}/test_samples.csv", header=True, index=False)
