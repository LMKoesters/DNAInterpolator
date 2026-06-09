import os
import pandas as pd
import random


def assign_splits(
    genus: str,
    recs: pd.DataFrame,
    data_dir: str,
    val_frac: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Assigns splits based on non interpolated data.
    The training split will be used to create new samples by interpolating.

    Args:
        genus: The genus currently processing
        recs: A dataframe with paired DNA samples and additional information
        data_dir: The parent directory where training/validation data for all genera can be found (data_dir > genus > val.csv)
        val_frac: The fraction of validation data to be aimed for
        random_state: A random state for reproducibility
    """
    if os.path.isfile(f"{data_dir}/{genus}/val.csv"):
        print(f"Skipping split assignment for {genus}...")
        return

    random.seed(random_state)

    all_individuals = list(
        pd.concat([recs["anchor_id"], recs["complement_id"]]).unique()
    )
    all_individuals = random.sample(all_individuals, len(all_individuals))

    recs.loc[:, "seq_anchor"] = (
        recs["seq_anchor"].str.replace(r"-|\?|N|n", "", regex=True).str.upper()
    )
    recs.loc[:, "seq_complement"] = (
        recs["seq_complement"].str.replace(r"-|\?|N|n", "", regex=True).str.upper()
    )

    chosen_individuals = []
    best_split = None
    best_diff = float("inf")
    for i, individual in enumerate(all_individuals):
        chosen_individuals.append(individual)
        val = len(
            recs[
                (recs["anchor_id"].isin(chosen_individuals))
                & (recs["complement_id"].isin(chosen_individuals))
            ].index
        )
        train = len(
            recs[
                (~recs["anchor_id"].isin(chosen_individuals))
                & (~recs["complement_id"].isin(chosen_individuals))
            ].index
        )
        together = val + train
        this_val_frac = val / together
        this_diff = abs(this_val_frac - val_frac)
        if this_diff < best_diff:
            best_split = i + 1
            best_diff = this_diff
        elif this_diff > best_diff:
            break

    chosen_individuals = all_individuals[:best_split]
    val = recs[
        (recs["anchor_id"].isin(chosen_individuals))
        & (recs["complement_id"].isin(chosen_individuals))
    ]
    train = recs[
        (~recs["anchor_id"].isin(chosen_individuals))
        & (~recs["complement_id"].isin(chosen_individuals))
    ]
    val.to_csv(f"{data_dir}/{genus}/val.csv", header=True, index=False)
    train.to_csv(f"{data_dir}/{genus}/train.csv", header=True, index=False)
