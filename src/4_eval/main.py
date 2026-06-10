import argparse
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from pathlib import Path
from typing import cast

import logging

from model_embeddings import EmbeddingWrapper
from calc_pred_dist import DistWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", "-y", type=str, help="Yaml file name")
    return parser.parse_args()


if __name__ == "__main__":
    # get user arguments
    user_args = get_args()

    # yaml settings/configuration
    yaml_path = f"yamls/{user_args.yaml}.yaml"

    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)  # for type checking
    if "loo_runs" not in cfg:
        cfg.loo_runs = cfg.genera_to_test

    # create a logger
    logger = logging.getLogger("speciesdelim_evaluate")
    Path(cfg.loggers_dir).mkdir(parents=True, exist_ok=True)
    logging_f = f"{Path(cfg.loggers_dir)}/{cfg.run_name}.log"

    fh = logging.FileHandler(filename=logging_f)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[fh],
    )

    logger.info("Calculating embeddings...")
    EmbeddingWrapper(cfg.copy(), logger).eval_test()
    if cfg.model_type == "custom":
        logger.info("Predicting distances...")
        DistWrapper(cfg.copy(), logger).calc_dists()
    logger.info("Finished...")
