import argparse
from composer.utils import dist
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from pathlib import Path
from typing import cast

import logging
import torch

from train import TrainWrapper


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-file", "-y", required=True, help="Name of yaml file")
    return parser.parse_args(args=None)


if __name__ == "__main__":
    # get user arguments
    user_args = get_args()

    # yaml settings/configuration
    yaml_path = f"yamls/{user_args.yaml_file}.yaml"

    with open(yaml_path) as f:
        cfg = om.load(f)
    cfg = cast(DictConfig, cfg)

    # create a logger
    logger = logging.getLogger("species_delim")

    Path(cfg.loggers.filelogger.filename).parent.mkdir(parents=True, exist_ok=True)
    logging_f = f"{Path(cfg.loggers.filelogger.filename).parent}/genetic_dist_{cfg.run_name}.log"

    fh = logging.FileHandler(filename=logging_f)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)

    logging.basicConfig(
        level=logging.INFO,
        handlers=[fh],
    )

    if not dist.is_initialized():
        dist.initialize_dist()

    TrainWrapper(cfg, logger).train()

    logger.info("Finished training...")
    torch.cuda.empty_cache()

    dist.destroy_process_group()
