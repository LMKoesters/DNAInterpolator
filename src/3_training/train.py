from composer import Trainer
from composer.utils import reproducibility
from glob import glob
import logging
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from omegaconf import errors as omerrors
import os
from pathlib import Path
import transformers
import torch

from train_modules import (
    TrainCommons,
    update_batch_size_info,
    build_algorithm,
    build_callback,
    build_logger,
    build_optimizer,
    build_scheduler,
    cleanup_cuda,
)


class TrainWrapper:
    def __init__(self, cfg: DictConfig, logger: logging.Logger):
        reproducibility.seed_all(cfg.seed)
        cfg = update_batch_size_info(cfg)

        self.cfg = cfg
        self.logger = logger

        # Tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name,
            model_max_length=cfg.max_seq_len,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

        self.train_commons = TrainCommons(cfg, self.tokenizer, logger)

    def train(self):
        """
        Sets up datasets, the model and a Trainer object and trains the model
        """
        torch.cuda.empty_cache()

        run_name = self.cfg.run_name
        logger_base_dir = Path(self.cfg.loggers.filelogger.filename).parent
        for loocv_idx, loo_genus in enumerate(
            self.cfg.get(
                "genera_to_test",
                ["Dactylorhiza", "Lomatium", "Palaquium", "Pterocarpus"],
            )
        ):
            # change run name throughout config
            self.cfg.run_name = f"{run_name}_{loo_genus}"
            self.cfg.loggers.filelogger.filename = os.path.join(
                logger_base_dir, f"training_log_{run_name}_{loo_genus}.txt"
            )

            if len(glob(f"{self.cfg.save_folder}/*-ba21978-rank0.pt")) > 0:
                self.logger.info(f"LOOCV run {loocv_idx} - Already trained {loo_genus}")
                continue

            self.logger.info(f"LOOCV run {loocv_idx} - Leaving out {loo_genus}")
            self.logger.info(f"Config of {loo_genus}: {om.to_yaml(self.cfg)}")

            # Model
            model = self.train_commons.get_model()

            # Datasets
            train_dataloader, eval_dataloader = self.train_commons.get_dataloaders(
                genus_to_test=loo_genus
            )

            # Optimizer
            optimizer = build_optimizer(self.cfg.optimizer, model)

            # Scheduler
            try:
                scheduler = build_scheduler(self.cfg.scheduler)
            except omerrors.ConfigAttributeError:
                scheduler = None

            # Loggers
            loggers = [
                build_logger(name, logger_cfg)
                for name, logger_cfg in self.cfg.get("loggers", {}).items()
            ]

            # Callbacks
            callbacks = [
                build_callback(name, callback_cfg)
                for name, callback_cfg in self.cfg.get("callbacks", {}).items()
            ]

            # Algorithms
            algorithms = [
                build_algorithm(name, algorithm_cfg)
                for name, algorithm_cfg in self.cfg.get("algorithms", {}).items()
            ]

            # cleanup cuda
            cleanup_cuda()

            if self.cfg.get("load_path", None):
                self.logger.info(
                    f"Loading weights from {self.cfg.get('load_path', None)}..."
                )

            trainer = Trainer(
                run_name=self.cfg.run_name,
                seed=self.cfg.seed,
                model=model,
                algorithms=algorithms,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                optimizers=optimizer,
                schedulers=scheduler,
                max_duration=self.cfg.max_duration,
                eval_interval=self.cfg.eval_interval,
                progress_bar=self.cfg.progress_bar,
                log_to_console=self.cfg.log_to_console,
                console_log_interval=self.cfg.console_log_interval,
                loggers=loggers,
                callbacks=callbacks,
                precision=self.cfg.precision,
                device=self.cfg.get("device", None),
                device_train_microbatch_size=self.cfg.get(
                    "device_train_microbatch_size", "auto"
                ),
                save_folder=self.cfg.get("save_folder", None),
                save_interval=self.cfg.get("save_interval", "5ep"),
                save_num_checkpoints_to_keep=self.cfg.get(
                    "save_num_checkpoints_to_keep", -1
                ),
                save_overwrite=self.cfg.get("save_overwrite", False),
                python_log_level=self.cfg.get("python_log_level", None),
                load_path=self.cfg.get("ckpt_load_path", None),
            )

            # load weights in path if pretrained model is defined
            if self.cfg.get("load_path", None):
                trainer.state.model = self.train_commons.load_model_weights(
                    trainer.state.model, self.cfg.load_path
                )

            self.logger.info("Starting training...")
            trainer.fit()
