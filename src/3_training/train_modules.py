from composer import algorithms
from composer.core import DataSpec
from composer.models import HuggingFaceModel, ComposerModel
from composer.callbacks import (
    LRMonitor,
    RuntimeEstimator,
)
from composer.loggers import TensorboardLogger, FileLogger
from composer.optim import DecoupledAdamW
from composer.optim.scheduler import (
    LinearWithWarmupScheduler,
)
from composer.utils import dist
import gc
from omegaconf import DictConfig
import transformers
import torch
from torch.utils.data import DataLoader
from typing import Optional
import build_datasets as data_module
import model_definition as custom_models
from metrics_and_callbacks import (
    CosineDistanceLoss,
    BiologicalDistance,
    SaveAfterEpochCallback,
    SaveBestContrastiveModelCallback,
    EmptyCacheBeforeEval
)


def get_num_samples_in_batch(batch: dict) -> int:
    """
    Calculates the number of samples within a batch

    Args:
        batch: The current batch

    Returns:
        The number of samples within the batch
    """
    return batch["distance"].shape[
        0
    ]  # // 2 # need to divide by 2 because anchor + complement


def update_batch_size_info(cfg: DictConfig) -> DictConfig:
    """
    Updates the device (micro)batch size based on the world size.

    Args:
        cfg: The configuration provided by the user

    Returns:
        The updated configuration
    """
    global_batch_size, device_microbatch_size = (
        cfg.global_train_batch_size,
        cfg.device_train_microbatch_size,
    )
    if global_batch_size % dist.get_world_size() != 0:
        raise ValueError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_train_batch_size = global_batch_size // dist.get_world_size()
    if isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_train_batch_size:
            print(
                f"WARNING: device_train_microbatch_size > device_train_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_train_batch_size}."
            )
            device_microbatch_size = device_train_batch_size
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_microbatch_size
    # Safely set `device_eval_batch_size` if not provided by user
    if "device_eval_batch_size" not in cfg:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_batch_size = 1
        else:
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
    return cfg


def build_algorithm(name: str, kwargs):
    """
    Creates algorithms for Trainer

    Args:
        name: name of algorithm to apply
        kwargs: Description
    """
    if name == "alibi":
        return algorithms.Alibi(**kwargs)
    elif name == "low_precision_layernorm":
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise ValueError(f"Not sure how to build algorithm: {name}")


def build_callback(name: str, kwargs):
    """
    Creates callbacks for Trainer

    Args:
        name: name of callback to apply
        kwargs: Description
    """
    if name == "lr_monitor":
        return LRMonitor()
    elif name == "runtime_estimator":
        return RuntimeEstimator()
    elif name == "checkpoint_saver_contrastive":
        return SaveBestContrastiveModelCallback(save_path=kwargs.get("save_path"))
    elif name == "epoch_saving":
        return SaveAfterEpochCallback(save_path=kwargs.get("save_path"))
    elif name == "empty_cache_eval":
        return EmptyCacheBeforeEval()
    else:
        raise ValueError(f"Not sure how to build callback: {name}")


def build_logger(name: str, kwargs):
    """
    Creates loggers for Trainer

    Args:
        name: name of the logger to apply
        kwargs: Description
    """
    if name == "filelogger":
        return FileLogger(**kwargs)
    else:
        raise ValueError(f"Not sure how to build logger: {name}")


def build_scheduler(cfg: DictConfig):
    """
    Creates scheduler for Trainer

    Args:
        cfg: Scheduler config provided by input YAML
    """
    if cfg.name == "linear_decay_with_warmup":
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise ValueError(f"Not sure how to build scheduler: {cfg.name}")


def build_optimizer(cfg: DictConfig, model: torch.nn.Module):
    """
    Creates optimizer for Trainer

    Args:
        cfg: Scheduler config provided by input YAML
        model: model to train
    """
    if cfg.name == "decoupled_adamw":
        return DecoupledAdamW(
            model.parameters(),
            lr=cfg.lr,
            betas=cfg.betas,
            eps=cfg.eps,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Not sure how to build optimizer: {cfg.name}")


def cleanup_cuda():
    """
    Helper method to cleanup cuda cache
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class TrainCommons:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: transformers.PreTrainedTokenizer,
        logger,
    ):
        self.cfg = cfg
        self.logger = logger
        self.tokenizer = tokenizer
        self.device_batch_sizes = self.get_device_batch_sizes()

    def get_device_batch_sizes(self) -> dict[str, int]:
        """
        Retrieves the device batch size for training

        :return: The device batch sizes for training/inference
        :rtype: dict[str, int]
        """
        device_train_batch_size = (
            self.cfg.global_train_batch_size // dist.get_world_size()
        )
        device_eval_batch_size = self.cfg.get(
            "device_eval_batch_size", device_train_batch_size
        )
        return {
            "test": device_eval_batch_size,
            "train": device_train_batch_size,
            "val": device_eval_batch_size,
        }

    def setup_dataloader(
        self, dataset: torch.utils.data.Dataset, split: str, use_data_spec: bool = True
    ) -> DataSpec | DataLoader:
        """
        Sets up the dataloader for model training/testing

        :param dataset: The training/testing dataset
        :type dataset: torch.utils.data.Dataset
        :param split: either train or val
        :type split: str
        :param use_data_spec: Whether to wrap the dataloader inside a DataSpec (for correct number of samples)
        :type use_data_spec: bool
        :return: A dataloader
        :rtype: Any
        """
        self.logger.info(
            f"Building {split} dataloader with a device batch size of {self.device_batch_sizes[split]}..."
        )
        dataloader = data_module.build_dataloader(
            dataset,
            split,
            self.cfg,
            self.tokenizer,
            self.device_batch_sizes[split],
        )
        self.logger.info(
            f"{split} dataloader has {len(dataloader)} batches per epoch..."
        )

        if use_data_spec:
            dataloader = DataSpec(
                dataloader=dataloader, get_num_samples_in_batch=get_num_samples_in_batch
            )
        return dataloader

    def renew_dataloader(
        self, dataloader: DataSpec, split: str
    ) -> DataSpec | DataLoader:
        """
        Renews the dataloader

        :param self: Description
        :param dataloader: The initial dataloader
        :type dataloader: DataLoader | DataSpec
        :return: A reset dataloader
        :rtype: Any
        """
        return self.setup_dataloader(
            dataloader.dataloader.dataset, split, self.device_batch_sizes[split]
        )

    def get_dataloaders(
        self,
        genus_to_test: Optional[str] = None,
    ):
        """
        Sets up the test dataloader

        :param genus_to_test: The current genus to test
        :type genus_to_test: str
        :return: The test dataloader
        :rtype: Any
        """
        train_dataloader = self.setup_dataloader(
            data_module.build_dataset(
                "train",
                self.cfg,
                self.tokenizer,
                genus_to_test=genus_to_test,
                logger=self.logger,
            ),
            "train",
        )

        eval_dataloader = self.setup_dataloader(
            data_module.build_dataset(
                "val",
                self.cfg,
                self.tokenizer,
                genus_to_test=genus_to_test,
                logger=self.logger,
            ),
            "val",
        )
        return train_dataloader, eval_dataloader

    def get_model(self):
        """
        Sets up a model using weights from DNABERT-2 for the transformer
        while adding a contrast head within GeneticDistanceModel

        :return: A transformer model with a contrast head
        :rtype: Any
        """
        self.logger.info("Initializing DNABert-2...")
        hf_model = transformers.AutoModel.from_pretrained(
            self.cfg.model.pretrained_model_name,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )

        hf_model.pooler = None  # we don't use the pooler in our model
        hf_model.encoder.gradient_checkpointing = True
        hf_model.config.gradient_checkpointing = True

        loss_fn = CosineDistanceLoss()
        metrics = [BiologicalDistance(loss_fn)]

        # Turn into composer model
        model = HuggingFaceModel(
            model=hf_model, tokenizer=self.tokenizer, metrics=metrics, use_logits=True
        )

        model = custom_models.GeneticDistanceModel(
            model.model,
            loss_fn,
            self.tokenizer,
            self.logger,
            feat_dim=self.cfg.get("feat_dim", 64),
        )

        return model

    def load_model_weights(self, model: ComposerModel, load_path: str) -> ComposerModel:
        """
        Loads model weights from a saved model

        :param model: The model to load weights for
        :type model: ComposerModel
        :param load_path: The path to the saved model
        :type load_path: str
        :return: The model with updated weights
        :rtype: Any
        """
        self.logger.info(f"Loading weights from {load_path}...")
        state = torch.load(load_path, weights_only=True)
        state = {
            k.replace(".bert.", ".dnabert.", 1).replace(".model.", ".", 1): v
            for k, v in state.items()
        }

        model.load_state_dict(state, strict=False)
        self.logger.info("Successfully loaded weights...")
        return model
