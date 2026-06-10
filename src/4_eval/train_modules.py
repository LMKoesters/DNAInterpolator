from composer.core import DataSpec
from composer.models import HuggingFaceModel, ComposerModel
from composer.utils import dist
import gc
from omegaconf import DictConfig
from transformers import PreTrainedTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader

import build_datasets as data_module
import model_definition as custom_models
from metrics_and_callbacks import BiologicalCosineDistance


def get_num_samples_in_batch(batch: dict) -> int:
    """
    Calculates the number of samples within a batch

    Args:
        batch: The current batch

    Returns:
        The number of samples within the batch
    """
    return batch["sample_idx"].shape[
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


def cleanup_cuda():
    """
    Helper method to cleanup cuda cache
    """
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


class TrainCommons:
    def __init__(self, cfg: DictConfig, tokenizer: PreTrainedTokenizer, logger):
        self.cfg = cfg
        self.logger = logger
        self.tokenizer = tokenizer
        self.device_batch_sizes = self.get_device_batch_sizes()

    def get_device_batch_sizes(self) -> dict[str, int]:
        """
        Retrieves the device batch size for training

        Returns:
            The device batch sizes for training/inference
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
        self, dataset: torch.utils.data.Dataset, use_data_spec: bool = True
    ) -> DataSpec | DataLoader:
        """
        Sets up the dataloader for model training/testing

        Args:
            dataset: The training/testing dataset
            use_data_spec: Whether to wrap the dataloader inside a DataSpec (for correct number of samples)

        Returns:
            A dataloader
        """
        self.logger.info(
            f"Building test dataloader with a device batch size of {self.device_batch_sizes['test']}..."
        )
        dataloader = data_module.build_dataloader(
            dataset, self.cfg, self.tokenizer, self.device_batch_sizes["test"]
        )
        self.logger.info(f"Test dataloader has {len(dataloader)} batches per epoch...")

        if use_data_spec:
            dataloader = DataSpec(
                dataloader=dataloader, get_num_samples_in_batch=get_num_samples_in_batch
            )
        return dataloader

    def renew_dataloader(
        self, dataloader: DataLoader | DataSpec
    ) -> DataLoader | DataSpec:
        """
        Renews the dataloader

        Args:
            dataloader: The initial dataloader

        Returns:
            A reset dataloader
        """
        return self.setup_dataloader(
            dataloader.dataloader.dataset, self.device_batch_sizes["test"]
        )

    def get_dataloader(self, genus_to_test: str) -> DataLoader | DataSpec:
        """
        Sets up the test dataloader

        Args:
            genus_to_test: The current genus to test

        Returns:
            The test dataloader
        """
        test_dataloader = self.setup_dataloader(
            data_module.build_dataset(
                self.cfg,
                self.tokenizer,
                genus_to_test=genus_to_test,
                logger=self.logger,
            ),
            use_data_spec=False,
        )
        return test_dataloader

    def get_model(self) -> ComposerModel:
        """
        Sets up a model using weights from DNABERT-2 for the transformer
        while adding a contrast head within GeneticDistanceModel

        Returns:
            A transformer model with a contrast head
        """
        self.logger.info("Initializing DNABert-2...")
        hf_model = AutoModel.from_pretrained(
            self.cfg.model.pretrained_model_name, trust_remote_code=True
        )

        hf_model.pooler = None  # we don't use the pooler in our model

        metrics = [BiologicalCosineDistance()]

        # Turn into composer model
        model = HuggingFaceModel(
            model=hf_model, tokenizer=self.tokenizer, metrics=metrics, use_logits=True
        )

        model = custom_models.GeneticDistanceModel(
            model.model,
            self.tokenizer,
            self.logger,
            feat_dim=self.cfg.get("feat_dim", 64),
        )

        return model

    def load_model_weights(self, model: ComposerModel, load_path: str) -> ComposerModel:
        """
        Loads model weights from a saved model

        Args:
            model: The model to load weights for
            load_path: The path to the saved model

        Returns:
            The model with updated weights
        """
        self.logger.info(f"Loading weights from {load_path}...")
        state = torch.load(load_path, weights_only=True, map_location="cuda:0")
        state = {
            k.replace("module.", "", 1): v for k, v in state.items()
        }  # .replace(".bert.", ".dnabert.", 1)

        model.load_state_dict(state, strict=True)
        self.logger.info("Successfully loaded weights...")
        return model
