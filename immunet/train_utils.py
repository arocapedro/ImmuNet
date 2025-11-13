import datetime
import gc
import json
import logging
import time

from enum import Enum

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from matplotlib import pyplot as plt

import subprocess as sp
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import v2
from tqdm import tqdm
from models import BaseImmunet, ImmuNet, UImmNet, DAPImmuNet
from annotations import Tile
from data_processing import (
    CustomImageDataset,
    GaussianNoise,
    RandomBrightness,
    RandomTransformations,
    RandomBlackoutChannels,
    RandomCrop,
    ToNHWC,
    Transformation,
    make_samples,
)

DEVICE_NUMBER = 0
DEVICE_TYPE_STR = (
    "cuda"
    if torch.cuda.is_available()
    # else "mps" if torch.backends.mps.is_available() else "cpu"
    else "cpu"
)

start_time = 0
logger = logging.getLogger(__name__)


class Backbone(Enum):
    ORIGINAL = 1
    UNET = 2
    DAPI = 3


## Check argparse in train.py for explanation of each field!
@dataclass
class TrainConfig:
    debug: bool
    train_annotations_path: Path
    val_annotations_path: Path
    images_path: Path
    in_channels_num: int
    out_markers_num: int
    cell_radius: int
    epochs: int
    output_folder: Path
    model_name: str
    model_path: str
    current_date: str
    freeze_batchnorm: str
    experiment_keyword: str
    batch_size: int
    val_batch_size: int
    seed: int
    window: int
    augmentations: dict
    optimizer_name: str
    freeze_backbone: bool
    freeze_distance: bool
    freeze_phenotype: bool
    backbone: Backbone
    backbone_features_size: int
    base_channels: int
    force_pixelwise: bool
    loss_function: str
    loss_weights: List[int]
    remove_channels: List[int]
    learning_rate: float
    max_examples_per_tile: int
    early_stopper_patience: int
    early_stopper_delta: float
    weights_path: Path
    history_path: Path
    autocast: bool
    log_graph: bool
    autocast_type: torch.dtype
    single_thread: bool
    device: str
    device_type: str
    device_id: int
    weighted_loss: bool
    positive_weights: bool
    background_types: List[str]


@dataclass
class TrainingResult:
    number_of_epochs: int
    total_time: str
    number_of_training_samples: int
    number_of_validation_samples: int
    saved_model_at_epoch: int
    max_memory: int


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Path, np.dtype, torch.dtype, Backbone)):
            return str(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


class HistoryVis:
    def __init__(self, history_name):
        super().__init__()
        self.losses = []
        self.val_losses = []
        self.history_name = history_name

    def from_file(self, file_path):
        import json

        with open(file_path) as f:
            file_json = json.load(f)
        self.losses = file_json["loss"]
        if "val_losss" in file_json:
            self.val_losses = file_json["val_losses"]

    def log_metrics(self, epoch, loss, val_loss=None):
        # print(f"logging metrics at ({self.history_name})...")
        self.losses.append(loss)
        if val_loss:
            self.val_losses.append(val_loss)
        # Plot train history
        self.plot_train_history(self.losses, self.val_losses, self.history_name)
        # Save train history
        with open("{}.json".format(self.history_name), "w") as f:
            hist_dict = {"loss": self.losses}
            if len(self.val_losses) > 0:
                hist_dict["val_loss"] = self.val_losses
            json.dump(hist_dict, f)

    def plot_train_history(self, losses, val_losses=None, file_name="history"):
        plt.figure()
        plt.plot(losses)
        has_val_losses = val_losses is not None and len(val_losses) > 0
        if has_val_losses:
            plt.plot(val_losses)
        plt.title("Training history")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        legend = ["train", "val"] if has_val_losses else ["train"]
        plt.legend(legend, loc="upper right")
        plt.savefig("{}.png".format(file_name))
        plt.close()


class EarlyStopper:
    def __init__(self, patience=1, min_delta=1e-2, start_from_epoch=None):
        self.patience = patience
        # A minimum increase in the score to qualify as an improvement,
        #  i.e. an increase of less than or equal to min_delta, will count as no improvement.
        self.min_delta = abs(min_delta)
        self.counter = 0
        self.min_loss = float("inf")
        self.start_from_epoch = start_from_epoch
        self.epoch_number = 1

    def _is_improvement(self, loss, reference_value):
        return (loss + self.min_delta) < reference_value

    def early_stop(self, loss):
        if self._is_improvement(loss, self.min_loss):
            # loss improved
            logger.info(
                f"[{self.epoch_number}] loss improved by {self.min_loss - loss} after {self.counter} epochs!"
            )
            self.min_loss = loss
            self.counter = 0
        else:
            print(
                f"running out of patience... {self.counter}/{self.patience} [loss: {loss}, best: {self.min_loss}]"
            )
            # loss did not improve
            self.counter += 1
            if self.counter >= self.patience:
                return True

        self.epoch_number += 1
        return False


def process_annotations(
    train_annotations: list[Tile],
    images_path: Path,
    max_examples_per_tile: int,
    in_channels_num: int,
    out_markers_num: int,
    cell_radius: int,
    window: int,
    window_label: int,
    batch_size: int,
    multithread: bool,
    augmentations=True,
    pixel_wise=True,
    remove_channels=[],
    background_celltypes=[],
    debug=False,
) -> DataLoader[CustomImageDataset]:
    """Creates Dataloader from list of Tiles, loaded beforehand using `load_annotations`.

    Parameters
    ----------
    train_annotations : list[Tile]
        List of Tiles
    images_path : Path
        Path to find component images
    max_examples_per_tile : int
        Maximum amount of examples per each tile
    in_channels_num : int
        Channel length of component images
    out_markers_num : int
        Output length of each label (how many markes in annotation phenotype)
    cell_radius : int
        Radius of label around cell center position
    window : int
        _description_
    window_label : int
        _description_
    batch_size : int
        _description_
    augmentations : bool, optional
        _description_, by default True
    pixel_wise : bool, optional
        _description_, by default True
    remove_channels : list, optional
        List used to ignore channels during training, by default []
    background_celltypes : list, optional
        Overrides the "background" field of an annotation, and gets assigned based on the type defined in this list.

    Returns
    -------
    tuple[DataLoader[CustomImageDataset], int]
        DataLoader and number of steps per epoch (length of samples divided by batch size)
    """
    logger.info(f"Processing annotations! Pixel wise: {pixel_wise}")
    num_threads = min(20, len(train_annotations)) if multithread else 1

    start_timer()
    patches, labels = make_samples(
        threads=num_threads,
        tiles=train_annotations,
        images_path=images_path,
        window=window,
        max_examples_per_tile=max_examples_per_tile,
        in_channels_num=in_channels_num,
        window_label=window_label,
        out_markers_num=out_markers_num,
        cell_radius=cell_radius,
        pixel_wise=pixel_wise,
        remove_channels=remove_channels,
        background_celltypes=background_celltypes,
    )

    if not pixel_wise:
        assert (
            patches.shape[1] == labels.shape[1]
        ), f"Patches: {patches.shape}, Labels: {labels.shape}"

    end_timer_and_print(
        f"Finished making samples! Multi-thread [{multithread}] - Augmentations: {augmentations}"
    )
    print("Number of patches/labels: ", len(patches))

    # Get augmentations
    input_aug, pair_aug = get_augmentations(augmentations)

    # Create dataset
    dataset = make_custom_dataset(
        patches=patches,
        labels=labels,
        in_channels_num=in_channels_num,
        out_markers_num=out_markers_num,
        input_visual_augmentation=input_aug,
        pair_augmentation=pair_aug,
        debug=debug,
    )

    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=augmentations,
        pin_memory=True,
        num_workers=8,
    )

    print("created dataset loaders...")
    return dataloader


def start_timer(device: Optional[str] = None):
    global start_time
    if device and "cuda" in device:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated(device=device)
        torch.cuda.synchronize(device=device)
    start_time = time.time()


def end_timer_and_print(local_msg, device=None):
    if device and "cuda" in device:
        torch.cuda.synchronize(device=device)
    end_time = time.time()
    logger.info("\n" + local_msg)
    logger.info("Total execution time = {:.3f} sec".format(end_time - start_time))
    if device and "cuda" in device:
        logger.info(
            "Max memory used by tensors = {} bytes".format(
                torch.cuda.max_memory_allocated(device=device)
            )
        )
    return "{:.3f}".format(end_time - start_time)


def get_gpus_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")
    memory_free_values = [int(x) for x in memory_free_info if x != ""]
    return memory_free_values


def build_output_folder(base: Path, keyword: str) -> Tuple[Path, str]:
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = base / f"{now}_{keyword}"
    out.mkdir(parents=True, exist_ok=True)
    return out, now


def set_up_training(
    logger, config: TrainConfig
) -> tuple[torch.device, torch.amp.GradScaler]:
    torch.manual_seed(seed=config.seed)
    device = torch.device(DEVICE_TYPE_STR)
    device_id = DEVICE_NUMBER

    if "cuda" in DEVICE_TYPE_STR:

        logger.info(f"Using device: {device} | {device_id}")
        # Check if selected GPU is busy
        used_memory_list = get_gpus_memory()
        selected_gpu_memory = used_memory_list[device_id]
        for i, mem in enumerate(used_memory_list):
            selected_string = ""
            if i == device_id:
                selected_string = "<-- [SELECTED]"
            logger.info(
                f"[{i}] {torch.cuda.get_device_name(i)}: used memory: {mem} {selected_string}"
            )
            if mem > (selected_gpu_memory * 1.2):
                raise Exception(
                    f"Currently GPU seems busy ({selected_gpu_memory} MB), perhaps try GPU[{i}]: {mem} MB"
                )

    # Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
    # If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
    # The same ``GradScaler`` instance should be used for the entire convergence run.
    # If you perform multiple convergence runs in the same script, each run should use
    # a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
    scaler = torch.amp.GradScaler(enabled=config.autocast)

    return device, scaler


def build_model(config: TrainConfig, device: torch.device) -> nn.Module:
    # model selection
    if config.backbone == Backbone.ORIGINAL:
        model = ImmuNet(
            n_inputs=config.in_channels_num,
            n_markers=config.out_markers_num,
            debug_flag=config.debug,
        ).to(device)
        logger.info(
            f"ImmuNet: n_inputs: {config.in_channels_num}, n_markers:{config.out_markers_num}"
        )
    elif config.backbone == Backbone.UNET:
        model = UImmNet(
            n_inputs=config.in_channels_num,
            n_markers=config.out_markers_num,
            debug_flag=config.debug,
            num_features=config.backbone_features_size,
            base_channels=config.base_channels,
        ).to(device)
        logger.info(
            f"UImmuNet: n_inputs: {config.in_channels_num}, n_markers:{config.out_markers_num}, num_features: {config.backbone_features_size}, base_channels: {config.base_channels}"
        )
    elif config.backbone == Backbone.DAPI:
        if config.freeze_distance or config.freeze_phenotype:
            raise ValueError(
                f"Cannot freeze distance branches while using {config.backbone} backbone!"
            )
        model = DAPImmuNet(
            num_features=config.backbone_features_size,
            base_channels=config.base_channels,
        ).to(device)
        logger.info(
            f"DAPImmuNet: num_features: {config.backbone_features_size}, base_channels: {config.base_channels}"
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")
    logger.info(model.print_network())
    # optionally load weights
    if config.weights_path:
        if not config.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {config.weights_path}")
        state = torch.load(config.weights_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"Loaded weights from {config.weights_path}")

    # freeze batchnorm if requested
    if config.freeze_batchnorm:
        raise NotImplementedError("Haven't tested this!")
        # for child in model.children():
        #         if isinstance(child, nn.BatchNorm2d) != -1:
        #             for param in child.parameters():
        #                 param.requires_grad = True
        #         else:
        #             for param in child.parameters():
        #                 param.requires_grad = False
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = False

    # freeze other parts
    if config.freeze_backbone:
        model.freeze_backbone()
    if config.freeze_distance:
        model.toggle_gradients_distance(freeze=True)
    if config.freeze_phenotype:
        model.toggle_gradients_phenotype(freeze=True)

    # test model with dummy input to make sure everything is correct!
    test_model_input(model, config, device, img=None)

    return model


def build_loss_list(
    config: TrainConfig, device: torch.device, **args
) -> list[nn.Module]:
    # loss
    loss_fn: list
    reduction = "none" if config.weighted_loss else "mean"
    logger.info(f"Using reduction: {reduction}")

    if config.loss_function == "MSE":
        loss_fn = [
            nn.MSELoss(reduction=reduction, **args).to(device),
            nn.MSELoss(reduction=reduction, **args).to(device),
        ]
    elif config.loss_function == "BCE":
        loss_fn = [nn.BCEWithLogitsLoss(reduction=reduction, **args).to(device)]
    else:
        raise ValueError(f"Unsupported loss: {config.loss_function}")
    return loss_fn


def build_optimizer(config: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    # optimizer
    if config.optimizer_name.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config.learning_rate, momentum=0.9
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")
    return optimizer


def make_dataloader(
    annotations, config: TrainConfig, train: bool
) -> DataLoader[CustomImageDataset]:

    batch_size = config.batch_size if train else config.val_batch_size
    loader = process_annotations(
        train_annotations=annotations,
        images_path=config.images_path,
        max_examples_per_tile=config.max_examples_per_tile,
        in_channels_num=config.in_channels_num,
        out_markers_num=config.out_markers_num,
        cell_radius=config.cell_radius,
        window=config.window,
        window_label=(config.window if config.backbone != Backbone.ORIGINAL else 1),
        batch_size=batch_size,
        augmentations=train,  # Use deterministic validation set!
        pixel_wise=(config.backbone == Backbone.ORIGINAL or config.force_pixelwise),
        remove_channels=config.remove_channels if train else [],
        multithread=not config.single_thread,
        background_celltypes=config.background_types,
    )
    if len(loader) == 0:
        raise ValueError(f"No {'training' if train else 'validation'} data found!")
    return loader


def get_augmentations(
    augmentations: bool,
) -> Tuple[Optional[list], Optional[list]]:
    """
    Returns input and pair augmentation pipelines.
    """
    if augmentations:
        pair_aug = [
            # Geometric data augmentation!
            RandomTransformations(
                t_type=Transformation.ROT90,
                p=0.5,
            ),
            RandomTransformations(
                t_type=Transformation.HORIZONTAL_FLIP,
                p=0.5,
            ),
            RandomTransformations(
                t_type=Transformation.VERTICAL_FLIP,
                p=0.5,
            ),
            # RandomBlackoutChannels(channels=[1, 2], p=0.2),
        ]

        input_aug = [
            RandomBrightness(),
            GaussianNoise(std=0.1),
        ]

        return input_aug, pair_aug
    else:
        return None, None


def make_custom_dataset(
    patches,
    labels,
    in_channels_num: int,
    out_markers_num: int,
    input_visual_augmentation: Optional[list],
    pair_augmentation: Optional[list],
    debug: bool = False,
) -> CustomImageDataset:
    """
    Create a CustomImageDataset with preprocessing and augmentations.
    """
    # Transformations to adapt input for ImmuNet
    preprocessing = [
        torch.from_numpy,  # convert to torch.Tensors
        ToNHWC(),  # Correct channel ordering!
    ]

    dataset = CustomImageDataset(
        patches=patches,
        labels=labels,
        input_c=in_channels_num,
        output_m=out_markers_num,
        input_preprocessing=preprocessing,
        input_postprocessing=input_visual_augmentation,
        target_preprocessing=preprocessing,
        pair_postprocessing=pair_augmentation,
        debug=debug,
    )

    # Get one sample to check everything is correct!
    dataset.visualize_sample(-1)

    return dataset


def test_model_input(
    model: torch.nn.Module, config, device: torch.device, img=None
) -> None:
    """
    Runs a single forward pass on a dummy (or real) sample to verify
    that the model's outputs have the shapes you expect.
    Raises an AssertionError if anything is off.
    """
    model.eval()

    patch_size = (config.window * 2) + (
        1 if (config.backbone == Backbone.ORIGINAL or config.force_pixelwise) else 2
    )
    if isinstance(img, type(None)):
        # reate a dummy tensor with the right shape:
        img = torch.randn(
            1, config.in_channels_num, patch_size, patch_size, device=device
        )

    logger.info(f"input shape {img.shape}")

    logger.info(
        f"Testing model forward with input shape {tuple(img.shape)} | {img.dtype}"
    )
    with torch.no_grad():
        distance, phenotype = model(img)

    expected_shape = (
        1,
        1,
        patch_size,
        patch_size,
    )
    # assert shapes
    assert (
        distance.shape == expected_shape
    ), f"Distance output shape mismatch: got {distance.shape}, expected {expected_shape}"

    if config.backbone != Backbone.DAPI:
        assert phenotype.shape == (
            1,
            config.out_markers_num,
            patch_size,
            patch_size,
        ), f"Phenotype output shape mismatch: got {phenotype.shape}"

    logger.info("Model input test passed.")

    model.train()


def add_file_handler_to_logger(config):
    # could be improved!
    rootLogger = logging.getLogger("__main__")

    logFormatter = logging.Formatter(
        "%(asctime)s [%(name)s] [%(levelname)-5.5s]:  %(message)s"
    )
    fileHandler = logging.FileHandler(
        "{0}/{1}.log".format(config.output_folder, "output")
    )
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    logger.addHandler(fileHandler)


def get_labels_stats(
    dataloader: DataLoader[CustomImageDataset],
) -> Tuple[int, int, int, int]:
    """Calculates foreground / background statistics for assesing the number of positives vs negatives examples.

    Parameters
    ----------
    dataloader : DataLoader
        Dataloader

    Returns
    -------
    Tuple[int, int, int, int]
        _description_
    """

    total = 0
    total_annotated = 0
    n_neg = 0
    n_pos = 0

    for _, labels in tqdm(dataloader):
        total += torch.numel(labels)
        total_annotated += torch.numel(labels[labels != -1])
        n_neg += torch.numel(labels[labels == -2])
        n_pos += torch.numel(labels[labels >= 0])

    return n_neg, n_pos, total_annotated, total
