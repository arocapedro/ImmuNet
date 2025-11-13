import contextlib
import gzip
import json
import os
import gzip
import argparse
import datetime
import shutil
import numpy as np
from pathlib import Path

from annotations import load_annotations
from train_utils import (
    EarlyStopper,
    EnhancedJSONEncoder,
    HistoryVis,
    TrainConfig,
    TrainingResult,
    add_file_handler_to_logger,
    get_labels_stats,
    set_up_training,
    build_model,
    build_loss_list,
    build_optimizer,
    end_timer_and_print,
    make_dataloader,
    start_timer,
    test_model_input,
    Backbone,
    DEVICE_NUMBER,
    DEVICE_TYPE_STR,
)

from typing import List, Optional, Tuple
from models import BaseImmunet, ImmuNet

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import numpy as np
from tqdm import tqdm
import logging
from config import IMAGES_FOLDER, TRAIN_ANNOTATONS_PATH, TRAIN_OUTPUT_PATH

torch.backends.cudnn.benchmark = True

# configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)


logger = logging.getLogger(__name__)


######## CONSTANTS #######
WINDOW = 31
MAX_EXAMPLES_PER_TILE = 1000
BATCH_SIZE = 64
IN_CHANNELS_NUM = 7
OUT_MARKERS_NUM = 5
CELL_RADIUS = 5
EPOCHS = 500

SEED = 42


def run_training(config: TrainConfig) -> TrainingResult:
    device, scaler = set_up_training(logger, config)

    model = build_model(config, device)
    optimizer = build_optimizer(config, model)
    loss_fn = build_loss_list(config, device)

    logger.info(
        f"BACKBONE {config.backbone}, Sparse: {not (config.backbone == 'ORIGINAL' or config.force_pixelwise)} | debug {config.debug}",
    )

    logger.info("loading training annotations...")
    train_annotations = load_annotations(config.train_annotations_path)
    if config.debug:
        train_annotations = train_annotations[:40]
    train_cells_num = sum([len(tile.annotations) for tile in train_annotations])

    train_loader = make_dataloader(train_annotations, config, train=True)
    val_loader = None
    if config.val_annotations_path:
        val_ann = load_annotations(config.val_annotations_path)
        if config.debug:
            val_ann = val_ann[:20]
        val_loader = make_dataloader(val_ann, config, train=False)
    elif config.debug:
        val_loader = make_dataloader(train_annotations, config, train=False)

    model_path = f"{config.model_path}.pth"
    logger.info(f"path to save model is at {model_path}")

    early_stopper = EarlyStopper(
        patience=config.early_stopper_patience, min_delta=config.early_stopper_delta
    )

    # TEST model
    logger.info("Testing input, evaluating on a sample...")
    dataiter = iter(train_loader.dataset)
    image: torch.Tensor = next(dataiter)[0]
    image = image.unsqueeze(0).to(config.device)

    test_model_input(model, config, device, image)

    # only create folders after the tests pass!
    os.makedirs(config.output_folder, exist_ok=True)
    add_file_handler_to_logger(config)

    training_config.augmentations = train_loader.dataset.get_augmentation_summary()

    logger.info("{} cells for training".format(train_cells_num))
    logger.info(f"image type and size {image.dtype}, {image.shape}")

    (n_neg, n_pos, total_annotated, total) = get_labels_stats(train_loader)
    logger.info(f"[Train] Annotated pixels: {total_annotated/total:.2%}%")
    logger.info(f"[Train] Background pixels: {n_neg/total_annotated:.2%}%")
    logger.info(f"[Train] Foreground pixels: {n_pos/total_annotated:.2%}%")

    if val_loader:
        (n_neg, n_pos, total_annotated, total) = get_labels_stats(val_loader)
        logger.info(f"[Val] Annotated pixels: {total_annotated/total:.2%}%")
        logger.info(f"[Val] Background pixels: {n_neg/total_annotated:.2%}%")
        logger.info(f"[Val] Foreground pixels: {n_pos/total_annotated:.2%}%")

    positive_weight = None
    if config.positive_weights:
        # binary_weights["pos"] = total_annotated / (2 * n_pos)
        # binary_weights["neg"] = total_annotated / (2 * n_neg)
        positive_weight = n_neg / n_pos
        logger.info(
            f"Assigning weight to positive class: {positive_weight} ({n_neg}/{n_pos}). Rebuilding loss..."
        )
        if config.loss_function != "BCE":
            raise ValueError
        loss_fn = build_loss_list(
            config, device, pos_weight=torch.tensor([positive_weight])
        )

    with open(
        training_config.output_folder
        / f"training_config_{training_config.current_date}.json",
        "w",
    ) as f:
        json.dump(vars(training_config), f, indent=2, cls=EnhancedJSONEncoder)

    # set up logging
    tb_dir = config.output_folder.parent / os.path.join(
        "runs",
        f"{config.current_date}_{config.experiment_keyword}_{config.batch_size}{'_debug' if config.debug else ''}",
    )
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(tb_dir)
    history_tracker = HistoryVis(tb_dir)
    if config.history_path:
        history_tracker.from_file(config.history_path)

    # Track the PyTorch model architecture
    if config.log_graph:
        dummy = torch.randn(
            1,
            config.in_channels_num,
            config.window * 2 + 1,
            config.window * 2 + 1,
            device=device,
        )
        writer.add_graph(model, dummy, use_strict_trace=False)

    # RUN PARAMS
    epoch_number = 1
    best_vloss = 1_000_000.0
    best_epoch = 1

    model.to(config.device)
    torch.autograd.set_detect_anomaly(mode=config.debug)

    start_timer(config.device)
    # Loop over epochs
    for epoch_number in range(1, config.epochs + 1):
        print("EPOCH {}/{}".format(epoch_number, config.epochs))
        # logger.info(f"Epoch {epoch_number}/{config.epochs}")

        train_loss = run_one_epoch(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            loss_weights=config.loss_weights,
            optimizer=optimizer,
            epoch_index=epoch_number,
            writer=writer,
            backbone=config.backbone,
            scaler=scaler,
            autocast_type=config.autocast_type,
            device=config.device,
            device_type=config.device_type,
            weighted_loss=config.weighted_loss,
            pixelwise=config.force_pixelwise,
            mode="train",
        )

        val_loss = 0.0
        if val_loader:
            val_loss = run_one_epoch(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                loss_weights=config.loss_weights,
                writer=writer,
                backbone=config.backbone,
                autocast_type=config.autocast_type,
                device=config.device,
                device_type=config.device_type,
                weighted_loss=config.weighted_loss,
                pixelwise=config.force_pixelwise,
                epoch_index=epoch_number,
                mode="eval",
            )

        if np.isnan(train_loss) | np.isnan(val_loss):
            raise RuntimeError(
                "Loss exploded! train loss [{train_loss}] | val loss [{val_loss}]"
            )

        logger.info(
            "LOSS [train {}] [val {}]".format(
                train_loss, val_loss if val_loader else "n/a"
            )
        )

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Loss/Weighted_Epoch",
            {"train_loss": train_loss, "validation_loss": val_loss},
            epoch_number,
        )

        history_tracker.log_metrics(
            epoch_number, train_loss, val_loss if val_loader else None
        )

        writer.flush()

        loss_to_track = val_loss if val_loader else train_loss

        # save best & latest
        model_path = config.output_folder / f"{config.model_name}.pth"
        if loss_to_track < best_vloss:
            best_vloss = loss_to_track
            best_epoch = epoch_number
            torch.save(model.state_dict(), model_path)
            logger.info(f"Epoch [{epoch_number}] Saved new best model to {model_path}")

        # save latest model always
        torch.save(
            model.state_dict(), config.output_folder / f"latest_{config.model_name}.pth"
        )

        if early_stopper.early_stop(loss_to_track):
            print(f"Early stopping after {early_stopper.counter} / {epoch_number}")
            break

    # Finish training!
    writer.close()

    total_time = end_timer_and_print(
        local_msg=f"{'Mixed' if config.autocast else 'Default'} precision:",
        device=device,
    )
    max_mem = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    logger.info("Done!")

    return TrainingResult(
        number_of_epochs=epoch_number,
        total_time=total_time,
        number_of_training_samples=len(train_loader.sampler),
        number_of_validation_samples=len(val_loader.sampler) if val_loader else 0,
        saved_model_at_epoch=best_epoch,
        max_memory=max_mem,
    )


def loss_mask(
    y_dist: torch.Tensor, y_pheno: torch.Tensor, sparse=False, weighted=False
):
    mask_dist = torch.ones_like(y_dist)
    mask_pheno = torch.ones_like(y_pheno)

    if weighted:
        eps = 0.01
        # distance shouldn't be masked
        mask_pheno = torch.zeros_like(y_pheno)
        mask_pheno[
            (torch.abs(y_pheno - 0.00) < eps) | (torch.abs(y_pheno - 1.00) < eps)
        ] = 1.0
        mask_pheno[
            (torch.abs(y_pheno - 0.25) < eps) | (torch.abs(y_pheno - 0.75) < eps)
        ] = 0.5

    if sparse:
        # only count annotated pixels torwards loss
        mask_dist *= (y_dist != -1).float()
        mask_pheno *= (y_dist != -1).float()

    return mask_dist, mask_pheno


def compute_dapi_loss(
    prediction: torch.Tensor,
    label: torch.Tensor,
    loss_fn: List[nn.Module],
    device: str,
) -> torch.Tensor:

    y_dist = label.to(device)

    # declare masks and weight
    weighted_mask = torch.ones_like(y_dist)

    # mask non-annotated pixels
    weighted_mask *= (y_dist != -1).float()

    # convert to binary labels
    y_dist = (y_dist > 0).float()

    loss_dist = loss_fn[0](prediction, y_dist)

    # apply mask / weights to loss
    loss_dist = loss_dist * weighted_mask

    # reduce loss using mask as number of elements
    loss_dist = loss_dist.sum() / weighted_mask.sum()

    return loss_dist


def calculate_loss(
    predictions: Tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    loss_fn: List[nn.Module],
    loss_weights: List[int],
    device: str,
    pixelwise: bool,
    backbone: Backbone,
    weight_mask=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculates the weighted loss between predictions and labels.

    Parameters
    ----------
    predictions : Tuple[torch.Tensor, torch.Tensor]
        Tuple containing distance predictions and phenotype predictions.
    labels : torch.Tensor
        Ground truth labels with shape (batch, channels, height, width). The first channel
        is assumed to be distance, the remaining channels phenotype data.
    loss_fn : List[nn.Module]
        List containing loss functions for distance and phenotype respectively.
    loss_weights : List[float]
        Weights to apply to distance and phenotype losses when computing total loss.
    device : str
        Device on which to perform computations (e.g., 'cpu' or 'cuda').
    sparse_mask : bool, optional
        Whether to apply a sparse mask to the loss computation (default is False).
    weight_mask : bool, optional
        Whether to apply a weighted mask to the loss computation (default is False).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tuple containing:
        - total_loss: weighted sum of distance and phenotype losses (scalar tensor)
        - loss_dist: distance loss (scalar tensor)
        - loss_pheno: phenotype loss (scalar tensor)
    """
    sparse_mask = (
        backbone == Backbone.UNET and not pixelwise
    )  # if usine pixelwise, this shouldn't make such a difference
    if backbone == Backbone.DAPI:
        return (
            compute_dapi_loss(
                prediction=predictions[0],
                label=labels,
                loss_fn=loss_fn,
                device=device,
            ),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )

    # unpack tuples
    pred_dist, pred_pheno = predictions

    y_dist: torch.Tensor = labels[:, 0, :, :]
    y_pheno: torch.Tensor = labels[:, 1:, :, :]

    # add empty dimension (after batch) to match num of axis when masking
    y_dist = y_dist.unsqueeze(1)

    y_dist, y_pheno = y_dist.to(device), y_pheno.to(device)

    loss_dist = loss_fn[0](pred_dist, y_dist)
    loss_pheno = loss_fn[1](pred_pheno, y_pheno)

    # apply mask / weights to loss
    if sparse_mask or weight_mask:
        mask_dist, mask_pheno = loss_mask(
            y_dist=y_dist, y_pheno=y_pheno, sparse=sparse_mask, weighted=weight_mask
        )
        loss_dist = loss_dist * mask_dist
        loss_pheno = loss_pheno * mask_pheno

        # reduce loss using mask as number of elements
        loss_dist = loss_dist.sum() / mask_dist.sum()
        loss_pheno = loss_pheno.sum() / mask_pheno.sum()

    else:
        loss_dist = loss_dist.mean()
        loss_pheno = loss_pheno.mean()

    # Compute the loss and its gradients
    total_loss = (loss_weights[0] * loss_dist) + (loss_weights[1] * loss_pheno)

    return total_loss, loss_dist, loss_pheno


def run_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: list,
    loss_weights: list[int],
    epoch_index: int,
    writer: Optional[SummaryWriter],  # tensorboard writer
    device: str,
    device_type: str,
    autocast_type: torch.dtype,
    backbone: Backbone,
    weighted_loss: bool,
    pixelwise: bool,
    scaler: Optional[torch.amp.GradScaler] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    mode="train",
):
    if mode not in ["train", "eval"]:
        raise ValueError(f"Invalid mode {mode}! Valid are: [train, eval]")

    train_mode = mode == "train"

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(train_mode)

    if not train_mode and isinstance(model, ImmuNet):
        # remove window padding when validating, only relevant when backbone is ORIGINAL
        model.window = 0

    steps_per_epoch = len(dataloader)
    total_num_samples = len(dataloader.sampler)

    # only report on first epoch!
    if epoch_index == 0:
        logger.info(
            f"[{mode.upper()}] Mini-batches/steps per epoch: {steps_per_epoch}. Total samples {total_num_samples}"
        )

    # Trackers for losses
    sum_loss = 0.0
    sum_dist = 0.0
    sum_pheno = 0.0

    progress_bar = tqdm(
        dataloader,
        total=steps_per_epoch,
        unit="batch",
        bar_format="{desc:<7.5} {percentage:3.0f}%|{bar:20}{r_bar}",
    )

    # Here, we use enumerate(dataloader) instead of
    # iter(dataloader) so that we can track the batch
    # index and do some intra-epoch reporting
    context = torch.no_grad() if not train_mode else contextlib.nullcontext()
    with context:
        for batch_idx, (inputs, labels) in enumerate(progress_bar, 1):
            progress_bar.set_description(f"Epoch {epoch_index}")

            batch_size = inputs.size(0)

            inputs = inputs.to(device)  # float32 [B, C, H, W]
            labels = labels.to(device)  # float16

            if train_mode:
                # Zero your gradients for every batch (even first backward)
                optimizer.zero_grad(set_to_none=True)

            # Runs the forward pass under ``autocast``.
            # it will downcast when numerically stable for some operations
            with torch.autocast(
                enabled=True, device_type=device_type, dtype=autocast_type
            ):
                # Make predictions for this batch, aka forward pass
                predictions = model(inputs)

                loss, loss_dist, loss_pheno = calculate_loss(
                    predictions,
                    labels,
                    loss_fn,
                    loss_weights=loss_weights,
                    weight_mask=weighted_loss,
                    device=device,
                    backbone=backbone,
                    pixelwise=pixelwise,
                )

            if train_mode:
                scaler.scale(loss).backward()

                # Adjust learning weights
                scaler.step(optimizer)
                scaler.update()

            ## Gather data and report ##
            # Accumulate for epoch‐wide average
            sum_loss += loss.item() * batch_size
            sum_dist += loss_dist.item() * batch_size
            sum_pheno += loss_pheno.item() * batch_size

            # log every mini-batch
            tb_step = epoch_index * total_num_samples + batch_idx
            if writer:
                mode_tag = "Train" if train_mode else "Val"
                writer.add_scalar(f"Loss/{mode_tag}", loss.item(), tb_step)
                writer.add_scalar(
                    f"Loss/Distance/{mode_tag}",
                    loss_dist.item(),
                    tb_step,
                )
                writer.add_scalar(
                    f"Loss/Phenotype/{mode_tag}",
                    loss_pheno.item(),
                    tb_step,
                )
                if (epoch_index - 1) % 10 == 0 and backbone == Backbone.DAPI:
                    sample_idx = torch.randint(low=0, high=batch_size - 1, size=(1,))[0]
                    sample_input = inputs[sample_idx]
                    sample_label = labels[sample_idx]
                    model.eval()
                    with torch.no_grad():
                        sample_pred_dist, sample_pred_pheno = model(
                            sample_input.unsqueeze(1)
                        )
                        sample_pred_dist = sample_pred_dist[0]

                    img_batch = torch.zeros((3, 3, *sample_pred_dist.shape[1:]))
                    # print("Size of image ", img_batch.shape)
                    img_batch[0, 0] = sample_input
                    img_batch[1, 1] = sample_label
                    img_batch[2, 2] = sample_pred_dist
                    writer.add_images(
                        "[INPUT, LABEL, PREDICTION]",
                        img_batch.cpu().detach().numpy(),
                        epoch_index,
                    )

        progress_bar.set_postfix(
            loss=loss.item(), dist_loss=loss_dist.item(), pheno_loss=loss_pheno.item()
        )

    # https://discuss.pytorch.org/t/on-running-loss-and-average-loss/107890
    # [...] multiplying the averaged batch loss by the batch size and dividing by the number of samples
    # gives you the correct average sample loss for this particular epoch.
    # The [...] approach of dividing the averaged batch loss by the number of batches would yield the same result,
    # if each batch in the epoch contains batch_size samples.
    # This might not always be the case, if the length of the dataset is not divisible by the batch_size without a remainder.
    # The last batch would thus contain less samples and the loss calculation would introduce a small bias. - ptrblck
    epoch_avg_loss = sum_loss / total_num_samples

    return epoch_avg_loss


def parse_args() -> TrainConfig:

    def path_type(input_str):
        if input_str is None:
            return input_str
        _path = Path(input_str)
        if not _path.exists():
            raise ValueError(f"Path not found! {input_str}")
        return _path

    parser = argparse.ArgumentParser(description="Training Configuration")

    # General Settings
    general_group = parser.add_argument_group("General Settings")
    general_group.add_argument("--seed", type=int, default=SEED, help="Random seed")
    general_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable additional checks and extra logging",
    )

    general_group.add_argument(
        "--output_folder",
        "-f",
        type=path_type,
        default=TRAIN_OUTPUT_PATH,
        help="Output folder for training artifacts",
    )
    general_group.add_argument(
        "--experiment_keyword",
        "-k",
        type=str,
        default="demo",
        help="Keyword to differentiate this experiment",
    )
    general_group.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="current_best_model",
        help="Model checkpoint filename",
    )
    general_group.add_argument(
        "--weights_path",
        "-w",
        type=path_type,
        default=None,
        help="Path to model weights for continued training",
    )
    general_group.add_argument(
        "--history_path",
        type=path_type,
        default=None,
        help="Path to training history for continued training",
    )

    # Data Settings
    data_group = parser.add_argument_group("Data Settings")
    data_group.add_argument(
        "--train_ann_path",
        "-t",
        dest="train_annotations_path",
        type=path_type,
        default=TRAIN_ANNOTATONS_PATH,
        help="Path to training annotations JSON",
    )
    data_group.add_argument(
        "--val_ann_path",
        "-v",
        dest="val_annotations_path",
        type=path_type,
        help="Path to validation annotations JSON",
    )
    data_group.add_argument(
        "--images_path",
        "-im",
        type=path_type,
        default=IMAGES_FOLDER,
        help="Path to image folder",
    )
    data_group.add_argument(
        "--cell_radius",
        "-r",
        type=int,
        default=5,
        help="Cell radius in pixels for label generation",
    )
    data_group.add_argument(
        "--remove_channel",
        dest="remove_channels",
        action="append",
        type=int,
        default=[],
        help="Channels to set as negative (zero-based index after DAPI)",
    )
    data_group.add_argument(
        "--background_types",
        action="append",
        type=str,
        default=[],
        help="Overrides annotation background assigment and uses the provide value instead (f.g., you want to declare T cells as background cells, and / or Tumor cell as foreground)",
    )

    # Model Settings
    model_group = parser.add_argument_group("Model Settings")
    model_group.add_argument(
        "--backbone",
        type=str,
        default="ORIGINAL",
        choices=["ORIGINAL", "UNET", "DAPI"],
        help="Model backbone architecture []",
    )
    model_group.add_argument(
        "--in_channels_num",
        "-c",
        type=int,
        default=IN_CHANNELS_NUM,
        help="Number of input channels",
    )
    model_group.add_argument(
        "--out_markers_num",
        "-o",
        type=int,
        default=5,
        help="Number of phenotype markers to predict",
    )
    model_group.add_argument(
        "--num_features",
        dest="backbone_features_size",
        type=int,
        default=-1,
        help="Number of features in the last layer of the backbone. For DAPI default is [32] and for UNET [128]",
    )
    model_group.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Starting number of channels in the CNNs (`N,Channels,H,W`) for UNET/DAPI networks. This value adjusts the total number of parameters in the network.",
    )
    model_group.add_argument(
        "--window", type=int, default=WINDOW, help="Window size for input patches"
    )

    # Training Settings
    training_group = parser.add_argument_group("Training Settings")
    training_group.add_argument(
        "--epochs", "-e", type=int, default=EPOCHS, help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    training_group.add_argument(
        "--val_batch_size", type=int, default=BATCH_SIZE, help="Validation batch size"
    )
    training_group.add_argument(
        "--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate"
    )
    training_group.add_argument(
        "--optimizer_name",
        type=str,
        default="adam",
        help="Optimizer name (refer to PyTorch documentation)",
    )
    training_group.add_argument(
        "--loss_function",
        type=str,
        default="MSE",
        choices=["MSE", "BCE"],
        help="Loss function (refer to PyTorch documentation)",
    )
    training_group.add_argument(
        "--weighted_loss",
        type=str,
        default="on",
        choices=["on", "off"],
        help="Applies weighs to the loss based on the confidence from the Likert-scale, where annotations with `5` and `1` have full weight, annotations with `3` are ignored and the rest is weighted by half",
    )
    training_group.add_argument(
        "--positive_weights",
        action="store_true",
        help="(Specially for DAPI backbone) Class weighting to trade off recall and precision by adding weights to positive examples [total_negatives / total_positives].",
    )
    training_group.add_argument(
        "--max_examples_per_tile",
        type=int,
        default=MAX_EXAMPLES_PER_TILE,
        help="Maximum examples per tile",
    )
    training_group.add_argument(
        "--autocast",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enables/disables autocast",
    )
    training_group.add_argument(
        "--log_graph",
        type=str,
        default="off",
        choices=["on", "off"],
        help="Enables/disables model graph in tensorboard",
    )

    # Early Stopping Settings
    early_stop_group = parser.add_argument_group("Early Stopping Settings")
    early_stop_group.add_argument(
        "--early_stopper_patience",
        type=float,
        default=50,
        help="Epochs without improvement before stopping",
    )
    early_stop_group.add_argument(
        "--early_stopper_delta",
        type=float,
        default=1e-3,
        help="Minimum delta for improvement",
    )

    # Advanced Settings
    advanced_group = parser.add_argument_group("Advanced Settings")
    advanced_group.add_argument(
        "--force_pixelwise",
        action="store_true",
        help="Force pixelwise data even if backbone is not ORIGINAL",
    )
    advanced_group.add_argument(
        "--freeze_backbone", action="store_true", help="Freeze backbone during training"
    )
    advanced_group.add_argument(
        "--freeze_distance",
        action="store_true",
        help="Freeze distance output during training",
    )
    advanced_group.add_argument(
        "--freeze_phenotype",
        action="store_true",
        help="Freeze phenotype output during training",
    )
    advanced_group.add_argument(
        "--single_thread",
        action="store_true",
        help="Disables parallel data loading",
    )
    advanced_group.add_argument(
        "--device_type",
        choices=["cpu", "cuda", "mps"],
        default=DEVICE_TYPE_STR,
        help="Select device type",
    )
    advanced_group.add_argument(
        "--device_id",
        default=DEVICE_NUMBER,
        type=int,
        help="Select device ID, if available",
    )

    args = parser.parse_args()

    assert os.path.exists(
        args.output_folder
    ), f"Output folder does not exist! [{args.output_folder}]"
    if args.force_pixelwise or args.backbone == Backbone.ORIGINAL:
        if args.window != 31:
            raise ValueError(
                "Window size must be 31 when using 'ORIGINAL' backbone or 'force_pixelwise' is set."
            )

    if args.backbone == "DAPI":
        if args.out_markers_num > 0:
            raise ValueError(
                f"{args.backbone} backbone cannot use phenotype predictions, please use out_markers_num == 0!"
            )
        if args.in_channels_num > 1:
            raise ValueError(
                f"{args.backbone} backbone shouldn't use more than 1 input for predictions, please use in_channels_num == 1!"
            )
        if len(args.background_types) == 0:
            raise ValueError(
                f"{args.backbone} backbone needs to override background types. Please specify background_types=['No cell']`"
            )

    # validate freeze flags
    if args.freeze_backbone and args.freeze_distance and args.freeze_phenotype:
        raise ValueError("All parts frozen: nothing to train")

    # validate remove_channels
    if args.remove_channels and max(args.remove_channels) >= args.out_markers_num:
        raise ValueError(
            "remove_channels index out of range, should be less than out_markers_num"
        )

    current_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_folder = args.output_folder / f"{current_date}_{args.experiment_keyword}"

    DEVICE_STR = (
        f"{args.device_type}:{args.device_id}"
        if args.device_type != "cpu"
        else args.device_type
    )

    backbone_feature_size = None
    # Default values!
    if args.backbone_features_size < 0:
        if args.backbone == "UNET":
            backbone_feature_size = 128
        elif args.backbone == "DAPI":
            backbone_feature_size = 32
    else:
        backbone_feature_size = args.backbone_features_size

    # Make necessary conversions to arguments
    training_config = vars(args)
    training_config_modification = {
        "output_folder": output_folder,
        "model_path": os.path.join(output_folder, args.model_name),
        "backbone_features_size": backbone_feature_size,
        "remove_channels": args.remove_channels if args.remove_channels else [],
        "loss_weights": [1, 20],  # [distance, phenotype]
        "freeze_batchnorm": False,
        # offers significant performance improvement but causes problems, dont use ?
        "autocast": args.autocast == "on",
        # since labels as float16, upcast to float32 them if autocast is disabled
        "autocast_type": (torch.float16 if args.autocast == "on" else torch.float32),
        "device": DEVICE_STR,
        "device_type": args.device_type,
        "epochs": args.epochs if not args.debug else 5,
        "augmentations": {},
        "backbone": Backbone[args.backbone],
    }
    for key in training_config_modification.keys():
        training_config[key] = training_config_modification[key]

    training_config["current_date"] = current_date

    logger.info(
        f'Autocast [{training_config["autocast"]}], {training_config["autocast_type"]}'
    )
    print("Training config:\n", training_config, "\n")
    return TrainConfig(**training_config)


if __name__ == "__main__":
    training_config = parse_args()

    try:
        with torch.device(training_config.device):
            train_result = run_training(training_config)
    except KeyboardInterrupt:
        logger.info("Manually stopped training!")
        logger.info("Done!")

        with open(
            training_config.output_folder
            / f"training_config_{training_config.current_date}_stopped_early.json",
            "w",
        ) as f:
            json.dump(vars(training_config), f, indent=2, cls=EnhancedJSONEncoder)
    else:
        logger.info("saving configs...")
        with open(
            training_config.output_folder
            / f"training_config_{training_config.current_date}.json",
            "w",
        ) as f:
            json.dump(
                vars(training_config) | vars(train_result),
                f,
                indent=2,
                cls=EnhancedJSONEncoder,
            )

    logger.info("saving annotations...")
    # save annotation file used for training
    with open(training_config.train_annotations_path, "rb") as f_in:
        with gzip.open(
            os.path.join(
                training_config.output_folder,
                f"annotations_training_{training_config.current_date}.json.gz",
            ),
            "wb",
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

    logger.info(
        f"Using val annotations? { training_config.val_annotations_path is not None}"
    )
    if (
        training_config.val_annotations_path is not None
        and training_config.backbone != Backbone.DAPI
    ):
        with open(training_config.val_annotations_path, "rb") as f_in:
            with gzip.open(
                os.path.join(
                    training_config.output_folder,
                    f"annotations_validation_{training_config.current_date}.json.gz",
                ),
                "wb",
            ) as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info(
            f'found model path: {os.path.exists(training_config.model_path + ".pth")} {training_config.model_path}',
        )
