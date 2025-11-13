from os import PathLike
import os
from pathlib import Path
import imageio
import numpy as np
import tifffile
import torch
from typing import Optional, Union
from csbdeep.utils import normalize
from models import BaseImmunet
import random
from scipy.spatial import KDTree
from annotations import load_annotations
from tqdm import tqdm
import pandas as pd
from models import load_model
from config import IMAGES_FOLDER
from skimage.feature import blob_log
from skimage import draw


class ImmuNetPredictionHandler:
    def __init__(
        self,
        model_name: str,
        model_path: str | Path,
        device: str,
        ph_thresh: float = 0.4,
        log_threshold: float = 0.07,
        min_sigma_log: int = 2,
        max_sigma_log: int = 5,
        dist_multiplier: int = 50,
        backbone: str = "ORIGINAL",
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.log_threshold = log_threshold
        self.ph_thresh = ph_thresh
        self.min_sigma_log = min_sigma_log
        self.max_sigma_log = max_sigma_log
        self.dist_multiplier = dist_multiplier
        self.backbone = backbone
        self.device = device
        self.model = load_model(
            which_model=self.model_path, device=device, backbone=backbone
        )
        if "cuda" in device:
            print("loaded model!", torch.cuda.memory_allocated(device=device))

    def phenotype_output(self, phenotype):
        return ",".join(["%.2f" % i for i in phenotype])

    def get_model_output(self, dataset, slide, tile):
        components = load_component(
            image_folder_path=IMAGES_FOLDER,
            dataset_id=dataset,
            slide_id=slide,
            tile_id=tile,
            in_channels_num=self.model.n_inputs,
            dtype="float32",
        )
        ph = create_pheno_image(
            components=components, model=self.model, device=self.device
        )

        return ph

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def get_prediction(self, dataset, slide, tile):
        data, _, _ = self.get_pos_pheno(self.get_model_output(dataset, slide, tile))
        return data

    def get_pos_pheno(self, ph, has_pheno_output=True):
        if isinstance(ph, type(None)):
            return [], -1, []
        ## !!! THIS CODE EXPECTS TO HAVE CHANNEL FIRST AGAIN FOR SOME REASON !!!
        channel_dim = 0
        if len(ph.shape) > 3:
            channel_dim = 1

        ph = np.moveaxis(ph, -1, channel_dim)
        assert ph.shape[channel_dim] < 10, f"Input must be C H W! {ph.shape}"

        distance = scale_image(ph[0], scalar=self.dist_multiplier)
        blobs = [
            (int(x[0]), int(x[1]))
            for x in blob_log(
                distance.astype("uint8"),
                min_sigma=self.min_sigma_log,
                max_sigma=self.max_sigma_log,
                threshold=self.log_threshold,
            )
        ]

        data = []
        for x in blobs:
            rr, cc = draw.disk((x[0], x[1]), radius=2, shape=distance.shape)
            if has_pheno_output:
                phen = [float(np.mean(ph[j, rr, cc])) for j in range(1, ph.shape[0])]

                # Filter out by minimum threhold of predicted phenotype
                if sum(np.array(phen) > self.ph_thresh) > 0:
                    data.append((x, phen))
            else:
                pred = float(np.mean(self.sigmoid(ph[0, rr, cc])))
                if pred > self.ph_thresh:
                    data.append((x, 0))

        return data, distance, ph[1:]


def create_pheno_image(
    components: np.ndarray,
    model,
    device: str,
    max_input_size=(996, 1332),
    window=30,
) -> np.ndarray:
    torch.cuda.empty_cache()

    # At the moment only support splitting an input image that is too large into halves
    # TODO: allow to call this recursively?
    if (
        components.shape[0] > max_input_size[0]
        or components.shape[1] > max_input_size[1]
    ):
        # Split an input image by x axis
        half_width = float(components.shape[1] / 2) + window
        half_width_multiple_8 = int(8 * round(half_width / 8)) - window

        # Increase width for of both halves by window size for a more accurate inference
        half1 = components[:, 0 : (half_width_multiple_8 + window), :]
        half2 = components[:, half_width_multiple_8 - window :, :]
        del components

        pred1 = get_model_output(model, half1, device)
        # del half1
        pred2 = get_model_output(model, half2, device)
        # del half2

        ph = np.concatenate(
            (pred1[:, :half_width_multiple_8, :], pred2[:, window:, :]), axis=1
        )
    else:
        ph = get_model_output(model, components, device)
        del components

    return np.array(ph)


def match_cells(
    annotations_path,
    prediction_handler: ImmuNetPredictionHandler,
    fout,
    out_markers_num=5,
    device: str = "cuda:0",
):
    """
    Matches annotations with predictions and saves results to a TSV file efficiently.
    Uses pandas for buffering and periodic flushing.
    """
    tiles = load_annotations(annotations_path)
    print("Matching", len(tiles), "tiles")
    random.shuffle(tiles)

    columns = [
        "panel",
        "dataset",
        "id",
        "ann_type",
        "ann_pheno",
        "pred_pheno",
        "distance",
    ]

    if os.path.exists(fout):
        answer = input(
            f"Prediction file already exists at {fout}. Do you want to overwrite? (Y/N)"
        )
        if answer.lower()[0] == "y":
            pass
        else:
            print("Skipping overwritting prediction...")
            return fout

    with open(fout, "w", buffering=1) as f:
        # Write header
        pd.DataFrame(columns=columns).to_csv(f, sep="\t", index=False)

        buffer = []
        print("Predicting:")

        for tile in tqdm(tiles):
            dataset_id = tile.dataset_id or ""
            slide_id = tile.slide_id or ""
            tile_id = tile.id
            panel = tile.panel or ""
            annotations = tile.annotations

            try:
                prediction = prediction = prediction_handler.get_prediction(
                    dataset_id, slide_id, tile_id
                )
            except FileNotFoundError as e:
                print(repr(e), dataset_id, slide_id, tile_id)
                continue

            if not prediction:
                prediction = [[[-100000, -100000], [0] * out_markers_num]]

            coords = np.array([cell[0] for cell in prediction])
            tr = KDTree(coords)

            for ann in annotations:
                # --- Safe key access (data integrity improvement) ---
                ann_id = ann.get("id", "")
                ann_type = ann.get("type", "")
                ann_pheno = ann.get("positivity", [0] * out_markers_num)
                ann_x = ann.get("x")
                ann_y = ann.get("y")

                if ann_x is None or ann_y is None:
                    print(f"Warning: missing coordinates in annotation {ann_id}")
                    continue

                dist, idx = tr.query([ann_y, ann_x])
                pred_pheno = prediction_handler.phenotype_output(prediction[idx][1])

                buffer.append(
                    [
                        panel,
                        dataset_id,
                        ann_id,
                        ann_type,
                        ",".join([f"{v:g}" for v in ann_pheno]),
                        pred_pheno,
                        f"{dist:.1f}",
                    ]
                )

            # Flush every tile
            pd.DataFrame(buffer, columns=columns).to_csv(
                f, sep="\t", index=False, header=False
            )
            buffer.clear()

        # Write remaining rows
        if buffer:
            pd.DataFrame(buffer, columns=columns).to_csv(
                f, sep="\t", index=False, header=False
            )

    print(f"Finished writing results to {fout}")
    return fout


def get_model_output(
    model: BaseImmunet,
    input_image_numpy: Union[torch.Tensor, np.ndarray],
    device: str = "cuda:0",
) -> np.ndarray:
    """Returns prediction image from model.

    Parameters
    ----------
    model : ImmuNet
        Model to run prediction.
    input_image : Union[torch.Tensor, np.ndarray]
        Input image prefered in shape [N, C, H, W]
    device : str
        CPU or CUDA (example: cuda:0)

    Returns
    -------
    np.ndarray
        Image prediction shape [N, input.shape[0], input.shape[1], 1 + out_markers] if input had a batch dimension, otherwise
        [input.shape[0], input.shape[1], 1 + out_markers]
    """
    input_image = torch.tensor(input_image_numpy, device=device)
    del input_image_numpy

    is_batch_mode = len(input_image.shape) > 3

    if not is_batch_mode:
        # add batch dimension
        input_image = input_image.unsqueeze(0)

    is_channels_first = input_image.shape[1] == model.n_inputs

    if not is_channels_first:
        input_image = input_image.permute(0, 3, 1, 2)

    assert not torch.isnan(input_image).any(), "Found NaN values on component..."

    with torch.inference_mode():
        with torch.autocast(
            enabled=True, device_type=torch.device(device).type, dtype=torch.float32
        ):
            d, u = model(input_image)

    del input_image

    if torch.isnan(d).any():
        raise RuntimeWarning("Encountered nan values in output!")

    if torch.isnan(u).any():
        raise RuntimeWarning("Encountered nan values in output!")

    im = torch.concat((d.detach(), u.detach()), dim=1)
    del d, u
    # permute back
    if not is_channels_first:
        im = im.permute(0, 2, 3, 1)

    im = im.detach().cpu().data.numpy()

    if not is_batch_mode:
        im = im[0]

    return im


def load_component(
    dataset_id: Optional[str] = None,
    slide_id: Optional[str] = None,
    tile_id: Optional[str] = None,
    image_folder_path: Optional[str | PathLike] = None,
    components_file_path: Optional[str | PathLike] = None,
    filename=Path("components.tiff"),
    in_channels_num: int = 7,
    dtype="float32",
    normalize_flag=True,
    ch_reordering: Optional[list] = None,
    channel_removals: Optional[list] = None,
) -> np.ndarray:
    """Load pipeline for a component file. Returns file with channels last [_, _, C], where `C = in_channels_num`.

    Parameters
    ----------

    Returns
    -------
    np.ndarray
        Components data, channels last format. Dtype float32 by default.

    Raises
    ------
    FileNotFoundError
        If path is invalid.
    """

    if None in [dataset_id, slide_id, tile_id]:
        if components_file_path is None:
            raise ValueError(
                "Please provide either dataset and slide and tile or components_file_path argument!"
            )
        else:
            filename = Path(components_file_path)
    else:
        assert image_folder_path, "Please provide image folder path!"
        tile_folder_path = Path(image_folder_path) / dataset_id / slide_id / tile_id
        filename = tile_folder_path / filename

    if not filename.exists():
        raise FileNotFoundError(
            "No file found in tile folder. There should be exactly one file with components name at {} {}.".format(
                filename, tile_folder_path
            )
        )

    if filename.suffix in [".tiff", ".tif"]:
        components = tifffile.imread(filename, key=range(0, in_channels_num))
    else:
        components = imageio.imread(filename)

    if normalize_flag:
        components = normalize(components)

    channel_index = list(components.shape).index(in_channels_num)
    if channel_index != components.shape[-1]:
        components = np.moveaxis(components, channel_index, -1)

    if ch_reordering is not None:
        components = components[[int(i) for i in ch_reordering]]

    # TODO
    if channel_removals is not None:
        raise NotImplementedError()

    components = components.astype(dtype)

    return components


def scale_image(image: np.ndarray, scalar: int, low_clip=0, up_clip=255) -> np.ndarray:
    """
    _summary_

    Parameters
    ----------
    image : np.ndarray
        _description_
    scalar : int
        _description_
    low_clip : int, optional
        _description_, by default 0
    up_clip : int, optional
        _description_, by default 255

    Returns
    -------
    np.ndarray
        _description_
    """
    image = scalar * image
    image[image < low_clip] = low_clip
    image[image > up_clip] = up_clip
    return image
