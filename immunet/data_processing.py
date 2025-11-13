import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from tqdm import tqdm
import tifffile
from csbdeep.utils import normalize
from enum import Enum
import math
from pathlib import Path
import random
from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional
from torchvision.transforms.v2 import Compose
from annotations import Tile
from tqdm import tqdm
from multiprocess import Pool
from prediction import load_component


def rnd(i):
    return int(round(i))


class Transformation(Enum):
    ROT90 = 1
    HORIZONTAL_FLIP = 2
    VERTICAL_FLIP = 3
    HORIZONTAL_SHIFT = 4
    VERTICAL_SHIFT = 5


class RandomTransformations(object):
    """Applies a random transformation given the Transformation enum (ROT90, HOR/VER flip)"""

    def __init__(self, t_type: Transformation | None, p=0.5):
        self.transform = t_type
        self.p = p

    def __call__(self, image: torch.Tensor, label: Optional[torch.Tensor] = None):
        pair_mode_flag = not isinstance(label, type(None))
        if torch.rand(1) <= self.p:
            if self.transform == Transformation.ROT90:
                # 0: no rotation, 1: 90 degrees clockwise, 2: 180 degrees, 3: 270 degrees
                rotation_direction = int(torch.randint(0, 4, (1,)).item())
                image = torch.rot90(image, rotation_direction, (1, 2))
                if pair_mode_flag:
                    label = torch.rot90(label, rotation_direction, (1, 2))
            elif self.transform == Transformation.HORIZONTAL_FLIP:
                image = functional.hflip(image)
                if pair_mode_flag:
                    label = functional.hflip(label)
            elif self.transform == Transformation.VERTICAL_FLIP:
                image = functional.vflip(image)
                if pair_mode_flag:
                    label = functional.vflip(label)
            elif self.transform == Transformation.HORIZONTAL_SHIFT:
                raise NotImplementedError()
            elif self.transform == Transformation.VERTICAL_SHIFT:
                raise NotImplementedError()

        if pair_mode_flag:
            return image, label
        else:
            return image

    def __repr__(self):
        return self.__class__.__name__ + f"(transform={self.transform},p={self.p})"


class RandomBlackoutChannels(object):
    """Randomly blackouts a channel and sets positivity to 1 (Index 0 is DAPI and cannot be blacked)"""

    def __init__(self, channels: list[int], p=0.2):
        self.p = p
        assert 0 not in channels, "Cannot blockout DAPI channel!"
        self.channels = channels

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        modified = False
        for i in self.channels:
            if torch.rand(1) <= self.p:
                modified = True
                image[i, :, :] = 0.0
                # replace  values greater than 0 with 0
                label[i, label[i, :, :] > 0.0] = 0.0

        # if there is no active phenotype, blackout the label
        # if sum([x.max() for x in label[1:]]) < 1.0 and label[1].max() > 0.0:

        # blackout dapi if all phenotype channels do not containe non-zero positive values
        # where ~x.max()~ the maximum value of a phenotype channel
        # and ~label[0].max() > 0.0~ indicates there was dapi channel
        sum_ph = sum([(1 if x.max() > 0.0 else 0) for x in label[1:]])
        if sum_ph < 1.0 and label[0].max() > 0.0:
            try:
                assert (
                    modified
                ), "Blacking out DAPI without applied augmentation, something went really wrong...."
            except Exception as A:
                raise
            # this cell is now background aka Other cell
            label[0, label[0, :, :] > 0.0] = -2

        return image, label

    def __repr__(self):
        return self.__class__.__name__ + f"(channel={self.channels},p={self.p})"


class RandomCrop(object):
    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        params = {"h_start": random.random(), "w_start": random.random()}
        return self.random_crop(
            image, self.height, self.width, **params
        ), self.random_crop(label, self.height, self.width, **params)

    def __repr__(self):
        return self.__class__.__name__ + f"(height={self.height},width={self.width})"

    @staticmethod
    def get_random_crop_coords(
        height: int,
        width: int,
        crop_height: int,
        crop_width: int,
        h_start: float,
        w_start: float,
    ) -> Tuple[int, int, int, int]:
        # h_start is [0, 1) and should map to [0, (height - crop_height)]  (note inclusive)
        # This is conceptually equivalent to mapping onto `range(0, (height - crop_height + 1))`
        # See: https://github.com/albumentations-team/albumentations/pull/1080
        y1 = int((height - crop_height + 1) * h_start)
        y2 = y1 + crop_height
        x1 = int((width - crop_width + 1) * w_start)
        x2 = x1 + crop_width
        return x1, y1, x2, y2

    @staticmethod
    def random_crop(
        img: np.ndarray | torch.Tensor,
        crop_height: int,
        crop_width: int,
        h_start: float,
        w_start: float,
    ) -> np.ndarray | torch.Tensor:
        height, width = img.shape[1:]
        if height < crop_height or width < crop_width:
            raise ValueError(
                f"Requested crop size ({crop_height}, {crop_width}) is larger than the image size ({height}, {width})",
            )
        x1, y1, x2, y2 = RandomCrop.get_random_crop_coords(
            height, width, crop_height, crop_width, h_start, w_start
        )
        return img[:, y1:y2, x1:x2]


class RandomShift(object):
    def __init__(self, shift_range: tuple[int, int], p: float = 0.5, axis=1):
        self.p = p
        self.range = shift_range
        self.axis = axis

    def __call__(self, image: torch.Tensor, label: torch.Tensor):
        if torch.rand(1) <= self.p:
            shifts = torch.randint(low=self.range[0], high=self.range[1])
            return torch.roll(image, shifts=shifts, dims=self.axis), torch.roll(
                label, shifts=shifts, dims=self.axis
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(shift_range={self.range},axis={self.axis}),p={self.p})"
        )


class ToNHWC(object):
    """Converts tensor to [C, H, W] from [H, W, C]. Warning! it expects to receive [HWC] and will not check."""

    def __call__(self, tensor: torch.Tensor):
        # tensor in (H, W, C), convert to (C, H, W)
        return tensor.permute(2, 0, 1)

    def __repr__(self):
        return self.__class__.__name__


class GaussianNoise(object):
    """Applies Gaussian noise with mean 0 and stdDev 0.2"""

    def __init__(self, std=0.1, mean=0.0, p=1.0):
        self.std = std
        self.mean = mean
        self.p = p

    def __call__(self, inputs: torch.Tensor):
        # return ins + torch.randn(ins.size()).cuda() * stddev
        if torch.rand(1) <= self.p:
            return inputs + torch.randn(inputs.size()) * self.std + self.mean
        return inputs

    def __repr__(self):
        return (
            self.__class__.__name__ + f"(std={self.std},mean={self.mean}),p={self.p})"
        )


class RandomBrightness(object):
    def __init__(
        self, multiply_range=(0.6, 2), additive_range=(-0.2, 0.2), p=1.0
    ) -> None:
        self.multiply_range = multiply_range
        self.additive_range = additive_range
        self.p = p

    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        for k in range(input.shape[0]):
            if torch.rand(1) <= self.p:
                # it comes from Stardist, random brightness with magic numbers
                scaling = torch.ones_like(input[0]).uniform_(*self.multiply_range)
                adding = torch.ones_like(input[0]).uniform_(*self.additive_range)
                input[k, :, :] = (input[k, :, :] * scaling) + adding

        return input

    def __repr__(self):
        return (
            self.__class__.__name__
            + f"(multiply_range={self.multiply_range},additive_range={self.additive_range}),p={self.p})"
        )


class CustomImageDataset(Dataset):
    """
    Custom dataset for image patches with optional preprocessing and postprocessing
    for both inputs and targets, and support for inference mode.

    Parameters
    ----------
    patches : np.ndarray or torch.Tensor
        Image data in shape (N, C, H, W).
    labels : np.ndarray or torch.Tensor
        Corresponding labels (targets).
    input_c : int
        Number of input channels.
    output_m : int
        Number of output markers/classes.
    input_preprocessing : list, optional
        List of preprocessing transforms for the input images.
    target_preprocessing : list, optional
        List of preprocessing transforms for the target labels.
    input_postprocessing : list, optional
        List of postprocessing transforms for the input images.
    target_postprocessing : list, optional
        List of postprocessing transforms for the target labels.
    pair_postprocessing : list, optional
        List of transforms applied jointly to both inputs and targets.
    debug : bool, optional
        Flag for enabling verbose/debug mode.
    inference : bool, optional
        If True, disables data augmentations during training.
    """

    def __init__(
        self,
        patches: Union[np.ndarray, torch.Tensor],
        labels: Union[np.ndarray, torch.Tensor],
        input_c: int,
        output_m: int,
        input_preprocessing: Optional[List] = None,
        target_preprocessing: Optional[List] = None,
        input_postprocessing: Optional[List] = None,
        target_postprocessing: Optional[List] = None,
        pair_postprocessing: Optional[List] = None,
        debug: bool = False,
        inference: bool = False,
    ):
        self.patches = patches
        self.labels = labels
        self.input_channels = input_c
        self.out_markers = output_m
        self.debug = debug
        self.inference = inference

        # Compose transforms if provided
        self.input_pre_t = Compose(input_preprocessing) if input_preprocessing else None
        self.target_pre_t = (
            Compose(target_preprocessing) if target_preprocessing else None
        )
        self.input_post_t = (
            Compose(input_postprocessing) if input_postprocessing else None
        )
        self.target_post_t = (
            Compose(target_postprocessing) if target_postprocessing else None
        )
        self.both_post_t = Compose(pair_postprocessing) if pair_postprocessing else None

        # Optional: Store augmentation setup summary
        self.augmentation_summary = {
            "input_preprocessing": str(input_preprocessing),
            "target_preprocessing": str(target_preprocessing),
            "input_postprocessing": str(input_postprocessing),
            "target_postprocessing": str(target_postprocessing),
            "pair_postprocessing": str(pair_postprocessing),
        }

    def get_augmentation_summary(self):
        return self.augmentation_summary

    def __len__(self):
        return len(self.patches)

    def get_unflatten_input(self, idx, postprocessing=True):
        image = self.patches[idx].copy()
        label = self.labels[idx].copy()

        if self.input_pre_t:
            image = self.input_pre_t(image)
        if self.target_pre_t:
            label = self.target_pre_t(label)

        if self.both_post_t:
            image, label = self.both_post_t(image, label)

        if postprocessing:
            if self.input_post_t:
                image = self.input_post_t(image)
            if self.target_post_t:
                label = self.target_post_t(label)

        if self.debug:
            assert (
                image.size(0) == self.input_channels
            ), f"Channels of image should come after batch dim! {image.size()}"
            assert image.size(1) >= image.size(
                2
            ), f"Height should come before width! {image.size()}"
            assert (
                label.size(0) == self.out_markers + 1
            ), f"Channels of labels should come after batch dim! {label.size()}"

        # Return modified image and y split into distances (first 9 channels) and phenotype labels (rest 45 channels)
        return image, label

    def __getitem__(self, idx):
        return self.get_unflatten_input(idx)

    def visualize_sample(
        self, idx: int, scaling=3, prediction=None, show_transformations=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Moves sample to CPU, converts to numpy and plots.

        Parameters
        ----------
        idx : int
            Index of sample to visualize.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, Micropanel]
            _description_
        """
        image, label = self.get_unflatten_input(idx, show_transformations)
        image = image.cpu().data.numpy()
        label = label.cpu().data.numpy()

        return (image, label)


def map_phenotype(c: dict, channels_num: int) -> np.ndarray:
    """Maps Likert scale to phenotype [0.0, 1.0]

    Parameters
    ----------
    c : dict
        Annotation dictionary which must include positivity field.
    channels_num : int
        Number of phenotype channels (without DAPI).

    Returns
    -------
    np.ndarray
        Phenotype array mapped from positivity,
    """
    # remove DAPI channel and slice until channels_num
    positivity = c["positivity"][:channels_num]

    # pad with 1 if positivity length is less than channels_num
    if len(positivity) < channels_num:
        positivity += list(np.ones(channels_num - len(positivity)).astype(int))

    return (np.array(positivity) - 1) / 4.0


def extract_labels(
    components: np.ndarray,
    annotations: list[dict],
    out_markers_num: int = 5,
    cell_radius: int = 5,
    remove_channels: list = [],
    background_celltypes: list[str] = [],
) -> np.ndarray:
    def background(c):
        return "background" in c and c["background"]

    """Create labels from components and annotations.

    Parameters
    ----------
    components : np.ndarray
        Component image array [`h, w, c`].
    annotations : list
        List of annotations dictionary.
    out_markers_num : int, optional
        Number of phenotype markers, by default 5.
    cell_radius : int, optional
        Radius of cells, by default 5.
    background_celltypes : list[str]
        Use this argument to override which cells are counted as background.

    Returns
    -------
    np.ndarray
        Array map with dimension [`h, w, 1 + out_markers_num`],
        where output channels are:

            [0,:,:]  -> distance map gradient to nearest cell center,
                         places without annotations have unknown status (`-1`),
                         and for background (`-2`).

            [1:,:,:] -> the rest are phenotype marker expression (range [`0`, `1`]) on corresponding channel
                         (no phenotypes equals `0`).
    """

    h = components.shape[0]
    w = components.shape[1]

    # The first output channel is a prediction of a distance map,
    # the rest are predictions of phenotype marker expression maps
    out_channels_num = 1 + out_markers_num
    out = np.zeros((h, w, out_channels_num), np.float16)
    known_status = np.zeros((h, w), np.uint8)

    # array containing cell positions where array containing cell positions where 0 ->> no cell,
    # and the rest i ->> cell number (just an order in which we got annotations)
    cell_at = np.zeros((h, w), np.uint32)
    if out_markers_num > 0:
        cell_ph = np.zeros((len(annotations), out_markers_num), np.float16)

    cell_fg = np.zeros(len(annotations))

    # iterate over annotations, assign position to cell_at
    # and phenotype to cell_ph based on the index of the cell
    for i, a in enumerate(annotations):
        # round coords to integer
        x, y = map(rnd, [a["x"], a["y"]])
        if x >= w:
            x = w - 1
        if y >= h:
            y = h - 1
        cell_at[y, x] = i + 1
        if out_markers_num > 0:
            cell_ph[i, :] = map_phenotype(a, out_markers_num)
            if len(remove_channels) > 0:
                for removal in remove_channels:
                    cell_ph[i, removal] = 0

        cell_fg[i] = not background(a)
        if len(background_celltypes) > 0:
            cell_fg[i] = not a["type"] in background_celltypes

        if out_markers_num > 0:
            # TODO: this expression can be simplified!
            sum_ph = sum(
                [
                    (1 if cell_ph[i, ch_index].max() > 0.0 else 0)
                    for ch_index in range(len(cell_ph[i]))
                ]
            )
            if sum_ph < 1.0 and cell_fg[i]:
                print(
                    "assigning background to cell because phenotypes were empty!",
                    cell_ph[i],
                    a["id"],
                    "sum pheno",
                    sum_ph,
                )
                cell_fg[i] = False

    # distance to the nearest cell (will be false in cell_at == 0 matrix) and indices
    # cell_at == 0 means >=1 - background, 0 - foreground
    # od - euclidean distance to the cell center (foreground points)
    # oi - indices of the closest cell center
    od, oi = distance_transform_edt(cell_at == 0, return_indices=True)
    in_cell = od <= cell_radius

    # for each x, y get cell index in order of cell_at (how we got annotations)
    which_cell = cell_at[oi[0][in_cell], oi[1][in_cell]] - 1

    # Sets 255 for all circles that represent cells, 0 otherwise
    known_status[in_cell] = 255

    # Distances channel
    # for places inside cells set ((cell_radius - distance)) to cell
    # so that max value were at the center of the cell
    # and -2 for background
    out[in_cell, 0] = np.where(cell_fg[which_cell], cell_radius - od[in_cell], -2)

    # Phenotypes channel - same ph value for all points inside the cell
    if out_markers_num > 0:
        out[in_cell, 1:] = cell_ph[which_cell, :]

    # places without annotations have unknown status
    out[known_status == 0, 0] = -1
    return out


def make_samples(
    threads: int,
    tiles: List[Tile],
    images_path: Path,
    window: int,
    max_examples_per_tile: int,
    in_channels_num: int,
    window_label: int = 1,
    out_markers_num: int = 5,
    cell_radius: int = 5,
    pixel_wise: bool = True,
    remove_channels: list = [],
    background_celltypes: list = [],
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample coordinates from tiles and creates patches and labels.

    Parameters
    ----------
    threads : int
        Number of threads. Use >1 for multi-threaded mode.
    tiles : List[Tile]
        List of tiles containing components path.
    images_path : Path
        Relative Path to dataset folder.
    window : int
        Size for calculating patches, where patch size is (2*window+1,2*window+1, C).
    max_examples_per_tile : int
        Upper limit for number of cell samples per tile.
    in_channels_num : int
        Number of channels in components image.
    window_label: int, optional
        Window around label, use to increase size of label.
    out_markers_num : int, optional
        Number of phenotype markers, by default 5.
    cell_radius : int, optional
        Cell size, by default 5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Returns tuple [patches, labels] [(N, 2*w+1, 2*w+1, C), (N, 2*w_l+1, 2*w_l+1, 1+out_markers_num)].
        Where `w` is window and `w_l` is window_label. NO LONGER FLATTENED!
    """
    patches_list = []
    labels_list = []

    func_inputs = [
        (
            tile,
            images_path,
            window,
            max_examples_per_tile,
            in_channels_num,
            window_label,
            out_markers_num,
            cell_radius,
            pixel_wise,
            remove_channels,
            background_celltypes,
        )
        for tile in tiles
    ]

    # single-threaded mode
    print(f"making samples with [{threads}] threads")
    if threads < 2:
        for input in tqdm(func_inputs):
            patches_and_labels = __make_a_sample(input)
            if patches_and_labels is not None:
                patches, labels = patches_and_labels
                patches_list += patches
                labels_list += labels

        patches = np.stack(patches_list)
        labels = np.stack(labels_list)

    # multi-threaded mode
    else:
        total = len(tiles)
        chunksize = max(1, len(func_inputs) // (threads * 4))

        with Pool(processes=threads) as pool:
            with tqdm(total=total) as pbar:
                results = []
                for result in pool.imap_unordered(
                    __make_a_sample, func_inputs, chunksize=chunksize
                ):
                    if result is not None:
                        results.append(result)
                    pbar.update()

        patches, labels = zip(*results)

        # Convert lists to tuples
        patches = np.concatenate(patches, axis=0)
        labels = np.concatenate(labels, axis=0)

    return patches, labels


def __make_a_sample(
    args,  # due to limitations of imap, using this single argument to wrap the rest
) -> Tuple[List[np.ndarray], List[np.ndarray]] | None:
    patches = []
    labels = []
    tile: Tile
    (
        tile,
        images_path,
        window,
        max_examples_per_tile,
        in_channels_num,
        window_label,
        out_markers_num,
        cell_radius,
        pixel_wise,
        remove_channels,
        background_celltypes,
    ) = args

    assert (
        window_label == window or window_label == 1
    ), "Error configuring window label! Should be 1 or equal to window"

    if len(tile.annotations) == 0:
        print("No annotations!")
        return None

    tile_path = tile.build_path(images_path)

    # get standardized components for this image
    if tile_path is None or not tile_path.is_file():
        raise FileNotFoundError(f"cannot find cached file for {tile_path}")

    components: np.ndarray
    components = load_component(
        components_file_path=tile_path,
        in_channels_num=in_channels_num,
        dtype="float32",
        normalize_flag=True,
    )

    h = components.shape[0]
    w = components.shape[1]
    assert (
        h > window and w > window
    ), f"Component is smaller than window! {components.shape}"

    label_maps = extract_labels(
        components=components,
        annotations=tile.annotations,
        out_markers_num=out_markers_num,
        cell_radius=cell_radius,
        remove_channels=remove_channels,
        background_celltypes=background_celltypes,
    )

    # places with an annotation
    known_status = label_maps[:, :, 0] != -1

    # sharpen cell edges a little more, assigning background
    # to values that are zero
    # NOTE/TODO: This adds a weird artifact where a circle of "background"
    # pixels is added even when the phenotype channel is 0,
    # i don't know where this comes from, maybe should be removed
    label_maps[label_maps[:, :, 0] == 0] = -1

    if pixel_wise:
        # list of [y,x] non-zero coordinates
        coordinates = np.transpose(np.nonzero(known_status))

        # sample training patches
        np.random.shuffle(coordinates)
        coordinates = coordinates[0 : int(round(coordinates.shape[0] / 8.0))]

        if max_examples_per_tile < coordinates.shape[0]:
            coordinates = coordinates[0:max_examples_per_tile]

        for i, j in coordinates:
            if i <= window or j <= window or i > h - window - 1 or j > w - window - 1:
                continue
            patches.append(
                components[i - window : i + window + 1, j - window : j + window + 1, :]
            )

            labels.append(
                label_maps[
                    i - window_label : i + window_label + 1,
                    j - window_label : j + window_label + 1,
                    :,
                ]
            )
    else:
        half_area_circle = cell_radius * cell_radius * 0.5 * math.pi
        threshold = int(math.ceil(half_area_circle))

        patch_size = window * 2
        # try to get closer to power of 2
        if (64 - patch_size) < 5:
            patch_size = 64

        # Calculate the padding required to make the image size divisible by the patch size
        padding_vertical = (math.ceil(h / patch_size) * patch_size) - h
        padding_horizontal = (math.ceil(w / patch_size) * patch_size) - w

        padding_up = padding_vertical // 2
        padding_down = padding_vertical - padding_up

        padding_left = padding_horizontal // 2
        padding_right = padding_horizontal - padding_left

        pad_width = ((padding_up, padding_down), (padding_left, padding_right), (0, 0))
        components_pad = np.pad(components, pad_width, "reflect")
        label_maps_pad = np.pad(label_maps, pad_width, "reflect")

        for i in range(components_pad.shape[0] // patch_size):  # loop over height
            for j in range(components_pad.shape[1] // patch_size):  # loop over width
                _label = label_maps_pad[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                # use only patches with at least half the annotation
                if np.sum(_label[:, :, 0] != -1) < threshold:
                    continue

                _patch = components_pad[
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches.append(_patch)
                labels.append(_label)

    # could not find valid patches/labels
    if len(patches) == 0 or len(labels) == 0:
        return None

    return patches, labels
