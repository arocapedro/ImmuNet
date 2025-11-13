import os
import sys
import numpy as np
import tifffile
from csbdeep.utils import normalize
from skimage.feature import blob_log
from skimage import draw
import argparse
from PIL import Image
from config import *
from models import load_model
import torch

# The colors used for the prediction visualisation
# Change according to the number of channels in your panel and preferred colors
panel_colors = [[255, 0, 0], [0, 255, 0], [255, 0, 255], [255, 255, 0], [0, 255, 255]]


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
            image_folder_path=IMAGE_PATH,
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


def mix(colors, fields):
    im = np.zeros(fields[0].shape + (3,))
    mxf = np.zeros(fields[0].shape)
    for i in range(len(fields)):
        exm = fields[i] > mxf
        exmf = fields[i][exm]
        for j in range(3):
            im[exm, j] = exmf * colors[i][j]
        mxf[exm] = exmf
    return im


def get_input(tile_path, in_channels_num):
    tile = tifffile.imread(tile_path, key=range(0, in_channels_num))
    return np.moveaxis(normalize(tile), 0, -1)


def scale_image(image, scalar, low_clip=0, up_clip=255):
    image = scalar * image
    image[image < low_clip] = low_clip
    image[image > up_clip] = up_clip
    return image


def predict(input, model, dist_scalar=50):
    dist_map, pheno_map = get_prediction(model, input)

    dist_map = dist_map[0][:, :, 0]
    scale_image(dist_map, dist_scalar)

    pheno_map = pheno_map[0]
    return dist_map, pheno_map


def find_cells(
    tile_path, model, log_threshold=0.07, min_log_std=3, max_log_std=5, dist_scalar=50
):
    # Determine number of input channels from the model
    in_channels_num = model.layers[0].input_shape[0][3]
    input = get_input(tile_path, in_channels_num)

    dist_map, pheno_map = predict(input, model, dist_scalar)

    cell_centers = [
        (int(x[0]), int(x[1]))
        for x in blob_log(
            dist_map,
            min_sigma=min_log_std,
            max_sigma=max_log_std,
            threshold=log_threshold,
        )
    ]

    out_markers_num = pheno_map.shape[2]
    predicted_cells = []
    for coords in cell_centers:
        rr, cc = draw.disk((coords[0], coords[1]), radius=2, shape=dist_map.shape)
        phenotype = [
            float(np.mean(pheno_map[rr, cc, j])) for j in range(out_markers_num)
        ]
        predicted_cells.append([coords, phenotype])

    return predicted_cells


# Extracts the central subregion of a specified size from the image
def crop_image_center(image, target_size=(512, 512)):
    h, w, _ = image.shape
    new_h, new_w = target_size
    origin_y = h // 2 - (new_h // 2)
    origin_x = w // 2 - (new_w // 2)

    return image[origin_y : (origin_y + new_h), origin_x : (origin_x + new_w), :]


def save_as_png(image_array, target_path):
    image = image_array.astype("uint8")
    image = Image.fromarray(image)
    image.type = "I"
    image.save(target_path, "PNG")


def draw_cell_coordinates(image, coordinates, vis_radius):

    for coordinate in coordinates:
        iy, ix = int(round(coordinate[0])), int(round(coordinate[1]))
        rr, cc = draw.disk((iy, ix), radius=vis_radius, shape=image.shape)
        image[rr, cc, :] = 255
        rr, cc = draw.circle_perimeter(iy, ix, vis_radius, shape=image.shape)
        image[rr, cc, :] = 0


def demo_inference(
    tile_path,
    model_path,
    output_path,
    display_image_path=None,
    log_threshold=0.07,
    min_log_std=3,
    max_log_std=5,
    dist_scalar=50,
    vis_radius=2,
):
    output_path.mkdir(exist_ok=True)

    model = load_model(device=DEVICE, which_model=model_path)

    in_channels_num = model.n_inputs
    input = get_input(tile_path, in_channels_num)

    try:
        dist_map, pheno_map = predict(input, model, 0)
    except ValueError:
        # If the whole sample image is too large to fit into a GPU memory during inference,
        # take image subregion
        print(
            "Warning: insufficient GPU memory to run inference for a single tile. Doing inference for a central 512x512 subimage"
        )
        subregion = crop_image_center(input)
        dist_map, pheno_map = predict(subregion, model, dist_scalar)

    cell_centers = [
        (int(x[0]), int(x[1]))
        for x in blob_log(
            dist_map,
            min_sigma=min_log_std,
            max_sigma=max_log_std,
            threshold=log_threshold,
        )
    ]
    print("{} lymphocytes has been detected".format(len(cell_centers)))

    # Output 1:
    # Make visualization of predicted cells
    pheno_channels = [pheno_map[..., i] for i in range(pheno_map.shape[2])]
    pheno_vis = mix(panel_colors, pheno_channels)
    draw_cell_coordinates(pheno_vis, cell_centers, vis_radius)
    save_as_png(pheno_vis, output_path / "phenotype_prediction.png")

    # Output 2:
    # Visualise prediction on a provided image
    if display_image_path is not None:
        if display_image_path.suffix in [".tiff", ".tif"]:
            display_image = tifffile.imread(display_image_path)
        else:
            display_image = imageio.imread(display_image_path)

        draw_cell_coordinates(display_image, cell_centers, vis_radius)

        im = Image.fromarray(display_image.astype("uint8"))
        im.type = "I"
        im.save(output_path / "prediction_vis.jpg", "JPEG", quality=90)

    prediction_path = output_path / "prediction.tsv"
    # Add pseudo-channels for phenotype identification
    with open(prediction_path, "w") as f:
        out_markers_num = pheno_map.shape[2]
        # Write column names first
        f.write(
            "\t".join(
                ["Y", "X"] + ["Marker %d" % (i + 1) for i in range(out_markers_num)]
            )
            + "\n"
        )
        for coords in cell_centers:
            rr, cc = draw.disk((coords[0], coords[1]), radius=2, shape=dist_map.shape)
            cell_prediction = list(coords) + [
                float(np.mean(pheno_map[rr, cc, j])) for j in range(out_markers_num)
            ]
            f.write("\t".join(["%g" % i for i in cell_prediction]) + "\n")


def demo(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tile_path",
        type=str,
        default=str(INPUT_IMAGE_PATH),
        required=False,
        help="a path to an image to use for demo inference",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(MODEL_PATH),
        required=False,
        help="a path to a model to use for demo inference",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=str(INFERENCE_PATH),
        required=False,
        help="a path to save output of demo inference",
    )
    parser.add_argument(
        "--display_image_path",
        type=str,
        default=None,
        required=False,
        help="a path to a file to use to display the detected cells",
    )
    parser.add_argument(
        "--log_th",
        type=float,
        default=0.07,
        required=False,
        help="a threshold for LoG blob detection algorithm",
    )
    parser.add_argument(
        "--min_log_std",
        type=float,
        default=3,
        required=False,
        help="the minimum standard deviation for Gaussian kernel of LoG, decrease to detect smaller cells",
    )
    parser.add_argument(
        "--max_log_std",
        type=float,
        default=5,
        required=False,
        help="the maximum standard deviation for Gaussian kernel of LoG, increase to detect larger cells",
    )
    parser.add_argument(
        "--vis_radius",
        type=int,
        default=2,
        required=False,
        help="the maximum standard deviation for Gaussian kernel of LoG, increase to detect larger cells",
    )

    args = parser.parse_args(argv)
    tile_path = Path(args.tile_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    display_image_path = (
        Path(args.display_image_path) if args.display_image_path is not None else None
    )
    log_th = args.log_th
    min_log_std = args.min_log_std
    max_log_std = args.max_log_std
    vis_radius = args.vis_radius

    demo_inference(
        tile_path,
        model_path,
        output_path,
        display_image_path,
        log_th,
        min_log_std,
        max_log_std,
        vis_radius,
    )


if __name__ == "__main__":
    # Very first argument determines action
    actions = {"demo": demo}

    try:
        action = actions[sys.argv[1]]
    except (IndexError, KeyError):
        print("Usage: " + "/".join(actions.keys()) + " ...")
    else:
        action(sys.argv[2:])
