import argparse
import json
import os
from typing import Optional
from annotations import load_annotations
from visualization import plot_confusion_matrix, plot_errorbar, plot_kde_auc_positivity
from panels import Panel, panels
from prediction import ImmuNetPredictionHandler, match_cells
from pathlib import Path
import pandas as pd
from config import VAL_ANNOTATONS_PATH, MODEL_FINAL_NAME, PREDICTION_FOLDER, EVALUATION_PATH
import numpy as np

def calculate_performance(results_maintypes, results_subtypes, label, target_path):
    print("\nMain cell types performance:")
    maintypes_performance = ["label,c_type,errors,cases,err_rate"]
    for type, detections in results_maintypes.items():
        cases = len(detections)
        errors = cases - np.sum(detections)
        maintypes_performance.append(
            ",".join((label, type, str(errors), str(cases), str(errors / cases)))
        )
        print("{} {} {}".format(type, errors / cases, cases))

    print("\nCell subtypes performance:")
    subtypes_performance = ["label,c_type,errors,cases,err_rate"]
    for type, detections in results_subtypes.items():
        cases = len(detections)
        errors = cases - np.sum(detections)
        subtypes_performance.append(
            ",".join((label, type, str(errors), str(cases), str(errors / cases)))
        )
        print("{} {} {}".format(type, errors / cases, cases))

    # Write performance stat
    with open(target_path / "perf_maintypes.csv", "w") as f:
        f.write("\n".join(maintypes_performance))

    with open(target_path / "perf_subtypes.csv", "w") as f:
        f.writelines("\n".join(subtypes_performance))


def get_pheno_and_cell(
    pred_pheno_str,
    panel: Panel,
    distance,
    activation_th: float,
    detection_th,
    mm_pp,
) -> tuple:
    pred_phen = [float(x) for x in pred_pheno_str.split(",")]
    try:
        cell = panel.cell_from_prediction(
            distance=distance, phenotype=pred_phen, activ_th=activation_th, detection_th=detection_th, mm_pp=mm_pp
        )
    except IndexError as e:
        raise ValueError(
            "Model output might not match data! Make sure to configure it correctly!"
        ) from e

    return pred_phen, cell

def get_error_dict(
    panel_id,
    ds_name,
    ann_id,
    annotated_type,
    distance,
    pred_phen,
    predicted_cell_type,
    annotated_positivity=None,
    type_from_ann=None,
):
    error_dict = {
        "panel": panel_id,
        "ds": ds_name,
        "id": ann_id,
        "ann_type": annotated_type,
        "dist": distance,
        "pred_ph": pred_phen,
        "pred_type": predicted_cell_type,
    }

    if annotated_positivity is not None:
        error_dict["ann_pos"] = annotated_positivity

    if type_from_ann is not None:
        error_dict["type_from_ann"] = type_from_ann

    return error_dict



def evaluate(
    prediction_file_path,
    label,
    target_path: Optional[os.PathLike] = EVALUATION_PATH,
    activation_th=0.4,
    detection_th=3.5,
    mm_pp=2,
    ann_list=None,
    ds_list=None,
    inForm=False,
    ann_types=None,
    force_panel: Optional[str] = None,
):
    """
    Evaluate pipeline predictions against annotated cells.

    Parameters
    ----------
    prediction_file_path: Path to matched annotations and predictions
    label: str, identifier for this evaluation (e.g., train/val or model name)
    target_path: Path, folder to save results (optional)
    activation_th: float, threshold to consider a phenotype marker active
    detection_th: float, radius to decide if a predicted cell matches an annotation
    mm_pp: float, micrometers per pixel
    ann_list: list of annotation IDs to include (optional)
    ds_list: list of dataset names to include (optional)
    inForm: bool, whether predictions come from inForm
    ann_types: list of annotation types to include (optional)
    force_panel: str, force all predictions to use this panel (optional)
    """

    # Read predictions file
    with open(prediction_file_path) as f:
        lines = [line.strip() for line in f.readlines()]

    results_main_types = {}
    results_subtypes = {}
    errors = []

    y_true_main, y_pred_main = [], []
    y_true_subtypes, y_pred_subtypes = [], []

    dataset_performance = {}

    # Skip header
    for line in lines[1:]:
        (
            panel_id,
            ds_name,
            ann_id,
            annotated_type,
            annotated_positivity_str,
            pred_phenotype_str,
            distance_str,
        ) = line.split("\t")

        # Filter by dataset or annotation list
        if ds_list is not None and ds_name not in ds_list:
            continue
        if ann_list is not None and ann_id not in ann_list:
            continue

        # Determine which panel to use
        panel: Panel = (
            panels[force_panel] if force_panel else panels[panel_id]
        )

        # Parse annotation positivity
        annotated_positivity = [int(x) for x in annotated_positivity_str.split(",")]
        annotated_positivity += ([3] * (len(panel.markers) - len(annotated_positivity)))

        ann_cell = panel.cell_from_annotation(annotated_type, annotated_positivity)

        # Skip unwanted annotation types or invalid annotations
        if ann_types is not None and annotated_type not in ann_types:
            continue
        if ann_cell.is_invalid:
            continue
        

        # Initialize dataset_performance entry
        dataset_performance.setdefault(ds_name, {})
        dataset_performance[ds_name].setdefault(
            annotated_type, {"cases": 0, "errors": 0}
        )

        # Track ground truth labels
        y_true_main.append(annotated_type)
        y_true_subtypes.append(ann_cell.name)

        distance = float(distance_str)
        pred_phen, predicted_cell = get_pheno_and_cell(
            pred_phenotype_str,
            panel,
            distance,
            activation_th,
            detection_th,
            mm_pp,
        )

        dataset_performance[ds_name][annotated_type]["cases"] += 1

        if annotated_type in panel.fg_cell_types:
            hit_main = predicted_cell.main_type_name == annotated_type
            y_pred_main.append(predicted_cell.main_type_name)

            if ann_cell.is_inconsistent:
                print("cell is inconsistent (?)")
                hit_subtype = hit_main
                y_pred_subtypes.append(ann_cell.name)
            else:
                hit_subtype = predicted_cell.name == ann_cell.name
                y_pred_subtypes.append(predicted_cell.name)

            if not hit_subtype:
                errors.append(
                    get_error_dict(
                        panel_id,
                        ds_name,
                        ann_id,
                        annotated_type,
                        distance,
                        pred_phen,
                        predicted_cell.name,
                        annotated_positivity,
                        ann_cell.name,
                    )
                )
                dataset_performance[ds_name][annotated_type]["errors"] += 1
        else:
            # Background / non-foreground cells
            hit = (
                distance > detection_th * mm_pp
                or predicted_cell.main_type_name == annotated_type
            )
            hit_subtype = hit

            if hit:
                y_pred_main.append(annotated_type)
                y_pred_subtypes.append(annotated_type)
            else:
                y_pred_main.append(predicted_cell.main_type_name)
                y_pred_subtypes.append(predicted_cell.name)
                errors.append(
                    get_error_dict(
                        panel_id,
                        ds_name,
                        ann_id,
                        annotated_type,
                        distance,
                        pred_phen,
                        predicted_cell.name,
                    )
                )
                dataset_performance[ds_name][annotated_type]["errors"] += 1

        # Record results for main types and subtypes
        results_main_types.setdefault(annotated_type, []).append(
            int(hit_main if annotated_type in panel.fg_cell_types else hit)
        )
        results_subtypes.setdefault(ann_cell.name, []).append(int(hit_subtype))

    # Save outputs if target_path is provided
    error_path = ""
    dataset_performance_path = ""
    if target_path:
        target_path.mkdir(exist_ok=True, parents=True)

        calculate_performance(results_main_types, results_subtypes, label, target_path)
        error_path = target_path / f"{label}_errors.json"
        with open(error_path, "w") as f:
            json.dump(errors, f)

        dataset_performance_path = target_path / f"{label}_dataset_performance.json"
        with open(dataset_performance_path, "w") as f:
            json.dump(dataset_performance, f)

    return (
        y_true_main,
        y_pred_main,
        y_true_subtypes,
        y_pred_subtypes,
        error_path,
        dataset_performance_path,
    )


def evaluate_pipeline(
    annotations_file_path: Path | str,
    model: str,
    target_path: Path | str,
    log_th: float,
    ph_thresh: float,
    out_markers_num: int | None = None,
    model_path: Path | str | None = None,
    prediction_file_path: Path | str | None = None,
    backbone: str = "ORIGINAL",
    min_sigma_log=2,
    max_sigma_log=5,
    dist_multiplier=50,
    device="cuda:0",
):
    """
    Run a model evaluation pipeline.

    This function loads ground-truth annotations, obtains or loads a trained model,
    performs predictions, applies thresholds and optional post-processing, and evaluates
    the results. Unlike the full pipeline, this does not create or manage a folder
    structure — output is written directly to `target_path`.

    Parameters
    ----------
    annotations_file_path : Path or str
        Path to the annotations file containing ground-truth labels.
    model : str
        Name or identifier of the model architecture to use.
    target_path : Path or str
        Path to store evaluation outputs (metrics, plots, etc.).
    log_th : float
        Threshold applied to the model's log output (e.g., log probability or logit).
    ph_thresh : float
        Probability threshold for detecting valid predictions.
    out_markers_num : int, optional
        Must match the output size of the network.
    model_path : Path or str, optional
        Path to a pre-trained model checkpoint. If None, the prediction is grabbed from the server.
    prediction_file_path : Path or str, optional
        If provided, uses an existing prediction file instead of running inference.
    backbone : str, optional
        Backbone network type to use in the model (e.g., "ORIGINAL", "RESNET").
    min_sigma_log : float, optional
        Minimum sigma value (in log space) for blob detection.
    max_sigma_log : float, optional
        Maximum sigma value (in log space) for blob detection.
    dist_multiplier : float, optional
        Distance multiplier used in post-processing for cell separation.
    device : str, optional
        Device for computation (e.g., "cuda:0", "cpu").

    Returns
    -------
    None
        Results are written to the `target_path` location.
    """
    if isinstance(target_path, str):
        target_path = Path(target_path)

    # visualize prediction first
    # to make sure everything looks okey

    if model_path and not prediction_file_path:
        print("missing peek!!!")
        pass
        # raise NotImplementedError()
        # tiles = load_annotations(annotations_file_path)
        # visualise_prediction(
        #     backbone=backbone,
        #     model_path=model_path,
        #     model_name=model,
        #     output_path=target_path,
        #     tile=tiles[0],
        #     log_thresh=log_th,
        #     device=device,
        # )

    target_path.mkdir(exist_ok=True)
    if not prediction_file_path:
        assert out_markers_num, "Please provide out markers!"
        prediction_request = ImmuNetPredictionHandler(
            model_name=model,
            model_path=model_path,
            log_threshold=log_th,
            ph_thresh=ph_thresh,
            backbone=backbone,
            device=device,
            min_sigma_log=min_sigma_log,
            max_sigma_log=max_sigma_log,
            dist_multiplier=dist_multiplier,
        )

        prediction_file_path = match_cells(
            annotations_path=annotations_file_path,
            prediction_handler=prediction_request,
            out_markers_num=out_markers_num,
            fout=target_path
            / f"prediction-immunet-{prediction_request.log_threshold}-{prediction_request.model_name}.tsv",
        )

    (
        y_true_main,
        y_pred_main,
        y_true_subtypes,
        y_pred_subtypes,
        error_path,
        dataset_performance_path,
    ) = evaluate(
        prediction_file_path=prediction_file_path,
        label=f"{model}",
        target_path=target_path,
        ann_types=None,
        activation_th=ph_thresh,
    )

    plot_confusion_matrix(
        y_true=y_true_subtypes,
        y_pred=y_pred_subtypes,
        target_path=target_path / "cm_subtypes.png",
    )
    plot_confusion_matrix(
        y_true=y_true_main,
        y_pred=y_pred_main,
        target_path=target_path / "cm_maintypes.png",
    )

    plot_confusion_matrix(
        y_true=y_true_subtypes,
        y_pred=y_pred_subtypes,
        target_path=target_path / "cm_subtypes.png",
    )
    plot_confusion_matrix(
        y_true=y_true_main,
        y_pred=y_pred_main,
        target_path=target_path / "cm_maintypes.png",
    )

    plot_errorbar(
        pd.read_csv(target_path / "perf_subtypes.csv"),
        output_figure_path=target_path / "bar_subtypes.png",
        output_table_path=target_path / "perf_subtypes.html",
    )
    plot_errorbar(
        pd.read_csv(target_path / "perf_maintypes.csv"),
        output_figure_path=target_path / "bar_maintypes.png",
        output_table_path=target_path / "perf_maintypes.html",
    )
    # misclass_stat_csv(
    #     error_path=error_path,
    #     out_folder=target_path,
    #     table_folder_path=target_path,
    #     out_markers_num=out_markers_num,
    # )
    plot_kde_auc_positivity(
        pred_file_path=prediction_file_path,
        target_path=target_path,
        out_markers_num=out_markers_num,
    )

    # with open(error_path) as f:
    #     errors = json.load(f)
    # plot_error_rate_per_dataset(
    #     dataset_performance_path,
    #     output_figure_folder=target_path,
    #     model_name=model,
    #     panel_name=errors[0]["panel"],
    # )

    # with open(annotations_file_path, "rb") as f_in:
    #     file_out_path = os.path.join(
    #         target_path,
    #         f"annotations_evaluation_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.json.gz",
    #     )
    #     with gzip.open(file_out_path, "wb") as f_out:
    #         shutil.copyfileobj(f_in, f_out)


def main_pipeline():
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")

    parser.add_argument(
        "--annotations_file_path",
        type=Path,
        help="Path to the annotations file containing ground-truth labels.",
        default=VAL_ANNOTATONS_PATH,
    )

    parser.add_argument(
        "--model_name",
        type=str,
        dest="model",
        help="Name or identifier of the model architecture to use.",
        default=MODEL_FINAL_NAME,
    )

    parser.add_argument(
        "--target_path",
        type=Path,
        help="Path to store evaluation outputs (metrics, plots, etc.).",
        default=PREDICTION_FOLDER,
    )

    parser.add_argument(
        "--log_th",
        type=float,
        help="Threshold applied to the model's log output.",
        default=0.07,
    )

    parser.add_argument(
        "--ph_thresh",
        type=float,
        help="Probability threshold for detecting valid predictions.",
        default=0.4,
    )

    parser.add_argument(
        "--out_markers_num",
        type=int,
        help="Limit the number of output markers.",
        default=5,
    )

    parser.add_argument(
        "--model_path",
        type=Path,
        help="Path to a pre-trained model checkpoint.",
        required=True,
    )

    parser.add_argument(
        "--prediction_file_path",
        type=Path,
        default=None,
        help="Path to an existing prediction file to use instead of running inference.",
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="ORIGINAL",
        help='Backbone network type to use in the model (default: "ORIGINAL").',
    )

    parser.add_argument(
        "--min_sigma_log",
        type=float,
        default=2,
        help="Minimum sigma value (in log space) for blob detection.",
    )

    parser.add_argument(
        "--max_sigma_log",
        type=float,
        default=5,
        help="Maximum sigma value (in log space) for blob detection.",
    )

    parser.add_argument(
        "--dist_multiplier",
        type=float,
        default=50,
        help="Distance multiplier used in post-processing for cell separation.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device for computation (e.g., "cuda:0", "cpu").',
    )

    args = parser.parse_args()

    evaluate_pipeline(
        annotations_file_path=args.annotations_file_path,
        model=args.model,
        target_path=args.target_path,
        log_th=args.log_th,
        ph_thresh=args.ph_thresh,
        out_markers_num=args.out_markers_num,
        model_path=args.model_path,
        prediction_file_path=args.prediction_file_path,
        backbone=args.backbone,
        min_sigma_log=args.min_sigma_log,
        max_sigma_log=args.max_sigma_log,
        dist_multiplier=args.dist_multiplier,
        device=args.device,
    )


if __name__ == "__main__":
    main_pipeline()
