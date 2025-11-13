import matplotlib.pyplot as plt
import scipy
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import pandas as pd
from pathlib import Path
import numpy as np
from skimage.measure import block_reduce
from panels import panels, Panel
from sklearn.neighbors import KernelDensity
from scipy.stats.contingency import association

def clopper_pearson(x: float, n: float, alpha: float = 0.05) -> tuple:
    """
    Estimate the confidence interval for a sampled Bernoulli random
    variable.

    Parameters
    ----------
    x : float
         number of successes
    n : float
        number trials (x <= n)
    alpha : float, optional
        confidence level (i.e., the true probability is inside the
        confidence interval with probability 1-alpha), by default 0.05

    Returns
    -------
    Tuple[float, float]
        returns a `(low, high)` pair of numbers indicating the
        interval on the probability.
    """
    b = scipy.stats.beta.ppf
    lo = b(alpha / 2, x, n - x + 1)
    hi = b(1 - alpha / 2, x + 1, n - x)
    hi = np.clip(hi, 0, 100)
    return np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=1.0)


def bar_plot(cell_subtypes, title, target_path, ylim=None):
    plt.figure()
    ax = plt.gca()
    plt.title(title)
    stat = cell_subtypes.items()
    stat = sorted(stat, key=lambda x: x[0])
    keys = [x[0] for x in stat]
    values = [x[1] for x in stat]
    ax.bar(keys, values)
    for i, v in enumerate(values):
        ax.text(i - 0.4, v + 10, str(v))
    plt.xticks(rotation=90)
    # Custom the subplot layout
    plt.subplots_adjust(bottom=0.3)
    if ylim is not None:
        plt.ylim(ylim)
    plt.savefig(target_path)
    plt.show()


def plot_error_rate_per_dataset(
    dataset_error_path: str | Path,
    output_figure_folder: Path,
    panel_name: str,
    model_name: str,
):
    panel: Panel = panels[panel_name]

    with open(dataset_error_path) as f:
        datasets_errors = json.load(f)

    errors_csv = []
    sample_dict = {
        "cases": 0,
        "errors": 0,
        "error_rate": 0.0,
        "upper_ci": 0.0,
        "lower_ci": 0.0,
        "minus_error": 0.0,
        "upper_ci": 0.0,
        "label": model_name,
    }
    for ds, errors in datasets_errors.items():
        bg_errors = dict([(key, value) for key, value in sample_dict.items()])
        fg_errors = dict([(key, value) for key, value in sample_dict.items()])
        for cell_type in errors:
            for k, v in errors[cell_type].items():
                if cell_type in panel.fg_cell_types:
                    fg_errors[k] += v
                else:
                    bg_errors[k] += v
        for dicts in [fg_errors, bg_errors]:
            if dicts["cases"] == 0:
                continue
            successes = dicts["cases"] - dicts["errors"]
            low_ci, high_ci = clopper_pearson(successes, dicts["cases"])
            dicts["ci"] = (1 - float(high_ci), 1 - float(low_ci))

            dicts["error_rate"] = dicts["errors"] / dicts["cases"]

            dicts["upper_ci"] = 1 - low_ci
            dicts["lower_ci"] = 1 - high_ci

            dicts["plus_error"] = dicts["upper_ci"] - dicts["error_rate"]
            dicts["minus_error"] = dicts["error_rate"] - dicts["lower_ci"]

        errors_csv.append(
            (ds, "foreground") + tuple([fg_errors[x] for x in sample_dict.keys()])
        )
        errors_csv.append(
            (ds, "background") + tuple([bg_errors[x] for x in sample_dict.keys()])
        )

    errors_df = pd.DataFrame(
        errors_csv, columns=["ds", "ann_type"] + list(sample_dict.keys())
    )
    for ann_type in ["background", "foreground"]:
        plot_errorbar(
            errors_df[errors_df.ann_type == ann_type],
            output_figure_folder / f"perf_dataset_{ann_type}.png",
            axis_value="ds",
            axis_name="Dataset",
            title=f"Error rate {ann_type}",
        )


def plot_errorbar(
    df: pd.DataFrame,
    output_figure_path: str | Path,
    output_table_path: str | Path | None = None,
    compare_df: pd.DataFrame | None = None,
    axis_value: str = "c_type",
    axis_name: str = "Cell value",
    title: str = "",
):
    """
    Plots error bar from df, and saves to figure. If specified, will save the table in html format.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe obtained with misclass_stat_csv from error_analysis
    output_figure_path : str
        Path to save figure.
    output_table_path : str, optional
        Path to save table in html, by default None
    """

    def add_error_margin(df, axis_value):
        cases = df["cases"].values
        errors = df["errors"].values

        # clopper-parsons
        alpha = 0.05
        lower_ci, upper_ci = clopper_pearson(cases - errors, cases, alpha)

        df["error_rate"] = df["error_rate"] * 100

        df["upper_ci"] = (1 - lower_ci) * 100
        df["lower_ci"] = (1 - upper_ci) * 100

        df["minus_error"] = df["error_rate"] - df["lower_ci"]
        df["plus_error"] = df["upper_ci"] - df["error_rate"]

        categories = df[axis_value].unique()
        df[axis_value] = pd.Categorical(df[axis_value], categories)
        return df

    comparing_flag = not isinstance(compare_df, type(None))
    font_size = 8

    plt.rcParams.update({"font.size": font_size})

    df_c = df.copy()
    if comparing_flag:
        df_c = pd.concat([df_c, compare_df])

    df_c = df_c[df_c["cases"] > 5]
    if axis_value == "c_type":
        df_c["cell_type"] = df_c["c_type"].apply(lambda x: x.replace(" cell", ""))
    if "err_rate" in df.columns:
        df_c["error_rate"] = df_c["err_rate"]

    df_processed = add_error_margin(df_c, axis_value=axis_value)
    df_processed.dropna(inplace=True)
    df_processed.sort_values(by=[axis_value], inplace=True, ascending=False)
    df_processed["error_rate"] = df_processed["error_rate"].apply(lambda x: round(x, 2))

    trimmed_table = df_processed[
        [axis_value, "error_rate", "errors", "cases", "label"]
    ].copy()

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    if title != "":
        ax.set_title(title)
    # fig.set_dpi(300)
    # fig.set_size_inches(6, 4.5)

    ax.axvline(10, alpha=0.3, linestyle="dashed")

    # Default behavior, not comparing models
    if not comparing_flag:
        data_color = [x / max(df_processed["cases"]) for x in df_processed["cases"]]
        my_cmap = plt.get_cmap("viridis")
        colors = my_cmap(data_color)

        cbar = fig.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(0, max(df_processed["cases"])), cmap=my_cmap
            ),
            ax=ax,
        )
        cbar.set_label("number of cases", rotation=270, labelpad=10)
        df_processed.plot.barh(
            x=axis_value,
            y="error_rate",
            color=colors,
            edgecolor="gray",
            alpha=0.75,
            linewidth=1.25,
            ax=ax,
            legend=False,
        )
        # add error bar
        x_coords = [p.get_width() for p in ax.patches]
        y_coords = [p.get_y() + 0.5 * p.get_height() for p in ax.patches]

        # error text
        for i, v in enumerate(trimmed_table["error_rate"]):
            ax.text(v + 1, i + 0.25, f"{v}%", fontsize=int(font_size) * 0.8)

        ax.errorbar(
            x=x_coords,
            y=y_coords,
            xerr=[df_processed["minus_error"], df_processed["plus_error"]],
            fmt="none",
            c="k",
            capsize=2.5,
        )
    else:
        _pd_dict = {}
        _ci_dict = {}
        for _label in df_processed.label.unique():
            _pd_current_label = df_processed[df_processed.label == _label].copy()
            _error_rate = _pd_current_label.error_rate.array
            _minus_error = _pd_current_label.minus_error.array
            _plus_error = _pd_current_label.plus_error.array
            _ci_dict[_label] = (_minus_error, _plus_error)
            _pd_dict[_label] = _error_rate

        index = df_processed[axis_value].unique()

        df = pd.DataFrame(_pd_dict, index=index)
        df.plot.barh(edgecolor="gray", alpha=0.75, linewidth=1.25, ax=ax)
        axbox = ax.get_position()
        ax.legend(
            loc="center",
            ncol=2,
            bbox_to_anchor=[
                axbox.x0 + 0.5 * axbox.width,
                axbox.y0 + axbox.height + 0.1,
            ],
            bbox_transform=fig.transFigure,
        )

    ax.set_ylabel(axis_name)
    ax.set_xlabel("Error rate (%)")

    ax.set_axisbelow(True)
    # max_ci_value = np.max(df_processed['upper_ci'])+5
    ax.set_xticks(np.arange(0, 101, 10))
    ax.xaxis.grid(color="gray", linestyle="dashed", alpha=0.2)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    fig.savefig(output_figure_path, bbox_inches="tight")
    print(output_figure_path)

    return ax


def plot_error_count_bar_per_dataset(dataset_performance_path, output_figure_path):
    with open(dataset_performance_path) as f:
        data = json.load(f)

    # Restructure the data into a DataFrame
    records = []
    for dataset, cells in data.items():
        for cell_type, counts in cells.items():
            records.append([dataset, cell_type, counts["errors"]])

    df = pd.DataFrame(records, columns=["Dataset", "Cell Type", "Errors"])

    # Pivot the DataFrame to prepare for stacking
    df_pivot = df.pivot(index="Dataset", columns="Cell Type", values="Errors").fillna(0)

    # Plot stacked horizontal bar chart for errors
    df_pivot.plot(kind="barh", stacked=True, figsize=(10, 6))

    # Set plot labels and title
    plt.xlabel("Number of errors")
    plt.title("Error distribution across datasets")
    plt.tight_layout()

    # Show plot
    plt.show()
    plt.savefig(output_figure_path)


def plot_confusion_matrix(y_true: list, y_pred: list, target_path: str | Path):
    """Plots confusion matrix

    Parameters
    ----------
    y_true : list
        List of true labels.
    y_pred : list
        List of predicted labels.
    target_path : str | Path
        Full path and file name to save figure.
    """

    plt.rcParams.update({"font.size": 16})

    labels = np.unique(np.array(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_v = association(pd.crosstab(y_true, y_pred), method="cramer")
    cm_v = round(cm_v, 3)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        # include_values=len(labels) < 15,
        ax=ax,
        xticks_rotation="vertical",
    )
    ax.set_title(f"Cramér's V: [{cm_v}]")
    plt.subplots_adjust(bottom=0.35)
    plt.savefig(
        target_path,
        bbox_inches="tight",
        pad_inches=0.1,
    )


def downsample_image(
    image: np.ndarray, factor: int = 2, channels_index=-1
) -> np.ndarray:
    """Downsamples an image by taking the mean of each block.

    This function applies a downsampling operation to the input image. The
    downsampling is done by taking the mean of each block in the image, where
    the size of each block is defined by the `factor`.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    factor : int, optional
        Factor for downsampling. Higher values will result in more aggressive
        downsampling., by default 2
    channels_index : int, optional
        Index of the channels axis in the input image., by default -1

    Returns
    -------
    np.ndarray
        Downsampled image array.

    """
    arrays = [
        block_reduce(image[:, :, c], block_size=factor, func=np.mean)
        for c in range(image.shape[channels_index])
    ]
    return np.stack(
        arrays,
        axis=-1,
    )


def create_subtype_col_from_pred(prediction_df, activation_th=0.4):
    df = prediction_df.copy()
    panel_obj: Panel = panels[df.iloc[0].panel]

    def assign_subtype(_type="", positivity=None, prediction=None, distance=0.0):
        if _type != "" and _type not in panel_obj.decorable_types:
            return str(_type)
        if not isinstance(positivity, type(None)):
            subtype = panel_obj.cell_from_annotation(_type, positivity)
            return subtype.name
        if not isinstance(prediction, type(None)):
            subtype = panel_obj.cell_from_prediction(
                distance=distance, phenotype=prediction, activ_th=activation_th
            )
            return subtype.name
        raise

    df["ann_subtype"] = df.apply(
        lambda x: assign_subtype(
            x.ann_type, positivity=[float(y) for y in x.ann_pheno.split(",")]
        ),
        axis=1,
    )
    df["pred_subtype"] = df.apply(
        lambda x: assign_subtype(
            prediction=[float(y) for y in x.pred_pheno.split(",")], distance=x.distance
        ),
        axis=1,
    )

    return df


def create_df_from_prediction(
    pred_file_path, panel_name, activation_th=0.4, out_markers_num=None
):
    df_input = pd.read_csv(pred_file_path, sep="\t")
    df_input = df_input[df_input.panel == panel_name]
    df = create_subtype_col_from_pred(df_input, activation_th)
    name_markers = [x for x in panels[panel_name].markers]

    # adjust name_markers with predicted phenotype length
    if out_markers_num:
        name_markers = name_markers[:out_markers_num]

    len_markers = len(name_markers)

    # Remove DAPI and background positivity
    # Predicted phenotype (probabilities)
    df[[f"p{x}" for x in name_markers]] = pd.DataFrame(
        [
            (str(x).split(",") + ["0"] * len_markers)[:len_markers]
            for x in df.pred_pheno.tolist()
        ],
        index=df.index,
        columns=[f"p{x}" for x in name_markers],
    ).astype("float32")

    # Annotation phenotype (ground truth, padded with "1")
    df[name_markers] = pd.DataFrame(
        [
            (str(x).split(",") + ["1"] * len_markers)[:len_markers]
            for x in df.ann_pheno.tolist()
        ],
        index=df.index,
        columns=name_markers,
    ).astype("int16")

    return df


def plot_kde_auc_positivity(
    pred_file_path: str | Path,
    target_path: Path,
    positivity: str | list | None = None,
    marker_to_condition: str | None = None,
    detection_th=3.5,
    mm_pp=2,
    out_markers_num: int | None = None,
):
    """
    Creates estimate density plot of predictions given a positive marker.

    Parameters
    ----------
    pred_file_path : Path
        Path to prediction (tsv) file.
    target_path : str
        Full path to save figure.
    positivity : str, optional
        Marker name(s) (e.g., cd3) to show prediction distribution (do not use predicted marker!). Leave empty for plotting all.
    marker_to_condition : str, optional
        If you want to condition on a positive marker, by default None
    """
    positivity_orig = positivity
    _tmp_df = pd.read_csv(pred_file_path, sep="\t")
    for panel_name in _tmp_df.panel.unique():
        df = create_df_from_prediction(
            pred_file_path, panel_name=panel_name, out_markers_num=out_markers_num
        )
        positivity = positivity_orig
        if isinstance(positivity, type(None)):
            positivity = [x for x in panels[panel_name].markers[:out_markers_num]]
        if isinstance(positivity, str):
            positivity = [positivity]

        for posit in positivity:
            _, (ax2, ax1) = plt.subplots(1, 2, figsize=(14, 4), dpi=300)
            labels = [f"real {posit}-", f"real {posit}+"]
            graphs = []
            colors = ["C0", "C1"]
            offset = [-0.6, -0.75]
            extra_condition = True
            condition_flag = not isinstance(marker_to_condition, type(None))
            if condition_flag:
                extra_condition = df[marker_to_condition] > 3
                df = df[extra_condition]
            x_range = []

            for i, condition in enumerate([(df[posit] <= 3), df[posit] > 3]):
                # if detected cell is far do not include predicted phenotype
                df.loc[df["distance"] > detection_th * mm_pp, [f"p{posit}"]] = 0.0
                _data = df[condition][[f"p{posit}"]].to_numpy()

                if _data.shape[0] == 0:
                    continue
                kde = KernelDensity(kernel="gaussian", bandwidth=0.05).fit(_data)
                # x-value range for plotting KDE
                x_range = np.linspace(_data.min() - 0.3, _data.max() + 0.3, num=200)

                # compute the log-likelihood of each sample
                log_density = kde.score_samples(x_range[:, np.newaxis])
                graphs.append(np.exp(log_density))
                # draw KDE curve
                ax1.plot(
                    x_range,
                    np.exp(log_density),
                    color=colors[i],
                    linewidth=2.5,
                    label=labels[i],
                )
                # draw boxes representing datapoints
                ax1.plot(
                    _data,
                    np.zeros_like(_data)
                    + offset[i]
                    + (np.random.random(size=_data.shape) * 0.1),
                    "s",
                    color=colors[i],
                    markersize=3,
                    alpha=0.01,
                    clip_on=False,
                )
            try:
                idx = np.argwhere(np.diff(np.sign(graphs[0] - graphs[1]))).flatten()
                # print("intersections", x_range[idx])
            except:
                pass

            ax1.set_ylim(0)
            ax1.set_title("Predicted phenotype density estimation")
            _x_label = (
                f"p{posit} (positive: {marker_to_condition})"
                if condition_flag
                else f"predicted {posit}"
            )

            ax1.set_xlabel(_x_label, fontsize=12, rotation="horizontal", labelpad=4)
            ax1.set_ylabel(
                "Estimated density", fontsize=16, rotation="vertical", labelpad=24
            )

            y_test = np.array(df[posit] > 3).astype(int)
            y_pred = df[f"p{posit}"].values

            fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
            metrics.RocCurveDisplay.from_predictions(
                y_true=y_test, y_pred=y_pred, ax=ax2
            )
            ax2.set_title(f"ROC Curve: {posit}")

            # Indicate the best threshold point on the plot
            arg_max = np.argmax(tpr - fpr)

            # Red dot at the best threshold
            ax2.scatter(fpr[arg_max], tpr[arg_max], color="red", zorder=5)
            ax2.text(
                (fpr[arg_max] + 0.04), tpr[arg_max], s="t=%f" % thresholds[arg_max]
            )

            gmeans = np.sqrt(tpr * (1 - fpr))
            arg_max_gmean = np.argmax(gmeans)
            if arg_max_gmean != arg_max:
                # Red dot at the best threshold
                ax2.scatter(
                    fpr[arg_max_gmean], tpr[arg_max_gmean], color="blue", zorder=5
                )
                ax2.text(
                    (fpr[arg_max_gmean] + 0.04),
                    tpr[arg_max_gmean],
                    s="t=%f" % thresholds[arg_max_gmean],
                )

            plt.legend()
            plt.savefig(target_path / f"kde_auc_{panel_name}_{posit}.png")
            # plt.show()
            plt.close()
