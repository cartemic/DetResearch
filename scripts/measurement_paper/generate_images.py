import os
import sys
from copy import copy
from functools import wraps
from time import time

import numpy as np
import pandas as pd
import seaborn as sns
import skimage.filters
import uncertainties as un
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import ks_2samp, t, ttest_ind, ttest_ind_from_stats
from skimage import io, transform
from uncertainties import unumpy as unp

import funcs
from funcs.post_processing.images.soot_foil import deltas as pp_deltas

d_drive = funcs.dir.d_drive
DF_SF_SPATIAL = pd.read_csv(
    os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "spatial_calibrations.csv",
    )
)
SF_DATE = "2020-12-27"
SF_SHOT = 3
SF_IMG_DIR = os.path.join(
    d_drive,
    "Data",
    "Processed",
    "Soot Foil",
    "foil images",
    SF_DATE,
    f"Shot {SF_SHOT:02d}",
)
SF_SPATIAL_SHOT_MASK = (DF_SF_SPATIAL["date"] == SF_DATE) & (DF_SF_SPATIAL["shot"] == SF_SHOT)
SF_DELTA_MM = DF_SF_SPATIAL[SF_SPATIAL_SHOT_MASK]["delta_mm"]
SF_DELTA_PX = DF_SF_SPATIAL[SF_SPATIAL_SHOT_MASK]["delta_px"]
PLOT_FILETYPE = "png"
# ibm color blind safe palette
# https://lospec.com/palette-list/ibm-color-blind-safe
# https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000
COLOR_SC = "#fe6100"
COLOR_SF = "#648fff"
SAVE_LOC = os.path.join(d_drive, "Measurement-Paper", "images")
DPI = 200


def hex2rgb(hex_color):
    hex_color = hex_color.replace("#", "")
    r = int(hex_color[:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:], 16)

    return r, g, b


def rgb2hex(r, g, b):
    out = f"#{hex(r)[2:]}{hex(g)[2:]}{hex(b)[2:]}"

    return out


def hex_add(c0, c1):
    """
    c0 + c1

    Parameters
    ----------
    c0
    c1

    Returns
    -------

    """
    r0, g0, b0 = hex2rgb(c0)
    r1, g1, b1 = hex2rgb(c1)
    r_out = min(255, r0 + r1)
    g_out = min(255, g0 + g1)
    b_out = min(255, b0 + b1)

    out = rgb2hex(r_out, g_out, b_out)

    return out


def hex_sub(c0, c1):
    """
    c0 - c1

    Parameters
    ----------
    c0
    c1

    Returns
    -------

    """
    r0, g0, b0 = hex2rgb(c0)
    r1, g1, b1 = hex2rgb(c1)
    r_out = max(0, r0 - r1)
    g_out = max(0, g0 - g1)
    b_out = max(0, b0 - b1)

    out = rgb2hex(r_out, g_out, b_out)

    return out


def timed(func):
    @wraps(func)
    def _timed(*args, **kwargs):
        name = func.__name__
        if name != "main":
            sys.stderr.write(f"{func.__name__} ")
        t0 = time()
        out = func(*args, **kwargs)
        t1 = time()
        if name != "main":
            sys.stderr.write(f"took {t1 - t0:6f} sec\n")
        else:
            sys.stderr.write(f"Done! {t1 - t0:6f} sec\n")
        return out

    return _timed


@timed
def set_plot_format():
    common_size = 7.5
    sns.set_color_codes("deep")
    sns.set_context(
        "paper",
        rc={
            "font.size": common_size,
            "axes.titlesize": common_size + 1.5,
            "axes.titleweight": "bold",
            "axes.labelsize": common_size,
            "xtick.labelsize": common_size,
            "ytick.labelsize": common_size,
        },
    )
    sns.set_style(
        {
            "font.family": "serif",
            "font.serif": "Computer Modern",
        }
    )
    # plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.dpi"] = DPI


def sf_imread(
    img_path,
    plot=True,
):
    """
    Thin wrapper around `skimage.io.imread` that rotates the image if it is
    to be used for plotting, but does not if it is to be used for measurements.

    Parameters
    ----------
    img_path : str
        Path to image
    plot : bool
        Determines whether or not image will be rotated 90 degrees

    Returns
    -------
    np.array
    """
    img_in = io.imread(img_path)
    if plot:
        img_in = transform.rotate(img_in, -90)  # show images going left-right
    return img_in


# noinspection PyTypeChecker
def get_scale_bar(
    delta_px,
    delta_mm,
    cell_size,
    text_color="#000",
    box_color="#fff",
    box_alpha=1,
    rotation="vertical",
):
    """
    Thin wrapper around ScaleBar that does a bit of standard formatting for
    my needs.

    Parameters
    ----------
    delta_px : float
        Calibration delta (px)
    delta_mm : float
        Calibration delta (mm)
    cell_size : float
        Fixed value to display in scale bar
    text_color : str
        Text color (hex)
    box_color: str
        Background box color (hex)
    box_alpha : float
        Box alpha -- NOTE: does not apply to border >:(
    rotation : str
        Which direction to place the scale bar: "vertical" or "horizontal"

    Returns
    -------
    ScaleBar
    """
    return ScaleBar(
        delta_mm / delta_px,
        "mm",
        location=3,
        fixed_value=cell_size,
        scale_formatter=(lambda x, u: f"{x:.1f} {u}"),
        border_pad=0.2,
        color=text_color,
        box_color=box_color,
        box_alpha=box_alpha,
        rotation=rotation,
    )


@timed
def get_schlieren_data(estimator):
    """
    Read in schlieren data from assorted .h5 stores and calculate cell sizes
    for individual shots.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
    """
    # read in data
    df_schlieren_tube = pd.DataFrame()
    df_schlieren_all_frames = pd.DataFrame()
    df_schlieren_frames = pd.DataFrame()
    for group in ("fffff", "hhhhh", "ggggg"):
        with pd.HDFStore(
            f"/d/Data/Processed/Data/data_{group}.h5",
            "r",
        ) as store:
            df_schlieren_tube = pd.concat((df_schlieren_tube, store.data))
        with pd.HDFStore(
            f"/d/Data/Processed/Data/schlieren_{group}.h5",
            "r",
        ) as store:
            df_schlieren_all_frames = pd.concat((df_schlieren_all_frames, store.data))

    # fix jacked up measurement
    with pd.HDFStore(
        "/d/Data/Processed/Data/tube_data_2020-08-07.h5",
        "r",
    ) as store:
        bad_shot = 3
        df_schlieren_tube[((df_schlieren_tube["date"] == "2020-08-07") & (df_schlieren_tube["shot"] == bad_shot))] = (
            store.data.iloc[0].to_numpy()
        )

    # After some analysis it looks like I was dumb and used u_delta_px as
    # u_loc_px. This gives an overly large estimate of uncertainty for
    # schlieren. Fix that before continuing with the analysis.
    df_schlieren_all_frames.loc[:, "u_delta_px"] = df_schlieren_all_frames["u_loc_px"].copy()
    df_schlieren_all_frames.loc[:, "u_loc_px"] = df_schlieren_all_frames["u_loc_px"].div(np.sqrt(2))

    # calculate cell size measurements
    df_schlieren_tube = df_schlieren_tube[
        np.isclose(df_schlieren_tube["phi_nom"], 1)
        & np.isclose(df_schlieren_tube["dil_mf_nom"], 0.2)
        & (df_schlieren_tube["fuel"] == "CH4")
        & (df_schlieren_tube["oxidizer"] == "N2O")
        & (df_schlieren_tube["diluent"] == "N2")
    ]
    df_schlieren_tube["cell_size"] = np.nan
    df_schlieren_tube["u_cell_size"] = np.nan

    u_delta_bias = np.sqrt(2) / 2
    deltas = unp.uarray(
        df_schlieren_all_frames["delta_px"],
        df_schlieren_all_frames["u_delta_px"],  # precision only
    ) + un.ufloat(0, u_delta_bias)
    spatials = unp.uarray(
        df_schlieren_all_frames["spatial_centerline"],
        df_schlieren_all_frames["u_spatial_centerline"],
    )
    cell_sizes = deltas * spatials
    df_schlieren_all_frames["cell_size"] = unp.nominal_values(cell_sizes)
    df_schlieren_all_frames["u_cell_size"] = unp.std_devs(cell_sizes)

    for (date, shot), _ in df_schlieren_tube.groupby(["date", "shot"]):
        _df_this_shot = df_schlieren_all_frames[
            ((df_schlieren_all_frames["date"] == date) & (df_schlieren_all_frames["shot"] == shot))
        ].dropna()
        df_schlieren_frames = pd.concat((df_schlieren_frames, _df_this_shot))
        if len(_df_this_shot):
            _deltas = unp.uarray(
                _df_this_shot["delta_px"],
                _df_this_shot["u_delta_px"],
            )
            _mm_per_px = unp.uarray(
                _df_this_shot["spatial_centerline"],
                _df_this_shot["u_spatial_centerline"],
            )
            _meas = estimator(_deltas * _mm_per_px) * 2
            # noinspection PyUnresolvedReferences
            df_schlieren_tube.loc[
                ((df_schlieren_tube["date"] == date) & (df_schlieren_tube["shot"] == shot)),
                ["cell_size", "u_cell_size"],
            ] = (_meas.nominal_value, _meas.std_dev)

    df_schlieren_tube = df_schlieren_tube[~pd.isna(df_schlieren_tube["cell_size"])]

    return df_schlieren_frames, df_schlieren_tube


@timed
def build_schlieren_images(
    cmap,
    df_meas,
    image_width=None,
    image_height=None,
    save=False,
    limits_x=(10, 110),
    limits_y=(10, 210),
):
    """
    Generates images of:

        * Raw schlieren image
        * Schlieren image with triple point locations identified

    Images will be rendered with an aspect ration of 0.5; only one of
    `image_width`, `image_height` should be given.

    Parameters
    ----------
    cmap : str
        Colormap to use for schlieren frame
    df_meas : pd.DataFrame
        DataFrame of schlieren measurements
    image_width : float or None
        Image width (in)
    image_height : float or None
        Image height (in)
    save : bool
        Whether or not to save images
    limits_x : tuple
        X limits of trimmed image
    limits_y : tuple
        Y limits of trimmed image

    Returns
    -------

    """
    aspect_ratio = 0.5  # w/h
    if image_width is None and image_height is None:
        raise ValueError("image_width or image_height must be given")

    if image_width is None:
        image_width = image_height * aspect_ratio
    elif image_height is None:
        image_height = image_width / aspect_ratio

    date = "2020-08-07"
    shot = 3
    frame = 0
    tube_data_h5_suffix = "fffff"
    with pd.HDFStore(f"/d/Data/Processed/Data/data_{tube_data_h5_suffix}.h5", "r") as store:
        schlieren_key_date = date.replace("-", "_")
        key = f"/schlieren/d{schlieren_key_date}/" + f"shot{shot:02d}/" f"frame_{frame:02d}"
        schlieren_raw = np.fliplr(store[key])

    # jankily shoehorn spatial calibration into existing function
    mm_per_px = df_meas[(df_meas["date"] == date) & (df_meas["shot"] == shot)]["spatial_centerline"].iloc[0]
    schlieren_scalebar = get_scale_bar(
        1,
        mm_per_px,
        cell_size=25.4,
    )

    # trim image to ROI
    limits_x = sorted(limits_x)
    limits_y = sorted(limits_y)
    schlieren_raw = schlieren_raw[np.arange(*limits_y), :][:, np.arange(*limits_x)]
    schlieren_raw /= schlieren_raw.max()
    schlieren_raw = skimage.filters.unsharp_mask(
        schlieren_raw,
        radius=1.5,
        amount=3,
    )
    df_meas = df_meas[(df_meas["loc_px"] >= limits_y[0]) & (df_meas["loc_px"] <= limits_y[1])]
    df_meas["loc_px"] -= limits_y[0]

    # raw frame
    name = "schlieren_frame_raw"
    fig, ax = plt.subplots(figsize=(image_width, image_height))
    fig.canvas.set_window_title(name)
    ax.imshow(schlieren_raw, cmap=cmap)
    ax.axis("off")
    # ax.set_title("Raw")
    ax.grid(False)
    ax.add_artist(
        copy(schlieren_scalebar),
    )
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )

    # frame with triple point measurements
    name = "schlieren_frame_measurements"
    fig, ax = plt.subplots(figsize=(image_width, image_height))
    fig.canvas.set_window_title(name)
    ax.imshow(schlieren_raw, cmap=cmap)
    ax.axis("off")
    # ax.set_title("Measurements")
    ax.grid(False)
    for loc_px in df_meas[(df_meas["date"] == date) & (df_meas["shot"] == shot) & (df_meas["frame"] == frame)][
        "loc_px"
    ]:
        plt.axhline(
            loc_px,
            c=COLOR_SC,
            lw=0.5,
        )

    ax.add_artist(
        copy(schlieren_scalebar),
    )
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def calculate_schlieren_cell_size(
    df_schlieren_frames,
    iqr_fencing=False,
    estimator=np.mean,
):
    """

    Parameters
    ----------
    df_schlieren_frames : pd.DataFrame
        DataFrame containing schlieren data
    iqr_fencing : bool
        Remove outliers using IQR fencing?
    estimator : func
        Estimator function to use

    Returns
    -------
    Tuple[float, float, unp.uarray, int]

        * Measured nominal cell size (mm)
        * Measured cell size uncertainty (mm)
        * Individual deltas with uncertainties (mm)
        * Number of measurements
    """
    df_schlieren_frames = df_schlieren_frames.copy().dropna()
    if iqr_fencing:
        # remove outliers
        meas_mean = df_schlieren_frames["cell_size"].mean()
        meas_std = df_schlieren_frames["cell_size"].std()
        mask = (meas_mean - 1.5 * meas_std <= df_schlieren_frames["cell_size"]) & (
            df_schlieren_frames["cell_size"] <= meas_mean + 1.5 * meas_std
        )
        del meas_std, meas_mean  # make sure we use reduced dataset!
    else:
        # leave em
        mask = np.ones_like(df_schlieren_frames["cell_size"], dtype=bool)
    meas = unp.uarray(
        df_schlieren_frames["cell_size"][mask],
        df_schlieren_frames["u_cell_size"][mask].values,
    )
    n_meas = len(meas)
    nominal_values = unp.nominal_values(meas)
    # cell_size_meas = np.sum(meas) / n_meas
    cell_size_meas = 2 * estimator(meas)
    cell_size_uncert_population = nominal_values.std() / np.sqrt(n_meas) * t.ppf(0.975, n_meas - 1)
    # noinspection PyUnresolvedReferences
    cell_size_uncert_schlieren = np.sqrt(
        np.sum(
            np.square(
                [
                    cell_size_uncert_population,
                    cell_size_meas.std_dev,
                ]
            )
        )
    )
    uncertainty = {
        "instrument": cell_size_meas.std_dev,
        "population": cell_size_uncert_population,
        "total": cell_size_uncert_schlieren,
    }

    # noinspection PyUnresolvedReferences
    return (
        cell_size_meas.nominal_value,
        uncertainty,
        meas,
        n_meas,
    )


@timed
def plot_schlieren_measurement_distribution(
    schlieren_meas,
    cell_size_meas,
    cell_size_uncert,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plot the distribution of schlieren measurements

    Parameters
    ----------
    schlieren_meas : np.array
        Array of individual schlieren nominal measurements (mm)
    cell_size_meas : float
        Nominal mean cell size measurement (mm)
    cell_size_uncert : float
        Uncertainty in cell size (mm)
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "schlieren_measurement_distribution"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    sns.distplot(
        schlieren_meas,
        hist=False,
        # rug=True,
        ax=ax,
        color=COLOR_SC,
    )
    ax_ylim = ax.get_ylim()
    plt.fill_between(
        [cell_size_meas + cell_size_uncert, cell_size_meas - cell_size_uncert],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color=COLOR_SC,
        ec=None,
        zorder=-1,
    )
    ax.axvline(
        cell_size_meas,
        c=COLOR_SC,
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    ax.set_ylim(ax_ylim)
    ax.set_xlabel("Measured Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    # ax.set_title("Schlieren Cell Size Measurement Distribution")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def plot_all_schlieren_deltas_distribution(
    df_schlieren_frames,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plot the distribution of all schlieren deltas in the dataset

    Parameters
    ----------
    df_schlieren_frames : pd.DataFrame
        Dataframe containing all schlieren frame deltas
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "schlieren_all_deltas_distribution"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    deltas = (df_schlieren_frames["spatial_centerline"] * df_schlieren_frames["delta_px"]).dropna()
    sns.kdeplot(
        deltas,
        ax=ax,
        color=COLOR_SC,
        clip=[0, 100],
    )
    ax.set_xlabel("Triple Point Delta (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    # ax.set_title("Schlieren Triple Point Delta Distribution")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )

    return deltas


@timed
def plot_all_soot_foil_deltas_distribution(
    soot_foil_meas,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plot the distribution of all schlieren deltas in the dataset

    Parameters
    ----------
    soot_foil_meas : list
        List of all soot foil deltas
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "soot_foil_all_deltas_distribution"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    sns.kdeplot(
        soot_foil_meas,
        ax=ax,
        color=COLOR_SF,
        clip=[0, 100],
    )
    ax.set_xlabel("Triple Point Delta (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    # ax.set_title("Soot Foil Triple Point Delta Distribution")
    ax.grid(False)
    plt.xlim([0, 100])
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def plot_both_delta_distributions(
    df_schlieren_frames,
    soot_foil_meas,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plot the distribution of all schlieren and soot foil deltas in the dataset

    Parameters
    ----------
    df_schlieren_frames : pd.DataFrame
        Dataframe containing all schlieren frame deltas
    soot_foil_meas : list
        List of all soot foil deltas
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "all_deltas_distributions"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    deltas = (df_schlieren_frames["spatial_centerline"] * df_schlieren_frames["delta_px"]).dropna()
    sns.kdeplot(
        deltas,
        ax=ax,
        color=COLOR_SC,
        label="Schlieren",
        clip=[0, 100],
    )
    sns.kdeplot(
        soot_foil_meas,
        ax=ax,
        color=COLOR_SF,
        label="Soot Foil",
        clip=[0, 100],
    )
    ax.set_xlabel("Triple Point Delta (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    plt.xlim([0, 100])
    # ax.set_title("Triple Point Delta Distributions")
    ax.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


# noinspection DuplicatedCode
@timed
def plot_schlieren_measurement_convergence(
    schlieren_meas,
    schlieren_uncert,
    schlieren_meas_all,
    n_schlieren_meas,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plots convergence of schlieren measurements vs. number of measurements.

    Parameters
    ----------
    schlieren_meas : float
        Actual measured schlieren value
    schlieren_uncert : float
        Uncertainty in measured value
    schlieren_meas_all : np.array
        Array of individual schlieren nominal measurements (mm)
    n_schlieren_meas : int
        Number of schlieren measurements
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "schlieren_measurement_convergence"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    n_meas = np.arange(1, n_schlieren_meas + 1)
    schlieren_meas_all = pd.Series(schlieren_meas_all)
    running_mean = schlieren_meas_all.rolling(
        n_schlieren_meas,
        min_periods=0,
    ).median()
    ax.plot(
        n_meas,
        np.abs(running_mean - schlieren_meas) * 100 / schlieren_meas,
        c=COLOR_SC,
    )
    ax.axhline(
        schlieren_uncert / schlieren_meas * 100,
        c="k",
        alpha=0.5,
        zorder=-1,
        lw=0.5,
        ls=(0, (5, 1, 1, 1)),
    )
    ax.set_xlim([2, len(running_mean)])
    ax.set_xlabel("Number of Triple Point Deltas")
    ax.set_ylabel("Absolute Difference\nFrom Final (%)")
    # ax.set_title("Schlieren Cell Size Measurement Convergence")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


# noinspection DuplicatedCode
@timed
def plot_soot_foil_measurement_convergence(
    nominal_measurement,
    uncert,
    all_measurements,
    plot_width,
    plot_height,
    save=False,
):
    """
    Plots convergence of soot foil measurements vs. number of measurements.

    Parameters
    ----------
    nominal_measurement : float
        Actual measured value
    uncert : float
        Uncertainty in measured value
    all_measurements : np.array
        Array of individual nominal measurements (mm)
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save the plot

    Returns
    -------

    """
    name = "soot_foil_measurement_convergence"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    n_meas = len(all_measurements)
    meas_range = np.arange(1, n_meas + 1)
    all_measurements = pd.Series(all_measurements)
    running_mean = all_measurements.rolling(
        n_meas,
        min_periods=0,
    ).median()
    uncert_pct = uncert / nominal_measurement * 100
    ax.plot(
        meas_range,
        np.abs(running_mean - nominal_measurement) * 100 / nominal_measurement,
        c=COLOR_SF,
    )
    ax.axhline(
        uncert_pct,
        c="k",
        alpha=0.5,
        zorder=-1,
        lw=0.5,
        ls=(0, (5, 1, 1, 1)),
    )
    ax.set_xlim([2, len(running_mean)])
    ax.set_xlabel("Number of Triple Point Deltas")
    ax.set_ylabel("Absolute Difference\nFrom Final (%)")
    plt.ticklabel_format(
        style="sci",
        axis="x",
        scilimits=(0, 0),
        useMathText=True,
    )
    # ax.set_title("Soot Foil Cell Size Measurement Convergence")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )

    return uncert_pct


@timed
def build_soot_foil_images(
    cmap,
    image_height,
    save=False,
):
    """
     Generates images of:

        * Raw soot foil image next to traced soot foil
        * Zoomed in trace with arrows to demonstrate measurements

    Parameters
    ----------
    cmap : str
        Colormap to use for schlieren frame
    image_height : float or None
        Image height (in)
    save : bool
        Whether or not to save images

    Returns
    -------

    """
    # settings
    aspect_ratio = 2  # w/h
    image_width = aspect_ratio * image_height
    sf_scalebar = get_scale_bar(
        SF_DELTA_PX,
        SF_DELTA_MM,
        cell_size=25.4,
    )

    # read in foil images
    sf_img = sf_imread(os.path.join(SF_IMG_DIR, "square.png"))
    sf_img_lines_thk = sf_imread(os.path.join(SF_IMG_DIR, "lines_thk.png"))

    # display foil images
    name = "soot_foil_images_main"
    fig, ax = plt.subplots(1, 2, figsize=(image_width, image_height))
    fig.canvas.set_window_title(name)
    ax[0].imshow(sf_img, cmap=cmap)
    ax[0].axis("off")
    # ax[0].set_title("Soot Foil")
    ax[1].imshow(sf_img_lines_thk, cmap=cmap)
    ax[1].axis("off")
    # ax[1].set_title("Traced Cells")
    for a in ax:
        a.add_artist(
            copy(sf_scalebar),
        )
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )

    # read in zoomed lines
    sf_img_lines_z = sf_imread(os.path.join(SF_IMG_DIR, "lines_zoomed.png"))
    sf_img_lines_z = np.rot90(np.rot90(sf_img_lines_z))  # don't want to redo this

    # plot zoomed lines
    name = "soot_foil_lines_zoomed"
    fig, ax = plt.subplots(figsize=(image_height, image_height))
    fig.canvas.set_window_title(name)
    ax.imshow(sf_img_lines_z, cmap=cmap)
    plt.axis("off")
    # plt.title("Traced Cells\n(Close-up)")
    lines_scale = 900 / 330  # scaled up for quality
    arrow_x = 160 * lines_scale
    arrow_length = np.array([36, 32, 86, 52, 88, 35, 50]) * lines_scale
    arrow_y_top = np.array([-10, 20, 46, 126, 172, 254, 282]) * lines_scale
    n_arrows = len(arrow_length)
    for i in range(n_arrows):
        if i == 0:
            arrowstyle = "-|>"
        elif i == n_arrows - 1:
            arrowstyle = "<|-"
        else:
            arrowstyle = "<|-|>"
        arrow = patches.FancyArrowPatch(
            (arrow_x, arrow_y_top[i]),
            (arrow_x, arrow_y_top[i] + arrow_length[i]),
            arrowstyle=arrowstyle,
            mutation_scale=5,
            linewidth=0.75,
            color=COLOR_SF,
        )
        plt.gca().add_artist(arrow)
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )

    # regular vs irregular
    irreg_scalebar = get_scale_bar(
        2268,
        300,
        cell_size=25.4,
    )
    irregular_image_path = os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "foil images",
        "2020-10-26",
        "Shot 01",
        "square.png",
    )
    img_irregular = sf_imread(irregular_image_path, plot=True)
    img_regular = sf_imread(os.path.join(SF_IMG_DIR, "square_regular.png"))

    # display foil images
    name = "soot_foil_irregular_cells"
    fig, ax = plt.subplots(1, 2, figsize=(image_width, image_height))
    fig.canvas.set_window_title(name)
    ax[0].imshow(img_regular, cmap=cmap)
    ax[0].axis("off")
    # ax[0].set_title("Regular")
    ax[0].add_artist(copy(sf_scalebar))
    ax[1].imshow(img_irregular, cmap=cmap)
    ax[1].axis("off")
    # ax[1].set_title("Irregular")
    ax[1].add_artist(copy(irreg_scalebar))
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def soot_foil_px_cal_uncertainty(
    plot_width,
    plot_height,
    save=False,
):
    """
    Calculate soot foil pixel location uncertainty and plot measurement
    distribution from repeatability test.

    NOTE: this function modifies DF_SF_SPATIAL, therefore this should be run
    before calculations referencing soot foil uncertainty!

    Parameters
    ----------
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save images

    Returns
    -------

    """
    # add pixel delta calibration precision uncertainty
    # estimate using IMG_1983 (2020-12-27 Shot 03)
    px_cal_deltas = np.array(
        [
            2344,  # this is what is saved in the .xcf
            2347,
            2345,
            2345,
            2345,
            2344,
            2344,
            2345,
            2344,
            2345,
        ]
    )
    u_px_cal_deltas = px_cal_deltas.std() / np.sqrt(len(px_cal_deltas)) * t.ppf(0.975, len(px_cal_deltas) - 1)

    # calculate and apply new calibration pixel uncertainty
    # existing measurement accounts for sqrt2 from delta
    # this applies directly without that because it is a direct delta
    # measurement
    DF_SF_SPATIAL["u_delta_px"] = np.sqrt(
        np.sum(
            np.square(
                np.array(
                    [
                        DF_SF_SPATIAL["u_delta_px"],  # bias (preexisting)
                        u_px_cal_deltas,  # precision (new)
                    ]
                )
            )
        )
    )

    # no need to do this for calibration mm uncertainty because it's a direct
    # ruler
    # reading, not a measurement of an existing quantity with a ruler
    # (i.e. bias only)

    name = "soot_foil_px_cal_uncertainty_distribution"
    fig = plt.figure(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    sns.distplot(
        px_cal_deltas,
        hist=False,
        color=COLOR_SF,
    )
    ax_ylim = plt.ylim()
    plt.fill_between(
        [
            px_cal_deltas.mean() + u_px_cal_deltas,
            px_cal_deltas.mean() - u_px_cal_deltas,
        ],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color=COLOR_SF,
        ec=None,
        zorder=-1,
    )
    plt.axvline(
        px_cal_deltas.mean(),
        c=COLOR_SF,
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    plt.ylim(ax_ylim)
    # plt.title(
    #     "Soot Foil Pixel Calibration Distance\nRepeatability Distribution"
    # )
    plt.grid(False)
    plt.xlabel("Ruler Distance (px)")
    plt.ylabel("Probability\nDensity (1/px)")
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


def find_row_px_loc(row):
    white = 255
    row_locs = np.where(row == white)[0]
    double_check = row_locs[np.abs(np.diff([row_locs, np.roll(row_locs, -1)], axis=0)).flatten() > 1]
    meas = double_check[0] if len(double_check) else row_locs[0]
    return meas


def get_all_image_px_locs(img):
    return np.apply_along_axis(find_row_px_loc, 1, img)


def soot_foil_px_delta_uncertainty():
    # add measurement pixel location precision uncertainty
    # estimate using IMG_1983 (2020-12-27 Shot 03)
    images = funcs.post_processing.images.schlieren.find_images_in_dir(
        os.path.join(
            d_drive,
            "Data",
            "Processed",
            "Soot Foil",
            "foil images",
            "2020-12-27",
            "Shot 03",
            "uncertainty",
        ),
        ".png",
    )
    img_size = io.imread(images[0]).shape[0]  # get image size
    n_repeatability_images = len(images)
    repeatability_px_locs = (
        np.ones(
            (
                img_size,
                n_repeatability_images,
            )
        )
        * np.nan
    )
    for i, img_loc in enumerate(images):
        img = io.imread(img_loc)
        repeatability_px_locs[:, i] = get_all_image_px_locs(img)

    # use max std of all rows as uncertainty estimate
    u_px_delta_precision = (
        np.std(
            repeatability_px_locs,
            axis=1,
        ).max()
        / np.sqrt(n_repeatability_images)
        * t.ppf(
            0.975,
            n_repeatability_images - 1,
        )
    ) * np.sqrt(2)  # accounts for propagation in delta

    u_px_delta_bias = 0.5 * np.sqrt(2)  # accounts for propagation in delta

    # calculate and apply new measurement pixel location precision uncertainty
    uncert_total = np.sqrt(np.sum(np.square(np.array([u_px_delta_bias, u_px_delta_precision]))))

    uncert = {"bias": u_px_delta_bias, "precision": u_px_delta_precision, "total": uncert_total}

    return uncert


@timed
# noinspection PyUnresolvedReferences
def calculate_soot_foil_cell_size(
    # n_schlieren_meas,
    iqr_fencing,
    estimator=np.mean,
    use_cache=True,
    save_cache=False,
):
    """
    Calculates the mean cell size from soot foil images

    Parameters
    ----------
    # n_schlieren_meas : int
        Number of schlieren measurements, which is used to trim down the data
        set after outliers have been removed -- NOTE: this is being left in
        in case it needs to be used again later, however the randomly selected
        batch of measurements from the first time this was run has been
        preserved and will be used for the sake of continuity.
    iqr_fencing : bool
        Remove outliers using IQR fencing?
    estimator : func
        Estimator function to use
    use_cache : bool
        Use cached data?
    save_cache : bool
        Overwrite cached data?

    Returns
    -------
    Tuple[np.array, float, float, pd.DataFrame]

        * Per-foil measurements (mm)
        * Mean cell size (mm)
        * Cell size uncertainty (mm)
    """
    cache_file = os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Data",
        "soot_foil_measurement_study.h5",
    )
    uncert_delta_px = soot_foil_px_delta_uncertainty()

    if use_cache:
        with pd.HDFStore(cache_file, "r") as store:
            all_meas = store.data["measurements"].to_numpy()
            all_total_uncerts = store.data["total_uncertainties"].to_numpy()
            all_cal_px_uncerts = store.data["u_cal_px"].to_numpy()
            all_cal_mm_uncerts = store.data["u_cal_mm"].to_numpy()
    else:
        date_shot = (
            # remove 4 at random
            # np.random.choice(range(19), 4, False)
            # Out[3]: array([16, 4, 5, 6])
            # date, shot
            ("2020-11-12", 0),
            ("2020-11-13", 8),
            ("2020-11-23", 3),
            # ("2020-11-23", 4),
            # ("2020-11-23", 6),
            # ("2020-11-23", 7),
            ("2020-11-24", 0),
            ("2020-11-24", 3),
            ("2020-11-24", 7),
            ("2020-11-25", 0),
            ("2020-12-20", 8),
            ("2020-12-21", 9),
            ("2020-12-27", 0),
            ("2020-12-27", 1),
            ("2020-12-27", 2),
            # ("2020-12-27", 3),
            ("2020-12-27", 6),
            ("2020-12-27", 7),
            ("2020-12-27", 8),
        )
        u_d_px = uncert_delta_px["total"]

        all_meas = []
        all_total_uncerts = []
        all_dates = []
        all_shots = []
        all_cal_mm_uncerts = []
        all_cal_px_uncerts = []
        all_n_deltas = np.ones(len(date_shot)) * np.nan
        for idx, (date, shot) in enumerate(date_shot):
            cal_mm, cal_px, u_cal_mm, u_cal_px = DF_SF_SPATIAL[
                (DF_SF_SPATIAL["date"] == date) & (DF_SF_SPATIAL["shot"] == shot)
            ][["delta_mm", "delta_px", "u_delta_mm", "u_delta_px"]].to_numpy[0]
            d_px = pp_deltas.get_px_deltas_from_lines(
                os.path.join(
                    d_drive,
                    "Data",
                    "Processed",
                    "Soot Foil",
                    "foil images",
                    f"{date}",
                    f"Shot {shot:02d}",
                    "composite.png",
                ),
                apply_uncertainty=False,
            )
            all_n_deltas[idx] = len(d_px)
            all_cal_mm_uncerts.append(u_cal_mm)
            all_cal_px_uncerts.append(u_cal_px)

            # apply uncertainties
            d_px = unp.uarray(d_px, u_d_px)
            cal_mm = un.ufloat(cal_mm, u_cal_mm)
            cal_px = un.ufloat(cal_px, u_cal_px)

            # calculate!
            d_mm = d_px * cal_mm / cal_px
            all_meas.extend(list(unp.nominal_values(d_mm)))
            all_total_uncerts.extend(list(unp.std_devs(d_mm)))
            n_current_meas = len(d_mm)
            all_dates.extend(list([date] * n_current_meas))
            all_shots.extend(list([shot] * n_current_meas))

        if save_cache:
            df_meas = pd.DataFrame(
                [
                    pd.Series(all_dates, name="date"),
                    pd.Series(all_shots, name="shot"),
                    pd.Series(all_meas, name="measurements"),
                    pd.Series(all_total_uncerts, name="total_uncertainties"),
                    pd.Series(all_cal_px_uncerts, name="u_cal_px"),
                    pd.Series(all_cal_mm_uncerts, name="u_cal_mm"),
                ]
            ).T
            with pd.HDFStore(cache_file, "w") as store:
                store.put("data", df_meas)

    measurements = unp.uarray(
        all_meas,
        all_total_uncerts,
    )
    meas_nominal = unp.nominal_values(measurements)

    if iqr_fencing:
        # remove outliers
        mean = meas_nominal.mean()
        std = meas_nominal.std()
        meas_mask = (meas_nominal <= mean + std * 1.5) & (meas_nominal >= mean - std * 1.5)
        measurements = measurements[meas_mask]
        meas_nominal = meas_nominal[meas_mask]
        del mean, std  # don't accidentally reuse these!

    # scale to match number of samples with schlieren
    # reduced_indices = sorted(np.random.choice(
    #     np.arange(len(measurements)),
    #     n_schlieren_meas,
    #     replace=False,
    # ))
    # reduced_indices = [0, 1, 3, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    #
    # measurements = measurements[reduced_indices]
    # meas_nominal = meas_nominal[reduced_indices]
    # date_shot_index = pd.MultiIndex.from_tuples(date_shot)[reduced_indices]

    # read in data
    with pd.HDFStore("/d/Data/Processed/Data/data_soot_foil.h5", "r") as store:
        df_tube = store.data.set_index(["date", "shot"], drop=True)

    # # trim down to only dates/shots currently in use
    # df_tube = df_tube.loc[date_shot_index]

    # collect population uncertainty
    n_measurements = len(measurements)
    if n_measurements // 2:
        # an even numbered dataset means that median is the mean of the two
        # middle numbers. This drops the uncertainty, which is rude. Since
        # this is a huge dataset with a small uncertainty, this is done to be
        # conservative.
        n_measurements -= 1
        measurements = measurements[1:]
    cell_size_meas = estimator(measurements)
    cell_size_uncert_population = meas_nominal.std() / np.sqrt(n_measurements) * t.ppf(0.975, n_measurements - 1)

    # combine uncertainties
    cell_size_uncert = np.sqrt(np.sum(np.square([cell_size_uncert_population, cell_size_meas.std_dev])))
    uncertainty = {
        "instrument": {
            "triple_point_delta_px": uncert_delta_px,
            "cal_px": np.mean(all_cal_px_uncerts),
            "cal_mm": np.mean(all_cal_mm_uncerts),
            "total_mm": cell_size_meas.std_dev,
        },
        "population": cell_size_uncert_population,
        "total": cell_size_uncert,
    }

    return (
        measurements,
        cell_size_meas.nominal_value,
        uncertainty,
        df_tube,
        all_meas,
    )


@timed
def plot_soot_foil_measurement_distribution(
    measurements,
    cell_size_meas,
    cell_size_uncert,
    plot_width,
    plot_height,
    save=False,
):
    """


    Parameters
    ----------
    measurements : np.array
        Nominal cell size measurements (mm)
    cell_size_meas : float
        Nominal mean cell size measurement (mm)
    cell_size_uncert : float
        Cell size uncertainty (mm)
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save images

    Returns
    -------

    """
    name = "soot_foil_measurement_distribution"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    sns.distplot(
        measurements,
        hist=False,
        # rug=True,
        ax=ax,
        color=COLOR_SF,
    )
    ax.axvline(
        cell_size_meas,
        color=COLOR_SF,
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    ax_ylim = ax.get_ylim()
    ax.fill_between(
        [cell_size_meas + cell_size_uncert, cell_size_meas - cell_size_uncert],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color=COLOR_SF,
        ec=None,
        zorder=-1,
    )
    ax.set_ylim(ax_ylim)
    ax.set_xlabel("Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    # ax.set_title("Soot Foil Measurement Distribution")
    ax.grid(False)
    plt.xlim([0, 100])
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def perform_soot_foil_measurement_study(
    plot_width,
    plot_height,
    save,
    uncert_pct,
):
    cache_file = os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Data",
        "soot_foil_measurement_study.h5",
    )
    with pd.HDFStore(cache_file, "r") as store:
        data = store.data

    data["date"] = pd.to_datetime(data["date"])

    grouped = data.groupby(["date", "shot"])
    median_per_shot = np.ones(len(grouped)) * np.nan
    running_median = np.ones_like(median_per_shot) * np.nan
    mean_of_medians = np.ones_like(median_per_shot) * np.nan
    date_shot = []

    for i, ((date, shot), df_current) in enumerate(grouped):
        median_per_shot[i] = df_current["measurements"].median()
        running_median[i] = data[(data["date"] < date) | ((data["date"] == date) & (data["shot"] <= shot))][
            "measurements"
        ].median()
        date_shot.append(f"{date.date().isoformat()} shot {shot}")
        mean_of_medians[i] = median_per_shot[: i + 1].mean()

    # normalize shot median by mean of the two methods
    median_per_shot /= np.mean((mean_of_medians[-1], median_per_shot[-1]))
    median_per_shot = np.abs(median_per_shot - 1) * 100
    # normalize
    mean_of_medians /= mean_of_medians[-1]
    mean_of_medians = np.abs(mean_of_medians - 1) * 100
    running_median /= running_median[-1]
    running_median = np.abs(running_median - 1) * 100

    n_foils = np.arange(len(grouped)) + 1
    # plot_title = "Soot Foil Measurement"
    name = "soot_foil_comparison"
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    ax.plot(
        n_foils,
        median_per_shot,
        label="foil median",
        color=COLOR_SF,
        ls="None",
        marker=".",
        zorder=-1,
    )
    ax.plot(
        n_foils,
        mean_of_medians,
        label="mean of foil medians",
        color=hex_add(COLOR_SF, "2f2f2f"),
        linestyle="--",
    )
    ax.plot(
        n_foils,
        running_median,
        label="pooled median",
        color=hex_sub(COLOR_SF, "2f2f2f"),
        ls=":",
    )
    ax.axhline(
        uncert_pct,
        c="k",
        alpha=0.5,
        zorder=-1,
        lw=0.5,
        ls=(0, (5, 1, 1, 1)),
    )
    ax.legend(
        prop={"size": 6},
        frameon=False,
        bbox_to_anchor=(0, 1, 1, 0),
        loc="lower left",
        mode="expand",
        ncol=3,
    )
    ax.set_xticks(n_foils)
    ax.set_xlabel("Number of Soot Foils Measured")
    ax.set_ylabel("Absolute Difference\nFrom Final (%)")
    # fig.suptitle(plot_title, y=0.92)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def plot_cell_size_comparison(
    deltas_schlieren,
    cell_size_meas_schlieren,
    cell_size_uncert_schlieren,
    measurements_foil,
    cell_size_meas_foil,
    cell_size_uncert_foil,
    plot_width,
    plot_height,
    save=False,
    zero_centered=False,
):
    """
    Plot comparison between schlieren and soot foil cell size measurement
    distributions

    Parameters
    ----------
    deltas_schlieren : np.array
        Nominal schlieren deltas (mm)
    cell_size_meas_schlieren : float
        Nominal measured schlieren cell size (mm)
    cell_size_uncert_schlieren : float
        Estimated schlieren measurement uncertainty (mm)
    measurements_foil : np.array
        Nominal soot foil measurements (mm)
    cell_size_meas_foil : float
        Nominal measured soot foil cell size (mm)
    cell_size_uncert_foil : float
        Estimated soot foil measurement uncertainty (mm)
    plot_width : float
        Width of plot (in)
    plot_height : float
        Height of plot (in)
    save : bool
        Whether or not to save images
    zero_centered : bool
        Whether or not to center about zero

    Returns
    -------

    """
    plot_title = "Cell Size Measurement Distributions"
    name = "cell_size_comparison"
    measurements_schlieren = 2 * deltas_schlieren
    if zero_centered:
        name += "_zero_centered"
        plot_title += "\n(Zero Centered)"
        measurements_foil -= cell_size_meas_foil
        measurements_schlieren -= cell_size_meas_schlieren
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title(name)
    sns.distplot(  # schlieren
        measurements_schlieren,
        hist=False,
        ax=ax,
        label="Schlieren",
        color=COLOR_SC,
        kde_kws={"ls": "-"},
    )
    sns.distplot(  # soot foil
        measurements_foil,
        hist=False,
        ax=ax,
        label="Soot Foil",
        color=COLOR_SF,
        kde_kws={"ls": "-"},
    )
    plt.legend(frameon=False)
    if not zero_centered:
        ax_ylim = ax.get_ylim()
        plt.fill_between(  # schlieren
            [
                cell_size_meas_schlieren + cell_size_uncert_schlieren,
                cell_size_meas_schlieren - cell_size_uncert_schlieren,
            ],
            ax_ylim[0],
            ax_ylim[1],
            alpha=0.25,
            color=COLOR_SC,
            ec=None,
            zorder=-1,
        )
        ax.fill_between(  # soot foil
            [
                cell_size_meas_foil + cell_size_uncert_foil,
                cell_size_meas_foil - cell_size_uncert_foil,
            ],
            ax_ylim[0],
            ax_ylim[1],
            alpha=0.25,
            color=COLOR_SF,
            ec=None,
            zorder=-1,
        )
        ax.axvline(  # schlieren
            cell_size_meas_schlieren,
            c=COLOR_SC,
            ls="--",
            alpha=0.7,
            zorder=-1,
        )
        ax.axvline(  # soot foil
            cell_size_meas_foil,
            color=COLOR_SF,
            ls="--",
            alpha=0.7,
            zorder=-1,
        )
        ax.set_ylim(ax_ylim)
        plt.xlim([0, 100])
    else:
        plt.xlim([-100, 100])
    ax.set_xlabel("Measured Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    # ax.set_title(plot_title)
    ax.grid(False)
    sns.despine()
    plt.tight_layout()
    if save:
        plt.savefig(
            os.path.join(SAVE_LOC, f"{name}.{PLOT_FILETYPE}"),
            dpi=DPI,
        )


@timed
def get_initial_conditions(df_data):
    """
    Extract initial conditions from filtered tube dataframe

    Parameters
    ----------
    df_data : pd.DataFrame
        Dataframe of tube data filtered down to desired measurements

    Returns
    -------
    dict
        dictionary of measurements with uncertainties stored in the following
        keys:

        * "p_0" / "u_p_0"
        * "t_0" / "u_t_0"
        * "phi" / "u_phi"
        * "dil_mf" / "u_dil_mf"
    """
    out = {}
    for item in ("p_0", "t_0", "phi", "dil_mf", "wave_speed"):
        out[item] = np.nanmean(
            unp.uarray(
                df_data[item],
                df_data[f"u_{item}"],  # instrument uncertainty
            )
        ) + un.ufloat(  # add population uncertainty
            0, df_data[item].sem() * t.ppf(0.975, len(df_data[item]) - 1)
        )

    return out


def check_null_hypothesis(p_value, alpha):
    indeterminate_threshold = 0.1  # indeterminate within 10%
    if np.abs(p_value - alpha) / alpha <= indeterminate_threshold:
        null_means = "Neither accept nor reject"
    elif p_value > alpha:
        null_means = "Fail to reject"
    else:
        null_means = "Reject"

    return null_means


def get_title_block(title):
    return f"{title}\n{'='*len(title)}\n"


def main(
    remove_outliers,
    save,
    estimator,
):
    cmap = "Greys_r"
    plot_width = 4
    plot_height = 2
    image_height_soot_foil = 2
    image_height_schlieren = 3
    set_plot_format()
    report = ""

    # do schlieren stuff
    df_schlieren_frames, df_schlieren_tube = get_schlieren_data(estimator)
    build_schlieren_images(
        cmap,
        df_schlieren_frames,
        image_height=image_height_schlieren,
        save=save,
    )
    plot_all_schlieren_deltas_distribution(
        df_schlieren_frames,
        plot_width,
        plot_height,
        save,
    )
    (
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren_dict,
        schlieren_meas_deltas,
        n_schlieren_meas,
    ) = calculate_schlieren_cell_size(
        df_schlieren_frames,
        remove_outliers,
        estimator,
    )
    cell_size_uncert_schlieren = cell_size_uncert_schlieren_dict["total"]
    schlieren_meas = 2 * schlieren_meas_deltas
    initial_conditions_schlieren = get_initial_conditions(df_schlieren_tube)
    plot_schlieren_measurement_distribution(
        unp.nominal_values(schlieren_meas),
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        plot_width,
        plot_height,
        save,
    )
    plot_schlieren_measurement_convergence(
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        unp.nominal_values(schlieren_meas),
        n_schlieren_meas,
        plot_width,
        plot_height,
        save,
    )
    report += get_title_block("Measurements")
    report += (
        f"schlieren: "
        f"{cell_size_meas_schlieren:.2f}+/-"
        f"{cell_size_uncert_schlieren:.2f} mm\n"
        f"    u_instrument: {cell_size_uncert_schlieren_dict['instrument']}\n"
        f"    u_population: {cell_size_uncert_schlieren_dict['population']}\n"
        f"    u_total: {cell_size_uncert_schlieren_dict['total']}\n"
    )

    # do soot foil stuff
    build_soot_foil_images(
        cmap,
        image_height_soot_foil,
        save,
    )
    soot_foil_px_cal_uncertainty(  # THIS UPDATES DF_SF_SPATIAL
        plot_width,
        plot_height,
        save,
    )
    (
        measurements_foil,
        cell_size_meas_foil,
        cell_size_uncert_foil_dict,
        df_tube_soot_foil,
        all_foil_meas,
    ) = calculate_soot_foil_cell_size(
        remove_outliers,
        estimator,
        # use_cache=False,
        # save_cache=True,
    )
    cell_size_uncert_foil = cell_size_uncert_foil_dict["total"]
    plot_all_soot_foil_deltas_distribution(
        all_foil_meas,
        plot_width,
        plot_height,
        save,
    )
    uncert_pct = plot_soot_foil_measurement_convergence(
        cell_size_meas_foil,
        cell_size_uncert_foil,
        all_foil_meas,
        plot_width,
        plot_height,
        save,
    )
    perform_soot_foil_measurement_study(
        plot_width,
        plot_height,
        save,
        uncert_pct,
    )
    plot_both_delta_distributions(
        df_schlieren_frames,
        all_foil_meas,
        plot_width,
        plot_height,
        save,
    )
    initial_conditions_soot_foil = get_initial_conditions(df_tube_soot_foil)
    plot_soot_foil_measurement_distribution(
        unp.nominal_values(measurements_foil),
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
        save,
    )
    cs_mean = np.mean([cell_size_meas_foil, cell_size_meas_schlieren])
    report += (
        f"soot foil: "
        f"{cell_size_meas_foil:.2f}+/-"
        f"{cell_size_uncert_foil:.2f} mm\n"
        f"    u_instrument: {cell_size_uncert_foil_dict['instrument']}\n"
        f"    u_population: {cell_size_uncert_foil_dict['population']}\n"
        f"    u_total: {cell_size_uncert_foil_dict['total']}\n"
        f"ID/cell size: {144.018 / cs_mean:.2f}\n"
        f"# SF deltas: {len(all_foil_meas)}\n"
        f"# schlieren deltas: {len(schlieren_meas_deltas)}\n"
    )

    # comparison
    plot_cell_size_comparison(
        unp.nominal_values(schlieren_meas_deltas),
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        unp.nominal_values(measurements_foil),
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
        save,
        zero_centered=False,
    )
    plot_cell_size_comparison(
        unp.nominal_values(schlieren_meas_deltas),
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        unp.nominal_values(measurements_foil),
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
        save,
        zero_centered=True,
    )
    foil_to_schlieren = un.ufloat(
        cell_size_meas_foil,
        cell_size_uncert_foil,
    ) / un.ufloat(
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
    )
    report += f"soot foil/schlieren ratio: {foil_to_schlieren:.2f}\n\n"

    alpha = 0.05
    # compare means using from_stats so we can apply extra uncertainties
    t_stat, t_p_value = ttest_ind_from_stats(
        cell_size_meas_foil,
        cell_size_uncert_foil,
        len(measurements_foil),
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        len(schlieren_meas),
        equal_var=False,
    )
    t_test_null = check_null_hypothesis(t_p_value, alpha)
    means = f"{t_test_null} the null hypothesis that means are equal"
    report += get_title_block("Means")
    report += f"{means}\n" f"    test statistic: {t_stat:0.2f}\n" f"    p: {t_p_value:0.3e}\n" f"    a: {alpha}\n\n"

    # compare distributions
    ks_stat, ks_p_value = ks_2samp(
        unp.nominal_values(measurements_foil),
        unp.nominal_values(schlieren_meas),
    )
    ks_test_null = check_null_hypothesis(ks_p_value, alpha)
    dists = f"{ks_test_null} the null hypothesis that distributions are equal"
    report += get_title_block("Distributions")
    report += f"{dists}\n" f"    test statistic: {ks_stat:0.2f}\n" f"    p: {ks_p_value:0.3e}\n" f"    a: {alpha}\n\n"

    # center distributions about zero and compare them
    ks_stat, ks_p_value = ks_2samp(
        unp.nominal_values(measurements_foil) - cell_size_meas_foil,
        unp.nominal_values(schlieren_meas) - cell_size_meas_schlieren,
    )
    ks_test_null = check_null_hypothesis(ks_p_value, alpha)
    dists = f"{ks_test_null} the null hypothesis that distributions are equal"
    report += get_title_block("Distributions (zero centered)")
    report += f"{dists}\n" f"    test statistic: {ks_stat:0.2f}\n" f"    p: {ks_p_value:0.3e}\n" f"    a: {alpha}\n\n"
    report += get_title_block("Initial Conditions")
    cj = 2030.0914212517014  # cj speed from previous calc (m/s)
    report += (
        "schlieren:\n"
        f"    P: {initial_conditions_schlieren['p_0']:.2f} Pa\n"
        f"    T: {initial_conditions_schlieren['t_0']:.2f} K\n"
        f"    phi: {initial_conditions_schlieren['phi']:.3f}\n"
        f"    dil mf: {initial_conditions_schlieren['dil_mf'] * 100:.2f} %\n"
        f"    speed: {initial_conditions_schlieren['wave_speed']:.3f} m/s\n"
        f"           {initial_conditions_schlieren['wave_speed'] / cj:.3f} "
        f"% CJ\n"
    )
    report += (
        "soot foil:\n"
        f"    P: {initial_conditions_soot_foil['p_0']:.2f} Pa\n"
        f"    T: {initial_conditions_soot_foil['t_0']:.2f} K\n"
        f"    phi: {initial_conditions_soot_foil['phi']:.3f}\n"
        f"    dil mf: {initial_conditions_soot_foil['dil_mf'] * 100:.2f} %\n"
        f"    speed: {initial_conditions_soot_foil['wave_speed']:.3f} m/s\n"
        f"           {initial_conditions_soot_foil['wave_speed'] / cj:.3f} "
        f"% CJ\n\n"
    )

    t_stat, t_p_value = ttest_ind(
        df_schlieren_tube["wave_speed"].dropna(),
        df_tube_soot_foil["wave_speed"].dropna(),
        equal_var=False,
    )
    t_test_speed = check_null_hypothesis(t_p_value, alpha)
    report += get_title_block("Speeds")
    report += f"CJ speed: {int(cj)} m/s\n"
    means = f"{t_test_speed} the null hypothesis that means are equal"
    report += f"{means}\n" f"    test statistic: {t_stat:0.2f}\n" f"    p: {t_p_value:0.3e}\n" f"    a: {alpha}\n\n"

    print(report)
    if save:
        with open(os.path.join(SAVE_LOC, "report"), "w") as f:
            f.write(report)


if __name__ == "__main__":
    print("hello")
    # remove_outliers_from_data = False
    # save_results = True
    # cell_size_estimator = np.median
    # main(remove_outliers_from_data, save_results, cell_size_estimator)
    # plt.show()
