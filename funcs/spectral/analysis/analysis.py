import numpy as np
import pandas as pd
from scipy.signal import argrelmax
from scipy.ndimage import gaussian_filter
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel

from . import image


def run(
        image_path,
        fft_pass,
        delta_px,
        delta_mm,
        bg_subtract=True,
        to_keep=None,
        return_plot_outputs=False
):
    """
    todo: clean up this documentation

    Parameters
    ----------
    image_path
    fft_pass : list or tuple or np.array
        list of:
            [0] - angular reject band
                reject base angle +/- angular reject band
            [1] - safe radius
                radius about center where band reject is ignored
    Returns
    -------

    """
    # read in image
    img_base = load_image(image_path)
    if bg_subtract:
        img_base -= gaussian_filter(img_base, 30)
    img_base = image.grayscale(img_base)

    # get power spectral density
    psd = image.calc_psd(img_base)

    # get x,y positions and angles for each pixel
    xc, yc = image.get_center(psd)
    psd_x, psd_y = image.get_xy(psd)
    psd_x_img = image.ax_to_img_coords(psd_x, xc)
    psd_y_img = -image.ax_to_img_coords(psd_y, yc)
    rad = image.get_radius(psd_x_img, psd_y_img)
    ang = image.get_angle(psd_x_img, psd_y_img)

    # build and apply FFT mask
    reject_band = 20
    reject_mask = image.get_reject_mask(
        ang,
        rad,
        [0, 90, 180, 270],
        reject_band,
        fft_pass[1]
    )

    fft = np.fft.fftshift(np.fft.fft2(img_base))
    psd_masked = image.grayscale(reject_mask * psd)  # reject only
    r_max = 250  # cut down on useless calcs by only looking at middle of psd
    best_angle_0 = image.find_best_angle(
        psd_masked[xc-r_max:xc+r_max, yc-r_max:yc+r_max],
        (reject_band, 90-reject_band)
    )
    best_angle_1 = image.find_best_angle(
        psd_masked[xc-r_max:xc+r_max, yc-r_max:yc+r_max],
        (90+reject_band, 180-reject_band)
    )
    pass_mask = image.get_pass_mask(
        ang,
        rad,
        [best_angle_0, best_angle_1, 180+best_angle_1, 180+best_angle_0],
        fft_pass[0],
        fft_pass[1]
    )
    psd_masked = image.grayscale(pass_mask * psd)  # pass only
    filtered = image.grayscale(
        np.real(np.fft.ifft2(np.fft.ifftshift(pass_mask * fft))),
    )

    # apply sobel filter
    edges = image.grayscale(sobel(filtered))

    # get edge detected PSD and recalculate best angles
    psd_final = image.calc_psd(edges)
    best_angle_0 = image.find_best_angle(
        psd_final[xc-r_max:xc+r_max, yc-r_max:yc+r_max],
        (reject_band, 90-reject_band)
    )
    best_angle_1 = image.find_best_angle(
        psd_final[xc-r_max:xc+r_max, yc-r_max:yc+r_max],
        (90+reject_band, 180-reject_band)
    )

    # perform radial intensity scan
    radius_0, int_radius_0 = image.get_radial_intensity(
        psd_final,
        best_angle_0,
    )
    radius_1, int_radius_1 = image.get_radial_intensity(
        psd_final,
        best_angle_1,
    )

    # find peaks
    dist_mask_0 = (radius_0 > 0)
    idx_pks_0 = argrelmax(int_radius_0[dist_mask_0])[0]
    dist_mask_1 = (radius_1 > 0)
    idx_pks_1 = argrelmax(int_radius_1[dist_mask_1])[0]

    # collect, filter and rescale measurements
    df_cells_0 = get_measurements_from_radial_scan(
        radius_0[dist_mask_0][idx_pks_0],
        int_radius_0[dist_mask_0][idx_pks_0],
        delta_px,
        delta_mm,
        best_angle_0,
        psd_final.shape[0],
        to_keep=to_keep,
        save_original=True
    )
    df_cells_0["Theta"] = best_angle_0
    df_cells_1 = get_measurements_from_radial_scan(
        radius_1[dist_mask_1][idx_pks_1],
        int_radius_1[dist_mask_1][idx_pks_1],
        delta_px,
        delta_mm,
        best_angle_1,
        psd_final.shape[0],
        to_keep=to_keep,
        save_original=True
    )
    df_cells_1["Theta"] = best_angle_1
    df_cells = pd.concat((df_cells_0, df_cells_1)).reset_index(drop=True)
    df_cells["Relative Energy"] = rescale_energy(df_cells["Intensity"])

    if return_plot_outputs:
        out = [
            df_cells.sort_values(
                ["Relative Energy"],
                ascending=False
            ).reset_index(drop=True),
            dict(
                image_filtering=[
                    img_base,
                    psd,
                    psd_masked,
                    filtered,
                    edges,
                    psd_final,
                ],
                measurements=[
                    (
                        radius_0[dist_mask_0],
                        radius_1[dist_mask_1]
                    ),
                    (
                        int_radius_0[dist_mask_0],
                        int_radius_1[dist_mask_1]
                    ),
                    df_cells,
                    to_keep
                ],
            )
        ]

    else:
        out = df_cells.sort_values(
            ["Relative Energy"],
            ascending=False
        ).reset_index(drop=True)

    return out


def load_image(image_path):
    return rgb2gray(io.imread(image_path))


def get_measurements_from_radial_scan(
        radii,
        intensities,
        delta_px,
        delta_mm,
        angle,
        img_size,
        to_keep=None,
        save_original=False,
):
    meas = image.peaks_to_measurements(
        radii,
        delta_px,
        delta_mm,
        angle,
        img_size
    )
    rescaled_intensities = rescale_energy(intensities)
    if save_original:
        meas = np.array([radii, intensities, meas, rescaled_intensities]).T
        cols = ["Radius", "Intensity", "Cell Size", "Relative Energy"]
    else:
        meas = np.array(meas, rescaled_intensities).T
        cols = ["Cell Size", "Relative Energy"]
    df_out = pd.DataFrame(
        data=meas,
        columns=cols
    )

    if to_keep is not None:
        if isinstance(to_keep, int):
            df_out = df_out.head(to_keep)
        elif isinstance(to_keep, float) and 0 < to_keep < 1:
            df_out = df_out[df_out["Relative Energy"] >= 100 * to_keep]
        else:
            raise ValueError("bad value of `to_keep`")

    return df_out


def rescale_energy(energy_peaks):
    return (energy_peaks - np.max(energy_peaks)) * \
        (100 - 10) / (np.max(energy_peaks) - np.min(energy_peaks)) + 100
