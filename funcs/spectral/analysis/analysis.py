import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import sobel

from . import image


def run(
        image_path,
        fft_reject,
        angular_scan_radius,
        angular_scan_window,
        radial_scan_window,
        radial_scan_half_steps,
        delta_px,
        delta_mm,
        max_measurement_mm,
        to_keep=None,
        return_plot_outputs=False
):
    """
    todo: clean up this documentation

    Parameters
    ----------
    image_path
    fft_reject : list or tuple or np.array
        list of:
            [0] - base angles to reject
                center of angular reject band
            [1] - angular reject band
                reject base angle +/- angular reject band
            [2] - safe radius
                radius about center where band reject is ignored
    Returns
    -------

    """
    # read in image
    img_base = load_image(image_path)

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
    mask = image.get_mask(
        ang,
        rad,
        fft_reject[0],
        fft_reject[1],
        fft_reject[2]
    )
    fft = np.fft.fftshift(np.fft.fft2(img_base))
    psd_masked = image.grayscale(mask * psd)
    filtered = image.grayscale(
        np.real(np.fft.ifft2(np.fft.ifftshift(mask * fft)))
    )

    # apply sobel filter
    edges = image.grayscale(sobel(filtered))

    # get edge detected PSD
    psd_final = image.calc_psd(edges)

    # Perform angular intensity scan
    psd_filtered = image.calc_psd(edges)
    angle, int_angle = image.get_angular_intensity(
        psd_final,
        angular_scan_radius,
        angular_scan_window
    )

    ind_ang_pks = find_peaks(int_angle, prominence=0.1, width=20)[0]
    pks_ang = angle[ind_ang_pks]
    pks_int = int_angle[ind_ang_pks]
    best_angle = pks_ang[pks_ang > 0].max()

    # perform radial intensity scan
    radius, int_radius = image.get_radial_intensity(
        psd,
        best_angle,
        radial_scan_window,
        radial_scan_half_steps
    )

    # convert radius to cell size and find peaks
    distance, _ = image.peaks_to_measurements(
        radius,
        int_radius,
        delta_px,
        delta_mm,
        best_angle,
        psd_final.shape[0]
    )
    distance = np.where(np.isnan(distance), 1e16, distance)
    dist_mask = (distance >= 0) & (distance <= max_measurement_mm)
    idx_pks = find_peaks(
        int_radius[dist_mask],
        prominence=0.025
    )[0][1:]

    # filter and rescale measurements
    if to_keep is not None:
        df_cells = get_measurements_from_radial_scan(
            radius[dist_mask][idx_pks],
            int_radius[dist_mask][idx_pks],
            delta_px,
            delta_mm,
            best_angle,
            psd_final.shape[0],
            to_keep=to_keep,
            save_original=True
        )
        # rescale top n measurements
        df_cells = get_measurements_from_radial_scan(
            df_cells["Radius"],
            df_cells["Intensity"],
            delta_px,
            delta_mm,
            best_angle,
            psd_final.shape[0],
            save_original=True
        )
    else:
        df_cells = get_measurements_from_radial_scan(
            radius[dist_mask][idx_pks],
            int_radius[dist_mask][idx_pks],
            delta_px,
            delta_mm,
            best_angle,
            psd_final.shape[0],
            to_keep=to_keep,
            save_original=True
        )

    if return_plot_outputs:
        out = [
            df_cells,
            dict(
                image_filtering=[
                    img_base,
                    psd,
                    xc,
                    yc,
                    psd_masked,
                    filtered,
                    edges,
                    psd_final,
                    angular_scan_radius
                ],
                scans=[
                    angular_scan_radius,
                    angle,
                    int_angle,
                    angular_scan_window,
                    best_angle,
                    radius,
                    int_radius,
                    radial_scan_window

                ],
                measurements=[
                    distance[dist_mask],
                    int_radius[dist_mask],
                    df_cells,
                    to_keep
                ],
            )
        ]

    else:
        out = df_cells

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
        intensities,
        delta_px,
        delta_mm,
        angle,
        img_size
    )
    if save_original:
        meas = np.array([radii, intensities, *meas]).T
        cols = ["Radius", "Intensity", "Cell Size", "Relative Energy"]
    else:
        meas = np.array(meas).T
        cols = ["Cell Size", "Relative Energy"]
    df_out = pd.DataFrame(
        data=meas,
        columns=cols
    ).sort_values("Relative Energy", ascending=False).reset_index(drop=True)

    if to_keep is not None:
        if isinstance(to_keep, int):
            df_out = df_out.head(to_keep)
        elif isinstance(to_keep, float) and 0 < to_keep < 1:
            df_out = df_out[df_out["Relative Energy"] >= 100 * to_keep]
        else:
            raise ValueError("bad value of `to_keep`")

    return df_out

# if __name__ == "__main__":
#     from skimage import io, color
#     img = io.imread("../scripts/spectral/1841.png")
#     img = color.rgb2gray(img)
#     from matplotlib import pyplot as plt
#     # plt.imshow(img)
#     # plt.figure()
#     psd = calc_psd(img)
#     plt.imshow(psd)
#     plt.figure()
#     x, y = get_angular_intensity(psd, 100, 5)
#     plt.plot(x, y)
#     plt.figure()
#     x, y = get_radial_intensity(psd, 47, 0, half_steps=2048)
#     plt.plot(x, y)
#     #
#     # # from matlab, for px mm conversion checking
#     # pks_x = np.array([4.419, 9.016, 13.61, 17.85])
#     # pks_y = np.array([230.5, 224.9, 217.4, 197.7])
#     # n_px = 85
#     # n_mm = 50
#
#
#     pks_x = np.array([4.71308, 10.3554, 13.9602, 17.8002])
#     pks_y = np.array([229.223, 211.307, 203.069, 210.483])
#     n_px = 2284.
#     n_mm = 300.
#
#
#     print(peaks_to_measurements(pks_x, pks_y, n_px, n_mm, 30, psd.shape[0]))
#     #
#     # plt.figure()
#     # plt.loglog(*peaks_to_measurements(x, y, n_px, n_mm, 30, psd.shape[0]))
#     plt.show()
