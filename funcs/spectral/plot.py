import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def all_results(run_output):
    pass


def measurements(
        line_cell_sizes,
        line_intensities,
        df_measurements,
        to_measure=None
):
    fig_meas, ax_meas = plt.subplots(
        1, 2,
        figsize=(12, 4),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    # left plot
    title_meas_pks = "Measurement Peaks"
    if isinstance(to_measure, float):
        title_meas_pks += r" (Relative Intensity $\geq$" + \
                          f" {to_measure * 100:.0f}%)"
    elif isinstance(to_measure, int):
        title_meas_pks += f" (Top {to_measure} Measurements" + \
                          " by Relative Intensity)"
    ax_meas[0].set_title(title_meas_pks)
    ax_meas[0].semilogx(
        line_cell_sizes,
        line_intensities
    )
    ax_meas[0].semilogx(
        df_measurements["Cell Size"],
        df_measurements["Intensity"],
        "ro"
    )
    ax_meas[0].set_xlabel("Cell Size (mm)")
    ax_meas[0].set_ylabel("Intensity")

    # right plot
    ax_meas[1].set_title("Measured Cell Sizes")
    ax_meas[1].bar(
        df_measurements["Cell Size"].to_string(
            float_format="%.2f",
            index=False
        ).split(),
        df_measurements["Relative Energy"],
    )
    ax_meas[1].set_xlabel("Cell Size (mm)")
    ax_meas[1].set_ylabel("Relative Intensity (%)")

    # both plots
    sns.despine()

    return fig_meas, ax_meas


def scans(
        angular_scan_radius,
        angular_scan_angles,
        angular_scan_intensities,
        angular_scan_window,
        radial_scan_angle,
        radial_scan_radii,
        radial_scan_intensities,
        radial_scan_window
):
    angular_scan_window = angular_scan_window ** 2 + 1
    radial_scan_window = radial_scan_window ** 2 + 1
    fig_scans, ax_scans = plt.subplots(1, 2, figsize=(8, 4))

    # left plot
    ax_scans[0].set_title(
        f"Intensity at R={angular_scan_radius} px\n" +
        f"{angular_scan_window} x {angular_scan_window} window"
    )
    ax_scans[0].set_xlabel("Angle (degrees)")
    ax_scans[0].set_ylabel("PSD Intensity")
    ax_scans[0].plot(angular_scan_angles, angular_scan_intensities)
    ax_scans[0].axvline(
        radial_scan_angle,
        c="k",
        ls="--",
        label="best angle",
        zorder=-1
    )

    # right plot
    ax_scans[1].set_title(
        r"Intensity at $\theta$=" +
        f"{radial_scan_angle:.1f} degrees\n" +
        f"{radial_scan_window} x {radial_scan_window} window"
    )
    ax_scans[1].set_xlabel("Radius (px)")
    ax_scans[1].set_ylabel("Intensity")
    ax_scans[1].plot(radial_scan_radii, radial_scan_intensities)

    # both plots
    sns.despine()

    return fig_scans, ax_scans


def image_filtering(
    base_image,
    base_psd,
    xc,
    yc,
    masked_psd,
    filtered_image,
    edge_detected,
    final_psd,
    scan_radius,
    figsize=None
):
    fig_images, ax_images = plt.subplots(2, 3, figsize=figsize)
    fig_images.canvas.set_window_title("Images")
    ax_images = ax_images
    for a in ax_images.flatten():
        a.axis("off")
    ax_images[0, 0].set_title("Base Image")
    ax_images[0, 0].imshow(base_image)

    ax_images[0, 1].set_title("Base PSD")
    ax_images[0, 1].imshow(base_psd)

    ax_images[0, 2].set_title("Masked PSD")
    ax_images[0, 2].imshow(masked_psd)

    ax_images[1, 0].set_title("FFT Filtered")
    ax_images[1, 0].imshow(filtered_image)

    ax_images[1, 1].set_title("Edge Detected")
    ax_images[1, 1].imshow(edge_detected)

    ax_images[1, 2].set_title("Final PSD")
    ax_images[1, 2].imshow(final_psd)
    ax_images[1, 2].add_artist(
        Circle(
            (xc, yc),
            scan_radius,
            color="k",
            ls="--",
            fill=False,
            label="Scan Circle"
        )
    )
    return fig_images, ax_images
