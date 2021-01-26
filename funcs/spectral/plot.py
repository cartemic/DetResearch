import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Circle


def all_results(run_output):
    pass


def measurements(
        line_radii,
        line_intensities,
        df_measurements,
        to_measure=None
):
    fig_meas, ax_meas = plt.subplots(
        1, 3,
        figsize=(16, 4),
    )
    fig_meas.canvas.set_window_title("Measurements")

    max_radius = df_measurements["Radius"].max()
    min_radius = df_measurements["Radius"].min()
    rad_distance = max_radius - min_radius
    max_radius += 0.1 * rad_distance
    min_radius -= 0.1 * rad_distance

    # left plot
    title_meas_pks = "Measurement Peaks"
    if isinstance(to_measure, float):
        title_meas_pks += r" (Relative Intensity $\geq$" + \
                          f" {to_measure * 100:.0f}%)"
    elif isinstance(to_measure, int):
        title_meas_pks += f" (First {to_measure})"
    ax_meas[0].set_title(title_meas_pks)
    for i, rads in enumerate(line_radii):
        ax_meas[0].plot(
            rads[
                (rads >= min_radius) &
                (rads <= max_radius)
            ],
            line_intensities[i][
                (rads >= min_radius) &
                (rads <= max_radius)
            ],
            alpha=0.5,
        )
    sns.scatterplot(
        x="Radius",
        y="Intensity",
        hue="Theta",
        data=df_measurements,
        palette=["C0", "C1"],
        ax=ax_meas[0],
    )
    ax_meas[0].set_xlim([min_radius, max_radius])
    ax_meas[0].set_xlabel("Distance from Center (px)")
    ax_meas[0].set_ylabel("Intensity")

    # right plot
    ax_meas[1].set_title("Measured Cell Sizes")
    sns.lineplot(
        x="Cell Size",
        y="Relative Energy",
        hue="Theta",
        data=df_measurements.sort_values(["Cell Size"]),
        ax=ax_meas[1],
        legend=False,
        palette=["C0", "C1"],
        alpha=0.5,
    )
    sns.scatterplot(
        x="Cell Size",
        y="Relative Energy",
        hue="Theta",
        data=df_measurements.sort_values(["Cell Size"]),
        ax=ax_meas[1],
        legend=False,
        palette=["C0", "C1"]
    )

    ax_meas[1].set_xlabel("Cell Size (mm)")
    ax_meas[1].set_ylabel("Relative Intensity (%)")
    plt.setp(
        ax_meas[1].get_xticklabels(),
        rotation=30,
        horizontalalignment="center"
    )

    labels = []
    for i, (th, df_th) in enumerate(df_measurements.groupby("Theta")):
        circle_plot(
            df_th,
            ax_meas[2],
            marker_scale=0.5,
            color=f"C{i}"

        )
        labels.append(fr"$\theta$ = {th:0.2f} deg")

    ax_meas[2].set_ylim([0, df_measurements["Cell Size"].max()*1.1])
    circles = [Circle((0, 0), color=f"C{i}") for i in range(len(labels))]
    for a in ax_meas:
        a.legend(circles, labels, frameon=False)
    # both plots
    plt.tight_layout()
    sns.despine()

    return fig_meas, ax_meas


def image_filtering(
    base_image,
    base_psd,
    masked_psd,
    filtered_image,
    edge_detected,
    final_psd,
    figsize=(16, 10)
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

    return fig_images, ax_images


def circle_plot(df_cells, ax, color="C0", marker_scale=1.):
    df_plot = df_cells.sort_values("Radius").reset_index(drop=True)
    for i, row in df_plot.iterrows():
        ax.plot(
            i+1,
            row["Cell Size"],
            "o",
            color=color,
            ms=row["Relative Energy"]*marker_scale,
            alpha=0.7,
            clip_on=False,
        )

    ax.set_xlim([0, len(df_cells)+1])
    ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_title("Measured Cell Sizes")
    ax.set_xlabel("Peak #")
    ax.set_ylabel("Cell Size (mm)")
