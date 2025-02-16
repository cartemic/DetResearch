import os

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar

# todo: maybe fix this later, idk
# ruff: noqa: A004
from funcs import dir
from funcs.plots import spectral as plot
from funcs.post_processing.images.soot_foil import spectral

DF_SPATIAL = pd.read_csv(
    os.path.join(
        dir.d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "spatial_calibrations.csv",
    )
)


def analyze_foil(
    date,
    shot,
    angular_band=20.0,
    safe_radius=5.0,
    file_name="square.png",
    add_scalebar=True,
    rotate_image=False,
):
    spatial_info = DF_SPATIAL[(DF_SPATIAL["date"] == date) & (DF_SPATIAL["shot"] == shot)]
    image_path = os.path.join(
        dir.get_drive("d"), "Data", "Processed", "Soot Foil", "foil images", date, f"Shot {shot:02d}", file_name
    )
    if not os.path.exists(image_path):
        # reads better with this at the top
        raise FileExistsError(f"{image_path} does not exist")
    fft_pass = [
        angular_band,  # angular band (+/-)
        safe_radius,  # safe radius
    ]
    delta_px = spatial_info["delta_px"].iloc[0]
    delta_mm = spatial_info["delta_mm"].iloc[0]
    to_keep = 10

    # run analysis
    df_cells, plot_args = spectral.analysis.run(
        image_path,
        fft_pass,
        delta_px,
        delta_mm,
        bg_subtract=spatial_info["bg_subtract"].iloc[0],
        to_keep=to_keep,
        return_plot_outputs=True,
    )

    fig_img, ax_img = plot.image_filtering(*plot_args["image_filtering"], rotate_image=rotate_image)
    # _ = spectral.plot.scans(*plot_args["scans"])
    _ = plot.measurements(*plot_args["measurements"])

    from matplotlib.transforms import Affine2D

    r = Affine2D().rotate_deg(90)
    t = ax_img[0][0].get_transform()
    ax_img[0][0].set_transform(r + t)

    ax_img = ax_img.flatten()
    if add_scalebar:
        if rotate_image:
            scalebar_rotation = "vertical"
            scalebar_location = 3
        else:
            scalebar_rotation = "horizontal"
            scalebar_location = None

        for a in ax_img[[0, 3, 4]]:
            scalebar = ScaleBar(
                delta_mm / delta_px,
                "mm",
                fixed_value=df_cells["Cell Size"].iloc[0],
                label_formatter=(lambda x, u: f"{x:.2f} {u}"),
                border_pad=0.2,
                color="#FFFFFF",
                box_color="#444444",
                box_alpha=0,
                rotation=scalebar_rotation,
                location=scalebar_location,
            )
            a.add_artist(scalebar)

    return df_cells


if __name__ == "__main__":
    date = "2020-12-27"
    shot = 3

    df_meas = analyze_foil(date, shot, rotate_image=False)
    print(df_meas.drop(["Radius", "Intensity"], axis=1))
    print(df_meas.drop(["Radius", "Intensity"], axis=1).iloc[0]["Cell Size"])

    plt.show()
