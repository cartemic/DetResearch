import os

import generate_images
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from skimage import io

from funcs.post_processing.images.soot_foil import deltas as pp_deltas


def main():
    image_location = os.path.join(
        generate_images.SF_IMG_DIR,
        "composite.png",
    )
    composite_image = io.imread(image_location, as_gray=True)
    actual_image_rows = composite_image.shape[1]
    deltas = []
    distance = []
    for i in range(composite_image.shape[0]):
        # ruff: noqa: SLF001
        row_deltas = pp_deltas._get_measurement_from_row(
            composite_image[i],
            255,
        )
        deltas.extend(row_deltas)
        distance.extend([i % actual_image_rows] * len(row_deltas))

    deltas = np.array(deltas)
    distance = np.array(distance)

    plt.figure()
    regression_all = linregress(distance, deltas)  # type: LinregressResult
    plt.plot(distance, deltas, label="deltas", marker=".", ls="", alpha=0.4)
    plt.plot(
        distance,
        regression_all.slope * distance + regression_all.intercept,
        label=f"regression R^2={regression_all.rvalue ** 2:.2f}",
    )
    plt.ylabel("triple point delta (px)")
    plt.xlabel("foil position (px)")
    plt.title("all measurements")
    plt.legend()

    plt.figure()
    delta_series = pd.Series(index=distance, data=deltas)
    delta_series = delta_series.groupby(delta_series.index).median()
    regression_median = linregress(
        delta_series.index,
        delta_series.values,
    )  # type: LinregressResult
    plt.plot(
        delta_series.index,
        delta_series.values,
        label="deltas",
        marker=".",
        ls="",
        alpha=0.4,
    )
    plt.plot(
        delta_series.index,
        regression_median.slope * delta_series.index + regression_median.intercept,
        label=f"regression R^2={regression_median.rvalue ** 2:.2f}",
    )
    plt.ylabel("triple point delta (px)")
    plt.xlabel("foil position (px)")
    plt.title("median measurements")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
