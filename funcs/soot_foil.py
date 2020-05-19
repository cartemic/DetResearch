import numpy as np
import uncertainties as un
from skimage import io
from uncertainties import unumpy as unp

from .uncertainty import add_uncertainty_terms, u_cell

u_cell = u_cell["soot_foil"]


def get_px_deltas_from_lines(
        img_path,
        apply_uncertainty=True
):
    """
    Returns an array of per-row triple point deltas (in pixels) from a given
    image. The spatial calibration factor, mm_per_px, is required in order to
    remove deltas larger than the tube diameter, which are unphysical and a
    result of the limitations of soot foil measurement.

    Parameters
    ----------
    img_path : str
        path of the image containing traced triple point lines
    apply_uncertainty : bool
        True returns array of nominal values with uncertainties; False returns
        only nominal values

    Returns
    -------
    deltas : np.array or unp.uarray
        triple point distances, in pixels.
    """
    img = io.imread(img_path)
    img_max = img.max()
    deltas = np.array([])
    for row in range(img.shape[0]):
        diffs = np.diff(np.where(img[row] == img_max)[0])
        deltas = np.append(deltas, diffs)

    if apply_uncertainty:
        uncert = add_uncertainty_terms([
            u_cell["delta_px"]["b"],
            u_cell["delta_px"]["p"]
        ])
        deltas = unp.uarray(
            deltas,
            uncert
        )

    return deltas


def get_cell_size_from_deltas(
        deltas,
        l_px_i,
        l_mm_i,
        estimator=np.median
):
    """
    Converts pixel triple point deltas to cell size

    Parameters
    ----------
    deltas : np.array or pandas.Series
    l_px_i : float
        nominal value of spatial calibration factor (px)
    l_mm_i : float
        nominal value of spatial calibration factor (mm)
    estimator : function
        function used to estimate cell size from triple point measurements

    Returns
    -------
    un.ufloat
        estimated cell size
    """
    l_px_i = un.ufloat(
        l_px_i,
        add_uncertainty_terms([
            u_cell["l_px"]["b"],
            u_cell["l_px"]["p"]
        ])
    )
    l_mm_i = un.ufloat(
        l_mm_i,
        add_uncertainty_terms([
            u_cell["l_mm"]["b"],
            u_cell["l_mm"]["p"]
        ])
    )
    return 2 * estimator(deltas) * l_mm_i / l_px_i
