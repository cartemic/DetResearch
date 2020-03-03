# third party imports
import numpy as np
from skimage import io


def get_px_deltas_from_lines(
        img_path,
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

    Returns
    -------
    deltas : np.array
        triple point distances, in pixels.
    """
    img = io.imread(img_path)
    img_max = img.max()
    deltas = np.array([])
    for row in range(img.shape[0]):
        diffs = np.diff(np.where(img[row] == img_max)[0])
        deltas = np.append(deltas, diffs)
    return deltas
