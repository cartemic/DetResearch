import os

import numpy as np
from skimage import color, io
from uncertainties import unumpy as unp

from funcs.uncertainty import add_uncertainty_terms, u_cell

from ._rust import _fast_get_deltas

u_cell = u_cell["soot_foil"]


class Shot:
    def __init__(
        self,
        date,
        shot_no,
        base_dir,
    ):
        """
        A cleaner-than-a-tuple organizer for date/shot info

        Parameters
        ----------
        date: str
            Shot date
        shot_no: int
            Shot number
        base_dir: str
            Base data directory for locating shot directory on disk
        """
        self.date = date
        self.shot_no = shot_no
        self._base_dir = base_dir

    def __repr__(self):
        return f"{self.date} {self.shot_string}"

    @property
    def shot_string(self):
        return f"Shot {self.shot_no:02d}"

    @shot_string.setter
    def shot_string(self, _):
        raise AttributeError("shot_string cannot be set")

    @property
    def dir_name(self):
        return os.path.join(self.date, self.shot_string)

    @dir_name.setter
    def dir_name(self, _):
        raise AttributeError("dir_name cannot be set")

    @property
    def directory(self):
        shot_dir = os.path.join(self._base_dir, self.dir_name)
        if not os.path.exists(shot_dir):
            raise NotADirectoryError(f"{shot_dir} does not exist")

        return shot_dir

    @directory.setter
    def directory(self, _):
        raise AttributeError("dir cannot be set")

    def __len__(self):
        return len(str(self))


def get_px_deltas_from_lines(img_path, mask_path=None, use_fast=True, apply_uncertainty=True):
    """
    Returns an array of per-row triple point deltas (in pixels) from a given
    image. The spatial calibration factor, mm_per_px, is required in order to
    remove deltas larger than the tube diameter, which are unphysical and a
    result of the limitations of soot foil measurement.

    Parameters
    ----------
    img_path : str
        path of the image containing traced triple point lines
    mask_path : Optional[str]
        Path to the damage mask image, which must have the same dimensions as the main image
    use_fast : bool
        If true, the faster compiled rust version of the processing script will be run. Probably linux x86_64 only.
    apply_uncertainty : bool
        True returns array of nominal values with uncertainties; False returns
        only nominal values

    Returns
    -------
    deltas : np.array or unp.uarray
        triple point distances, in pixels.
    """
    if use_fast:
        deltas = _fast_get_deltas(img_path, mask_path)
    else:
        img = color.rgb2gray(io.imread(img_path))
        mask = color.rgb2gray(io.imread(mask_path)) if mask_path else np.zeros_like(img)
        deltas = _slow_get_deltas(img, mask)

    if apply_uncertainty:
        uncert = add_uncertainty_terms([u_cell["delta_px"]["b"], u_cell["delta_px"]["p"]])
        deltas = unp.uarray(deltas, uncert)
    else:
        deltas = np.array(deltas)

    return deltas


def _load_image(img_path):
    """
    Loads in an image as a black and white array

    Parameters
    ----------
    img_path: str
        Path of image to be loaded in.

    Returns
    -------

    """
    return _binarize_array(color.rgb2gray(io.imread(img_path)))


def _binarize_array(arr_in):
    """
    Converts an input array to have integer values of 1 at max, and 0
    everywhere else.

    Parameters
    ----------
    arr_in: np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Converted array (copy)
    """
    arr_max = np.nanmax(arr_in)
    # this can still be NaN if the input array is all NaN
    if np.isnan(arr_max) or arr_max <= 0:
        raise ValueError("cannot process image with max <= 0 or max == NaN")

    arr_out = (arr_in.copy() / arr_in.max()).astype(int)
    arr_out[arr_out < 1] = 0

    return arr_out


def _slow_get_deltas(
    img,
    mask,
):
    """
    Returns an array of per-row triple point deltas (in pixels) from a given
    image. The spatial calibration factor, mm_per_px, is required in order to
    remove deltas larger than the tube diameter, which are unphysical and a
    result of the limitations of soot foil measurement.

    Parameters
    ----------
    img : np.ndarray
        Array containing traced triple point lines for a single direction.
    mask : np.ndarray
        Array containing traced triple point lines for a single direction.

    Returns
    -------
    deltas : np.ndarray
        Triple point distances, in pixels.
    """
    _img = img.copy()

    if mask.shape != _img.shape:
        raise ValueError(f"image shape mismatch: {mask.shape} vs. {_img.shape}")
    _mask = mask.copy()

    # NaN out exclusion zones
    _img = np.where(_mask == 1, np.nan, _img)

    deltas = []
    for i in range(_img.shape[0]):
        diffs = _get_diffs_from_row(_img[i])
        deltas.extend(diffs)

    deltas = np.array(deltas)

    return deltas


def _get_diffs_from_sub_row(sub_row):
    """
    The actual diff-getter. If two measurements are in adjacent pixels, this
    method selects the leftmost and rightmost adjacent locations and throws out
    the others (i.e. it only accepts measurements where the boundary location
    is >1 px away from the previous boundary location). The distance between
    the leftmost and rightmost adjacent locations is not counted.

    Parameters
    ----------
    sub_row: np.ndarray
        Portion of a single row of image pixels within a given exclusion zone.

    Returns
    -------
    np.ndarray
        Distances between cell boundary locations.
    """
    # locate cell boundaries
    cell_boundary_indices = np.where(sub_row == 1)[0]

    # find how far apart adjacent boundaries are
    cell_boundary_index_diffs = np.abs(cell_boundary_indices - np.roll(cell_boundary_indices, -1))[
        :-1
    ]  # throw out last diff -- roll wraps the first location around!

    # throw out adjacent boundaries
    cell_boundary_index_diffs = cell_boundary_index_diffs[cell_boundary_index_diffs > 1]

    return cell_boundary_index_diffs


def _get_diffs_from_row(row):
    """
    Get all pixel distances between cell boundaries for a single row in an image

    Parameters
    ----------
    row: np.ndarray
        Row containing 1 where cell boundaries exist, NaN where exclusion zones
        exist, and 0 otherwise.

    Returns
    -------
    np.ndarray
        Concatenated array of cell boundary distances within exclusion zones.
    """
    diffs = []
    # skip rows without useful data
    if not (np.all(np.isnan(row)) or np.allclose(row, 0)):
        # split into sub-arrays on NaN to enforce exclusion zones
        split_row = np.split(row, np.where(np.isnan(row))[0])
        for sub_row in split_row:
            diffs.extend(_get_diffs_from_sub_row(sub_row))

    return np.array(diffs)
