import dataclasses
import datetime as dt
import multiprocessing as mp
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tqdm

from funcs.dir import d_drive
from funcs.post_processing.images.soot_foil.deltas import Shot, get_px_deltas_from_lines

IMAGES_BASE_DIR: str = os.path.join(d_drive, "Data", "Processed", "Soot Foil", "foil images")
DF_SPATIAL: pd.DataFrame = pd.read_csv(
    os.path.join(d_drive, "Data", "Processed", "Soot Foil", "spatial_calibrations.csv")
)


@dataclasses.dataclass(frozen=True)
class _ImageName:
    image: str
    mask: str

    def has_a_mask_file(self, directory: str):
        return os.path.exists(os.path.join(directory, self.mask))


# probably not worth fixing this
# ruff: noqa: RUF009
@dataclasses.dataclass(frozen=True)
class ImageNames:
    dir0: _ImageName = _ImageName(image="dir0.png", mask="mask0.png")
    dir1: _ImageName = _ImageName(image="dir1.png", mask="mask1.png")


def read_all_shots() -> List[Shot]:
    all_shots = []
    with open(os.path.join(os.path.dirname(__file__), "shots_to_measure.csv"), "r") as f:
        good_n_cols = 2
        for line in f.readlines():
            date: str
            shot_no: str
            possible_info = line.strip().split(",")
            if len(possible_info) == good_n_cols:
                date, shot_no = possible_info
                shot_no: int = int(shot_no)
                all_shots.append(Shot(date=date, shot_no=shot_no, base_dir=IMAGES_BASE_DIR))

    return all_shots


ALL_SHOTS: List[Shot] = read_all_shots()
with pd.HDFStore(os.path.join(d_drive, "Data", "Processed", "Soot Foil", "tube_data.h5"), "r") as _store:
    TUBE_DATA: pd.DataFrame = _store.data.filter(
        [
            "date",
            "shot",
            "fuel",
            "oxidizer",
            "diluent",
            "t_0",
            "u_t_0",
            "p_0_nom",
            "p_0",
            "u_p_0",
            "phi_nom",
            "phi",
            "u_phi",
            "dil_mf_nom",
            "dil_mf",
            "u_dil_mf",
            "wave_speed",
            "u_wave_speed",
        ]
    )


def get_spatial(shot: Shot) -> float:
    """
    Collect the spatial calibration factor for a given shot

    Parameters
    ----------
    shot: Shot instance

    Returns
    -------
    pixel-to-mm conversion factor (mm/px)
    """
    this_spatial = DF_SPATIAL[(DF_SPATIAL["date"] == shot.date) & (DF_SPATIAL["shot"] == shot.shot_no)]
    px_to_mm = (this_spatial["delta_mm"] / this_spatial["delta_px"]).iloc[0] if len(this_spatial) else np.nan

    return px_to_mm


def measure_shot_image(shot: Shot) -> Tuple[float, float]:
    """
    Measure a single shot

    Parameters
    ----------
    shot: Shot instance

    Returns
    -------
    Median triple point distance, uncertainty (mm)
    """
    px_to_mm: float = get_spatial(shot)
    directions = [ImageNames.dir0, ImageNames.dir1]
    image_paths: List[str] = []
    mask_paths: List[Optional[str]] = []
    for direction in directions:
        image_paths.append(os.path.join(shot.directory, direction.image))
        if direction.has_a_mask_file(shot.directory):
            mask_paths.append(os.path.join(shot.directory, direction.mask))
        else:
            mask_paths.append(None)

    deltas = np.hstack((get_px_deltas_from_lines(img, mask) for img, mask in zip(image_paths, mask_paths)))
    delta_mm = np.median(deltas) * px_to_mm

    # noinspection PyUnresolvedReferences
    return delta_mm.nominal_value, delta_mm.std_dev


def collect_shot_info(shot: Shot) -> pd.Series:
    """
    Collect all tube data and shot measurement into a series

    Parameters
    ----------
    shot: Shot instance

    Returns
    -------
    Tube data + cell size information
    """
    tube_data: pd.DataFrame = TUBE_DATA[(TUBE_DATA["date"] == shot.date) & (TUBE_DATA["shot"] == shot.shot_no)]
    if tube_data.shape[0] != 1:
        raise ValueError(f"Incorrect number of rows found for {shot}: {tube_data.shape[0]} (1 required)")

    tube_data: pd.Series = tube_data.iloc[0].copy()
    cell_size, u_cell_size = measure_shot_image(shot)
    tube_data["cell_size"] = cell_size
    tube_data["u_cell_size"] = u_cell_size

    return tube_data


def measure_all_shots() -> pd.DataFrame:
    with mp.Pool() as p:
        results = (
            pd.DataFrame(tqdm.tqdm(p.imap(collect_shot_info, ALL_SHOTS), total=len(ALL_SHOTS)))
            .sort_index()
            .reset_index(drop=True)
        )

    return results


def main(output_location: str):
    with pd.HDFStore(output_location, "w") as store:
        store["data"] = measure_all_shots()


if __name__ == "__main__":
    # todo: get simulation results and add those in as well
    today = dt.date.today().isoformat()
    final_data_loc = os.path.join(os.path.dirname(__file__), f"measurements_{today}.h5")
    main(final_data_loc)
