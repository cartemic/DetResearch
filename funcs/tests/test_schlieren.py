import os
import numpy as np
import funcs.schlieren as sc


BG_DIR = os.path.join(
    "D:\\",
    "Data",
    "Raw",
    "2019-10-22",
    "Camera",
    "shot 00",
    "bg"
)


# noinspection PyProtectedMember
def test_find_images_in_dir():
    # noinspection PyTypeChecker
    assert len(sc._find_images_in_dir(BG_DIR)) == 101


def test_collect_shot_images():
    correct_counts = [101, 1]
    assert all([
        len(dir_count) == correct
        for dir_count, correct in zip(
            sc.collect_shot_images(os.path.dirname(BG_DIR)),
            correct_counts
        )
    ])


def test_average_frames():
    # TODO: write a test for this
    # sc.average_frames()
    assert False


def test_bg_subtract_all():
    # TODO: write a test for this
    # sc.bg_subtract_all()
    assert False


# noinspection PyProtectedMember
def test_calibrate():
    assert np.isclose(sc._calibrate([0, 1], [0, 1], 1), 1 / np.sqrt(2))


# noinspection PyProtectedMember
def test_save_spatial_calibration():
    # TODO: write a test for this
    # sc._save_spatial_calibration()
    assert False
