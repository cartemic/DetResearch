import os

import numpy as np
import pandas as pd

import funcs.schlieren as sc

BG_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    "data",
    "schlieren",
    "bg"
)


def test__find_images_in_dir():
    # noinspection PyTypeChecker
    assert len(sc._find_images_in_dir(BG_DIR)) == 2


def test_collect_shot_images():
    correct_counts = [2, 1]
    assert np.allclose(
        list(map(
            lambda x: len(x),
            sc.find_shot_images(os.path.dirname(BG_DIR))
        )),
        correct_counts
    )


def test_average_frames():
    assert np.allclose(
        sc.average_frames(sc._find_images_in_dir(BG_DIR)),
        np.ones((4, 4)) * 127.5
    )


def test_bg_subtract_all():
    assert np.allclose(
        sc.bg_subtract_all_frames(os.path.dirname(BG_DIR)),
        np.ones((4, 4)) * 127.5 + 2**15
    )


def test__calibrate():
    assert np.isclose(
        sc._calibrate([0, 1], [0, 1], 1).nominal_value,
        1 / np.sqrt(2)
    )


def test__filter_df_day_shot():
    df_test = pd.DataFrame(
        columns=["date", "shot"],
        data=[[date, shot]
              for shot in range(4)
              for date in ("2020-01-01", "2020-01-02", "2020-01-03")]
    )
    desired = [("2020-01-01", 1, 2),
               ("2020-01-03", 2, 3),
               ("2020-01-02", 2, 2)]
    good = (("2020-01-01", 1),
            ("2020-01-01", 2),
            ("2020-01-02", 2),
            ("2020-01-03", 2),
            ("2020-01-03", 3))
    results = sc._filter_df_day_shot(
        df_test,
        desired
    )[0]
    checks = np.zeros((len(good), 2)).astype(bool)
    for idx, ((date_g, shot_g), (_, s_row)) in enumerate(
            zip(good, results.iterrows())):
        checks[idx, 0] = s_row["date"] == date_g
        checks[idx, 1] = s_row["shot"] == shot_g

    assert np.all(checks)


class TestCheckStoredCalibration:
    def test_single_row(self):
        df_test = pd.DataFrame(
            columns=["spatial_near", "spatial_far", "spatial_centerline"]
        )
        df_test.loc[0, "spatial_near"] = 2

        results = sc._check_stored_calibrations(df_test)
        good = dict(
            near=dict(
                any=True,
                all=True,
                equal=True,
            ),
            far=dict(
                any=False,
                all=False,
                equal=True,
            ),
            centerline=dict(
                any=False,
                all=False,
                equal=True,
            ),
        )
        assert results == good

    def test_multi_row(self):
        df_test = pd.DataFrame(
            columns=["spatial_near", "spatial_far", "spatial_centerline"]
        )
        df_test.loc[0, "spatial_near"] = 2
        df_test.loc[1, "spatial_near"] = np.NaN
        df_test.loc[0, "spatial_far"] = 5.
        df_test.loc[1, "spatial_far"] = 5.
        df_test.loc[0, "spatial_centerline"] = 5.
        df_test.loc[1, "spatial_centerline"] = 4.
        results = sc._check_stored_calibrations(df_test)

        good = dict(
            near=dict(
                any=True,
                all=False,
                equal=False,
            ),
            far=dict(
                any=True,
                all=True,
                equal=True,
            ),
            centerline=dict(
                any=True,
                all=True,
                equal=False,
            ),
        )
        assert results == good
