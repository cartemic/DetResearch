from unittest import TestCase

import numpy as np

from funcs.post_processing.images.soot_foil import deltas


class TestBinarizeArray(TestCase):
    def test_bad_value(self):
        """
        binarize_array raises an error on bad values
        """
        with self.subTest("zero"), self.assertRaises(ValueError):
            deltas._binarize_array(np.array([0]))

        with self.subTest("negative"), self.assertRaises(ValueError):
            deltas._binarize_array(np.array([0]))

        with self.subTest("NaN"), self.assertRaises(ValueError), self.assertWarns(RuntimeWarning):
            # numpy throws a runtime warning for all-NaN slice
            deltas._binarize_array(np.array([np.nan]))

    def test_good_value(self):
        """
        binarize_array returns an array with only 0, 1
        """
        arr_in = np.array([0.5, 0.5, 0.0, 0.1])
        good = np.array([1, 1, 0, 0], dtype="int")

        test = deltas._binarize_array(arr_in)

        with self.subTest("values"):
            np.testing.assert_allclose(test, good)

        with self.subTest("dtypes"):
            self.assertEqual(test.dtype, good.dtype)


class TestGetDiffsFromSubRow(TestCase):
    def test_good_value(self):
        """
        get_diffs_from_sub_row works properly
        """
        sub_row = np.array([np.nan, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        good = np.array([3, 2, 2])

        test = deltas._get_diffs_from_sub_row(sub_row)

        np.testing.assert_array_equal(test, good)


class TestGetDiffsFromRow(TestCase):
    def test_all_nan(self):
        """
        get_diffs_from_row handles all-NaN input
        """
        test = deltas._get_diffs_from_row(np.array([np.nan]))

        np.testing.assert_array_equal(test, np.array([]))

    def test_all_zero(self):
        """
        get_diffs_from_row handles all-zero input
        """
        test = deltas._get_diffs_from_row(np.array([0]))

        np.testing.assert_array_equal(test, np.array([]))

    def test_good_value(self):
        """
        get_diffs_from_row works properly
        """
        row = np.array([1, 0, 0, 1, np.nan, 1, 0, 1, np.nan, np.nan, 1, np.nan])
        good = np.array([3, 2])

        test = deltas._get_diffs_from_row(row)

        np.testing.assert_array_equal(test, good)


class TestSlowGetDeltas(TestCase):
    def test_shape_mismatch(self):
        """
        _slow_get_deltas raises an error when image shapes don't match
        """
        img = np.array([1, 0])
        mask = np.array([1, 0, 1])

        with self.assertRaises(ValueError):
            deltas._slow_get_deltas(
                img=img,
                mask=mask,
            )

    def test_good_value_masked(self):
        """
        _slow_get_deltas works properly
        """
        img = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
            ]
        )
        mask = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 1],  # NaNs out both 0 and 1
                [0, 1, 1, 1, 1, 0, 0, 0],  # NaNs out both 0 and 1
            ]
        )
        good = np.array([2, 3, 2, 3, 2])  # second row only has the 3 px gap

        test = deltas._slow_get_deltas(
            img=img,
            mask=mask,
        )

        np.testing.assert_array_equal(test, good)

    def test_good_value_unmasked(self):
        """
        _slow_get_deltas works properly
        """
        img = np.array(
            [
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
            ]
        )
        mask = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        good = np.array([2, 3, 2, 2, 3, 2])

        test = deltas._slow_get_deltas(
            img=img,
            mask=mask,
        )

        np.testing.assert_array_equal(test, good)
