import os

import numpy as np
import pytest

from funcs.post_processing.tube import diodes


class TestFindDiodeData:
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fake_diode_dirs")

    def test_not_empty(self):
        good_dirs = ["test1", "test3"]
        pattern = "test.tdms"
        pth_test = os.path.abspath(os.path.join(self.base_dir, "good"))
        good = [os.path.abspath(os.path.join(pth_test, g, pattern)) for g in good_dirs]
        result = diodes.find_diode_data(pth_test, pattern)
        assert all(np.char.compare_chararrays(result, good, "==", True))

    def test_empty(self):
        pattern = "test.tdms"
        pth_test = os.path.abspath(os.path.join(self.base_dir, "bad"))
        with pytest.raises(FileNotFoundError, match=f"No instances of {pattern} found"):
            diodes.find_diode_data(pth_test, pattern)
