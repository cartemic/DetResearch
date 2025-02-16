import numpy as np

from funcs.post_processing.images.soot_foil.spectral import image


def test_grayscale():
    test_img = np.array([100, 150, 175, 200])
    expected = np.array([0, 127.5, 191.25, 255])
    test_result = image.grayscale(test_img)
    assert np.allclose(test_result, expected)


def test_fspecial_gauss():
    # check against octave output
    good = np.array(
        [
            [1.134373655849507e-02, 8.381950580221061e-02, 1.134373655849507e-02],
            [8.381950580221061e-02, 6.193470305571772e-01, 8.381950580221061e-02],
            [1.134373655849507e-02, 8.381950580221061e-02, 1.134373655849507e-02],
        ]
    )
    assert np.allclose(good, image.fspecial_gauss())


def test_get_center():
    good_x = 112
    good_y = 65
    test_img = np.zeros((300, 729))
    test_img[good_y, good_x] = 1
    result = image.get_center(test_img)
    assert result == (good_x, good_y)


class TestFix:
    # NOTE: because of numba I can only send fix numpy arrays
    def test_positive(self):
        assert image.fix(np.array([1.78])) == 1

    def test_negative(self):
        assert image.fix(np.array([-37.8])) == -37


def test_get_radial_intensity():
    img = np.zeros((7, 7))
    img[:, 3] = 1
    good = (np.arange(img.shape[0]), np.ones(img.shape[0]))
    result = image.get_radial_intensity(img, 90)
    assert np.allclose(result, good)


def test_get_radius():
    assert image.get_radius(3, 4) == 5
