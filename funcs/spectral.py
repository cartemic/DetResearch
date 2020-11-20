import numpy as np
from numba import jit, njit
from warnings import catch_warnings, simplefilter


def grayscale(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return 255 / (img_max - img_min) * (img - img_min)


def fspecial_gauss(
        shape=(3, 3),
        sigma=0.5
):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])

    from: https://stackoverflow.com/a/17201686
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sum_h = h.sum()
    if sum_h != 0:
        h /= sum_h
    return h


def calc_psd(img):
    wind = img * fspecial_gauss(img.shape, 0.2*min(img.shape))
    wind = grayscale(wind)
    fourtrans = np.fft.fft2(wind)
    absolu = np.abs(fourtrans)
    powspecden = np.square(absolu)
    loga = np.log(powspecden)
    final = np.fft.fftshift(loga)
    return grayscale(final)


@jit
def get_center(psd):
    cols = psd.shape[1]
    max_loc = np.argmax(psd)
    return max_loc % cols, max_loc // cols


@jit
def fix(n):
    if n < 0:
        sign = -1
    else:
        sign = 1
    return int(np.floor(abs(n)) * sign)


@njit
def get_angular_intensity(
        psd,
        radius,
        window,
        n_steps=1024
):
    half_steps = n_steps // 2
    intensity = np.zeros(n_steps)
    img_h, img_w = psd.shape
    buffer = np.zeros((img_w, img_w))
    win_size = (2 * window + 1) ** 2
    x_c, y_c = get_center(psd)

    for idx_int in range(n_steps):
        theta = np.pi / half_steps * (idx_int + 1 - half_steps)
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        x = x_c - radius * sin_th + 1
        y = y_c + radius * cos_th + 1
        m = fix(x)
        n = fix(y)

        if (0 <= m) & (m <= img_h) & (0 <= n) & (n <= img_w):
            for i in (m, m+1):
                for j in (n, n+1):
                    local_sum = 0
                    for k in range(-window, window + 1):
                        for l in range(-window, window + 1):
                            if (0 <= i + k) & (i + k < img_h) & \
                                    (0 <= j + l) & (j + l < img_w):
                                local_sum += psd[i + k - 1, j + l - 1] / \
                                    win_size
                    buffer[i, j] = local_sum

            p = x - m
            q = y - n
            comp_p = 1 - p
            comp_q = 1 - q
            intensity[idx_int] = \
                comp_p * comp_q * buffer[m, n] + \
                p * comp_q * buffer[m + 1, n] + \
                q * comp_p * buffer[m, n + 1] + \
                p * q * buffer[m + 1, n + 1]

    return 180 / half_steps * (np.arange(1, 1025) - half_steps), intensity


@njit
def get_radial_intensity(
        psd,
        theta,
        window,
        half_steps=1024
):
    n_steps = half_steps * 2 + 1  # mirror about 0
    img_h, img_w = psd.shape
    buffer = np.zeros((img_w, img_w)) * np.NaN
    x_c, y_c = get_center(psd)
    win_size = (2 * window + 1) ** 2

    intensity = np.zeros(n_steps)
    cos_theta = np.cos(theta * np.pi / 180)
    sin_theta = np.sin(theta * np.pi / 180)

    radii = np.sqrt(
        np.sum(
            np.square(
                np.array([img_h, img_w]) / 2
            )
        )
    ) / half_steps * np.arange(-half_steps, half_steps+1)

    for idx_int, radius in enumerate(radii):
        x = x_c - radius * sin_theta + 1
        y = y_c + radius * cos_theta + 1
        m = fix(x)
        n = fix(y)

        if (0 <= m) & (m < img_h) & (0 <= n) & (n < img_w):
            for i in (m, m+1):
                for j in (n, n+1):
                    local_sum = 0
                    for k in range(-window, window + 1):
                        for l in range(-window, window + 1):
                            if (0 <= i + k) & \
                                    (i + k < img_h) & \
                                    (0 <= j + l) & \
                                    (j + l < img_w):
                                local_sum += psd[i + k - 1, j + l - 1] / \
                                    win_size

                    buffer[i, j] = local_sum

            p = x - m
            q = y - n
            comp_p = 1 - p
            comp_q = 1 - q
            intensity[idx_int] = \
                comp_p * comp_q * buffer[m, n] + \
                p * comp_q * buffer[m + 1, n] + \
                q * comp_p * buffer[m, n + 1] + \
                p * q * buffer[m + 1, n + 1]

    return radii, intensity


def peaks_to_measurements(
        peak_x,
        peak_y,
        delta_px,
        delta_mm,
        theta,
        img_square_size
):
    px_perpendicular = img_square_size / np.where(
        np.isclose(peak_x, 0),
        np.NaN,
        peak_x
    )
    px_projected = px_perpendicular / np.abs(np.cos(theta * np.pi / 180))
    cell_size = delta_mm / delta_px * px_projected
    rescaled_energy = (peak_y - np.max(peak_y)) * \
        (100 - 10) / (np.max(peak_y) - np.min(peak_y)) + 100
    return cell_size, rescaled_energy


def img_to_ax_coords(points, center):
    return points + center


def ax_to_img_coords(points, center):
    return points - center


def get_xy(img):
    x = np.ones_like(img) * np.arange(img.shape[1])[:, None].T
    y = np.ones_like(img) * np.arange(img.shape[0])[:, None]
    return x, y


def get_radius(x, y):
    return np.sqrt(x**2 + y**2)


def get_angle(x, y):
    with catch_warnings():
        simplefilter("ignore")
        angle = 90 - (np.arctan(x / y) * 180 / np.pi)
        angle[y < 0] += 180
        angle[np.isnan(angle)] = 0
        return angle


def get_mask(thetas, radii, mask_angles, angle_pm, safe_rad):
    mask_out = np.ones_like(thetas).astype(bool)
    for a in mask_angles:
        if a < angle_pm:
            mask_out[
                ((thetas < np.mod(a + angle_pm, 360)) |
                 (thetas > np.mod(a - angle_pm, 360))) &
                (radii > safe_rad)
            ] = False
        else:
            mask_out[
                ((thetas < np.mod(a + angle_pm, 360)) &
                 (thetas > np.mod(a - angle_pm, 360))) &
                (radii > safe_rad)
            ] = False
    return mask_out


# if __name__ == "__main__":
#     from skimage import io, color
#     img = io.imread("../scripts/spectral/1841.png")
#     img = color.rgb2gray(img)
#     from matplotlib import pyplot as plt
#     # plt.imshow(img)
#     # plt.figure()
#     psd = calc_psd(img)
#     plt.imshow(psd)
#     plt.figure()
#     x, y = get_angular_intensity(psd, 100, 5)
#     plt.plot(x, y)
#     plt.figure()
#     x, y = get_radial_intensity(psd, 47, 0, half_steps=2048)
#     plt.plot(x, y)
#     #
#     # # from matlab, for px mm conversion checking
#     # pks_x = np.array([4.419, 9.016, 13.61, 17.85])
#     # pks_y = np.array([230.5, 224.9, 217.4, 197.7])
#     # n_px = 85
#     # n_mm = 50
#
#
#     pks_x = np.array([4.71308, 10.3554, 13.9602, 17.8002])
#     pks_y = np.array([229.223, 211.307, 203.069, 210.483])
#     n_px = 2284.
#     n_mm = 300.
#
#
#     print(peaks_to_measurements(pks_x, pks_y, n_px, n_mm, 30, psd.shape[0]))
#     #
#     # plt.figure()
#     # plt.loglog(*peaks_to_measurements(x, y, n_px, n_mm, 30, psd.shape[0]))
#     plt.show()
