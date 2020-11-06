import numpy as np
from numba import jit, njit
from scipy import fftpack, ndimage


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
    fourtrans = fftpack.fft2(wind)
    absolu = np.abs(fourtrans)
    powspecden = np.square(absolu)
    loga = np.log(powspecden)
    final = fftpack.fftshift(loga)
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
    return int(round(abs(n)) * sign)


@njit
def get_angular_intensity(
        psd,
        radius,
        window,
        n_steps=1024
):
    half_steps = n_steps // 2
    intensity = np.zeros(n_steps)
    buffer = np.zeros((n_steps, n_steps)) * np.NaN
    win_size = (2 * window + 1) ** 2
    x_c, y_c = get_center(psd)
    img_h, img_w = psd.shape

    for idx_int in range(n_steps):
        theta = np.pi / half_steps * (idx_int - half_steps)
        cos_th = np.cos(theta)
        sin_th = np.sin(theta)
        x = x_c - radius * sin_th
        y = y_c + radius * cos_th
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
                                local_sum += psd[i + k, j + l] / win_size
                    buffer[i, j] = local_sum

            p = x - m
            q = y - n
            comp_p = 1 - p
            comp_q = 1 - q
            intensity[idx_int] = comp_p * comp_q * buffer[m, n] + \
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
    buf = np.zeros((n_steps, n_steps)) * np.NaN
    x_c, y_c = get_center(psd)
    img_h, img_w = psd.shape
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

    for idx, radius in enumerate(radii):
        x = x_c - radius * sin_theta
        y = y_c + radius * cos_theta
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
                                local_sum += psd[i + k, j + l] / win_size

                    buf[i, j] = local_sum

            p = x - m
            q = y - n
            comp_p = 1 - p
            comp_q = 1 - q
            intensity[idx] = \
                comp_p * comp_q * buf[m, n] + \
                p * comp_q * buf[m+1, n] + \
                q * comp_p * buf[m, n+1] + \
                p * q * buf[m+1, n+1]

    return radii, intensity


if __name__ == "__main__":
    from skimage import io, color
    img = io.imread("../scripts/spectral/celledges.jpg")
    img = color.rgb2gray(img)
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.figure()
    psd = calc_psd(img)
    plt.imshow(psd)
    plt.figure()
    x, y = get_angular_intensity(psd, 50, 3)
    plt.plot(x, y)
    plt.figure()
    x, y = get_radial_intensity(psd, 30, 0)
    plt.plot(x, y)
    plt.show()
