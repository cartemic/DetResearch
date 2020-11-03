import numpy as np
from numba import jit, njit
from skimage import filters, exposure, transform
from scipy import fftpack, ndimage


def calc_psd(img):
    wind = img * filters.gaussian(img, 0.2*img.shape[0])
    wind = exposure.equalize_hist(wind)
    fourtrans = fftpack.fft2(wind)
    absolu = np.abs(fourtrans)
    powspecden = np.square(absolu)
    loga = np.log(powspecden)
    final = fftpack.fftshift(loga)
    return exposure.equalize_hist(final)


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
def get_angular_intensity(psd, radius, window, n_steps=1024):
    half_steps = n_steps / 2
    intensity = np.zeros(n_steps)
    buffer = np.zeros((n_steps, n_steps))
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


if __name__ == "__main__":
    from skimage import io, color
    img = io.imread("https://shepherd.caltech.edu/EDL/PublicResources"
                  "/CellImageProcessing/cellsize/celledges.jpg")
    img = color.rgb2gray(img)
    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.figure()
    psd = calc_psd(img)
    plt.imshow(psd)
    plt.figure()
    x, y = get_angular_intensity(psd, 150, 2)
    print(x, y)
    plt.plot(x, y)
    plt.show()
