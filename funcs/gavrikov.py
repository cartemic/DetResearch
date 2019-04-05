import numpy as np
from matplotlib import pyplot as plt


def check_validity(ea_rtps, tvn_t0):
    ea_check = 5 <= ea_rtps <= 10
    tvn_check = 4.5 <= tvn_t0 <= 6.5

    if not ea_check and not tvn_check:
        print('Both E_a/(R*T_ps) and T_vn/T_0 are out of range')
        return False
    elif not ea_check:
        print('E_a/(R*T_ps) is out of range')
        return False
    elif not tvn_check:
        print('T_vn/T_0 is out of range')
        return False
    else:
        print('Gavrikov conditions are met')
        return True


def get_ea_r(time_1, time_2, temp_1, temp_2):
    # equation 4
    return np.log(time_1 / time_2) / (1 / temp_1 - 1 / temp_2)


def lambda_delta_ratio(gav_x, gav_y):
    # equation 5
    a = -0.007843787493
    b = 0.1777662961
    c = 0.02371845901
    d = 1.477047968
    e = 0.1545112957
    f = 0.01547021569
    g = -1.446582357
    h = 8.730494354
    i = 4.599907939
    j = 7.443410379
    k = 0.4058325462
    m = 1.453392165

    return np.power(
        10,
        gav_y * (a * gav_y - b) +
        gav_x * (c * gav_x - d + (e - f * gav_y) * gav_y) +
        g * np.log(gav_y) +
        h * np.log(gav_x) +
        gav_y * (i / gav_x - k * gav_y / np.power(gav_x, m)) -
        j
    )


if __name__ == "__main__":
    # x = Ea/RTps
    # y = Tvn/T0
    # x = np.linspace(2, 10, 100)
    # y = 3
    # ans = model(x, y)
    # plt.semilogy(x, ans)
    # plt.show()
    check_validity(5, 5)
