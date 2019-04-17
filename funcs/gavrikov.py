import numpy as np
import cantera as ct
import sdtoolbox as sd


def _calculate_ea_r(time_1, time_2, temp_1, temp_2):
    # equation 4
    if time_1 == 0 or time_2 == 0:
        return 0
    else:
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


def get_ea_r(temp_init, press_init, q, mech, cj_speed, time_end):
    gas = sd.postshock.PostShock_fr(
        cj_speed,
        press_init,
        temp_init,
        q,
        mech
    )
    temp_ps, press_ps = gas.TP
    temps = temp_ps * np.array([1.02, 0.98])
    ind_times = []
    for temp in temps:
        gas.TPX = temp, press_ps, q
        cv_results = sd.cv.cvsolve(gas, t_end=time_end)
        ind_times.append(cv_results['ind_time'])
    return _calculate_ea_r(*ind_times, *temps)


if __name__ == "__main__":
    # x = Ea/RTps
    # y = Tvn/T0
    # x = np.linspace(2, 10, 100)
    # y = 3
    # ans = model(x, y)
    # plt.semilogy(x, ans)
    # plt.show()
    # check_validity(5, 5)
    import sys

    init_temp = 300
    init_press = 1 * ct.one_atm
    phi = 1
    fuel = 'H2'
    oxidizer = 'O2:1, N2:3.76'
    mechanism = 'gri30.cti'
    # cj_speed = 2197.0853146306313
    # build initial state gas
    init_gas = ct.Solution(mechanism)

    # my gas
    init_gas.set_equivalence_ratio(phi, fuel, oxidizer)
    init_gas.TP = init_temp, init_press
    q = init_gas.mole_fraction_dict()

    print('calculating cj speed...', end='')
    sys.stdout.flush()
    d_cj = sd.postshock.CJspeed(
        init_press,
        init_temp,
        q,
        mechanism
    )
    print('done')

    print('getting reflected gas properties...', end=' ')
    sys.stdout.flush()
    # get reflected gas (frozen props)
    ps_gas = sd.postshock.PostShock_fr(
        d_cj,
        init_press,
        init_temp,
        init_gas.mole_fraction_dict(),
        mechanism
    )
    print('done')

    end_time = 1e-3  # default 1e-3
    print('getting znd state...', end=' ')
    sys.stdout.flush()
    znd_results = sd.znd.zndsolve(
        ps_gas,
        init_gas,
        d_cj,
        t_end=end_time,
        advanced_output=True
    )
    print('done')
    vn_temp = max(znd_results['T'])

    print('getting post-shock state...', end=' ')
    sys.stdout.flush()
    ps_temp = sd.postshock.PostShock_eq(
        U1=1.3 * d_cj,
        P1=init_press,
        T1=init_temp,
        q=init_gas.mole_fraction_dict(),
        mech=mechanism
    ).T
    print('done')

    print('calculating stability parameters...', end=' ')
    sys.stdout.flush()
    ea_r = get_ea_r(300, 101325, q, mechanism, d_cj, end_time)
    print('done')

    print()

    print('T_vn: {:1.0f} K'.format(vn_temp))
    print('T_ps: {:1.0f} K'.format(ps_temp))

    print()

    gav_x = ea_r / ps_temp
    gav_y = vn_temp / init_temp
    print('X:    {:f}'.format(ea_r / ps_temp))
    print('Y:    {:f}'.format(vn_temp / init_temp))

    print()

    lambda_delta = lambda_delta_ratio(gav_x, gav_y)
    delta = znd_results['ind_len_ZND']
    print('lam/del:   {:f}'.format(lambda_delta))
    print('cell size: {:f} cm'.format(lambda_delta * delta * 100))
