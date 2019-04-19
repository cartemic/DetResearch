"""
Original script: demo_ZND_CJ_cell.m found at
http://shepherd.caltech.edu/EDL/PublicResources/sdt/nb/sdt_intro.slides.html

Modified for python by Mick Carter
cartemic@oregonstate.edu

Shock and Detonation Toolbox Demo Program

Computes ZND and CV models of detonation with the shock front
traveling at the CJ speed.  Evaluates various measures of the reaction
zone thickness and exothermic pulse width, effective activation energy
and Ng stability parameter.

################################################################################
Theory, numerical methods and applications are described in the following report:

    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    California Institute of Technology Revised September, 2018.

Please cite this report and the website if you use these routines.

Please refer to LICENCE.txt or the above report for copyright and disclaimers.

http://shepherd.caltech.edu/EDL/PublicResources/sdt/

################################################################################
Updated January 2019
Tested with:
    python 3.6.7
Under these operating systems:
    Windows 10
"""
import numpy as np
import cantera as ct
import sdtoolbox as sd


def gavrikov(delta, theta, Tvn, T0):
    # Correlation function for detonation cell width
    # proposed by Gavrikov et al COMBUSTION AND FLAME 120:19ï¿½33 (2000)
    # based on using a reaction zone length based on time to 50% limiting
    # reactant consumption in constant volume explosion approximation using vn
    # postshock velocity to convert time to distance.   Tested against a range
    # of fuel-oxidizer diluent mixtures
    #
    # Inputs:
    # delta = reaction zone length based on time to 50% consumption of limiting
    # reactant from CV computation and delta = time * w_VN
    # theta = Ea/RT_VN,  effective reduced activation energy based on CV
    # computation
    # Tvn = von Neumann (postshock temperature behind CJ shock wave)
    # T0 = initial temperature
    #
    # Constants
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
    #  define nondimensional parameters
    X = theta
    Y = Tvn / T0
    z = Y * (a * Y - b) + X * (c * X - d + (e - f * Y) * Y) + g * np.log(
        Y) + h * np.log(X) + Y * (i / X - k * Y / X ** m) - j
    return delta * 10 ** z


def ng(delta, chi):
    # correlation function for detonation cell size from
    # Ng, Hoi Dick, Yiguang Ju, and John H. S. Lee. 2007. Assessment of
    # Detonation Hazards in High-Pressure Hydrogen Storage from Chemical
    # Sensitivity Analysis. INTERNATIONAL JOURNAL OF HYDROGEN ENERGY 32 (1):
    # 93-99.
    # Tested only against low pressure H2-air data
    # Inputs:
    # delta = reaction zone length based on peak thermicity in ZND simulation
    # chi = theta*Delta_i/Delta_r where
    #       theta = reduced effective activation energy from CV computation
    #       Delta_i = distance to peak thermicity from ZND computation
    #       Delta_r = w_vN/\sigmadot_max from ZND computation
    # See Ng et al.  Combustion Theory and Modeling 2005 for a discussion of
    # the chi parameter.
    #
    # Constants
    A0 = 30.465860763763
    a1 = 89.55438805808153
    a2 = -130.792822369483
    a3 = 42.02450507117405
    b1 = -0.02929128383850
    b2 = 1.0263250730647101E-5
    b3 = -1.031921244571857E-9
    return delta * (A0 + ((a3 / chi + a2 / chi) / chi + a1) / chi + (
                (b3 * chi + b2 * chi) * chi + b1) * chi)


if __name__ == '__main__':
    print('demo_ZND_CJ_cell')

    P1 = 100000
    T1 = 300
    q = 'H2:2 O2:1 N2:3.76'
    mech = 'Mevel2017.cti'
    fname = 'h2air'

    cj_speed = sd.postshock.CJspeed(P1, T1, q, mech)

    gas1 = ct.Solution(mech)
    gas1.TPX = T1, P1, q

    # FIND EQUILIBRIUM POST SHOCK STATE FOR GIVEN SPEED
    gas = sd.postshock.PostShock_eq(cj_speed, P1, T1, q, mech)
    u_cj = cj_speed * gas1.density / gas.density

    # FIND FROZEN POST SHOCK STATE FOR GIVEN SPEED
    gas = sd.postshock.PostShock_fr(cj_speed, P1, T1, q, mech)

    # SOLVE ZND DETONATION ODES
    out = sd.znd.zndsolve(
        gas,
        gas1,
        cj_speed,
        advanced_output=True,
        t_end=2e-3
    )

    # Find CV parameters including effective activation energy
    gas.TPX = T1, P1, q
    gas = sd.postshock.PostShock_fr(cj_speed, P1, T1, q, mech)
    Ts, Ps = gas.TP
    Ta = Ts * 1.02
    gas.TPX = Ta, Ps, q
    CVout1 = sd.cv.cvsolve(gas)
    Tb = Ts * 0.98
    gas.TPX = Tb, Ps, q
    CVout2 = sd.cv.cvsolve(gas)

    # Approximate effective activation energy for CV explosion
    taua = CVout1['ind_time']
    taub = CVout2['ind_time']
    if taua == 0 or taub == 0:
        theta_effective_CV = 0
    else:
        theta_effective_CV = 1 / Ts * (
                np.log(taua / taub) / ((1 / Ta) - (1 / Tb))
        )

    #  Find Gavrikov induction length based on 50% limiting species consumption,
    #  fuel for lean mixtures, oxygen for rich mixtures
    #  Westbrook time based on 50% temperature rise
    limit_species = 'H2'
    limit_species_loc = gas.species_index(limit_species)
    gas.TPX = Ts, Ps, q
    X_initial = gas.mole_fraction_dict()[limit_species]
    gas.equilibrate('UV')
    X_final = gas.mole_fraction_dict()[limit_species]
    T_final = gas.T
    X_gav = 0.5*(X_initial - X_final) + X_final
    T_west = 0.5*(T_final - Ts) + Ts
    bb = len(CVout1['speciesX'][:, limit_species_loc])

    t_gav = 0
    for loc in range(bb):
        if CVout1['speciesX'][loc, limit_species_loc] > X_gav:
            t_gav = CVout1['time'][loc]

    x_gav = t_gav*out['U'][0]

    t_west = 0
    for loc in range(bb):
        if CVout1['T'][loc] < T_west:
            t_west = CVout1['time'][loc]

    x_west = t_west*out['U'][0]

    # Ng et al definition of max thermicity width
    max_thermicity_width_ZND = u_cj/max(out['thermicity'])
    chi_ng = theta_effective_CV*out['ind_len_ZND']/max_thermicity_width_ZND
    cell_gav = gavrikov(x_gav, theta_effective_CV, Ts, T1)
    cell_ng = ng(out['ind_len_ZND'], chi_ng)

    print('ZND computation results ')
    print('Mixture ', q)
    print('Mechanism ', mech)
    print('Initial temperature {0:8.3e} K'.format(T1))
    print('Initial pressure {0:8.3e} Pa'.format(P1))
    print('CJ speed {0:8.3e} m/s'.format(cj_speed))
    print(' ')
    print(
        'Reaction zone computation end time = {0:8.3e} s'.format(
            out['tfinal']
        )
    )
    print(
        'Reaction zone computation end distance = {0:8.3e} m'.format(
            out['xfinal']
        )
    )
    print(' ')
    print('T (K), initial = {0:1.5f}, final {1:1.5f}, max {2:1.5f}'.format(
            out['T'][0], out['T'][-1], max(out['T'])
        )
    )
    print(
        'P (Pa), initial = {0:1.3f}, final {1:1.3f}, max {2:1.3f}'.format(
            out['P'][0], out['P'][-1], max(out['P'])
        )
    )
    print(
        'M, initial = initial = {0:1.3f}, final {1:1.3f}, max {2:1.3f}'.format(
            out['M'][0], out['M'][-1], max(out['M'])
        )
    )
    print(
        'u (m/s), initial = {0:1.5f}, final {1:1.5f}, cj {2:1.5f}'.format(
            out['U'][0], out['U'][-1], u_cj
        )
    )
    print(' ')
    print(
        'Reaction zone thermicity half-width = {0:8.3e} m'.format(
            out['exo_len_ZND']
        )
    )
    print(
        'Reaction zone maximum thermicity distance = {0:8.3e} ,'.format(
            out['ind_len_ZND']
        )
    )
    print(
        'Reaction zone thermicity half-time = {0:8.3e} s'.format(
            out['exo_time_ZND']
        )
    )
    print(
        'Reaction zone maximum thermicity time = {0:8.3e} s'.format(
            out['ind_time_ZND']
        )
    )
    print(
        'Reaction zone width (u_cj/sigmadot_max) = {0:8.3e} m'.format(
            max_thermicity_width_ZND
        )
    )
    print(' ')
    print('CV computation results ')
    print('Time to dT/dt_max = {0:8.3e} s'.format(CVout1['ind_time']))
    print(
        'Distance to dT/dt_max = {0:8.3e} m'.format(
            CVout1['ind_time'] * out['U'][0]
        )
    )
    print('Reduced activation energy) = {0:8.3e}'.format(theta_effective_CV))
    print('Time to 50% consumption = {0:8.3e} s'.format(t_gav))
    print('Distance to 50% consumption = {0:8.3e} m'.format(x_gav))
    print('Time to 50% temperature rise = {0:8.3e} s'.format(t_west))
    print('Distance to 50% temperature = {0:8.3e} m'.format(x_west))
    print(' ')
    print('Cell size predictions ')
    print('Gavrikov correlation {0:8.3e} m'.format(cell_gav))
    print('Ng et al Chi Parameter {0:8.3e} m'.format(chi_ng))
    print('Ng et al correlation {0:8.3e} m'.format(cell_ng))
    print('Westbrook correlation {0:8.3e} m'.format(29*x_west))

    sd.utilities.znd_plot(
        out,
        maxx=0.002,
        major_species={'H2', 'O2', 'H2O'},
        minor_species={'H', 'O', 'OH', 'H2O2', 'HO2'}
    )
