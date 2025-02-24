"""
Shock and Detonation Toolbox
"znd" module

Calculates ZND explosions.
 
This module defines the following functions:

    zndsolve
    getThermicity
    
and the following classes:
    
    ZNDSys
    
################################################################################
Theory, numerical methods and applications are described in the following report:

    Numerical Solution Methods for Shock and Detonation Jump Conditions, S.
    Browne, J. Ziegler, and J. E. Shepherd, GALCIT Report FM2006.006 - R3,
    California Institute of Technology Revised September, 2018

Please cite this report and the website if you use these routines. 

Please refer to LICENCE.txt or the above report for copyright and disclaimers.

http://shepherd.caltech.edu/EDL/PublicResources/sdt/


################################################################################ 
Updated August 2018
Tested with: 
    Python 3.5 and 3.6, Cantera 2.3 and 2.4
Under these operating systems:
    Windows 8.1, Windows 10, Linux (Debian 9)
"""
import typing
from typing import Optional

import cantera as ct
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from sdtoolbox.thermo import soundspeed_fr
from scipy.integrate import solve_ivp
from sdtoolbox.output import SimulationDatabase, ReactionData, SpeciesData, BulkPropertiesData


class ZNDSys(object):
    def __init__(
        self,
        gas: ct.Solution,
        U1: float,
        r1: float,
    ):
        self.gas = gas
        self.U1 = U1
        self.r1 = r1

    def __call__(self, t, y):
        """
        Set of ODEs to solve ZND Detonation Problem.
    
        INPUT:
            t = time
            y = solution array [pressure, density, position, species mass 1, 2, ..]
            gas = working gas object
            U1 = shock velocity (m/s)
            r1 = initial density (kg/m^3)
        
        OUTPUT:
            An array containing time derivatives of:
                pressure, density, distance and species mass fractions, 
            formatted in a way that the integrator in zndsolve can recognize.
            
        """
        self.gas.DPY = y[1], y[0], y[3:]
        c = soundspeed_fr(self.gas)
        U = self.U1*self.r1/self.gas.density
        M = U/c
        eta = 1-M**2 
        
        sigmadot = getThermicity(self.gas)
        Pdot = -self.gas.density*U**2*sigmadot/eta
        rdot = -self.gas.density*sigmadot/eta
        
        dYdt = self.gas.net_production_rates*self.gas.molecular_weights/self.gas.density
    
        return np.hstack((Pdot, rdot, U, dYdt))


def getThermicity(gas):
    """
    Returns the thermicity = sum ( (w/wi-hsi/(cp*T))*dyidt ). Used by zndsys,
    as well as the stagnation module.
    
    FUNCTION SYNTAX:
        thermicity = getThermicity(gas)
        
    INPUT:
        gas = Cantera gas object (not modified by this function)
        
    OUTPUT:
        thermicity (1/s)
    """
    w = gas.molecular_weights
    hs = gas.standard_enthalpies_RT*ct.gas_constant*gas.T/w
    dydt = gas.net_production_rates*w/gas.density
    
    thermicity = sum((gas.mean_molecular_weight/w
                      -hs/(gas.cp_mass*gas.T))*dydt)
       
    return thermicity


def zndsolve(
    gas: ct.Solution,
    gas1: ct.Solution,
    U1: float,
    t_end: float = 1e-3,
    max_step: float = 1e-4,
    t_eval=None,
    relTol=1e-5,
    absTol=1e-8,
    advanced_output=False,
    rxn_indices: Optional[list[int]] = None,
    spec_indices: Optional[list[int]] = None,
    db: Optional[SimulationDatabase] = None,
    run_no: int = 1,
):
    """
    ZND Model Detonation Struction Computation
    Solves the set of ODEs defined in ZNDSys.
    
    FUNCTION SYNTAX:
    output = zndsolve(gas,gas1,U1,**kwargs)
    
    INPUT
        gas = Cantera gas object - postshock state
        gas1 = Cantera gas object - initial state
        U1 = shock velocity (m/s)
        
    OPTIONAL INPUT:
        t_end = end time for integration, in sec
        max_step = maximum time step for integration, in sec
        t_eval = array of time values to evaluate the solution at.
                    If left as 'None', solver will select values.
                    Sometimes these may be too sparse for good-looking plots.
        relTol = relative tolerance
        absTol = absolute tolerance
        advanced_output = calculates optional extra parameters such as induction lengths
    
    
    OUTPUT:
        output = a dictionary containing the following results:
            time = time array
            distance = distance array
            
            T = temperature array
            P = pressure array
            rho = density array
            U = velocity array
            thermicity = thermicity array
            species = species mass fraction array
            
            M = Mach number array
            af = frozen sound speed array
            g = gamma (cp/cv) array
            wt = mean molecular weight array
            sonic = sonic parameter (c^2-U^2) array
            
            tfinal = final target integration time
            xfinal = final distance reached
            
            gas1 = a copy of the input initial state
            U1 = shock velocity
            
            and, if advanced_output=True:
            ind_time_ZND = time to maximum thermicity gradient
            ind_len_ZND = distance to maximum thermicity gradient
            exo_time_ZND = pulse width (in secs) of thermicity  (using 1/2 max)
            ind_time_ZND = pulse width (in meters) of thermicity (using 1/2 max)
            max_thermicity_width_ZND = according to Ng et al definition
    """
    ###########################################################
    # Define initial information
    ###########################################################
    r1 = gas1.density

    x_start = 0.
    y0 = np.hstack((gas.P, gas.density, x_start, gas.Y))

    tel = [0., t_end]  # Timespan
    
    output = {}

    out: OdeResult = typing.cast(
        OdeResult,
        solve_ivp(
            ZNDSys(gas, U1, r1),
            tel,
            y0,
            method='Radau',
            atol=absTol,
            rtol=relTol,
            max_step=max_step,
            t_eval=t_eval,
        )
    )
    
    output['time'] = out.t    
    output['P'] = out.y[0, :]
    output['rho'] = out.y[1, ]
    output['distance'] = out.y[2, :]
    output['species'] = out.y[3:, :]
    
    output['tfinal'] = t_end
    output['xfinal'] = output['distance'][-1]
        
    # Initialize additional output matrices where needed
    b = len(output['time'])
    output['T'] = np.zeros(b)
    output['U'] = np.zeros(b)
    output['thermicity'] = np.zeros(b)
    output['af'] = np.zeros(b)
    output['g'] = np.zeros(b)
    output['wt'] = np.zeros(b)    
    if advanced_output:
        output['ind_len_ZND'] = 0
        output['ind_time_ZND'] = 0
        output['exo_len_ZND'] = 0
        output['exo_time_ZND'] = 0

    #############################################################################
    # Extract TEMPERATURE, WEIGHT, GAMMA, SOUND SPEED, VELOCITY, MACH NUMBER, 
    # c^2-U^2, THERMICITY, and TEMPERATURE GRADIENT
    #############################################################################

    if db is not None:
        for reaction_no in rxn_indices or []:
            for species_no in spec_indices or []:
                db.clear_results(reaction_no, species_no)
    # Have to loop for operations involving the working gas object
    for i, P in enumerate(output['P']):
        gas.DPY = output['rho'][i], P, output['species'][:, i]
        af = soundspeed_fr(gas)
        U = U1*r1/gas.density
       
        output['T'][i] = gas.T
        output['U'][i] = U
        output['thermicity'][i] = getThermicity(gas)
        output['af'][i] = af
        output['g'][i] = gas.cp/gas.cv
        output['wt'][i] = gas.mean_molecular_weight

        if db is not None:
            db.bulk_properties.insert(
                BulkPropertiesData(
                    condition_id=db.conditions_id,
                    time=output["time"][i],
                    temperature=gas.T,
                    pressure=gas.P,
                    velocity=U,
                    cp=gas.cp_mass,
                    cv=gas.cv_mass,
                ), commit=False)
            if spec_indices is not None:
                for idx_spec in spec_indices:
                    species = gas.species(idx_spec)
                    db.species.insert(SpeciesData(
                        condition_id=db.conditions_id,
                        time=output["time"][i],
                        species_no=idx_spec,
                        species=species,
                        mole_frac=gas.mole_fraction_dict().get(species.name, 0),
                        concentration=gas.concentrations[idx_spec],
                        creation_rate=gas.creation_rates[idx_spec],
                        destruction_rate=gas.destruction_rates[idx_spec],
                        net_production_rate=gas.net_production_rates[idx_spec],
                    ), commit=False)
            if rxn_indices is not None:
                rxn_eqns = gas.reaction_equations()
                net_rates_of_progress = gas.net_rates_of_progress
                for idx_rxn in rxn_indices:
                    this_net_rate_of_progress = net_rates_of_progress[idx_rxn]
                    db.reactions.insert(ReactionData(
                        condition_id=db.conditions_id,
                        time=output["time"][i],
                        reaction_no=idx_rxn,
                        reaction=rxn_eqns[idx_rxn],
                        fwd_rate_constant=gas.forward_rate_constants[idx_rxn],
                        fwd_rate_of_progress=gas.forward_rates_of_progress[idx_rxn],
                        rev_rate_constant=gas.reverse_rate_constants[idx_rxn],
                        rev_rate_of_progress=gas.reverse_rates_of_progress[idx_rxn],
                        net_rate_of_progress=gas.net_rates_of_progress[idx_rxn],
                        relative_chemical_contribution=(
                                np.abs(this_net_rate_of_progress) / np.sum(np.abs(net_rates_of_progress))
                        ),
                    ), commit=False)

    if db is not None:
        db.commit_all()
        
    # Vectorize operations where possible    
    output['M'] = output['U']/output['af']
    eta = 1- output['M']**2
    output['sonic'] = eta*output['af']**2
    
    if advanced_output:
        ################################################################################################
        # Find INDUCTION TIME and LENGTH based on MAXIMUM THERMICITY
        ################################################################################################
        n = output['thermicity'].argmax()
        
        output['ind_time_ZND'] = output['time'][n]
        output['ind_len_ZND'] = output['distance'][n]
        output['max_thermicity_ZND'] = max(output['thermicity']) # required for Ng et al Chi parameter
        
        #######################################################
        # Check for eigenvalue detonation
        #######################################################
        
        if n == b:
            raise ValueError(
                'Error: Maximum thermicity occurs at the end of the reaction zone'
                '       You may have an eigenvalue detonation, your final integration length may be too short,\n'
                '       your mixture may be too rich/lean, or something else may be wrong\n'
                '\n'
                'Mach Number (end of reaction): '
                f"{output['M'][b]}"
                ' - if close to 1, check for eigenvalue detonation\n'
                'Induction Time: '
                f"{output['ind_time_ZND']}\n"
                'Exothermic Pulse Time: '
                f"{output['exo_time_ZND']}"
            )
        
        elif n == 0:
            raise ValueError(
                'Error: Maximum thermicity occurs at the beginning of the reaction zone\n'
                '       You may have an eigenvalue detonation, your final integration length may be too short,\n'
                '       your mixture may be too rich/lean, or something else may be wrong\n'
                '\n'
                'Mach Number (end of reaction): '
                f"{output['M'][b]}"
                ' - if close to 1, check for eigenvalue detonation\n'
                'Induction Time: '
                f"{output['ind_time_ZND']}\n"
                'Exothermic Pulse Time: '
                f"{output['exo_time_ZND']}"
            )

        # I don't care about exo time or len
        # else:
        #     max_sigmadot = max(output['thermicity'])
        #     half_sigmadot_flag1 = 0
        #     half_sigmadot_flag2 = 0
        #     # Go into a loop to find two times when sigma_dot is half its maximum
        #     tstep2 = 0 # JML temporary
        #     for j,thermicity in enumerate(list(output['thermicity'])):
        #         if half_sigmadot_flag1 == 0:
        #             if thermicity > 0.5*max_sigmadot:
        #                 half_sigmadot_flag1 = 1
        #                 tstep1 = j
        #
        #         elif half_sigmadot_flag2 == 0:
        #             if thermicity < 0.5*max_sigmadot:
        #                 half_sigmadot_flag2 = 1
        #                 tstep2 = j
        #             else:
        #                 tstep2 = 0
        #
        # if tstep2 == 0:
        #     raise ValueError(
        #         'Error: No pulse in the thermicity\n'
        #         '       You may have an eigenvalue detonation, your final integration length may be too short,\n'
        #         '       your mixture may be too rich/lean, or something else may be wrong\n'
        #     )
        # else:
        #     output['exo_time_ZND'] = output['time'][tstep2] - output['time'][tstep1];
        #     output['exo_len_ZND'] = output['distance'][tstep2] - output['distance'][tstep1]
        
    
    #################################################################
    # Append extra data used to make output file (via znd_fileout)
    output['gas1'] = gas1
    output['U1'] = U1
    
    return output
    
