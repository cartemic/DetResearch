"""
Defines configuration settings that are globally available.
Used by:    
    "postshock" module:    
        PostShock_fr
        PostShock_eq
        shock_calc
        shk_eq_calc
        
    "reflections" module:
        reflected_fr
        reflected_eq
        PostReflectedShock_fr
        PostReflectedShock_eq
"""
from typing import Literal

ERRFT = 1e-4
ERRFV = 1e-4
volumeBoundRatio = 5
Solver = Literal["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
