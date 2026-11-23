#!/usr/bin/env python3
"""
This module consists of functions:
    -to derive variables like
            specific humidity,
            cloud cover,
            turbulent fluxes like sensible and latent heat fluxes
    -to correct varibles like
            the incoming shortwave radiation due station tilting
            relative humidity below the freezing temperature

Author: Sonika Shahi
Updated: 28 Aug 2025

Environment: xeofs_env2 (conda activate xeofs_env2 on walrus server)
"""

import numpy as np
import pandas as pd

# Constants declaration
eps_overcast = 1   # Overcast sky emissivity
eps_clear = 9.36508e-7 # Clear sky emissivity
sigma = 5.67e-8 # 5.67*10**(-8) #STEFAN_BOLTZMAN_CONSTANT
eps = 0.622 # Ratio of molecular weights of water and dry air
T_0_abs = -273.15 # Absolute zero degree Celsius
T_0 = 273.15 # Freezing point temperature in K
T_100 = T_0 + 100 # Steam point temperature in K

def calculate_cloud_cover(dlr, Tair):
    """Calculate cloud cover using incoming longwave and air temperature.

    References:
    https://doi.org/10.5194/essd-2021-80
    https://doi.org/10.1002/qj.49708938105

    Parameters
    ----------
    dlr = downward/incoming longwave radiation in W/m2
    Tair = air temperature in °C

    Returns
    ----------
    CloudCov = the cloud cover (limited to the [0:1] range)
    # DifFrac = diffuse fraction; 0.2 for clear skies to 1 for overcast conditions

    """

    # Overcast conditions assuming the black-body radiation
    LR_overcast = eps_overcast * sigma * (Tair + T_0)**4
    # Theoretical downward longwave radiation flux corresponding to clear sky conditions using the equation from Swinbank (1963)
    LR_clear = eps_clear * sigma * (Tair + T_0)**6

    CloudCov = (dlr - LR_clear) / (LR_overcast - LR_clear)

    overcast = CloudCov > 1
    Clear = CloudCov < 0
    CloudCov[overcast] = 1
    CloudCov[Clear] = 0
    #DifFrac = 0.2+0.8*CloudCov #diffuse fraction
    return CloudCov