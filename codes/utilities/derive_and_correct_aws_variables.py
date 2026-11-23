"""
This module consists of functions:
    -to derive variables like
            specific humidity,
            cloud cover,
            turbulent fluxes like sensible and latent heat fluxes
    -to correct varibles like
            the incoming shortwave radiation due station tilting
            relative humidity below the freezing temperature

Created on 28 August 2025 by Sonika Shahi.
"""
import numpy as np
import pandas as pd

# Constants declaration
eps_overcast = 1   #overcast sky emissivity
eps_clear = 9.36508e-7 # clear sky emissivity
sigma = 5.67e-8 # 5.67*10**(-8) #STEFAN_BOLTZMAN_CONSTANT
eps = 0.622 # ratio of molecular weights of water and dry air
T_0_abs = -273.15 # Absolute zero degree celcius
T_0 = 273.15 # freezing point temperature in K
T_100 = T_0+100 # steam point temperature in K
# z_0=0.001 # Aerodynamic surface roughness length for momentum, assumed constant for all ice/snow surfaces.
R_d = 287.05 # Gas constant of dry air.
es_0 = 6.1071 # Saturation vapour pressure at the melting point (hPa).
es_100 = 1013.246 # Saturation vapour pressure at steam point temperature (hPa).
kappa = 0.4 # Von Karman constant (0.35-0.42).
WS_lim = 1. #
g = 9.82 # Gravitational acceleration (m/s2).
gamma = 16. #Flux profile correction (Paulson & Dyer).
L_sub = 2.83e6 # Latent heat of sublimation (J/kg).
L_dif_max = 0.01,
c_pd = 1005. # Specific heat of dry air (J/kg/K).
aa = 0.7 # Flux profile correction constants (Holtslag & De Bruin '88).
bb = 0.75 # Flux profile correction constants (Holtslag & De Bruin '88).
cc = 5. # Flux profile correction constants (Holtslag & De Bruin '88).
dd = 0.35 # Flux profile correction constants (Holtslag & De Bruin '88).
rho_w = 1000  # water density at 0 °C, kg/cubic meter  - taken from P. 142 Cuffey and Paterson
rho_i = 917  # pure ice density at 0 °C, kg/cubic meter  - taken from P. 142 Cuffey and Paterson
L_fus = 3.34e5 # latent heat of fusion at 0 C, J/kg  - taken from P. 142 Cuffey and Paterson
epsilon = 0.97 # longwave emmissivity of snow and ice act like a black body in the infrared
deg2rad = np.pi / 180 # Degrees to radians conversion
rad2deg = 1 / deg2rad # Radians to Degrees conversion

def _calculate_cloud_cover(dlr, Tair):
    '''Calculate cloud cover using incoming longwave and air temperature.

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

    '''
    # overcast conditions assuming the black-body radiation
    LR_overcast = eps_overcast * sigma * (Tair + T_0)**4
    # theoretical downward longwave radiation flux corresponding to clear sky conditions using the equation from Swinbank (1963)
    LR_clear = eps_clear * sigma * (Tair + T_0)**6

    CloudCov = (dlr - LR_clear) / (LR_overcast - LR_clear)

    overcast = CloudCov > 1
    Clear = CloudCov < 0
    CloudCov[overcast] = 1
    CloudCov[Clear] = 0
    #DifFrac = 0.2+0.8*CloudCov #diffuse fraction
    return CloudCov

def _calculate_tilt(tilt_x, tilt_y):
    '''Calculate station tilt.
    The inclinometer measures the tilt (unit: °) across (left-right) and along (up-down)
    the sensor boom, which translates into tilt-to-east and tilt-to-north when the sensor boom is
    perfectly oriented north-south.

    Calculating the tilt angle and direction of sensor and rotating to a north-south aligned coordinate system.

    https://doi.org/10.5194/essd-2021-80

    Parameters
    ----------
    tilt_x : X tilt inclinometer measurements
    tilt_y : Y tilt inclinometer measurements

    Returns
    -------
    phi_sensor_rad : Spherical tilt coordinates
    theta_sensor_rad : Total tilt of sensor, where 0 is horizontal
    '''
    # Tilt as radians
    tx = tilt_x * deg2rad
    ty = tilt_y * deg2rad

    # Calculate cartesian coordinates
    X = np.sin(tx) * np.cos(tx) * np.sin(ty)**2 + np.sin(tx) * np.cos(ty)**2
    Y = np.sin(ty) * np.cos(ty) * np.sin(tx)**2 + np.sin(ty) * np.cos(tx)**2
    Z = np.cos(tx) * np.cos(ty) + np.sin(tx)**2 * np.sin(ty)**2

    # Calculate spherical coordinates/ radiometer tilt direction
    phi_sensor_rad = -np.pi / 2 - np.arctan(Y/X)
    phi_sensor_rad[X > 0] += np.pi
    phi_sensor_rad[(X == 0) & (Y < 0)] = np.pi
    phi_sensor_rad[(X == 0) & (Y == 0)] = 0
    phi_sensor_rad[phi_sensor_rad < 0] += 2 * np.pi

    # Total tilt of the sensor, i.e. 0 when horizontal /radiometer tilt angle
    theta_sensor_rad = np.arccos(Z / (X**2 + Y**2 + Z**2)**0.5)
    # phi_sensor_deg = phi_sensor_rad * rad2deg                                #TODO take these out if not needed
    # theta_sensor_deg = theta_sensor_rad * rad2deg
    return phi_sensor_rad, theta_sensor_rad

def calculate_declination(doy, hour, minute):
    '''Calculate sun declination based on time.

    Parameters
    ----------
    doy : int
        Day of year
    hour : int
        Hour of day
    minute : int
        Minute of hour

    Returns
    -------
    float
        Sun declination
    '''
    d0_rad = 2 * np.pi * (doy + (hour + minute / 60) / 24 -1) / 365
    return np.arcsin(0.006918 - 0.399912
                    * np.cos(d0_rad) + 0.070257
                    * np.sin(d0_rad) - 0.006758
                    * np.cos(2 * d0_rad) + 0.000907
                    * np.sin(2 * d0_rad) - 0.002697
                    * np.cos(3 * d0_rad) + 0.00148
                    * np.sin(3 * d0_rad))

def calculate_hour_angle(hour, minute, lon):
    '''Calculate hour angle of sun based on time and longitude. Make sure that
    time is set to UTC and longitude is positive when west. Hour angle should
    be 0 at noon

    Parameters
    ----------
    hour : int
        Hour of day
    minute : int
        Minute of hour
    lon : float, Longitude

    Returns
    -------
    float
        Hour angle of sun
    '''
    # return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5))# - lon/360)#- 15.*timezone/360.)
    return 2 * np.pi * (((hour + minute / 60) / 24 - 0.5) - lon/360)

def calculate_zenith(lat, Declination_rad, HourAngle_rad):
    '''Calculate sun zenith in radians and degrees.

    Parameters
    ----------
    lat : float
        Latitude
    Declination_Rad : float
        Sun declination in radians
    HourAngle_rad : float
        Sun hour angle in radians
    deg2rad : float
        Degrees to radians conversion
    rad2deg : float
        Radians to degrees conversion

    Returns
    -------
    ZenithAngle_rad : float
        Zenith angle in radians
    ZenithAngle_deg : float
        Zenith angle in degrees
    '''
    ZenithAngle_rad = np.arccos(np.cos(lat * deg2rad)
                                * np.cos(Declination_rad)
                                * np.cos(HourAngle_rad)
                                + np.sin(lat * deg2rad)
                                * np.sin(Declination_rad))

    ZenithAngle_deg = ZenithAngle_rad * rad2deg
    return ZenithAngle_rad, ZenithAngle_deg

def _calculate_angle_difference(ZenithAngle_rad, HourAngle_rad, phi_sensor_rad,
                theta_sensor_rad):
    '''Calculate angle between sun and upper sensor (to determine when sun is
    in sight of upper sensor

    Parameters
    ----------
    ZenithAngle_rad : float
        Zenith angle in radians
    HourAngle_rad : float
        Sun hour angle in radians
    phi_sensor_rad : Spherical tilt coordinates
    theta_sensor_rad : Total tilt of sensor, where 0 is horizontal

    Returns
    -------
    float
        Angle between sun and sensor
    '''
    return 180 / np.pi * np.arccos(np.sin(ZenithAngle_rad)
                                * np.cos(HourAngle_rad + np.pi)
                                * np.sin(theta_sensor_rad)
                                * np.cos(phi_sensor_rad)
                                + np.sin(ZenithAngle_rad)
                                * np.sin(HourAngle_rad + np.pi)
                                * np.sin(theta_sensor_rad)
                                * np.sin(phi_sensor_rad)
                                + np.cos(ZenithAngle_rad)
                                * np.cos(theta_sensor_rad))

def _calculate_albedo(usr, dsr_cor, AngleDif_deg, ZenithAngle_deg):
    '''Calculate surface albedo based on upwelling and downwelling shorwave
    flux, the angle between the sun and sensor, and the sun zenith

    Parameters
    ----------
    usr : Upwelling shortwave radiation
    dsr_cor : Downwelling shortwave radiation corrected
    AngleDif_def : float
        Angle between sun and sensor in degrees
    ZenithAngle_deg: float
        Zenith angle in degrees

    Returns
    -------
    albedo : Derived albedo
    OKalbedos : Valid albedo measurements
    '''
    albedo = usr / dsr_cor

    # NaN bad data
    OKalbedos = (AngleDif_deg < 70) & (ZenithAngle_deg < 70) & (albedo < 1) & (albedo > 0)
    albedo[~OKalbedos] = np.nan

    # Interpolate all. Note "use_coordinate=False" is used here to force
    # comparison against the GDL code when that is run with *only* a TX file.
    # Should eventually set to default (True) and interpolate based on time,
    # not index.
    # albedo = albedo.interpolate_na(dim='time', use_coordinate=False)
    # albedo = albedo.ffill(dim='time').bfill(dim='time')                        #TODO remove this line and one above?
    albedo = albedo.interpolate(axis=0, method='linear')
    albedo = albedo.ffill(axis=0).bfill(axis='rows')
    return albedo, OKalbedos

def _calculate_toa(ZenithAngle_deg, ZenithAngle_rad):
    '''Calculate incoming shortwave radiation at the top of the atmosphere,
    accounting for sunset periods

    Parameters
    ----------
    ZenithAngle_deg : float
        Zenith angle in degrees
    ZenithAngle_rad : float
        Zenith angle in radians

    Returns
    -------
    isr_toa : float
        Incoming shortwave radiation at the top of the atmosphere
    '''
    sundown = ZenithAngle_deg >= 90

    # Incoming shortware radiation at the top of the atmosphere
    isr_toa = 1372 * np.cos(ZenithAngle_rad)
    isr_toa[sundown] = 0
    return isr_toa

def _calculate_correction_factor(Declination_rad, phi_sensor_rad, theta_sensor_rad,
                        HourAngle_rad, ZenithAngle_rad, ZenithAngle_deg,
                        lat, DifFrac):
    """Calculate correction factor for direct beam radiation, as described
    here: http://solardat.uoregon.edu/SolarRadiationBasics.html

    Offset correction (where solar zenith angles larger than 110 degrees) not
    implemented as it should not improve the accuracy of well-calibrated
    instruments. It would go something like this:
    ds['dsr'] = ds['dsr'] - ds['dwr_offset']
    SRout = SRout - SRout_offset

    Parameters
    ----------
    Declination_rad : float
        Declination in radians
    phi_sensor_rad : Spherical tilt coordinates
    theta_sensor_rad : Total tilt of sensor, where 0 is horizontal
    HourAngle_rad : float
        Sun hour angle in radians
    ZenithAngle_rad : float
        Zenith angle in radians
    ZenithAngle_deg : float
        Zenith Angle in degrees
    lat :  float
        Latitude
    DifFrac : float
        Fractional cloud cover

    Returns
    -------
    CorFac_all : Correction factor
    """
    CorFac = np.sin(Declination_rad) * np.sin(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        - np.sin(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        + np.cos(Declination_rad) \
        * np.cos(lat * deg2rad) \
        * np.cos(theta_sensor_rad) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(lat * deg2rad) \
        * np.sin(theta_sensor_rad) \
        * np.cos(phi_sensor_rad + np.pi) \
        * np.cos(HourAngle_rad) \
        + np.cos(Declination_rad) \
        * np.sin(theta_sensor_rad) \
        * np.sin(phi_sensor_rad + np.pi) \
        * np.sin(HourAngle_rad) \

    CorFac = np.cos(ZenithAngle_rad) / CorFac
    # sun out of field of view upper sensor
    CorFac[(CorFac < 0) | (ZenithAngle_deg > 90)] = 1

    # Calculating dsr over a horizontal surface corrected for station/sensor tilt
    CorFac_all = CorFac / (1 - DifFrac + CorFac * DifFrac)

    return CorFac_all

def correct_dsr_for_tilt(dsr, usr, dlr, Tair, tiltx, tilty, dates, lat):
    """Tilt correction of downwelling shortwave radiation (DSR).

    Downwelling shortwave radiation (dsr) consists of a diffuse and direct beam part.
    It is only the direct beam part of dsr that requires tilt correction.

    Tilt correction of dsr using uncorrected dsr,
    uncorrected upwelling shortwave radiation (usr),
    station tilt by inclinometer (tiltx and tilty),
    diffuse radiation,
    measurement period (for coding ease), latitute, and longitude of the station.

    Reference: https://doi.org/10.5194/essd-2021-80

    Parameters
    ----------
    dsr = downwelling shortwave radiation in W/m2
    usr = upwelling shortwave radiation in W/m2
    dlr = downwelling longwave radiation in W/m2
    Tair = air temperature in °C
    tiltx & tilty = tilt sensor readings in degrees
    dates = mecodes/derive_and_correct_variables.pyasurement period (for coding ease)
    lat = latitute of the station
    # lon = longitude of the station

    Returns
    -------
    dsr_cor = corrected downwelling shortwave radiation in W/m2
    usr_cor = corrected upwelling shortwave radiation in W/m2
    albedo = corrected albedo
    """
    # Determine station position relative to sun
    # Calculate station tilt angle and direction of sensor and rotating to a north-south aligned coordinate system
    phi_sensor_rad, theta_sensor_rad = _calculate_tilt(tiltx, tilty)

    # Calculate declination, zenith, and hour angle of the sun
    dates = pd.DataFrame(dates)
    doy =dates.index.dayofyear.values
    hour = dates.index.hour.values
    minute = dates.index.minute.values

    Declination_rad = calculate_declination(doy, hour, minute)                      # Calculate declination
    HourAngle_rad = calculate_hour_angle(hour, minute) # TODO with lon or with out   # Calculate hour angle
    ZenithAngle_rad, ZenithAngle_deg = calculate_zenith(lat, Declination_rad,       # Calculate zenith
                                                   HourAngle_rad)

    # Determine cloud cover
    cc = _calculate_cloud_cover(dlr=dlr, Tair=Tair)
    # calculate diffuse fraction
    DifFrac = 0.2 + 0.8 * cc
    CorFac_all = _calculate_correction_factor(Declination_rad, phi_sensor_rad,        # Calculate correction
                                       theta_sensor_rad, HourAngle_rad,
                                       ZenithAngle_rad, ZenithAngle_deg,
                                       lat, DifFrac)
    # Correct Downwelling shortwave radiation
    # dsr over a horizontal surface corrected for station/sensor tilt
    dsr_cor = dsr.copy(deep=True) * CorFac_all                     # Apply correction
    # Calculate angle between sun and sensor
    AngleDif_deg = _calculate_angle_difference(ZenithAngle_rad, HourAngle_rad,
                                  phi_sensor_rad, theta_sensor_rad)
    # Determine albedo
    # Calculate albedo based on albedo values when sun is in sight of the upper sensor
    albedo, OKalbedos = _calculate_albedo(usr, dsr_cor, AngleDif_deg, ZenithAngle_deg)
    # Correct upwelling and downwelling shortwave radiation
    sunonlowerdome =(AngleDif_deg >= 90) & (ZenithAngle_deg <= 90)  # Determine when sun is in FOV of lower sensor, assuming sensor measures only diffuse radiation
    dsr_cor[sunonlowerdome] = dsr[sunonlowerdome] / DifFrac[sunonlowerdome] # Apply to downwelling

    usr_cor = usr.copy(deep=True)
    usr_cor[sunonlowerdome] = albedo[sunonlowerdome] * dsr[sunonlowerdome] / DifFrac[sunonlowerdome]# Apply to upwelling

    # Setting dsr and usr to zero for solar zenith angles larger than 95 deg or either dsr or usr are (less than) zero
    no_SR = (ZenithAngle_deg > 95) | (dsr_cor <= 0) | (usr_cor <= 0)
    dsr_cor[no_SR] = 0
    usr_cor[no_SR] = 0

    # Correcting dsr using more reliable usr when sun not in sight of upper sensor
    dsr_cor = usr_cor.copy(deep=True)/albedo

    # Remove data where TOA shortwave radiation invalid
    isr_toa = _calculate_toa(ZenithAngle_deg, ZenithAngle_rad)   # Calculate TOA shortwave radiation
    TOA_crit_nopass = (dsr_cor > (0.9 * isr_toa + 10))     # Determine filter
    dsr_cor[TOA_crit_nopass] = np.nan                      # Apply filter and interpolate
    usr_cor[TOA_crit_nopass] = np.nan

    dsr_cor = dsr_cor.interpolate(axis=0,method='linear')
    dsr_cor = dsr_cor.ffill(axis=0).bfill(axis='rows')

    usr_cor = usr_cor.interpolate(axis=0,method='linear')
    usr_cor = usr_cor.ffill(axis=0).bfill(axis='rows')

    return  dsr_cor, usr_cor, albedo, OKalbedos

def _calculate_saturation_vapor_pressure(T):
    """ Calculate saturation vapor pressure from temperature.

    Parameters
    ----------
    T : temperature in degree celcius (°C)

    Returns
    ----------
    es : saturation vapor pressure in hectoPascal (hPa)
    esw: saturation vapor pressure with respect to water (hPa)
    esi: saturation vapor pressure with respect to ice (hPa)

    References:
    Goff, J. A. and Gratch, S.: Low-pressure properties of water-from 160 to 212ˆ° F., Trans. Am. Heat. Vent. Eng., 52, 95–121, 1946.

    """
    #saturation vapor pressure over water # in hPa (Goff & Gratch)
    esw = 10**(-7.90298 * (T_100 / (T + T_0) - 1) + 5.02808 * np.log10(T_100 / (T + T_0))
                - 1.3816e-7 * (10**(11.344 * (1. -(T + T_0) / T_100)) - 1.)
                + 8.1328e-3 * (10**(-3.49149 * (T_100 / (T + T_0) - 1)) -1.) + np.log10(es_100))
    #saturation vapor pressure over ice
    esi = 10**(-9.09718 * (T_0 / (T + T_0) - 1.) - 3.56654 * np.log10(T_0 / (T + T_0))
                + 0.876793 * (1. - (T + T_0) / T_0) + np.log10(es_0)) # in hPa (Goff & Gratch)

    # combine esi and esw
    freezing = T <= 0
    es = esw.copy()     # saturation vapor pressure over water in hPa (incorrect below melting point)
    es[freezing] = esi[freezing]   # saturation vapor pressure over ice in hPa
    #es = es *100 #hPa into Pa
    # es = pd.DataFrame(index=T.index,columns={'es'})
    return es, esw, esi

def correct_relative_humidity(RH, Tair):
    """ Correcting relative humidity readings for T below 0 to give value with respect to ice.

    RH is measured relative to water.
    For temperatures below freezing, RH is recalculated relative to ice in post-processing.
    Reference: https://doi.org/10.5194/essd-2021-80

    Inputs:
    RH = uncorrected relative humidity
    Tair = air temperature
    Tair is used to calculate saturation vapor pressure with respect to water and ice;
    esw and esi, saturation vapor pressure with respect to water and ice (hPa) respectively

    Output:
    RH_corr = corrected relative humidity with respect to ice
    """
    RH_corr = RH.copy(deep=True)
    es, esw, esi = _calculate_saturation_vapor_pressure(Tair)
    freezing = Tair <= 0
    RH_corr[freezing] = RH[freezing] * esw[freezing] / esi[freezing]
    RH_corr[RH_corr > 100] = 100
    RH_corr[RH_corr < 0] = 0
    return RH_corr

def calculate_surface_temperature(ulr, dlr, surface_type='ice'):
    """ Calculate surface temperature from the outgoing longwave radiation.

    Parameters
    ----------
    ulr: upwelling longwave radiation in W/m2
    dlr: downwelling longwave radiation in W/m2

    Returns:
    --------
    tsurf: surface temperature in °C
    """
    # I = e * STEFAN_BOLTZMANN_CONSTANT * T**4
    tsurf = ((ulr.abs() - ((1 - epsilon) * dlr)) / (sigma * epsilon))**(0.25) + T_0_abs # 4th sqrt

    if surface_type == 'ice':
        tsurf[tsurf > 0] = 0 # ice cant have positive degrees

    return tsurf

def calculate_turbulent_heat_flux(T, Tsurf, rho_atm, WS, z_WS, z_T, nu, q, p,z_0):
    """ Calculate latent and sensible heat flux using the bulk calculation method.

    Parameters
    ----------
    T : Air temperature
    Tsurf : Surface temperature
    rho_atm : Atmopsheric density
    WS : Wind speed
    z_WS : Height of anemometer
    z_T : Height of thermometer
    nu  : Kinematic viscosity of air
    q : Specific humidity
    p : Air pressure
    z_0 : Aerodynamic surface roughness length for momention,
          assumed constant for all ice/snow surfaces. Default is 0.001.

    Constants declaration
    ---------------------
    kappa : Von Karman constant (0.35-0.42). Default is 0.4.
    WS_lim : Default is 1.
    g : Gravitational acceleration (m/s2). Default is 9.82.
    es_0 : Saturation vapour pressure at the melting point (hPa). Default is 6.1071.
    eps :  Default is 0.622.
    gamma : Flux profile correction (Paulson & Dyer). Default is 16..
    L_sub : Latent heat of sublimation (J/kg). Default is 2.83e6.
    L_dif_max : Default is 0.01.
    c_pd : Specific heat of dry air (J/kg/K). Default is 1005..
    aa : Flux profile correction constants (Holtslag & De Bruin '88). Default is 0.7.
    bb : Flux profile correction constants (Holtslag & De Bruin '88). Default is 0.75.
    cc : Flux profile correction constants (Holtslag & De Bruin '88). Default is 5.
    dd : Flux profile correction constants (Holtslag & De Bruin '88). Default is 0.35.

    Returns
    -------
    SHF : Sensible heat flux
    LHF : Latent heat flux
    """
    SHF = T.applymap(lambda x: 0)
    LHF = T.applymap(lambda x: 0)
    L = T.applymap(lambda x: 1E5)

    u_star = kappa * WS / np.log(z_WS / z_0)
    Re = u_star * z_0 / nu
    z_0h = z_0 * np.exp(1.5 - 0.2 * np.log(Re) - 0.11 * np.log(Re)**2) # Surface roughness length for heat and moisture, from Smeets & Van den Broeke 2008
    z_0h[WS <= 0] = 1e-10
    # in hPa (Goff & Gratch)
    es_ice_surf = 10**(-9.09718
                    * (T_0 / (Tsurf + T_0) -1) - 3.56654
                    * np.log10(T_0 / (Tsurf + T_0)) + 0.876793
                    * (1 - (Tsurf + T_0) / T_0)
                    + np.log10(es_0))
    q_surf = eps * es_ice_surf / (p - (1 - eps) * es_ice_surf)
    theta = T + z_T * g / c_pd
    stable = (theta > Tsurf) & (WS > WS_lim)
    unstable = (theta < Tsurf) & (WS > WS_lim)      #TODO: check if unstable = ~stable? And if not why not
                                                    #no_wind  = (WS <= WS_lim)
    # Calculate stable stratification
    for i in np.arange(0,31):
        psi_m1 = -(aa*         z_0/L[stable] + bb*(         z_0/L[stable]-cc/dd)*np.exp(-dd*         z_0/L[stable]) + bb*cc/dd)
        psi_m2 = -(aa*z_WS[stable]/L[stable] + bb*(z_WS[stable]/L[stable]-cc/dd)*np.exp(-dd*z_WS[stable]/L[stable]) + bb*cc/dd)
        psi1 = -(aa*z_0h[stable]/L[stable] + bb*(z_0h[stable]/L[stable]-cc/dd)*np.exp(-dd*z_0h[stable]/L[stable]) + bb*cc/dd)
        psi2 = -(aa* z_T[stable]/L[stable] + bb*( z_T[stable]/L[stable]-cc/dd)*np.exp(-dd* z_T[stable]/L[stable]) + bb*cc/dd)
        u_star[stable] = kappa*WS[stable]/(np.log(z_WS[stable]/z_0)-psi_m2+psi_m1)
        Re[stable] = u_star[stable]*z_0/nu[stable]
        z_0h[stable] = z_0*np.exp(1.5-0.2*np.log(Re[stable])-0.11*(np.log(Re[stable]))**2)

        # If n_elements(where(z_0h[stable] < 1e-6)) get 1 then
        # z_0h[stable[where(z_0h[stable] < 1e-6)]] = 1e-6
        z_0h[stable][z_0h[stable] < 1E-6] == 1E-6
        th_star = kappa \
            * (theta[stable] - Tsurf[stable]) \
            / (np.log(z_T[stable] / z_0h[stable]) - psi2 + psi1)
        q_star  = kappa *(q[stable] - q_surf[stable]) \
            / (np.log(z_T[stable] / z_0h[stable]) - psi2 + psi1)
        SHF[stable] = rho_atm[stable] * c_pd * u_star[stable] * th_star
        LHF[stable] = rho_atm[stable] * L_sub * u_star[stable] * q_star
        L_prev = L[stable]
        L[stable] = u_star[stable]**2 \
            * (theta[stable] + T_0)\
            * (1 + ((1-eps) / eps) * q[stable]) \
            / (g * kappa * th_star * (1 + ((1-eps)/eps) * q_star))
        L_dif = np.abs((L_prev-L[stable])/L_prev)

        # If n_elements(where(L_dif > L_dif_max)) eq 1 then break
        if np.all(L_dif <= L_dif_max):
            break

    # Calculate unstable stratification
    if len(unstable) > 0:
        for i in np.arange(0,21):
            x1  = (1-gamma*z_0           /L[unstable])**0.25
            x2  = (1-gamma*z_WS[unstable]/L[unstable])**0.25
            y1  = (1-gamma*z_0h[unstable]/L[unstable])**0.5
            y2  = (1-gamma*z_T[unstable] /L[unstable])**0.5
            psi_m1 = np.log(((1+x1)/2)**2*(1+x1**2)/2)-2*np.arctan(x1)+np.pi/2
            psi_m2 = np.log(((1+x2)/2)**2*(1+x2**2)/2)-2*np.arctan(x2)+np.pi/2
            psi1 = np.log(((1+y1)/2)**2)
            psi2 = np.log(((1+y2)/2)**2)
            u_star[unstable] = kappa*WS[unstable]/(np.log(z_WS[unstable]/z_0)-psi_m2+psi_m1)
            Re[unstable] = u_star[unstable]*z_0/nu[unstable]
            z_0h[unstable] = z_0 * np.exp(1.5 - 0.2 * np.log(Re[unstable]) - 0.11 \
                                        * (np.log(Re[unstable]))**2)

            # If n_elements(where(z_0h[unstable] < 1e-6)) > 1 then
            # z_0h[unstable[where(z_0h[unstable] < 1e-6)]] = 1e-6
            z_0h[stable][z_0h[stable] < 1E-6] == 1E-6
            th_star = kappa * (theta[unstable] - Tsurf[unstable]) \
                / (np.log(z_T[unstable] / z_0h[unstable]) - psi2 + psi1)
            q_star  = kappa * (q[unstable] - q_surf[unstable]) \
                / (np.log(z_T[unstable] / z_0h[unstable]) - psi2 + psi1)
            SHF[unstable] = rho_atm[unstable] * c_pd * u_star[unstable] * th_star
            LHF[unstable] = rho_atm[unstable] * L_sub * u_star[unstable] * q_star
            L_prev = L[unstable]
            L[unstable] = u_star[unstable]**2 * (theta[unstable]+T_0) \
                * ( 1 + ((1-eps) / eps) * q[unstable]) \
                / (g * kappa * th_star * ( 1 + ((1-eps) / eps) * q_star))
            L_dif = abs((L_prev-L[unstable])/L_prev)

            # If n_elements(where(L_dif > L_dif_max)) eq 1 then break
            if np.all(L_dif <= L_dif_max):
                break

    HF_nan = np.isnan(p) | np.isnan(T) | np.isnan(Tsurf)| np.isnan(q) | np.isnan(WS)
    SHF[HF_nan] = np.nan
    LHF[HF_nan] = np.nan
    return SHF, LHF

def calculate_atmospheric_density(p, T):
    """Calculate Atmospheric density

    Parameters
    ----------
    T : Air temperature
    p : Air pressure

    Returns
    -------
    rho_atm : Atmopsheric density
    """
    rho_atm = 100 * p / R_d / (T + T_0)
    return rho_atm

def calculate_kinematic_viscosity(T, rho_atm):
    """Calculate kinematic viscosity of air

    Parameters
    ----------
    T : Air temperature
    T_0 : Steam point temperature
    rho_atm : Atmopsheric density

    Returns
    -------
    nu : Kinematic viscosity
    """
    # Dynamic viscosity of air in Pa s (Sutherlands' equation using C = 120 K)
    nu = 18.27e-6 * (291.15 + 120) / ((T + T_0) + 120) * ((T + T_0) / 291.15)**1.5

    # Kinematic viscosity of air in m^2/s
    return nu / rho_atm

def calculate_specific_humidity(T, p, RH_cor):
    """Calculate specific humidity

    Parameters
    ----------
    T : Air temperature
    p : Air pressure
    RH_cor : Relative humidity corrected

    Constants
    ----------
    T_0 : Steam point temperature. Default is 273.15.
    T_100 : Steam point temperature in Kelvin
    es_0 : Saturation vapour pressure at the melting point (hPa)
    es_100 : Saturation vapour pressure at steam point temperature (hPa)
    eps

    Returns
    -------
    Specific humidity data array
    """
    # Saturation vapour pressure above 0 C (hPa)
    es_wtr = 10**(-7.90298 * (T_100 / (T + T_0) - 1) + 5.02808 * np.log10(T_100 / (T + T_0))
                - 1.3816E-7 * (10**(11.344 * (1 - (T + T_0) / T_100)) - 1)
                + 8.1328E-3 * (10**(-3.49149 * (T_100 / (T + T_0) -1)) - 1) + np.log10(es_100))

    # Saturation vapour pressure below 0 C (hPa)
    es_ice = 10**(-9.09718 * (T_0 / (T + T_0) - 1) - 3.56654
                * np.log10(T_0 / (T + T_0)) + 0.876793
                * (1 - (T + T_0) / T_0)
                + np.log10(es_0))

    # Specific humidity at saturation (incorrect below melting point)
    q_sat = eps * es_wtr / (p - (1 - eps) * es_wtr)

    # Replace saturation specific humidity values below melting point
    freezing = T < 0
    q_sat[freezing] = eps * es_ice[freezing] / (p[freezing] - (1 - eps) * es_ice[freezing])

    # Convert to kg/kg
    q= RH_cor * q_sat / 100

    #clean specific humidity
    q_nan = np.isnan(T) | np.isnan(RH_cor) | np.isnan(p)
    q[q_nan] = np.nan
    return q

def calculate_energy_balance(usr, dsr, ulr, dlr, shf, lhf):
    """Caculate net energy balance of a glacier
    Assuming no horizontal transfers of heat on the ground,
    the net energy flux into the surface,eb, sums the following components:

    Parameters
    ----------
    usr: upwelling shortwave radiation, W/m2
    dsr: downwelling shortwave radiation, W/m2; positive sign
    ulr: upwelling longwave radiation, W/m2; positive sign
    dlr: downwelling longwave radiation, W/m2
    shf: sensible heat flux, W/m2
    lhf: latent heat flux, W/m2

    Returns
    ----------
    Enet: the net energy flux into the surface, W/m2
    """
    # calculate net shortwave radiation
    srnet = dsr - usr # usr should be positive
    # calculate net longwave radiation
    lrnet = dlr - ulr # ulr should be positive
    # calculate net radiation
    rnet = srnet + lrnet
    # Assuming no horizontal transfers of heat on the ground,
    # calculate the net energy flux into the surface
    Enet = rnet + shf + lhf
    return Enet

def energy_balance_to_melt_rate_per_m2(energy_balance):
    """ Converts a given energy balance into a theoretic meltrate, which results of the energy balance

    Parameters
    ----------
    energy_balance: in W/m2

    Returns
    ----------
    Melt rate in cubic meters per second per square meter (m3/s/m2)

    """
    return energy_balance/(rho_w * L_fus)

# def _meltRatePerM2_to_meltWaterPerM2(melt_rate_per_m2, time_interval: dt.timedelta):
#     """ Converts the melt rate per square meter over a given time interval into how much melt water this would give.
#     Parameters
#     ----------

#     melt_rate_per_m2: in cubic liters per second for one square meter, positive value
#     time_interval: datetime timedelta

#     Returns
#     -------
#     Melt water per square meter in liters, will be positive if input parameter melt_rate_per_m2 is as well

#     '''
#     return melt_rate_per_m2 * time_interval.total_seconds() * 1000  # * 1000 cause result is liters


# def meltWaterPerM2_to_Ablation(melt_water_per_m2):
#     '''Converts a given melt water per square meter into ablation. Ablation means how much ice
#     would have been melted.

#     Parameters
#     ----------
#     melt_water_per_m2: in liters, must be positive

#     Returns
#     ---
#     Positive value of how much ice has been melted in meters
#     '''
#     if melt_water_per_m2 < 0:
#         exit("Melt water must be positive")
#     return melt_water_per_m2 / rho_i


# def meltRatePerM2_to_mmWePerDay(melt_rate_per_m2):
#     """
#     Converts the given melt rate per square meter in millimeter water equivalent per day

#     :param melt_rate_per_m2: in cubic liters per second for one square meter
#     :return:
#     """

#     return cls.meltRatePerM2_to_meltWaterPerM2(melt_rate_per_m2, ONE_DAY)  # 1 liter equals one mm
