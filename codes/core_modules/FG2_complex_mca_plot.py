#!/usr/bin/env python3
"""
Freya Glacier Analysis Level 2 (FG2) - Plot the SVD results

Analyze relationship between Greenland Ice Sheet SMB, sea ice, and geopotential height.

Author: Sonika Shahi
Updated: 9 Nov 2025

Environment: xeofs_env (conda activate xeofs_env on walrus server)
"""
# %%
from __future__ import annotations
from turtle import mode
import warnings
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from shapely import bounds
warnings.filterwarnings('ignore')
import os
import sys
import numpy as np
import xarray as xr
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica')
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Helvetica'
plt.rcParams['mathtext.it'] = 'Helvetica:italic'
plt.rcParams['mathtext.bf'] = 'Helvetica:bold'
import matplotlib.patheffects as pe
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.patches import FancyArrowPatch
from typing import Optional, Literal, Iterable, Union
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.signal as signal
import seaborn as sns
from dataclasses import dataclass, replace
from pyproj import Transformer, CRS
import pandas as pd

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utility helpers
# -------------------------
os.chdir('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/codes/core_modules')
print(os.getcwd())
from mca_data_loader import load_coarsed_mjjas_data

# Import modules from different directory
# Reference: https://www.geeksforgeeks.org/python-import-module-from-different-directory/
module_path = "/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/codes/utilities/"
if os.path.exists(module_path):
    sys.path.append(module_path)
    from figure_properties import FigureProperties as fp
else:
    raise ImportError(f"Module path '{module_path}' does not exist.")

def _get_latlon_meshgrid(variable: str, coarsen_level: Optional[int] = None):
    if variable == 'smb':
        if coarsen_level == 4:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level_4.nc')
        else:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
        x = da.coords['x'].values
        y = da.coords['y'].values
    elif (variable == 'sif') | (variable == 'z0500'):
        da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        x = da.coords['rlon'].values
        y = da.coords['rlat'].values
    return np.meshgrid(x, y)

def _get_mean_std_margin_of_error(df, var_name, critical_value=1.96):
    # Resample to uniform frequency
    df['doy'] = df.index.dayofyear
    # Remove values with doy > 365 (e.g., 366 for leap years) to avoid issues in grouping by day of year
    df.loc[df.doy > 365, 'doy'] = np.nan
    df_mean = pd.DataFrame(df[var_name].groupby(df['doy']).mean())
    df_std = pd.DataFrame(df[var_name].groupby(df['doy']).std())
    df_count = pd.DataFrame(df[var_name].groupby(df['doy']).count())
    df_std_error = df_std / np.sqrt(df_count)
    df_margin_of_error = critical_value * df_std_error
    return df_mean, df_std, df_margin_of_error

def _rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

@dataclass
class ComplexMCAConfig:
    """Configuration for building Complex MCA output paths."""
    name: str
    left_var_name: str
    right_var_name: str
    coarsen_level_left: int
    coarsen_level_right: int
    rotation: str
    total_modes: int
    rotation_modes: int
    def with_name(self, new_name: str) -> "ComplexMCAConfig":
        """Return a copy of this config with a different name."""
        return replace(self, name=new_name)

    def with_updates(self, **kwargs) -> "ComplexMCAConfig":
        """Return a copy with selected fields updated."""
        return replace(self, **kwargs)

def build_complex_rotated_mca_path(config: ComplexMCAConfig) -> Path:
    """Build the path to the complex rotated mca output file without loading it."""

    base_input_path = Path(
        f"/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results/")
    suffix = (f"_{config.left_var_name}_{config.coarsen_level_left}_vs_{config.right_var_name}_{config.coarsen_level_right}"
        f"_mjjas_1990to2019_xeofs_{config.rotation}_total_mode_{config.total_modes}_rotation_mode_{config.rotation_modes}.nc")
        # f"_mjjas_1990to2019_xeofs_{config.rotation}.nc")
        # f"_monthly_mean_1990to2019_xeofs_{config.rotation}.nc")
        # f"_mjjas_1990to2019_xeofs.nc")
        # f"_mjjas_1990to2019_xeofs_detrended.nc")
        # f"_mjjas_1990to2019_xeofs.nc")
        # f"_amjjaso_1990to2019_xeofs.nc")
    return base_input_path / f"{config.name}{suffix}"

def complex_correlation(scores):
    """
    Calculate Hermitian complex correlation matrix.

    Parameters
    ----------
    scores : np.ndarray
        Complex array with shape (time, modes)

    Returns
    -------
    corr : np.ndarray
        Complex correlation matrix
    """

    # Remove temporal mean
    scores_centered = scores - np.mean(scores, axis=0)

    # Hermitian covariance
    C = (
        scores_centered.conj().T
        @ scores_centered
        / (scores.shape[0] - 1)
    )

    # Normalize
    std = np.sqrt(np.real(np.diag(C)))

    corr = C / np.outer(std, std)

    return corr

def load_complex_mca_output(
    mode: int,
    path: Optional[Path] = None,
    config: Optional[ComplexMCAConfig] = None,
) -> xr.DataArray:
    """ Load the Complex MCA output file and extract a mode.

    Usage:
    # Two-step
    path = build_complex_rotated_mca_path(config)
    da = load_complex_mca_output(mode=1, path=path)

    # One-step
    da = load_complex_mca_output(mode=1, config=config)
    """

    if path is None:
        if config is None:
            raise ValueError("Must supply either `path` or `config`.")
        path = build_complex_rotated_mca_path(config)

    if not path.exists():
        raise FileNotFoundError(path)

    da = xr.open_dataarray(path)
    return da.sel(mode=mode)

def get_greenland_mask_latitude_longitude(variable: str):
    if variable == 'smb':
        # racmo1km_lat_lon_coarsened_4 = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level4.nc')
        # racmo1km_latitude_coarsened_4  = racmo1km_lat_lon_coarsened_4['LAT']
        # racmo1km_longitude_coarsened_4  = racmo1km_lat_lon_coarsened_4['LON']

        gr_masks_1km = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
        land_sea_mask = gr_masks_1km['Promicemask']
        land_sea_mask.values[land_sea_mask.values > 1] = 1

        # Latitude and longitude
        racmo_latitude = gr_masks_1km['LAT']
        racmo_longitude = gr_masks_1km['LON']

        x = racmo_longitude.values
        y = racmo_latitude.values
    else:
        # Coarsen Greenland mask and merge it with Svalbard and Iceland
        gr_masks_5km = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        land_sea_mask = gr_masks_5km['LSM_GR']
        topo = gr_masks_5km['Topography']
        topo_c = topo.copy()
        # Mask Svalbard and Iceland
        topo_c.values[(topo_c.values > 0) & (land_sea_mask.values == 0)] = 0
        # Mask Greenland to seperate Iceland and Svalbard
        topo.values[(topo.values == topo_c.values)] = 0
        # Add Iceland and Svalbard in land sea mask
        land_sea_mask.values[(topo.values > 100)] = 1

        # Latitude and longitude
        racmo_latitude = gr_masks_5km['lat']
        racmo_longitude = gr_masks_5km['lon']

    if variable == 'ci':
        land_sea_mask = xr.where((land_sea_mask == 0), np.nan, land_sea_mask) # to avoid hiding the colors below
        colors_landseamask = [(255, 255, 255), (169, 169, 169)]
    else:
        colors_landseamask = [(169, 169, 169),(0, 0, 0)]
    cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
    bounds_landseamask = np.array([-0.01, 0.01, 1.5])
    norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

    return land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask

def get_latlon_meshgrid(variable: str, coarsen_level: Optional[int] = None):
    if variable == 'smb':
        if coarsen_level == 4:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level_4.nc')
        else:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
        x = da.coords['x'].values
        y = da.coords['y'].values
    else:
        if coarsen_level == 2:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates_level_2.nc')
        else:
            da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        x = da.coords['rlon'].values
        y = da.coords['rlat'].values
    return np.meshgrid(x, y)

def get_bounds_cbtick_cbticklabel(data_type: str):
    if data_type == 'amplitudes':
        colors = [(68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141), (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # cbticklabel = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # cbticklabel = [0, '', 0.2, '', 0.4, '', 0.6, '', 0.8, '', 1]
        cbticklabel = [0, '', '', '', '', '', '', '', '', '', 1]
    elif data_type == 'phases':
        colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 228, 225), (173, 216, 230), (135, 206, 235), (70, 130, 180), (65, 105, 225)]
        # colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (224, 242, 241), (173, 216, 230), (135, 206, 235), (70, 130, 180), (65, 105, 225)]
        # colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 255, 255), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        # colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (153, 153, 153), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        # colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (140, 153, 179), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-3.14159, -2.35619, -1.5708, -0.785398, -0.0001, 0.0001, 0.785398, 1.5708, 2.35619, 3.14159])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-3.14159, -2.35619, -1.5708, -0.785398, 0, 0.785398, 1.5708, 2.35619, 3.14159]
        # cbticklabel = ['-π', '-3/4π', '-1/2π', '-1/4π', '0', '1/4π', '1/2π', '3/4π', 'π']
        # cbticklabel = ['-π', '', '-1/2π', '', '0', '', '1/2π', '', 'π']
        cbticklabel = ['-π', '', '', '', '0', '', '', '', 'π']

    return bounds, cmap, norm, cbtick, cbticklabel

def get_expansion_coefficient_ls(variable):
    if variable == 'ci':
        ls = '-'
    elif variable == 'smb':
        ls = '--'
    elif variable == 'z0500':
        ls = ':'
    return ls

def get_lonlat_params(variable):
    # Color longitude
    if variable == 'smb':
        bounds_lon = [-65, -55, -45, -35, -25, -15]
    else:
        bounds_lon = [-60, -50, -40, -30, -20, -10, 0]
    colors_lon = [(190, 190, 190), (190, 190, 190), (190, 190, 190), (190, 190, 190)]
    cmap_lon = fp.make_cmap(colors_lon, bit=True)
    norm_lon = mpl.colors.BoundaryNorm(bounds_lon, cmap_lon.N)

    # Color latitude
    bounds_lat = [60, 65, 70, 75, 80]
    colors_lat = [(190, 190, 190), (190, 190, 190), (190, 190, 190), (190, 190, 190)]
    cmap_lat = fp.make_cmap(colors_lat, bit=True)
    norm_lat = mpl.colors.BoundaryNorm(bounds_lat, cmap_lat.N)
    return bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat

def get_lonlat_ticks(variable, bounds_lon, bounds_lat):
    if variable == 'smb':
        xcoord = 'x'
        ycoord = 'y'
        da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
    else:
        xcoord = 'rlon'
        ycoord = 'rlat'
        da = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')

    xcoord_min = np.min(da.coords[xcoord].values)
    xcoord_max = np.max(da.coords[xcoord].values)

    ycoord_min = np.min(da.coords[ycoord].values)
    ycoord_max = np.max(da.coords[ycoord].values)

    if variable == 'smb':
        # Convert x,y into lat lon for x-y tick label
        xlims = [xcoord_min, xcoord_max]
        ylims = [ycoord_min, ycoord_max]
        bounds_lat_label = bounds_lat
    else:
        # Convert rlat rlon into lat lon for x-y tick label
        xlims = [xcoord_min, xcoord_max]
        ylims = [ycoord_min + 2, xcoord_max]
        bounds_lat_label = bounds_lat[:-1]   # The 80 N latitude goes beyond map frame, so exclude it.

    bounds_lon_label = bounds_lon[1:-2] # The longitude goes beyond map frame, so exclude it.

    # Convert the lowest lower left corner lat and lon from y, ylims[0] and x, xlims[0]
    # Define Polar stereographic EPSG:3413
    crs_xy = CRS.from_epsg(3413)
    # Define the custom rotated pole CRS
    crs_rpolar = CRS.from_proj4('+proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +datum=WGS84')
    # EPSG:4326 is the standard WGS84 lat/lon
    crs_lonlat = CRS.from_epsg(4326)

    # Create the transformer
    if variable == 'smb':
        # Create the transformer from xy (Polar stereographic EPSG:3413) to latlon (WGS84 lat/lon)
        transformer_to_lonlat = Transformer.from_crs(crs_xy, crs_lonlat, always_xy=True)
        # Create the transformer from latlon (WGS84 lat/lon) to xy (Polar stereographic EPSG:3413)
        transformer_from_lonlat = Transformer.from_crs(crs_lonlat, crs_xy, always_xy=True)
    else:
        # Create the transformer from rotated polar to latlon (WGS84 lat/lon)
        transformer_to_lonlat = Transformer.from_crs(crs_rpolar, crs_lonlat, always_xy=True)
        # Create the transformer from latlon (WGS84 lat/lon) to rotated polar
        transformer_from_lonlat = Transformer.from_crs(crs_lonlat, crs_rpolar, always_xy=True)

    # Perform the transformation from given coordinate to lonlat
    lon_low, lat_low = transformer_to_lonlat.transform(xlims[0], ylims[0])

    # Longitude labels
    xticks = []
    xticklabels = []
    for bound_lon_label in bounds_lon_label:
        x_tick, _ = transformer_from_lonlat.transform(bound_lon_label, lat_low + np.cos(lat_low))
        xticks.append(x_tick)
        # xticklabels.append(f"{abs(bound_lon_label)}$^\\circ$W")
        xticklabels.append(f"{abs(bound_lon_label)}°W")

    # Latitude labels
    yticks = []
    yticklabels = []
    for bound_lat_label in bounds_lat_label:
        _, y_tick = transformer_from_lonlat.transform(lon_low, bound_lat_label)
        yticks.append(y_tick)
        # yticklabels.append(f"{bound_lat_label}$^\\circ$N")
        yticklabels.append(f"{bound_lat_label}°N")

    # Rescale the xticks
    if variable == 'smb':
        # Rescale the xticks
        scaler = 9000
        for i, _ in enumerate(xticks):
            if i == 0:
                xticks[i] = xticks[i] + scaler
            if i == 2:
                xticks[i] = xticks[i] - 2 * scaler
            else:
                xticks[i] = xticks[i] + i

        # Rescale the yticks
        scaler = 9000
        for i, _ in enumerate(yticks):
            yticks[i] = yticks[i] + (i * i * 9000) + 26000
    else:
        # Rescale the yticks
        scaler = 0.09999999999999964
        for i, _ in enumerate(yticks):
            if i == 0:
                yticks[i] = yticks[i] + 2 * scaler
            if i == 2:
                yticks[i] = yticks[i] + i - 2 * scaler
            else:
                yticks[i] = yticks[i] + i


    return xlims, xticks, xticklabels, ylims, yticks, yticklabels

def get_labels_heterogenous_correlation(variable_pair, variable):
    # Add text between main variable and second variable, and percentage of var explained name
    if variable_pair == ('ci', 'smb'):
        if variable == 'ci':
            fig_label = 'a)'
            het_pat_text = r'r(SIF, EC$\mathdefault{_{SMB}}$)'
        elif variable == 'smb':
            fig_label = 'd)'
            het_pat_text = r'r(SMB, EC$\mathdefault{_{SIF}}$)'
    elif variable_pair == ('ci', 'z0500'):
        if variable == 'ci':
            fig_label = 'b)'
            het_pat_text = r'r(SIF, EC$\mathdefault{_{Z500}}$)'
        elif variable == 'z0500':
            fig_label = 'e)'
            het_pat_text = r'r(Z500, EC$\mathdefault{_{SIF}}$)'
    elif variable_pair == ('z0500', 'smb'):
        if variable == 'z0500':
            fig_label = 'c)'
            het_pat_text = r'r(Z500, EC$\mathdefault{_{SMB}}$)'
        elif variable == 'smb':
            fig_label = 'f)'
            het_pat_text = r'r(SMB, EC$\mathdefault{_{Z500}}$)'
    return fig_label, het_pat_text

def get_labels_singular_vectors(variable_pair, variable, method_name):
    # Add text between main variable and second variable, and percentage of var explained name
    if variable_pair == ('ci', 'smb'):
        if variable == 'ci':
            if method_name == 'rotxeofs':
                svd_pat = r'U$^{*}_{\mathrm{SIF}}$'
            else:
                svd_pat = r'U$\mathdefault{_{SIF}}$'
            fig_label = 'a)'
        elif variable == 'smb':
            if method_name == 'rotxeofs':
                svd_pat = r'V$^{*}_{\mathrm{SMB}}$'
            else:
                svd_pat = r'V$\mathdefault{_{SMB}}$'
            fig_label = 'd)'
    elif variable_pair == ('ci', 'z0500'):
        if variable == 'ci':
            if method_name == 'rotxeofs':
                svd_pat = r'U$^{*}_{\mathrm{SIF}}$'
            else:
                svd_pat = r'U$\mathdefault{_{SIF}}$'
            fig_label = 'b)'
        elif variable == 'z0500':
            if method_name == 'rotxeofs':
                svd_pat = r'V$^{*}_{\mathrm{Z500}}$'
            else:
                svd_pat = r'V$\mathdefault{_{Z500}}$'
            fig_label = 'e)'
    elif variable_pair == ('z0500', 'smb'):
        if variable == 'z0500':
            if method_name == 'rotxeofs':
                svd_pat = r'U$^{*}_{\mathrm{Z500}}$'
            else:
                svd_pat = r'U$\mathdefault{_{Z500}}$'
            fig_label = 'c) '
        elif variable == 'smb':
            if method_name == 'rotxeofs':
                svd_pat = r'V$^{*}_{\mathrm{SMB}}$'
            else:
                svd_pat = r'V$\mathdefault{_{SMB}}$'
            fig_label = 'f) '
    return f"{fig_label} {svd_pat}"

def get_bounds_cbtick_label(step, max_val, zero_replace, skip_step):
    # step = 0.001; max_val=0.007; zero_replace=step * 0.01; skip_step=3
    if step > max_val:
        raise ValueError(f"Step '{step}' must be less than max_val '{max_val}'")
    # Define symmetric bounds around zero
    bounds = np.arange(-max_val, max_val + step, step)
    # Find index close to zero
    zero_index = np.where(np.isclose(bounds, 0))[0].item()
    bounds = np.insert(bounds, zero_index, zero_replace * (-1))
    bounds[zero_index + 1] = zero_replace # replace 0

    # Define cbtick
    cbtick = np.arange(-max_val, max_val + step, step)
    cbtick[np.where(np.isclose(cbtick, 0))[0][0]] = 0
    # Add tick to only some values
    # Take absolute values of positives (to mirror negatives)
    vals = cbtick[cbtick >= 0]  # includes 0
    neg_vals = -vals[::-1][::-1] # mirror to negative
    # Downsample both sides equally
    pos = vals[::skip_step]
    neg = neg_vals[::skip_step] # reverse back to increasing order
    # Ensure lengths match
    m = min(len(neg), len(pos))
    neg, pos = neg[:m], pos[:m]
    # Combine
    cbtick_reduced = np.concatenate([neg[::-1][:-1], [0], pos[1:]])  # avoid duplicating 0

    # Define cbticklabel
    cbticklabel = cbtick_reduced.copy() * 10**3
    cbticklabel = cbticklabel.round(1)
    cbticklabel_format = [str(int(v)) if v.is_integer() else str(v) for v in cbticklabel]

    return bounds, cbtick_reduced, cbticklabel_format

def get_station_location(station, variable):

    # EPSG:4326 is the standard WGS84 lat/lon
    source_crs = CRS.from_epsg(4326)
    # -------------
    # Using pyproj
    # -------------
    if variable == 'smb':
        # Define the target CRS (Polar stereographic EPSG:3413)
        target_crs = CRS.from_epsg(3413)
    else:
        # Define the custom rotated pole CRS
        target_crs = CRS.from_proj4("+proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +datum=WGS84")
    # Create the transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Add the location of Freya Glacier
    station_metadata_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/station_data/northeast_stations_metadata.csv'
    stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
    stations_data.index = stations_data.Stations
    stations_data.drop('Stations', axis=1, inplace=True)
    stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
    stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)
    # Perform the transformation
    station_X, station_Y = transformer.transform(stations_data.loc['lon (W)', station], stations_data.loc['lat (N)', station])
    return station_X, station_Y

def reconstruct_complex(da):
    """Reconstruct complex DataArray from real and imaginary parts.

    """
    if isinstance(da.dtype, np.dtype) and da.dtype.names == ('r', 'i'):
        return xr.DataArray(
            da.data['r'] + 1j * da.data['i'],
            coords=da.coords,
            dims=da.dims,
            attrs=da.attrs)
    return da

def add_subplot_frac(fig, gs, row, col_start, col_end, row_span=1, adjust=True):
    """
    Create a subplot using fractional column indices.
    Example: col_start=0.5, col_end=2.5 will center the plot between columns 0–3.
    """
    import math
    # Determine integer range of cells covering the requested fraction
    col_start_i = math.floor(col_start)
    col_end_i = math.ceil(col_end)

    # Create initial subplot
    ax = fig.add_subplot(gs[row:row+row_span, col_start_i:col_end_i])

    if adjust:
        # Compute adjustment fraction relative to full subplot cell width
        ncols = gs.get_geometry()[1]
        total_width = 1.0 / ncols
        dx_left = (col_start - col_start_i) * total_width
        dx_right = (col_end_i - col_end) * total_width

        # Adjust subplot position
        pos = ax.get_position()
        new_x0 = pos.x0 + dx_left
        new_width = pos.width - (dx_left + dx_right)
        ax.set_position([new_x0, pos.y0, new_width, pos.height])

    return ax

# %%
class ComplexVarimaxRotatedMCA:
    """Plot the amplitude functions, phase functions, and real part of PC at phase 0.

    """
    def plot_analytical_svd(
        left_var_name: str,
        right_var_name: str,
        coarsen_level_left: int, # Left variables are always at original resolution; 'ci' and 'z0500'
        coarsen_level_right: int,
        mode: int,
        total_modes: int,
        rotation: str,
        rotation_modes: int,
        figure_name: str):
        """Plot the complex varimax rotated MCA results.

        """
        # %%
        # left_var_name = 'z0500'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'ci'
        # coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # left_var_name = 'z0500'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'smb'
        # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # left_var_name = 'ci'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'smb'
        # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # mode = 1
        # rotation = 'varimax'
        # # # rotation = 'promax'
        # total_modes = 3
        # rotation_modes = 3

        print(f"Plotting analytical SVD for {left_var_name} vs {right_var_name} with coarsen levels {coarsen_level_left} and {coarsen_level_right} and mode {mode}")

        # Base config with all shared parameters
        base_cfg = ComplexMCAConfig(name='',  # will override per-call
                                    left_var_name=left_var_name,
                                    right_var_name=right_var_name,
                                    coarsen_level_left=coarsen_level_left,
                                    coarsen_level_right=coarsen_level_right,
                                    rotation=rotation,
                                    total_modes=total_modes,
                                    rotation_modes=rotation_modes)
        tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=8, width=0.5)

        # fig = plt.figure(figsize=(3, 2.5), dpi=600)
        if (left_var_name == 'z0500') and (right_var_name == 'smb'):
            # fig = plt.figure(figsize=(2.3, 2.1), dpi=600)
            fig = plt.figure(figsize=(2.9, 3.1), dpi=600)
        elif (left_var_name == 'ci') and (right_var_name == 'smb'):
            # fig = plt.figure(figsize=(3.5, 3.8), dpi=600)
            fig = plt.figure(figsize=(3, 3.2), dpi=600)
        else:
            fig = plt.figure(figsize=(3.3, 3.1), dpi=600)
        # if left_var_name == 'z0500':
        #     fig = plt.figure(figsize=(2.3, 2.1), dpi=600)
        # else:
        #     fig = plt.figure(figsize=(2, 2.2), dpi=600)

        for col, variable in [(0, left_var_name), (1, right_var_name)]:
            if variable == left_var_name:
                pattern = 'left'
            else:
                pattern = 'right'
            # Get lat lon meshgrid and land sea mask
            if variable == 'smb':
                lonlat = get_latlon_meshgrid(variable=variable)
                lonlat_coarsened = get_latlon_meshgrid(variable=variable, coarsen_level=4)
            else:
                lonlat = get_latlon_meshgrid(variable=variable)
                lonlat_coarsened = get_latlon_meshgrid(variable=variable, coarsen_level=0)
            # Get land sea mask, latitude and longitudes
            land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = get_greenland_mask_latitude_longitude(variable=variable)

            # Plot amplitude and phase functions
            for component_type in ['amplitudes', 'phases']:
                if component_type == 'amplitudes':
                    row = 0
                else:
                    row = 1

                if ((left_var_name == 'ci') and (right_var_name == 'smb')) | ((left_var_name == 't2m') and (right_var_name == 'smb')) | ((left_var_name == 'psurf') and (right_var_name == 'smb')):
                    if (row == 0) and (col == 0):
                        # ax = fig.add_axes([0.12, 0.54, 0.55, 0.45])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                        ax = fig.add_axes([0.13, 0.53, 0.525, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 0) and (col == 1):
                        # ax = fig.add_axes([0.58, 0.54, 0.55, 0.45])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                        ax = fig.add_axes([0.59, 0.53, 0.525, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 0):
                        ax = fig.add_axes([0.13, 0.045, 0.525, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 1):
                        ax = fig.add_axes([0.59, 0.045, 0.525, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                elif (left_var_name == 'z0500') and (right_var_name == 'smb'):
                    if (row == 0) and (col == 0):
                        ax = fig.add_axes([0.14, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 0) and (col == 1):
                        # ax = fig.add_axes([0.535, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                        ax = fig.add_axes([0.61, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 0):
                        ax = fig.add_axes([0.14, 0.045, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 1):
                        ax = fig.add_axes([0.61, 0.045, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                elif (left_var_name == 'z0500') and (right_var_name == 'ci'):
                    if (row == 0) and (col == 0):
                        ax = fig.add_axes([0.09, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 0) and (col == 1):
                        # ax = fig.add_axes([0.61, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                        ax = fig.add_axes([0.55, 0.535, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 0):
                        ax = fig.add_axes([0.09, 0.045, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                    elif (row == 1) and (col == 1):
                        ax = fig.add_axes([0.55, 0.045, 0.52, 0.425])   # [left, bottom, width, height] in figure coordinates (0 to 1)

                if component_type == 'amplitudes':
                    # Read component amplitude
                    component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{pattern}_component_amplitudes"))
                    component = component_amplitude/component_amplitude.max()
                elif component_type == 'phases':
                    # Read component phase
                    component = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{pattern}_component_phases"))

                    component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{pattern}_component_amplitudes"))
                    component_amplitude_max_norm = component_amplitude/component_amplitude.max()

                # Define plotting parameters
                bounds, cmap, norm, _, _ = get_bounds_cbtick_cbticklabel(data_type=component_type)
                # if variable == 'smb':
                #     if component_type == 'amplitudes':
                #         ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                #     elif component_type == 'phases':
                #         ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component_amplitude_max_norm > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                # else:
                #     if component_type == 'amplitudes':
                #         ax.contourf(lonlat[0], lonlat[1], component.where(component > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                #         # ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                #     elif component_type == 'phases':
                #         ax.contourf(lonlat[0], lonlat[1], component.where(component_amplitude_max_norm > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                #         # ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component_amplitude_max_norm > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)

                if component_type == 'amplitudes':
                    ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)
                elif component_type == 'phases':
                    ax.contourf(lonlat_coarsened[0], lonlat_coarsened[1], component.where(component_amplitude_max_norm > 0.25, np.nan), cmap=cmap, norm=norm, extend='neither', levels=bounds)

                # Plot land sea mask
                if variable == 'ci':
                    ax.contourf(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, zorder=20)
                else:
                    ax.contour(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, linewidths=0.05, zorder=5)

                # Plot latitude and longitudes
                bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat = get_lonlat_params(variable=variable)
                ax.contour(lonlat[0], lonlat[1], racmo_longitude, cmap=cmap_lon, norm=norm_lon, levels=bounds_lon, linewidths=0.2, linestyles='dashed', zorder=10)
                ax.contour(lonlat[0], lonlat[1], racmo_latitude, cmap=cmap_lat, norm=norm_lat, levels=bounds_lat, linewidths=0.2, linestyles='dashed', zorder=10)

                # Set x-y labels
                xlims, xticks, xticklabels, ylims, yticks, yticklabels = get_lonlat_ticks(variable=variable, bounds_lon=bounds_lon, bounds_lat=bounds_lat)
                ax.set_xlim(xlims)
                ax.set_ylim(ylims)

                ax.set_xticks(xticks)
                ax.set_yticks(yticks)

                ax.set_xticklabels(xticklabels, color='k')
                ax.set_yticklabels(yticklabels, color='k')

                if left_var_name == 'z0500':
                    labelsize=7
                else:
                    labelsize=7
                tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=labelsize, width=0.5)
                ax.tick_params(**tick_params)
                [i.set_linewidth(0.3) for i in ax.spines.values()]

                ax.set_aspect('equal', adjustable='box')

                if variable == 'smb':
                    freya_X, freya_Y = get_station_location(station='FRE2 (promice type, tripod)', variable=variable)
                    ax.plot(freya_X, freya_Y, 'o', markersize=3, markeredgewidth=0.8, markerfacecolor='none', markeredgecolor='0.3', zorder=20)

                if (variable == 'ci') | (variable == 'z0500'):
                    # # Read the sea ice buffer
                    mask_file_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks'
                    # sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
                    # Greenland Sea mask for z500
                    gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
                    binary_mask = xr.where(gsea_mask > 0.5, 1, 0)
                    cs = ax.contour(lonlat[0], lonlat[1], binary_mask, colors='k', linewidths=0.2, linestyles='dashed', zorder=1)
                    cs.set_dashes([(0, (10, 10))])

                if pattern == 'left':
                    if component_type == 'amplitudes':
                        # fig_label = r'$\mathbf{a}$'
                        fig_label = '(a)'
                    else:
                        # fig_label = r'$\mathbf{c}$'
                        fig_label = '(c)'
                if pattern == 'right':
                    if component_type == 'amplitudes':
                        # fig_label = r'$\mathbf{b}$'
                        fig_label = '(b)'
                    else:
                        # fig_label = r'$\mathbf{d}$'
                        fig_label = '(d)'

                if (left_var_name == 'ci') and (right_var_name == 'smb'):
                    ax.text(0.025, 0.93, fig_label, transform=ax.transAxes, horizontalalignment='left', size=8, fontweight='bold', color='k', zorder=30, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
                elif (left_var_name == 'z0500') and (right_var_name == 'smb'):
                    ax.text(0.025, 0.925, fig_label, transform=ax.transAxes, horizontalalignment='left', size=8, fontweight='bold', color='k', zorder=30, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
                elif (left_var_name == 'z0500') and (right_var_name == 'ci'):
                    ax.text(0.025, 0.925, fig_label, transform=ax.transAxes, horizontalalignment='left', size=8, fontweight='bold', color='k', zorder=30, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
        # Variance explained by the mode
        main_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results'
        variance_explained = xr.open_dataarray(f"{main_path}/squared_covariance_fraction_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_unrotated.nc")
        # variance_explained = xr.open_dataarray(f"{main_path}/squared_covariance_fraction_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_monthly_mean_1990to2019_xeofs_unrotated.nc")
        variance_explained_mode = variance_explained.sel(mode=mode).values
        # variance_explained = load_complex_mca_output(mode=mode, config=base_cfg.with_name('squared_covariance_fraction'))
        variance_explained_text = f"{variance_explained_mode * 100:.0f}%"
        if (left_var_name == 'ci') and (right_var_name == 'smb'):
            fig.text(0.65, 0.965, f"Mode {mode} ({variance_explained_text})", transform=fig.transFigure, horizontalalignment='right', size=8, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
        elif (left_var_name == 'z0500') and (right_var_name == 'smb'):
            fig.text(0.65, 0.97, f"Mode {mode} ({variance_explained_text})", transform=fig.transFigure, horizontalalignment='right', size=8, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
        elif (left_var_name == 'z0500') and (right_var_name == 'ci'):
            fig.text(0.65, 0.97, f"Mode {mode} ({variance_explained_text})", transform=fig.transFigure, horizontalalignment='right', size=8, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]

        # # targets = dict(left=290, right=0, top=10, bottom=55, hspace=250, wspace=0)
        # targets = dict(left=290, right=0, top=10, bottom=55, hspace=0, wspace=0)
        # # Hook up a function to adjust the borders when the window is resized
        # fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        # fp.adjust_borders(fig, targets)

        # Colorbar for amplitude plots
        bounds, cmap, norm, cbtick, cbticklabel = get_bounds_cbtick_cbticklabel(data_type='amplitudes')
        if right_var_name == 'ci':
            cbar_ax_1 = fig.add_axes([0.055, 0.55, 0.022, 0.4]) #[left, bottom, width, height]
        else:
            cbar_ax_1 = fig.add_axes([0.06, 0.55, 0.022, 0.4]) #[left, bottom, width, height]
        cb1 = mpl.colorbar.ColorbarBase(cbar_ax_1, orientation='vertical', cmap=cmap, norm=norm)
        cb1.set_ticks(cbtick)
        cb1.minorticks_off()
        cb1.update_ticks()
        cb1.set_ticklabels(cbticklabel)
        # cb1.ax.tick_params(labelsize=5, pad=0.5, length=2, direction='in', width=0.5)
        cb1.ax.tick_params(labelsize=8, pad=0.5, length=2, direction='in', width=0.5)
        cb1.outline.set_linewidth(0.4)
        cb1.set_label('Amplitude', size=8, labelpad=-3)
        cb1.ax.yaxis.set_label_position('left')
        cb1.ax.yaxis.tick_left()

        # Colorbar for phase plots
        bounds, cmap, norm, cbtick, cbticklabel = get_bounds_cbtick_cbticklabel(data_type='phases')
        if right_var_name == 'ci':
            cbar_ax_2 = fig.add_axes([0.055, 0.06, 0.0222, 0.4]) #[left, bottom, width, height]
        else:
            cbar_ax_2 = fig.add_axes([0.06, 0.06, 0.0222, 0.4]) #[left, bottom, width, height]
        cb2 = mpl.colorbar.ColorbarBase(cbar_ax_2, orientation='vertical', cmap=cmap, norm=norm)
        cb2.set_ticks(cbtick)
        cb2.minorticks_off()
        cb2.update_ticks()
        cb2.set_ticklabels(cbticklabel)
        # cb2.ax.tick_params(labelsize=5, pad=0.5, length=2, direction='in', width=0.5)
        cb2.ax.tick_params(labelsize=8, pad=0.5, length=2, direction='in', width=0.5)
        cb2.outline.set_linewidth(0.4)
        cb2.set_label('Phase', size=8, labelpad=-2)
        cb2.ax.yaxis.set_label_position('left')
        cb2.ax.yaxis.tick_left()

        # Save the plot
        logger.info(f"Saving the analytical svd plot for ...")
        base_output_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures'
        # # fig.savefig(f"{base_output_path}{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mode_{mode}_monthly_mean_{rotation}.png", dpi=600)
        # fig.savefig(f"{base_output_path}{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mode_{mode}_mjjas_{rotation}.png", dpi=600)
        fig.savefig(f"{base_output_path}/{figure_name}.png", dpi=300)
    # %%
    def plot_principal_component_and_frequency_spectrum_(
        left_var_name: str,
        right_var_name: str,
        coarsen_level_left: int, # Left variables are always at original resolution; 'ci' and 'z0500'
        coarsen_level_right: int,
        mode: int):
        """Plot the complex varimax rotated MCA results.

        """
        # %%
        print(f"Plotting principal component and frequency spectrum for {left_var_name} vs {right_var_name} with coarsen levels {coarsen_level_left} and {coarsen_level_right} and mode {mode}")
        # left_var_name = 'ci'
        left_var_name = 'z0500'
        coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        right_var_name = 'ci'
        coarsen_level_right = 0 # 'smb' is always coarsened to level 4
        mode = 2

        # Base config with all shared parameters
        base_cfg = ComplexMCAConfig(name='',  # will override per-call
                                    left_var_name=left_var_name,
                                    right_var_name=right_var_name,
                                    coarsen_level_left=coarsen_level_left,
                                    coarsen_level_right=coarsen_level_right)
        tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=5, width=0.5)

        fig = plt.figure(figsize=(3.2, 2.25), dpi=600)
        ax_fs = fig.add_axes([0.58, 0.3, 0.4, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)

        for row, col, variable in [(0, 0, left_var_name), (1, 0, right_var_name)]:
            if variable == left_var_name:
                pattern = 'left'
                color = 'tab:blue'
            else:
                pattern = 'right'
                color = 'tab:orange'

            if (row == 0) and (col == 0):
                # ax = fig.add_axes([0.08, 0.59, 0.45, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)
                ax = fig.add_axes([0.08, 0.59, 0.4, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)
            elif (row == 1) and (col == 0):
                ax = fig.add_axes([0.08, 0.09, 0.4, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)

            score = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{pattern}_scores"))
            reconstruct_score = reconstruct_complex(score)
            reconstruct_score_max_norm = reconstruct_score/reconstruct_score.max()

            # Find separate positive and negative extrema
            max_pos = reconstruct_score.where(reconstruct_score > 0).max()
            max_neg = reconstruct_score.where(reconstruct_score < 0).min()

            # Normalize separately for positive and negative values
            reconstruct_score_max_norm = xr.where(reconstruct_score > 0,
                                                  reconstruct_score / max_pos,
                                                  xr.where(reconstruct_score < 0,
                                                           reconstruct_score / abs(max_neg),
                                                           reconstruct_score))

            # Reset the time coordinate to a sequential index
            reconstruct_score_max_norm = reconstruct_score_max_norm.assign_coords(day_index=('time', range(1, reconstruct_score_max_norm.time.size + 1)))
            reconstruct_score_max_norm.real.plot(x='day_index', ax=ax, color=color, lw=0.5)

            # Set ticks and labels
            # --- Define major and minor ticks ---
            years = reconstruct_score_max_norm.day_index.values
            major_years = [1, 153 * 5, 153 * 10, 153 * 15, 153 * 20, 153 * 25, 153 * 30]
            # major_years_labels = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
            major_years_labels = ['90', '95', '00', '05', '10', '15', '20']
            minor_years = [153, 153 * 2, 153 * 3, 153 * 4, 153 * 6, 153 * 7, 153 * 8, 153 * 9, 153 * 11, 153 * 12, 153 * 13, 153 * 14, 153 * 16, 153 * 17, 153 * 18, 153 * 19, 153 * 21, 153 * 22, 153 * 23, 153 * 24, 153 * 26, 153 * 27, 153 * 28, 153 * 29]
            ax.set_xlim([years.min(), years.max()])
            ax.set_xticks(major_years)
            ax.set_xticklabels(major_years_labels, color='k')
            ax.set_xticks(minor_years, minor=True)
            ax.set_xlabel('')

            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, 0, 1])
            ax.set_yticklabels([-1, 0, 1], color='k')

            if (left_var_name == 'ci') & (right_var_name == 'smb'):
                if pattern == 'left':
                    pc_name = fr"PC$_\mathdefault{{SIF}}$"
                elif pattern == 'right':
                    pc_name = fr"PC$_\mathdefault{{SMB}}$"

            if (left_var_name == 'z0500') & (right_var_name == 'smb'):
                if pattern == 'left':
                    pc_name = fr"PC$_\mathdefault{{Z500}}$"
                elif pattern == 'right':
                    pc_name = fr"PC$_\mathdefault{{SMB}}$"

            if (left_var_name == 'z0500') & (right_var_name == 'ci'):
                if pattern == 'left':
                    pc_name = fr"PC$_\mathdefault{{Z500}}$"
                elif pattern == 'right':
                    pc_name = fr"PC$_\mathdefault{{SIF}}$"

            ax.set_ylabel(f"{pc_name} (-)", size=5)
            ax.yaxis.set_label_coords(-0.12, 0.5)

            ax.set_title('')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax_spine_shift = 3
            ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
            ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
            [i.set_linewidth(0.3) for i in ax.spines.values()]

            # Style x, y ticks
            ax.tick_params(axis='both', which='major', direction='out', length=5, width=0.4, labelsize=5, pad=1, color='k')
            # ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax.tick_params(axis='x', which='minor', length=2.5, width=0.25, color='gray')

            if pattern == 'left':
                fig_label = r'$\mathbf{a}$'
            if pattern == 'right':
                fig_label = r'$\mathbf{b}$'
            ax.text(0.005, 0.94, fig_label, transform=ax.transAxes, horizontalalignment='left', size=7, color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

            # Plot frequency spectrum of PC ------------------------------------------------
            time_steps = len(reconstruct_score)

            # Apply Hanning window
            window = signal.windows.hann(len(reconstruct_score))
            windowed_pc = reconstruct_score * window

            # Compute FFT
            # nperseg can be adjusted for frequency resolution vs. time resolution tradeoff
            # if fs=1 (samples per day; Daily data so 1 sample/day); then periods_years = periods / 12
            # fs=1 assumes daily data, so frequency is in cycles/day. Convert to cycles/year.
            # frequencies, power = scipy.signal.welch(windowed_pc, fs=1, nperseg=1024, scaling='density')
            # fs = 12  # samples per year; Monthly data so 12 samples/year
            frequencies, power = signal.welch(windowed_pc, fs=153, nperseg=1024, scaling='density')

            # Convert frequency (cycles/day) to period (days)
            # Avoid division by zero for 0 frequency (DC component)
            periods = np.zeros_like(frequencies)
            valid_freq_mask = frequencies > 1e-6 # Avoid division by zero or very small frequencies
            periods[valid_freq_mask] = 1 / frequencies[valid_freq_mask]

            # Filter out the PC component (period = infinity) for plotting clarity if needed
            valid_period_mask = periods < (time_steps / 2) # Nyquist limit related
            periods = periods[valid_period_mask]
            power = power[valid_period_mask]
            # power = power / np.sum(power)  # Normalize power for better comparison
            # power = power / (365 * 24 * 60 * 60)  # 365 days * 24 hours * 60 minutes * 60 seconds # SI units (m²/Hz, i.e. cycles per second); 12 cycles/year to Hz (divide frequency by 86400 s/day)
            power = power / (153 * 24 * 60 * 60)  # 365 days * 24 hours * 60 minutes * 60 seconds # SI units (m²/Hz, i.e. cycles per second); 12 cycles/year to Hz (divide frequency by 86400 s/day)
            # (already in years) periods_years = periods / 12  # Convert days to years for monthly data
            ax_fs.plot(periods[1:], power[1:], color=color, lw=0.5) # Log-log plot is common for spectra
            # ax.plot(periods_years[1:50], power[1:50], marker='o') # Log-log plot is common for spectra

        ax_fs.set_xscale('log')
        if mode == 1:
            xlim = [1.e-01, 9]
            xticks = [1.e-01, 1.e+00, 1.e+01]
            xticklabels = [f"$10^{{-1}}$", f"$10^{{0}}$", f"$10^{{1}}$"]

            ylim = [0, 0.0001]
            yticks = [0, 0.002, 0.004, 0.006, 0.008, 0.01]
            yticklabels = [0, 2, 4, 6, 8, 10]
            ylabel = fr"Power (x 10$^\mathdefault{{-3}}$ Hz$^\mathdefault{{-1}}$)"

        elif mode == 2:
            xlim = [0.009, 9]
            xticks = [1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]
            xticklabels = [f"$10^{{-4}}$", f"$10^{{-3}}$", f"$10^{{-2}}$", f"$10^{{-1}}$", f"$10^{{0}}$", f"$10^{{1}}$", f"$10^{{2}}$"]

            ylim = [0, 7.556100075972494e-05]
            yticks = [0.e+00,  5.e-05,  1.e-04]
            yticklabels = [0, 5, 10]
            ylabel = fr"Power (x 10$^\mathdefault{{-5}}$ Hz$^\mathdefault{{-1}}$)"

        ax_fs.set_xlim(xlim)
        ax_fs.set_xticks(xticks)
        # ax_fs.set_xticklabels([f"$10^{{-4}}$", f"$10^{{-3}}$", f"$10^{{-2}}$", f"$10^{{-1}}$", f"$10^{{0}}$", f"$10^{{1}}$", f"$10^{{2}}$", f"$10^{{3}}$"])
        ax_fs.set_xticklabels(xticklabels, color='k')
        ax_fs.set_xlabel('Period (years)', size=5)
        ax_fs.xaxis.set_label_coords(0.5, -0.25)

        ax_fs.set_ylim(ylim)
        ax_fs.set_yticks(yticks)
        ax_fs.set_yticklabels(yticklabels, color='k')
        # ax_fs.set_ylabel('Power (x 10$^\mathdefault{-3}$ units$^\mathdefault{2}$ Hz$^\mathdefault{-1}$)', size=6, labelpad=-2.5)
        ax_fs.set_ylabel(ylabel, size=5) # labelpad=-2.5
        ax_fs.yaxis.set_label_coords(-0.13, 0.5)

        # Add specific period lines if relevant
        ax_fs.axvline(1, color='k', linestyle='--', lw=0.5)
        ax_fs.spines['right'].set_visible(False)
        ax_fs.spines['top'].set_visible(False)
        ax_spine_shift = 3
        ax_fs.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax_fs.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        # ax_fs.spines['bottom'].set_bounds(1.e-01, 1.e+01)
        [i.set_linewidth(0.3) for i in ax_fs.spines.values()]

        # Style x, y ticks
        ax_fs.tick_params(axis='both', which='major', direction='out', length=5, width=0.4, labelsize=5, pad=1, color='k')
        # ax_fs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_fs.tick_params(axis='x', which='minor', length=2.5, width=0.25, color='gray')

        ax_fs.text(0.005, 0.94, r'$\mathbf{c}$', transform=ax_fs.transAxes, horizontalalignment='left', size=7, color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

        # Variance explained by the mode
        # variance_explained = load_complex_mca_output(mode=mode, config=base_cfg.with_name('squared_covariance_fraction'))
        main_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/{left_var_name}_vs_{right_var_name}")
        variance_explained = xr.open_dataarray(f"{main_path}/squared_covariance_fraction_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_unrotated.nc")
        variance_explained_mode = variance_explained.sel(mode=mode).values
        variance_explained_text = f"{variance_explained_mode * 100:.0f}%"
        fig.text(0.85, 0.75, f"Mode {mode} ({variance_explained_text})", transform=fig.transFigure, horizontalalignment='right', size=6, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]

        # Legend items
        if (left_var_name == 'ci') & (right_var_name == 'smb'):
            left_var_label = 'SIF'
            right_var_label = 'SMB'
        elif (left_var_name == 'z0500') & (right_var_name == 'smb'):
            left_var_label = 'Z500'
            right_var_label = 'SMB'
        elif (left_var_name == 'z0500') & (right_var_name == 'ci'):
            left_var_label = 'Z500'
            right_var_label = 'SIF'
        left = mlines.Line2D([], [], label=left_var_label, lw=0.5, color='tab:blue')
        right = mlines.Line2D([], [], label=right_var_label, lw=0.5, color='tab:orange')
        legend_param = dict(loc='center', fontsize=5, frameon=False, handletextpad=0.5)
        fig.legend(handles=[left, right], bbox_to_anchor=(0.78, 0.1), ncol=2, handlelength=1, labelspacing=1.3, markerscale=1.5, **legend_param)

        # Save the plot
        logger.info(f"Saving plot_principal_component_and_frequency_spectrum for ...")
        base_output_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG2_complex_mca_plots-plot_principal_component_and_frequency_spectrum_'
        fig.savefig(f"{base_output_path}{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mode_{mode}_mjjas.png", dpi=600)

    def plot_principal_component_and_frequency_spectrum(
        mode: int,
        total_modes: int,
        rotation: str,
        rotation_modes: int,
        figure_name: str,):
        """Plot the complex varimax rotated MCA results.

        """
        # %%
        # mode = 2
        # rotation = 'varimax'
        # # rotation = 'promax'
        # total_modes = 3
        # rotation_modes = 3

        fig = plt.figure(figsize=(4, 2.6), dpi=600)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.01], hspace=0.3)   # large spacing between row groups
        gs_top = outer[0].subgridspec(3, 3, hspace=0.7, wspace=0.3)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        vars = ['ci', 'smb', 'z0500']
        colors = [(27, 158, 119), (217, 95, 2), (117, 112, 179)]
        rgb_hex = []
        for color in colors:
            rgb_hex.append(_rgb_to_hex(color))
        var_styles = {var: {'color': color, 'linestyle': '-', 'marker': 'o', 'markersize': 2, 'lw': 0.9} for var, color in zip(vars, rgb_hex)}

        for left_var_name, coarsen_level_left, left_label, right_var_name, coarsen_level_right, right_label, row, fs_label in [('ci', 0, '(a)', 'smb', 4, '(b)', 0, '(c)'), ('z0500', 0, '(d)', 'smb', 4, '(e)', 1, '(f)'), ('z0500', 0, '(g)', 'ci', 0, '(h)', 2, '(i)')]:
            # Base config with all shared parameters
            base_cfg = ComplexMCAConfig(name='',  # will override per-call
                                        left_var_name=left_var_name,
                                        right_var_name=right_var_name,
                                        coarsen_level_left=coarsen_level_left,
                                        coarsen_level_right=coarsen_level_right,
                                        rotation=rotation,
                                        total_modes=total_modes,
                                        rotation_modes=rotation_modes)

            ax_fs = fig.add_subplot(gs_top[row, 2])
            for col, variable, pattern in [(0, left_var_name, 'left'), (1, right_var_name, 'right')]:
                ax = fig.add_subplot(gs_top[row, col])

                score = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{pattern}_scores"))
                reconstruct_score = reconstruct_complex(score)
                reconstruct_score_max_norm = reconstruct_score/reconstruct_score.max()

                # Find separate positive and negative extrema
                max_pos = reconstruct_score.where(reconstruct_score > 0).max()
                max_neg = reconstruct_score.where(reconstruct_score < 0).min()

                # Normalize separately for positive and negative values
                reconstruct_score_max_norm = xr.where(reconstruct_score > 0,
                                                    reconstruct_score / max_pos,
                                                    xr.where(reconstruct_score < 0,
                                                            reconstruct_score / abs(max_neg),
                                                            reconstruct_score))

                # Reset the time coordinate to a sequential index
                reconstruct_score_max_norm = reconstruct_score_max_norm.assign_coords(day_index=('time', range(1, reconstruct_score_max_norm.time.size + 1)))
                reconstruct_score_max_norm.real.plot(x='day_index', ax=ax, lw=0.5, color=var_styles[variable]['color'])

                # Set ticks and labels
                # --- Define major and minor ticks ---
                years = reconstruct_score_max_norm.day_index.values
                major_years = [1, 153 * 5, 153 * 10, 153 * 15, 153 * 20, 153 * 25, 153 * 30]
                # major_years_labels = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
                major_years_labels = ['90', '95', '00', '05', '10', '15', '20']
                minor_years = [153, 153 * 2, 153 * 3, 153 * 4, 153 * 6, 153 * 7, 153 * 8, 153 * 9, 153 * 11, 153 * 12, 153 * 13, 153 * 14, 153 * 16, 153 * 17, 153 * 18, 153 * 19, 153 * 21, 153 * 22, 153 * 23, 153 * 24, 153 * 26, 153 * 27, 153 * 28, 153 * 29]
                ax.set_xlim([years.min(), years.max()])
                ax.set_xticks(major_years)
                ax.set_xticklabels(major_years_labels, color='k')
                ax.set_xticks(minor_years, minor=True)
                ax.set_xlabel('Year', size=5)
                ax.xaxis.set_label_coords(0.5, -0.4)

                ax.set_ylim(-1, 1)
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels([-1, 0, 1], color='k')

                if (left_var_name == 'ci') & (right_var_name == 'smb'):
                    if pattern == 'left':
                        pc_name = fr"PC$_\mathdefault{{SIC}}$"
                    elif pattern == 'right':
                        pc_name = fr"PC$_\mathdefault{{SMB}}$"

                if (left_var_name == 'z0500') & (right_var_name == 'smb'):
                    if pattern == 'left':
                        pc_name = fr"PC$_\mathdefault{{Z500}}$"
                    elif pattern == 'right':
                        pc_name = fr"PC$_\mathdefault{{SMB}}$"

                if (left_var_name == 'z0500') & (right_var_name == 'ci'):
                    if pattern == 'left':
                        pc_name = fr"PC$_\mathdefault{{Z500}}$"
                    elif pattern == 'right':
                        pc_name = fr"PC$_\mathdefault{{SIC}}$"

                ax.set_ylabel(f"{pc_name} (-)", size=5)
                ax.yaxis.set_label_coords(-0.17, 0.5)

                ax.set_title('')

                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax_spine_shift = 3
                ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
                ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
                [i.set_linewidth(0.3) for i in ax.spines.values()]

                # Style x, y ticks
                ax.tick_params(axis='both', which='major', direction='out', length=4, width=0.4, labelsize=5, pad=1, color='k')
                # ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                ax.tick_params(axis='x', which='minor', length=2.5, width=0.25, color='gray')

                if pattern == 'left':
                    fig_label = f"{left_label}"
                if pattern == 'right':
                    fig_label = f"{right_label}"
                ax.text(0.005, 0.94, fig_label, transform=ax.transAxes, horizontalalignment='left', size=7, fontweight='bold', color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

                # Plot frequency spectrum of PC ------------------------------------------------
                time_steps = len(reconstruct_score)

                # Apply Hanning window
                window = signal.windows.hann(len(reconstruct_score))
                windowed_pc = reconstruct_score * window

                # Compute FFT
                # nperseg can be adjusted for frequency resolution vs. time resolution tradeoff
                # if fs=1 (samples per day; Daily data so 1 sample/day); then periods_years = periods / 12
                # fs=1 assumes daily data, so frequency is in cycles/day. Convert to cycles/year.
                # frequencies, power = scipy.signal.welch(windowed_pc, fs=1, nperseg=1024, scaling='density')
                # fs = 12  # samples per year; Monthly data so 12 samples/year
                frequencies, power = signal.welch(windowed_pc, fs=153, nperseg=1024, scaling='density')

                # Convert frequency (cycles/day) to period (days)
                # Avoid division by zero for 0 frequency (DC component)
                periods = np.zeros_like(frequencies)
                valid_freq_mask = frequencies > 1e-6 # Avoid division by zero or very small frequencies
                periods[valid_freq_mask] = 1 / frequencies[valid_freq_mask]

                # Filter out the PC component (period = infinity) for plotting clarity if needed
                valid_period_mask = periods < (time_steps / 2) # Nyquist limit related
                periods = periods[valid_period_mask]
                power = power[valid_period_mask]
                # power = power / np.sum(power)  # Normalize power for better comparison
                # power = power / (365 * 24 * 60 * 60)  # 365 days * 24 hours * 60 minutes * 60 seconds # SI units (m²/Hz, i.e. cycles per second); 12 cycles/year to Hz (divide frequency by 86400 s/day)
                power = power / (153 * 24 * 60 * 60)  # 365 days * 24 hours * 60 minutes * 60 seconds # SI units (m²/Hz, i.e. cycles per second); 12 cycles/year to Hz (divide frequency by 86400 s/day)
                # (already in years) periods_years = periods / 12  # Convert days to years for monthly data
                ax_fs.plot(periods[1:], power[1:], lw=0.5, color=var_styles[variable]['color']) # Log-log plot is common for spectra
                # ax.plot(periods_years[1:50], power[1:50], marker='o') # Log-log plot is common for spectra

            ax_fs.set_xscale('log')
            if mode == 1:
                xlim = [1.e-01, 9]
                xticks = [1.e-01, 1.e+00, 1.e+01]
                xticklabels = [f"$10^{{-1}}$", f"$10^{{0}}$", f"$10^{{1}}$"]

                if (left_var_name == 'ci') & (right_var_name == 'smb'):
                    ylim = [0, 0.01]
                    yticks = [0, 0.002, 0.004, 0.006, 0.008, 0.01]
                    yticklabels = [0, 2, 4, 6, 8, 10]
                elif (left_var_name == 'z0500') & (right_var_name == 'smb'):
                    ylim = [0, 0.01]
                    yticks = [0, 0.002, 0.004, 0.006, 0.008, 0.01]
                    # ylim = [0, 0.004]
                    # yticks = [0, 0.001, 0.002, 0.003, 0.004]
                    yticklabels = [0, 2, 4, 6, 8, 10]
                elif (left_var_name == 'z0500') & (right_var_name == 'ci'):
                    ylim = [0, 0.014]
                    yticks = [0, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014]
                    yticklabels = [0, 2, 4, 6, 8, 10, 12, 14]
                ylabel = fr"Power (x 10$^\mathdefault{{-3}}$ Hz$^\mathdefault{{-1}}$)"

            elif mode == 2:
                xlim = [0.09, 9]
                xticks = [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02]
                xticklabels = [f"$10^{{-3}}$", f"$10^{{-2}}$", f"$10^{{-1}}$", f"$10^{{0}}$", f"$10^{{1}}$", f"$10^{{2}}$"]

                ylim = [0, 7.556100075972494e-05]
                yticks = [0.e+00,  5.e-05,  1.e-04]
                yticklabels = [0, 5, 10]

                if (left_var_name == 'ci') & (right_var_name == 'smb'):
                    ylim = [0, 7.556100075972494e-05]
                    yticks = [0.e+00,  5.e-05,  1.e-04]
                    yticklabels = [0, 5, 10]
                    ylabel = fr"Power (x 10$^\mathdefault{{-5}}$ Hz$^\mathdefault{{-1}}$)"
                elif (left_var_name == 'z0500') & (right_var_name == 'smb'):
                    ylim = [0, 9.e-04]
                    yticks = [0.e+00, 3.e-04, 6.e-04, 9.e-04]
                    yticklabels = [0, 3, 6, 9]
                    ylabel = fr"Power (x 10$^\mathdefault{{-4}}$ Hz$^\mathdefault{{-1}}$)"
                elif (left_var_name == 'z0500') & (right_var_name == 'ci'):
                    ylim = [0, 5e-04]
                    yticks = [0.e+00, 2.5e-04,  5.e-04]
                    yticklabels = [0, 2.5, 5]
                    ylabel = fr"Power (x 10$^\mathdefault{{-4}}$ Hz$^\mathdefault{{-1}}$)"

            # Variance explained by the mode
            # variance_explained = load_complex_mca_output(mode=mode, config=base_cfg.with_name('squared_covariance_fraction'))
            main_path = Path(f"/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results")
            variance_explained = xr.open_dataarray(f"{main_path}/squared_covariance_fraction_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_unrotated.nc")
            variance_explained_mode = variance_explained.sel(mode=mode).values
            variance_explained_text = f"{variance_explained_mode * 100:.0f}%"
            # ax_fs.text(0.4, 0.77, rf"$\alpha_\mathdefault{{{mode} ({left_var_name}_vs_{right_var_name})}}$ = {variance_explained_text}", transform=ax_fs.transAxes, horizontalalignment='right', size=6, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
            # ax_fs.text(0.4, 0.77, rf"$\alpha_\mathdefault{{{mode}}}$ $_\mathdefault{{({left_var_name}_vs_{right_var_name})}}$ = {variance_explained_text}", transform=ax_fs.transAxes, horizontalalignment='right', size=6, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
            if (left_var_name == 'ci') & (right_var_name == 'smb'):
                # alpha_label = rf"$\alpha_\mathdefault{{{mode}}}$ $_\mathdefault{{(SIF}}$$_\mathdefault{{-}}$$_\mathdefault{{SMB)}}$ = {variance_explained_text}"
                alpha_label = rf"$\psi_\mathdefault{{{mode}}}$ $_\mathdefault{{(SIC}}$$_\mathdefault{{-}}$$_\mathdefault{{SMB)}}$"
            elif (left_var_name == 'z0500') & (right_var_name == 'smb'):
                alpha_label = rf"$\psi_\mathdefault{{{mode}}}$ $_\mathdefault{{(Z500}}$$_\mathdefault{{-}}$$_\mathdefault{{SMB)}}$"
            elif (left_var_name == 'z0500') & (right_var_name == 'ci'):
                alpha_label = rf"$\psi_\mathdefault{{{mode}}}$ $_\mathdefault{{(Z500}}$$_\mathdefault{{-}}$$_\mathdefault{{SIC)}}$"
            ax_fs.text(0.8, 0.97, alpha_label, transform=ax_fs.transAxes, horizontalalignment='center', verticalalignment='center', size=5, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]
            ax_fs.text(0.8, 0.8, f"{variance_explained_text}", transform=ax_fs.transAxes, horizontalalignment='center', verticalalignment='center', size=5, color='k', zorder=11) # path_effects=[pe.withStroke(linewidth=1.5, foreground='white')]

            ax_fs.set_xlim(xlim)
            ax_fs.set_xticks(xticks)
            ax_fs.set_xticklabels(xticklabels, color='k')
            # ax_fs.set_xlabel(f"Period (years, {variance_explained_text})", size=5)
            ax_fs.set_xlabel(f"Period (years)", size=5)
            ax_fs.xaxis.set_label_coords(0.5, -0.4)

            ax_fs.set_ylim(ylim)
            ax_fs.set_yticks(yticks)
            ax_fs.set_yticklabels(yticklabels, color='k')
            # ax_fs.set_ylabel('Power (x 10$^\mathdefault{-3}$ units$^\mathdefault{2}$ Hz$^\mathdefault{-1}$)', size=6, labelpad=-2.5)
            ax_fs.set_ylabel(ylabel, size=5) # labelpad=-2.5
            ax_fs.yaxis.set_label_coords(-0.2, 0.5)

            # Add specific period lines if relevant
            ax_fs.axvline(1, color='k', linestyle='--', lw=0.5)
            ax_fs.spines['right'].set_visible(False)
            ax_fs.spines['top'].set_visible(False)
            ax_spine_shift = 3
            ax_fs.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
            ax_fs.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
            # ax_fs.spines['bottom'].set_bounds(1.e-01, 1.e+01)
            [i.set_linewidth(0.3) for i in ax_fs.spines.values()]

            # Style x, y ticks
            ax_fs.tick_params(axis='both', which='major', direction='out', length=4, width=0.4, labelsize=5, pad=1, color='k')
            # ax_fs.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
            ax_fs.tick_params(axis='x', which='minor', length=2.5, width=0.25, color='gray')
            ax_fs.text(0.005, 0.94, rf'$\mathbf{{{fs_label}}}$', transform=ax_fs.transAxes, horizontalalignment='left', size=7, color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

        # Legend items
        mode_label = mlines.Line2D([], [], label=f"Mode {mode}", lw=0, color='k')
        ci_label = mlines.Line2D([], [], label='SIC', lw=0.5, color=var_styles['ci']['color'])
        smb_label = mlines.Line2D([], [], label='SMB', lw=0.5, color=var_styles['smb']['color'])
        z500_label = mlines.Line2D([], [], label='Z500', lw=0.5, color=var_styles['z0500']['color'])
        legend_param = dict(loc='center', fontsize=5, frameon=False, handletextpad=0.5)

        ax_legend = fig.add_subplot(gs_bot[0, 0])
        ax_legend.axis('off')  # Hide the axis for the legend subplot
        ax_legend.legend(handles=[mode_label, ci_label, smb_label, z500_label], bbox_to_anchor=(0.5, 0.1), ncol=4, handlelength=1, labelspacing=1.3, markerscale=1.5, **legend_param)

        targets = dict(left=157, right=30, top=35, bottom=30, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Save the plot
        logger.info(f"Saving plot_principal_component_and_frequency_spectrum for ...")
        base_output_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures'
        fig.savefig(f"{base_output_path}/{figure_name}.png", dpi=600)

        # %%
    def plot_covariance_of_modes():
        # %%
        print("Plotting covariance of modes for ci vs smb, z0500 vs ci, and z0500 vs smb...")

        vars = ['ci_vs_smb', 'z0500_vs_ci', 'z0500_vs_smb']
        colors = [(27, 158, 119), (217, 95, 2), (117, 112, 179)]
        rgb_hex = []
        for color in colors:
            rgb_hex.append(_rgb_to_hex(color))
        var_styles = {var: {'color': color, 'linestyle': '-', 'marker': 'o', 'markersize': 2, 'lw': 0.9} for var, color in zip(vars, rgb_hex)}

        fig = plt.figure(figsize=(4, 2.5), dpi=600)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.01], hspace=0.33)   # large spacing between row groups
        gs_top = outer[0].subgridspec(1, 2, hspace=0, wspace=0.2)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        ax_ind_mode = fig.add_subplot(gs_top[0, 0])
        ax_cum_mode = fig.add_subplot(gs_top[0, 1])

        # Read data
        for left_var_name, coarsen_level_left, right_var_name, coarsen_level_right, color in [('ci', 0, 'smb', 4, (27, 158, 119)), ('z0500', 0, 'smb', 4, (217, 95, 2)), ('z0500', 0, 'ci', 0, (117, 112, 179))]:
            main_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results'
            squared_covariance_fraction = xr.open_dataarray(f"{main_path}/squared_covariance_fraction_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_unrotated.nc")
            squared_covariance_fraction = squared_covariance_fraction.sel(mode=squared_covariance_fraction.mode.values[:38])  # Limit to first 40 modes for plotting

            modes = squared_covariance_fraction['mode'].values
            squared_covariance_fraction_data = squared_covariance_fraction.values * 100
            cumsum_squared_covariance_fraction_data = np.cumsum(squared_covariance_fraction.values) * 100

            # Individual mode and cumulative mode plots
            ax_ind_mode.plot(modes, squared_covariance_fraction_data, marker='o', markersize=1, lw=0.6, color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'])
            ax_cum_mode.plot(modes, cumsum_squared_covariance_fraction_data, marker='o', markersize=1, lw=0.6, color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'])

            # Choose the mode where cumulative covariance fraction reaches 99% for annotation
            cumsum_squared_covariance_fraction_data_round = np.round(cumsum_squared_covariance_fraction_data, 0)
            # idx_99 = np.where(cumsum_squared_covariance_fraction_data_round == 99)[0][0]
            idx_99 = np.where(cumsum_squared_covariance_fraction_data >= 99)[0][0]

            # ax_ind_mode.axvline(modes[idx_99], color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'], linestyle='--', lw=0.6)
            # ax_cum_mode.axvline(modes[idx_99], color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'], linestyle='--', lw=0.6)
            ax_ind_mode.vlines(modes[idx_99], ymin=1.e-05, ymax=squared_covariance_fraction_data[idx_99], color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'], linestyle='--', lw=0.6)
            ax_cum_mode.vlines(modes[idx_99], ymin=75, ymax=cumsum_squared_covariance_fraction_data[idx_99], color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'], linestyle='--', lw=0.6)

            ax_cum_mode.hlines(cumsum_squared_covariance_fraction_data[idx_99], xmin=-0.5, xmax=modes[idx_99], color=var_styles[f"{left_var_name}_vs_{right_var_name}"]['color'], linestyle='--', lw=0.6)

        # Set ticks and labels
        # For individual mode plot
        ax_ind_mode.set_xlim(modes.min() - 0.5, 40)
        ax_ind_mode.set_xticks([1, 10, 20, 30, 40])
        ax_ind_mode.set_xticks([2.5, 5, 7.5, 12.5, 15, 17.5, 22.5, 25, 27.5, 32.5, 35, 37.5], minor=True)
        ax_ind_mode.set_xticklabels([1, 10, 20, 30, 40], color='k')
        ax_ind_mode.set_xlabel('Mode', size=7)
        ax_ind_mode.xaxis.set_label_coords(0.5, -0.095)

        ax_ind_mode.set_yscale('log')
        # ax_ind_mode.yaxis.set_minor_locator(AutoMinorLocator())
        ax_ind_mode.set_ylim(1.e-05, 1.e+02 + 10)
        ax_ind_mode.set_yticks([1.e-05, 1.e-04, 1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02])
        ax_ind_mode.set_ylabel(rf'$\psi_\mathdefault{{i}}$ (%)', size=7) # labelpad=-2.5
        ax_ind_mode.yaxis.set_label_coords(-0.17, 0.5)

        # Add specific period lines if relevant
        ax_ind_mode.spines['right'].set_visible(False)
        ax_ind_mode.spines['top'].set_visible(False)
        ax_spine_shift = 3
        ax_ind_mode.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax_ind_mode.spines['left'].set_bounds(1.e-05, 1.e+02)
        ax_ind_mode.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax_ind_mode.spines['bottom'].set_bounds(1, 40)
        [i.set_linewidth(0.3) for i in ax_ind_mode.spines.values()]

        # Style x, y ticks
        ax_ind_mode.tick_params(axis='both', which='major', direction='out', length=3, width=0.3, labelsize=7, pad=1, color='k')
        # ax_ind_mode.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_ind_mode.tick_params(axis='both', which='minor', length=1.5, width=0.2, color='gray')
        ax_ind_mode.tick_params(axis='y', which='minor', direction='out', length=1.5, width=0.2, color='gray', left=True)

        # Add subplot label
        # ax_ind_mode.text(-0.225, 0.97, '(a)', transform=ax_ind_mode.transAxes, horizontalalignment='left', size=9, fontweight='bold', color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])
        ax_ind_mode.text(-0.235, 0.98, '(a)', transform=ax_ind_mode.transAxes, horizontalalignment='left', size=8, fontweight='bold', color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

        # For cumulative mode plot
        ax_cum_mode.set_xlim(modes.min() - 0.5, 40)
        ax_cum_mode.set_xticks([1, 10, 20, 30, 40])
        ax_cum_mode.set_xticks([2.5, 5, 7.5, 12.5, 15, 17.5, 22.5, 25, 27.5, 32.5, 35, 37.5], minor=True)
        ax_cum_mode.set_xticklabels([1, 10, 20, 30, 40], color='k')
        ax_cum_mode.set_xlabel('Mode', size=7)
        ax_cum_mode.xaxis.set_label_coords(0.5, -0.095)

        ax_cum_mode.set_ylim(74.5, 100 + 0.5)
        ax_cum_mode.set_yticks([75, 80, 85, 90, 95, 99, 100])
        ytick_labels = [75, 80, 85, 90, 95, 99, 100]
        ax_cum_mode.set_yticklabels([f"{label}" if label != 99 else f"$\\bf{{{label}}}$" for label in ytick_labels])
        # ax_cum_mode.set_ylabel('Cumulated Covariance Fraction (%)', size=7) # labelpad=-2.5
        # ax_cum_mode.set_ylabel(r'$CCF_k$ ($\frac{\sum_{i=1}^{k}\lambda_i}{\sum_j\lambda_j}\times100$ %)', size=7)
        ax_cum_mode.set_ylabel(r'$\sum_{i=1}^{k}\psi_i$ (%)', size=7)
        ax_cum_mode.yaxis.set_label_coords(-0.13, 0.5)

        # Add specific period lines if relevant
        ax_cum_mode.spines['right'].set_visible(False)
        ax_cum_mode.spines['top'].set_visible(False)
        ax_spine_shift = 3
        ax_cum_mode.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax_cum_mode.spines['left'].set_bounds(75, 100)
        ax_cum_mode.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax_cum_mode.spines['bottom'].set_bounds(1, 40)
        [i.set_linewidth(0.3) for i in ax_cum_mode.spines.values()]

        # Style x, y ticks
        ax_cum_mode.tick_params(axis='both', which='major', direction='out', length=3, width=0.4, labelsize=7, pad=1, color='k')
        # ax_cum_mode.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        ax_cum_mode.tick_params(axis='both', which='minor', length=1.5, width=0.2, color='gray')

        # Add subplot label
        ax_cum_mode.text(-0.245, 0.98, '(b)', transform=ax_cum_mode.transAxes, horizontalalignment='left', size=8, fontweight='bold', color='k', zorder=11, path_effects=[pe.withStroke(linewidth=1.5, foreground='w')])

        # Add legend
        ax_legend = fig.add_subplot(gs_bot[0, 0])
        # ax_legend = fig.add_axes([0.3, 0.01, 0.6, 0.1])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_legend.axis('off')  # Hide the axes for the legend

        sif_vs_smb = Line2D([], [], label='SIC–SMB', **var_styles['ci_vs_smb'])
        z0500_vs_smb = Line2D([], [], label='Z500–SMB', **var_styles['z0500_vs_smb'])
        z0500_vs_sif = Line2D([], [], label='Z500–SIC', **var_styles['z0500_vs_ci'])

        legend_param = dict(loc='center', fontsize=7, frameon=False, handletextpad=0.5)
        ax_legend.legend(handles=[sif_vs_smb, z0500_vs_smb, z0500_vs_sif], bbox_to_anchor=(0.5, 0.5), ncol=3, columnspacing=1, handlelength=1, labelspacing=1.3, markerscale=1, **legend_param)

        targets = dict(left=230, right=30, top=30, bottom=30, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Save the plot
        logger.info(f"Saving plot_covariance_of_modes for ...")
        base_output_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/manuscript_figures'
        fig.savefig(f"{base_output_path}/Figure_4.png", dpi=300)

    # %%
    def calculate_and_print_phase_shift_ci_smb_z500(left_var_name: str,
                                                    right_var_name: str,
                                                    coarsen_level_left: int,
                                                    coarsen_level_right: int,
                                                    mode: int,
                                                    total_modes: int,
                                                    rotation: str,
                                                    rotation_modes: int,
                                                    periodicity: int):

        # %%
        # left_var_name = 'z0500'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'ci'
        # coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # left_var_name = 'z0500'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'smb'
        # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # left_var_name = 'ci'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'smb'
        # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # mode = 1
        # rotation = 'varimax'
        # # Find lag days
        # periodicity = 153 # days/months; dominant periodicity of mode 1
        # main_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/{left_var_name}_vs_{right_var_name}")
        # total_modes = 3
        # rotation_modes = 3
        # Base config with all shared parameters
        base_cfg = ComplexMCAConfig(name='',  # will override per-call
                                    left_var_name=left_var_name,
                                    right_var_name=right_var_name,
                                    coarsen_level_left=coarsen_level_left,
                                    coarsen_level_right=coarsen_level_right,
                                    rotation=rotation,
                                    total_modes=total_modes,
                                    rotation_modes=rotation_modes)
        # Variable at location
        # Calculate time lags from phase patterns --------------------------------
        # Left pattern
        # Amplitude
        left_component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"left_component_amplitudes"))
        left_component_amplitude = left_component_amplitude/left_component_amplitude.max()
        left_component_amplitude_sig = left_component_amplitude.where(left_component_amplitude > 0.25, np.nan)

        # Phase
        left_component_phase = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"left_component_phases"))
        left_component_phase_sig = left_component_phase.where(left_component_amplitude_sig)
        left_time_lag_days = (periodicity * left_component_phase_sig) / (2 * np.pi)

        # Right pattern
        # Amplitude
        right_component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"right_component_amplitudes"))
        right_component_amplitude = right_component_amplitude/right_component_amplitude.max()
        right_component_amplitude_sig = right_component_amplitude.where(right_component_amplitude > 0.25, np.nan)

        # Phase
        right_component_phase = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"right_component_phases"))
        right_component_phase_sig = right_component_phase.where(right_component_amplitude_sig)
        right_time_lag_days = (periodicity * right_component_phase_sig) / (2 * np.pi)

        # Sea ice buffer mask for ci and z500
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        ci_buffer = xr.open_dataarray(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")
        ci_buffer = ci_buffer.sel(radius=800)

        # Greenland Sea mask for z500
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")

        if (left_var_name == 'ci') & (right_var_name == 'smb'):
            # For sea ice
            # Interpolate left to 100 km buffer of Sea Ice Location
            # left_time_lag_days_ci = left_time_lag_days.where(ci_buffer == 1)

            # Interpolate left to Greenland Sea
            left_time_lag_days_ci = left_time_lag_days.where(gsea_mask == 1)
            left_time_lag_days_ci = np.abs(left_time_lag_days_ci) # * (-1) # Take absolute value to ensure positive time lag
            left_time_lag_days_ci = left_time_lag_days_ci.where(left_time_lag_days_ci > 0, np.nan) # Remove negative values
            # left_time_lag_days_on_point = left_time_lag_days_ci.mean(dim=['rlat', 'rlon'])
            left_time_lag_days_on_point = left_time_lag_days_ci.median(dim=['rlat', 'rlon'])

            # For SMB
            # Change in moisture transfered from ocean to ZAC
            station = 'FRE2 (promice type, tripod)' #ZAC
            # Interpolate left to Freya Location
            station_x, station_y = get_station_location(station=station, variable='smb')
            right_time_lag_days_on_point = right_time_lag_days.sel(x=station_x, y=station_y, method='nearest')

        elif (left_var_name == 'z0500') & (right_var_name == 'smb'):
            # For z0500
            # Interpolate left to 100 km buffer of Sea Ice Location
            # left_time_lag_days_on_point = left_time_lag_days.where(ci_buffer == 1)
            left_time_lag_days_on_point = left_time_lag_days.where(gsea_mask == 1)
            left_time_lag_days_on_point = np.abs(left_time_lag_days_on_point)  # Take absolute value to ensure positive time lag
            # left_time_lag_days_on_point = left_time_lag_days_on_point.mean(dim=['rlat', 'rlon'])
            left_time_lag_days_on_point = left_time_lag_days_on_point.median(dim=['rlat', 'rlon'])

            # For SMB
            # Change in moisture transfered from ocean to ZAC
            station = 'FRE2 (promice type, tripod)' #ZAC
            # Interpolate left to Freya Location
            station_x, station_y = get_station_location(station=station, variable='smb')
            right_time_lag_days_on_point = right_time_lag_days.sel(x=station_x, y=station_y, method='nearest')

        elif (left_var_name == 'z0500') & (right_var_name == 'ci'):
            # For z0500
            # Interpolate left to 100 km buffer of Sea Ice Location
            # left_time_lag_days_on_point = left_time_lag_days.where(ci_buffer == 1)
            left_time_lag_days_on_point = left_time_lag_days.where(gsea_mask == 1)
            left_time_lag_days_on_point = np.abs(left_time_lag_days_on_point)  # Take absolute value to ensure positive time lag
            # left_time_lag_days_on_point = left_time_lag_days_on_point.mean(dim=['rlat', 'rlon'])
            left_time_lag_days_on_point = left_time_lag_days_on_point.median(dim=['rlat', 'rlon'])

            # For ci
            # right_time_lag_days_on_point = right_time_lag_days.where(ci_buffer == 1)
            right_time_lag_days_on_point = right_time_lag_days.where(gsea_mask == 1)
            right_time_lag_days_on_point = np.abs(right_time_lag_days_on_point) #* (-1) # Take absolute value to ensure positive time lag
            right_time_lag_days_on_point = right_time_lag_days_on_point.where(right_time_lag_days_on_point > 0, np.nan) # Remove negative values
            # right_time_lag_days_on_point = right_time_lag_days_on_point.mean(dim=['rlat', 'rlon'])
            right_time_lag_days_on_point = right_time_lag_days_on_point.median(dim=['rlat', 'rlon'])

        print(f"\nCalculating {left_var_name} and {right_var_name} ...")
        print(f"{left_var_name}: {left_time_lag_days_on_point.values} days")
        print(f"{right_var_name}: {right_time_lag_days_on_point.values} days")
        print(f"Time lag difference {right_var_name} and {left_var_name}: {right_time_lag_days_on_point.values - left_time_lag_days_on_point.values} days")
        # %%

    def plot_annual_cycle_smb_within_significant_insignificance_amplitude():
        """Plot the annual cycle of SMB within the significant and insignificant amplitude function of the right component of the SVD analysis.

        """
        # %%
        from matplotlib.patches import FancyArrowPatch
        from matplotlib.path import Path

        # read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        read_file_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var/from_freya_glacier'
        # Read the masked SMB dataset within significant amplitude
        masked_smb_within_significant_amplitude = pd.read_csv(f"{read_file_path}/masked_smb_within_significant_amplitude.csv", parse_dates=['time'], index_col='time')
        df_smb_mean_sig, df_smb_std_sig, df_smb_margin_of_error_sig = _get_mean_std_margin_of_error(masked_smb_within_significant_amplitude, 'smb', critical_value=1.96)
        # Read the masked SMB dataset within insignificant amplitude
        masked_smb_within_insignificant_amplitude = pd.read_csv(f"{read_file_path}/masked_smb_within_insignificant_amplitude.csv", parse_dates=['time'], index_col='time')
        df_smb_mean_insig, df_smb_std_insig, df_smb_margin_of_error_insig = _get_mean_std_margin_of_error(masked_smb_within_insignificant_amplitude, 'smb', critical_value=1.96)

        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        # colors = [(107, 174, 214), (8, 48, 107)]
        colors = [(86, 180, 233), (230, 159, 0)]
        colors_helix = [_rgb_to_hex(c) for c in colors]

        ax.plot(df_smb_mean_sig.index, df_smb_mean_sig['smb'], color=colors_helix[1], lw=0.5, zorder=12)
        ax.fill_between(df_smb_mean_sig.index, df_smb_mean_sig['smb'] - df_smb_margin_of_error_sig['smb'], df_smb_mean_sig['smb'] + df_smb_margin_of_error_sig['smb'], color=colors_helix[1], alpha=0.4, edgecolor='none', label="95% confidence interval")
        ax.fill_between(df_smb_mean_sig.index, df_smb_mean_sig['smb'] - df_smb_std_sig['smb'], df_smb_mean_sig['smb'] + df_smb_std_sig['smb'], color=colors_helix[1], alpha=0.2, edgecolor='none', label="±1 standard deviation")
        ax.plot(df_smb_mean_insig.index, df_smb_mean_insig['smb'], color=colors_helix[0], lw=0.5, zorder=12)
        ax.fill_between(df_smb_mean_insig.index, df_smb_mean_insig['smb'] - df_smb_margin_of_error_insig['smb'], df_smb_mean_insig['smb'] + df_smb_margin_of_error_insig['smb'], color=colors_helix[0], alpha=0.4, edgecolor='none', label="95% confidence interval")
        ax.fill_between(df_smb_mean_insig.index, df_smb_mean_insig['smb'] - df_smb_std_insig['smb'], df_smb_mean_insig['smb'] + df_smb_std_insig['smb'], color=colors_helix[0], alpha=0.2, edgecolor='none', label="±1 standard deviation")

        # Customize ticks and labels
        ax.set_xlim([1, 365])
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticklabels([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticks([15, 46, 74, 105, 135, 166, 196, 225, 258, 288, 319, 349], minor=True)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        ax.set_ylim([-20, 5])
        smb_yticks = [-20, -15, -10, -5, 0, 5]
        ax.set_yticks(smb_yticks)
        ax.set_yticks([-17.5, -12.5, -7.5, -2.5, 2.5], minor=True)
        ax.set_yticklabels(smb_yticks)
        ax.set_ylabel('SMB (mm w.e.)', color='k', fontsize=9)
        ax.yaxis.set_label_coords(-0.07, 0.5)

        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(-20, 5)
        ax.spines['bottom'].set_bounds(1, 365)
        ax.spines['bottom'].set_position(('outward', 3))
        ax.spines['left'].set_position(('outward', 3))
        [i.set_linewidth(0.4) for i in ax.spines.values()]

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=9, color='k')
        ax.tick_params(axis='both', which='major', **ytick_params)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=9)
        ax.tick_params(axis='y', which='minor', direction='out', width=0.5, length=2.5, pad=2)

        # Draw grid
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.2, alpha=0.7, zorder=0)
        # # Get spine bounds
        x0, x1 = ax.spines['bottom'].get_bounds()
        y0, y1 = ax.spines['left'].get_bounds()
        # x0, x1 = ax.get_xlim()
        # y0, y1 = ax.get_ylim()
        # Create clipping rectangle based on spine bounds
        grid_clip = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData)
        # Clip gridlines
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_clip_path(grid_clip)

        # Resizeable plot with fixed aspect ratio
        # targets = dict(left=250, right=270, top=30, bottom=300, hspace=0, wspace=0)
        targets = dict(left=275, right=70, top=30, bottom=1250, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Add a map of Greenland
        ax_map = fig.add_axes([0, 0, 0.45, 0.45])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        [i.set_visible(False) for i in ax_map.spines.values()]
        ax_map.axis('off')

        # Set parameters for the SVD analysis
        total_modes = 3
        rotation_modes = 3
        mode = 1
        rotation = 'varimax'
        left_var_name = 'ci'
        coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        right_var_name = 'smb'
        coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        main_path = f"/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results"
        # Read component amplitude
        right_component_amplitude = xr.open_dataarray(f"{main_path}/right_component_amplitudes_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_{rotation}_total_mode_{total_modes}_rotation_mode_{rotation_modes}.nc")
        right_component_amplitude = right_component_amplitude.sel(mode=1)
        right_component_amplitude_sig = right_component_amplitude/right_component_amplitude.max()

        right_component_phase = xr.open_dataarray(f"{main_path}/right_component_phases_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_{rotation}_total_mode_{total_modes}_rotation_mode_{rotation_modes}.nc")
        right_component_phase = right_component_phase.sel(mode=1)
        right_component_phase_sig = xr.where(right_component_amplitude_sig > 0.25, 1, 0)
        # right_component_phase_sig = right_component_phase_sig.where(right_component_phase_sig == 1, np.nan)
        mask_1km_file = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level_4.nc')['Promicemask']
        right_component_phase_sig = xr.where(mask_1km_file != 0, right_component_phase_sig, np.nan)

        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-0.5, 0.5, 1.5])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        lonlat = _get_latlon_meshgrid(variable='smb', coarsen_level=coarsen_level_right)
        ax_map.contourf(lonlat[0], lonlat[1], right_component_phase_sig, cmap=cmap, norm=norm, zorder=5)

        [i.set_visible(False) for i in ax_map.spines.values()]  # Set visible spines to have a linewidth of 0.3
        # ax_map.set_xlim([-7.5, 12.5])
        # ax.set_ylim([-13, 13])
        # ax_map.set_ylim([-12, 11.6])
        ax_map.axis('off')
        ax_map.set_aspect('equal', adjustable='box')

        gr_masks_1km = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level_4.nc')
        gr_masks_1km_topo = gr_masks_1km['Topography']

        gr_masks_1km_topo_0 = gr_masks_1km_topo.where(right_component_phase_sig == 0, np.nan)
        topo_0 = gr_masks_1km_topo_0.values[~np.isnan(gr_masks_1km_topo_0.values)].tolist()

        gr_masks_1km_topo_1 = gr_masks_1km_topo.where(right_component_phase_sig == 1, np.nan)
        topo_1 = gr_masks_1km_topo_1.values[~np.isnan(gr_masks_1km_topo_1.values)].tolist()

        # Create dataframe for topography values and their corresponding phase
        df_topo_0 = pd.DataFrame({'Elevation': topo_0})
        df_topo_0['topo_phase'] = 'Phase_0'
        df_topo_1 = pd.DataFrame({'Elevation': topo_1})
        df_topo_1['topo_phase'] = 'Phase_1'

        df_topo = pd.concat([df_topo_0, df_topo_1], ignore_index=True)
        df_topo['type'] = 'Elevation (m a.s.l.)'

        # Add elevation distribution plot
        ax_elevation = fig.add_axes([0.5, 0.05, 0.2, 0.35])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        pos = 0.08
        # Box plot properties
        boxplot_prop = dict(showcaps=False, widths=0.12, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1.5, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
        boxplot_prop_0 = dict(boxprops=dict(facecolor=colors_helix[0], color='k', linewidth=0.5))
        boxplot_prop_1 = dict(boxprops=dict(facecolor=colors_helix[1], color='k', linewidth=0.5))

        # Create box plot properties based on wind type
        ax_elevation.boxplot(df_topo.loc[df_topo['topo_phase'] == 'Phase_0', 'Elevation'], whis=[5, 95], positions=np.array([-pos]), **boxplot_prop_0, **boxplot_prop)
        ax_elevation.boxplot(df_topo.loc[df_topo['topo_phase'] == 'Phase_1', 'Elevation'], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_1, **boxplot_prop)
        # Create violin plot
        ax_violin = sns.violinplot(x='type', y='Elevation', hue='topo_phase', hue_order=['Phase_0', 'Phase_1'],
                                   split=True, inner=None, palette={'Phase_1':colors_helix[1], 'Phase_0':colors_helix[0]},
                                   data=df_topo, density_norm='count', ax=ax_elevation, linewidth=0.4)
        # Remove default labels and legend
        plt.setp(ax_violin.collections, zorder=2, label='')
        ax_elevation.get_legend().set_visible(False)
        ax_elevation.set_ylabel('')
        ax_elevation.set_xlabel('')

        ax_elevation.set_xlim([-0.5, 0.5])
        ax_elevation.set_xticks([])

        ax_elevation.set_ylim([0, 3694])
        topo_yticks = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]
        ax_elevation.set_yticks(topo_yticks)
        ax_elevation.set_yticks([250, 750, 1250, 1750, 2250, 2750, 3250], minor=True)
        ax_elevation.set_yticklabels(topo_yticks)
        ax_elevation.set_ylabel('Elevation (m a.s.l.)', color='k', fontsize=9)
        ax_elevation.yaxis.set_label_coords(-0.425, 0.5)

        # Customize spines
        ax_elevation.spines['top'].set_visible(False)
        ax_elevation.spines['right'].set_visible(False)
        ax_elevation.spines['bottom'].set_visible(False)
        ax_elevation.spines['left'].set_bounds(0, 3500)
        ax_elevation.spines['left'].set_position(('outward', 3))
        [i.set_linewidth(0.4) for i in ax_elevation.spines.values()]

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=9, color='k')
        ax_elevation.tick_params(axis='y', which='major', **ytick_params)
        ax_elevation.tick_params(axis='y', which='minor', direction='out', width=0.5, length=2.5, pad=2)

        # # Draw grid
        ax_elevation.grid(True, which='major', axis='y', linestyle='--', linewidth=0.2, alpha=0.7, zorder=0)

        # ------------------------------------
        # Add idealized violin plot in new axis
        # ------------------------------------
        # Generate skewed data (right-skewed)
        np.random.seed(42)
        # core = np.random.gamma(shape=1.0, scale=1.0, size=950)
        # tail = np.random.gamma(shape=2.0, scale=2.0, size=50)  # long tail
        # example_high = pd.DataFrame(np.concatenate([core, tail]), columns=['example_variable'])
        example_high = pd.DataFrame(np.random.gamma(shape=1.2, scale=2, size=1000), columns=['example_variable'])
        example_high['sif'] = 'High'
        example_low = pd.DataFrame(np.random.gamma(shape=2, scale=2, size=1000), columns=['example_variable'])
        example_low['sif'] = 'Low'
        example_data = pd.concat([example_high, example_low], ignore_index=True)
        example_data['time'] = 'idealized'
        # Plot only one side of the violin plot
        example_data.loc[example_data['sif'] == 'Low', 'example_variable'] = np.nan

        pos = 0.08
        # Create box plot properties based on wind type
        boxplot_prop_ideal = dict(boxprops=dict(facecolor='w', color='k', linewidth=0.5))

        ax_violin_idealized = fig.add_axes([0.725, 0.2, 0.15, 0.18])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_violin_idealized.boxplot(example_data.loc[example_data['sif'] == 'High', 'example_variable'], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_ideal, **boxplot_prop)
        # Create violin plot
        ax_violin = sns.violinplot(x='time', y='example_variable', hue='sif', hue_order=['High', 'Low'], split=True, inner=None,
                        palette={'High':'w', 'Low':'royalblue'}, data=example_data, density_norm='area',
                        ax=ax_violin_idealized, linewidth=0.4)

        [sp.set_visible(False) for sp in ax_violin_idealized.spines.values()]
        plt.setp(ax_violin.collections, zorder=2, label='')
        ax_violin_idealized.get_legend().set_visible(False)
        ax_violin_idealized.set_ylabel('')
        ax_violin_idealized.set_xlabel('')

        ax_violin_idealized.set_xlim([-0.4, 0.5])
        ax_violin_idealized.set_ylim([-2, 9])
        ax_violin_idealized.axis('off')

        # # Plot arrows and labels
        pos = 0.3
        quantile_fontsize = 5
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.95), '95%', fontsize=quantile_fontsize)
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.75) + 1.5, '75%', fontsize=quantile_fontsize)
        ax_violin_idealized.text(pos, example_high['example_variable'].mean() + 0.8, 'Mean', color='k', fontsize=quantile_fontsize, fontweight='bold')
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.50) - 0.4, '50%', color='grey', fontsize=quantile_fontsize, fontweight='bold')
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.25) - 1, '25%', fontsize=quantile_fontsize)
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.05) - 1.8, '5%', fontsize=quantile_fontsize)
        # ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.95) - 0.35, '95%', fontsize=quantile_fontsize)
        # ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.75) + 0.2, '75%', fontsize=quantile_fontsize)
        # ax_violin_idealized.text(pos, example_high['example_variable'].mean() - 0.3, 'Mean', color='k', fontsize=quantile_fontsize, fontweight='bold')
        # ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.50) - 0.9, '50%', color='grey', fontsize=quantile_fontsize, fontweight='bold')
        # ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.25) - 1.2, '25%', fontsize=quantile_fontsize)
        # ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.05) - 1.8, '5%', fontsize=quantile_fontsize)

        # ax_violin_idealized.annotate('', xy=(0.05, 0.1), xytext=(0.24, -1.25), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='k', mutation_scale=8), clip_on=False)
        # ax_violin_idealized.annotate('', xy=(0.1, 1.2), xytext=(0.23, 0), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='k', mutation_scale=8), clip_on=False)
        # ax_violin_idealized.annotate('', xy=(0.1, 3), xytext=(0.23, 4), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='k', mutation_scale=8), clip_on=False)
        # ax_violin_idealized.annotate('', xy=(0.1, 1.9), xytext=(0.24, 1.6), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='gray', mutation_scale=8), clip_on=False)
        # ax_violin_idealized.annotate('', xy=(0.1, 2.5), xytext=(0.24, 2.5), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='k', mutation_scale=8), clip_on=False)
        # ax_violin_idealized.annotate('', xy=(0.1, 6.5), xytext=(0.24, 6.5), arrowprops=dict(arrowstyle='-|>', lw=0.3, ls='-', color='k', mutation_scale=8), clip_on=False)

        ax_violin_idealized.annotate('', xy=(-0.4, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle='<|-|>', lw=0.3, ls='--', color='gray'), clip_on=False)
        ax_violin_idealized.plot([0.005, 0.005], [0.05, 0.22], transform=ax_violin_idealized.transAxes, color='gray', ls='--', lw=0.3)
        ax_violin_idealized.text(0.3, -0.09, 'Relative density', transform=ax_violin_idealized.transAxes, va='center', ha='center', fontsize=6, clip_on=False)

        # Define start and end points
        # Control points for the connector
        for percent in ['95','75', 'mean', '50', '25', '5']:
            # (start at label, first horizontal segment, bend downward, final straight segment to feature)
            if percent == '95':
                verts = [ (0.275, 6.75), (0.275, 6.75), (0.275, 6.75), (0.08, 6.75)]
            elif percent == '75':
                verts = [ (0.275, 5.0), (0.23, 5.0), (0.23, 3.3), (0.14, 3.3)]
            elif percent == 'mean':
                verts = [ (0.275, 3.4), (0.26, 3.4), (0.26, 2.45), (0.14, 2.45)]
            elif percent == '50':
                verts = [ (0.275, 1.9), (0.275, 1.9), (0.275, 1.9), (0.14, 1.9)]
            elif percent == '25':
                verts = [ (0.275, 0.4), (0.26, 0.4), (0.26, 0.9), (0.14, 0.9)]
            elif percent == '5':
                verts = [ (0.275, -1), (0.23, -1), (0.23, 0.205), (0.08, 0.205)]

            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
            path = Path(verts, codes)

            if percent == '50':
                color = 'grey'
            else:
                color = 'k'

            arrow = FancyArrowPatch(path=path, arrowstyle='-|>', mutation_scale=7, linewidth=0.3, color=color, clip_on=False)
            ax_violin_idealized.add_patch(arrow)

        logger.info(f"Saving plot of 'plot_annual_cycle_smb_within_significant_insignificance_amplitude'...")
        # base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        base_output_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/manuscript_figures'
        fig_name = 'Figure_B4.png'
        fig.savefig(f"{base_output_path}/{fig_name}", dpi=600)
    # %%
    def calculate_and_print_correlation_between_promax_rotated_pcs():
        """Calculate the spearman correlation between the principal components of the left and right variables in the promax rotation.

        """
        # %%
        # left_var_name = 'z0500'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'ci'
        # coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # # left_var_name = 'z0500'
        # # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # # right_var_name = 'smb'
        # # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        # left_var_name = 'ci'
        # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        # right_var_name = 'smb'
        # coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

        total_modes = 3
        rotation_modes = 3
        # ----------------------------------------------------
        # Load complex MCA scores
        # ----------------------------------------------------

        for rotation in ['varimax', 'promax']:
            for left_var_name, coarsen_level_left, right_var_name, coarsen_level_right in [
                ('ci', 0, 'smb', 4),
                ('z0500', 0, 'smb', 4),
                ('z0500', 0, 'ci', 0)]:
                print(f"Calculating correlation between PC of analytical SVD for \n{left_var_name} vs {right_var_name} with rotation: {rotation} ...")

                # Base config with all shared parameters
                base_cfg = ComplexMCAConfig(name='',  # will override per-call
                                            left_var_name=left_var_name,
                                            right_var_name=right_var_name,
                                            coarsen_level_left=coarsen_level_left,
                                            coarsen_level_right=coarsen_level_right,
                                            rotation=rotation,
                                            total_modes=total_modes,
                                            rotation_modes=rotation_modes)

                for component in ['left', 'right']:
                    print(f"Loading {component} scores for {left_var_name} vs {right_var_name} with rotation: {rotation} ...")
                    for mode in [1, 2, 3]:
                        score = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"{component}_scores"))
                        reconstructed_score = reconstruct_complex(score)
                        # print(f"{component.capitalize()} score shape for mode {mode}: {reconstructed_score.shape}")
                    complex_scores = []
                    for mode in [1, 2, 3]:
                        left_score = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"left_scores"))
                        score = reconstruct_complex(left_score)
                        complex_scores.append(score.values)

                    # Shape: time x modes
                    scores = np.column_stack(complex_scores)
                    # print("Complex score shape:")
                    # print(scores.shape)

                    # ----------------------------------------------------
                    # 1. Complex correlation
                    # ----------------------------------------------------

                    complex_corr = complex_correlation(scores)

                    # print("\nComplex correlation matrix:")
                    # print(complex_corr)

                    # ----------------------------------------------------
                    # 2. Magnitude of complex correlation
                    # ----------------------------------------------------

                    complex_corr_abs = np.abs(complex_corr)

                    print("\nMagnitude of complex correlation |r|:")
                    print(
                        pd.DataFrame(
                            complex_corr_abs,
                            index=[1, 2, 3],
                            columns=[1, 2, 3]
                        )
                    )


                # spearman_corr = df_pcs[[1, 2, 3]].corr(method="pearson")
                # print(f"Pearson correlation between left principal components: \n{spearman_corr}")

                # df_pcs = pd.DataFrame(columns=[1, 2, 3])
                # for mode in [1, 2, 3]:
                #     right_score = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"right_scores"))
                #     reconstruct_right_score = reconstruct_complex(right_score)
                #     df_pcs[mode] = reconstruct_right_score.real.values
                # spearman_corr = df_pcs[[1, 2, 3]].corr(method="pearson")
                # print(f"Pearson correlation between right principal components: \n{spearman_corr}")


        # %%
if __name__ == "__main__":
    # for left_var_name, coarsen_level_left, right_var_name, coarsen_level_right in [
    #     # ('ci', 0, 'smb', 4),]:
    #     # ('z0500', 0, 'smb', 4)]:
    #     ('z0500', 0, 'ci', 0)]:
    #     for mode in [2]: #[1, 2, 3]
    #         # for rotation in ['varimax', 'promax']:
    #         for rotation in ['varimax',]:
    #             ComplexVarimaxRotatedMCA.plot_analytical_svd(left_var_name=left_var_name,
    #                                                         right_var_name=right_var_name,
    #                                                         coarsen_level_left=coarsen_level_left,
    #                                                         coarsen_level_right=coarsen_level_right,
    #                                                         mode=mode,
    #                                                         total_modes=3,
    #                                                         rotation=rotation,
    #                                                         rotation_modes=3,
    #                                                         # figure_name='manuscript_figures/Figure_5') # ('ci', 0, 'smb', 4)
    #                                                         # figure_name='manuscript_figures/Figure_B5') # ('z0500', 0, 'ci', 0)
    #                                                         # figure_name='manuscript_figures/Figure_B6') # ('z0500', 0, 'smb', 4)
    #                                                         # figure_name='supplementary_figures/Figure_S10') # ('z0500', 0, 'smb', 4) mode=2
    #                                                         figure_name='supplementary_figures/Figure_S11') # ('z0500', 0, 'ci', 0) mode=2

    for mode in [2]: #[1, 2, 3]
        for rotation in ['varimax',]: #  'promax'
            ComplexVarimaxRotatedMCA.plot_principal_component_and_frequency_spectrum(mode=mode,
                                                                                 total_modes=3,
                                                                                 rotation=rotation,
                                                                                 rotation_modes=3,
                                                                                #  figure_name='manuscript_figures/Figure_B3') # mode=1
                                                                                 figure_name='supplementary_figures/Figure_S12') # mode=2

    # ComplexVarimaxRotatedMCA.plot_covariance_of_modes()


    # for left_var_name, coarsen_level_left, right_var_name, coarsen_level_right in [
    #     ('ci', 0, 'smb', 4),
    #     ('z0500', 0, 'smb', 4),
    #     ('z0500', 0, 'ci', 0)]:
    #     for mode in [1,]: #[1, 2, 3]:
    #         for rotation in ['varimax',]: #['varimax', 'promax']:
    #             print(f"\nRunning function calculate_and_print_phase_shift_ci_smb_z500 ...")
    #             ComplexVarimaxRotatedMCA.calculate_and_print_phase_shift_ci_smb_z500(left_var_name=left_var_name,
    #                                                                                 right_var_name=right_var_name,
    #                                                                                 coarsen_level_left=coarsen_level_left,
    #                                                                                 coarsen_level_right=coarsen_level_right,
    #                                                                                 mode=mode,
    #                                                                                 total_modes=3,
    #                                                                                 rotation=rotation,
    #                                                                                 rotation_modes=3,
    #                                                                                 periodicity=153)














# I am calculating the time lag difference between ci, smb, and z0500 using the complex MCA output. The results are as follows:
# The result shows that ci max is leading the smb max by 30.27 days,
# z0500 min is leading ci max by 106.33 days, and
# z0500 min is leading smb max by 78.27 days.
# However, the difference between smb - z0500 and ci - z0500 is -28.06 days. I was expecting this to be +28.06 days,
# whicj probably indicate that ci max is lagging behind the smb max by 28.06 days when compared to their respective lags with z0500.
# However, it doesnot match with the previous result of ci max leading smb max by 30.27 days.

# Does the result of -28 days indicate that the smb max is lagging behind the ci max by 28.06 days
# when compared to their respective lags with z0500. or the other way around


# left_var_name = 'ci'
# coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
# right_var_name = 'smb'
# coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

# left_var_name = 'z0500'
# coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
# right_var_name = 'ci'
# coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

# left_var_name = 'z0500'
# coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
# right_var_name = 'smb'
# coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

# print(f"\nCalculating {left_var_name} and {right_var_name} ...")
# print(f"{left_var_name}: {left_time_lag_days_on_point.values} days")
# print(f"{right_var_name}: {right_time_lag_days_on_point.values} days")
# print(f"Time lag difference {right_var_name} and {left_var_name}: {right_time_lag_days_on_point.values - left_time_lag_days_on_point.values} days")

# Calculating ci and smb ...
# ci: 14.348677131715412 days
# smb: 44.62331023144897 days
# Time lag difference smb and ci: 30.274633099733556 days

# The results:
#     Calculating z0500 and ci ...
# z0500: -70.67168610503782 days
# ci: 35.65824944970547 days
# Time lag difference ci and z0500: 106.32993555474329 days

# Calculating z0500 and smb ...
# z0500: -60.24478761572227 days
# smb: 18.023633324436204 days
# Time lag difference smb and z0500: 78.26842094015848 days


# smb and z0500: 78.26842094015848 days
# ci and z0500: 106.32993555474329 days
# smb - z0500 - (ci - z0500) = 78.26842094015848 - 106.32993555474329 = -28.06151461458481 days


# def calculate_phase_shift4_ci_smb_relation():
#     # %%
#     left_var_name = 't2m'
#     coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
#     right_var_name = 'smb'
#     coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

#     # left_var_name = 't2m'
#     # coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
#     # right_var_name = 'ci'
#     # coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

#     left_var_name = 'ci'
#     coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
#     right_var_name = 'smb'
#     coarsen_level_right = 0 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.

#     mode = 1
#     rotation = 'varimax'
#     # Find lag days
#     # periodicity = 153 # days/months; dominant periodicity of mode 1
#     periodicity = 12 # days/months; dominant periodicity of mode 1
#     main_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/{left_var_name}_vs_{right_var_name}")

#     # Base config with all shared parameters
#     base_cfg = ComplexMCAConfig(name='',  # will override per-call
#                                 left_var_name=left_var_name,
#                                 right_var_name=right_var_name,
#                                 coarsen_level_left=coarsen_level_left,
#                                 coarsen_level_right=coarsen_level_right,
#                                 rotation=rotation)
#     # Variable at location
#     # Calculate time lags from phase patterns --------------------------------
#     # Left pattern
#     # Amplitude
#     left_component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"left_component_amplitudes"))
#     left_component_amplitude = left_component_amplitude/left_component_amplitude.max()
#     left_component_amplitude_sig = left_component_amplitude.where(left_component_amplitude > 0.25, np.nan)

#     # Phase
#     left_component_phase = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"left_component_phases"))
#     left_component_phase_sig = left_component_phase.where(left_component_amplitude_sig)
#     left_time_lag_days = (periodicity * left_component_phase_sig) / (2 * np.pi)

#     # Right pattern
#     # Amplitude
#     right_component_amplitude = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"right_component_amplitudes"))
#     right_component_amplitude = right_component_amplitude/right_component_amplitude.max()
#     right_component_amplitude_sig = right_component_amplitude.where(right_component_amplitude > 0.25, np.nan)

#     # Phase
#     right_component_phase = load_complex_mca_output(mode=mode, config=base_cfg.with_name(f"right_component_phases"))
#     right_component_phase_sig = right_component_phase.where(right_component_amplitude_sig)
#     right_time_lag_days = (periodicity * right_component_phase_sig) / (2 * np.pi)


#     if right_var_name == 'smb':
#         # Change in moisture transfered from ocean to ZAC
#         station = 'FRE2 (promice type, tripod)' #ZAC
#         # Interpolate left to Freya Location
#         station_x, station_y = get_station_location(station=station, variable='t2m')
#         left_time_lag_days_on_point = left_time_lag_days.sel(rlon=station_x, rlat=station_y, method='nearest')
#         # Interpolate right to Freya Location
#         station_x, station_y = get_station_location(station=station, variable='smb')
#         right_time_lag_days_on_point = right_time_lag_days.sel(x=station_x, y=station_y, method='nearest')
#     if right_var_name == 'ci':
#         mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
#         ci_buffer = xr.open_dataarray(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")
#         ci_buffer = ci_buffer.sel(radius=100)
#         # Interpolate left to 100 km buffer of Sea Ice Location
#         left_time_lag_days_on_point = left_time_lag_days.where(ci_buffer == 1)
#         left_time_lag_days_on_point = left_time_lag_days_on_point.mean(dim=['rlat', 'rlon'])
#         # Interpolate right to 100 km buffer of Sea Ice Location
#         right_time_lag_days_on_point = right_time_lag_days.where(ci_buffer == 1)
#         right_time_lag_days_on_point = right_time_lag_days_on_point.mean(dim=['rlat', 'rlon'])

#     if right_var_name == 'ci':
#         location = '100 km buffer of Sea Ice'
#     if right_var_name == 'smb':
#         location = 'FRE'
#     print(f"{left_var_name} Mode 1 Time Lag at {location} (days): {left_time_lag_days_on_point.values} days")
#     print(f"{right_var_name} Mode 1 Time Lag at {location} (days): {right_time_lag_days_on_point.values} days")

#     print(f"Time lag difference {right_var_name} and {left_var_name} at {location} (days): {right_time_lag_days_on_point.values - left_time_lag_days_on_point.values} days")
