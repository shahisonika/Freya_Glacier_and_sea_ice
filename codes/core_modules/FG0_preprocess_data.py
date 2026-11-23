#!/usr/bin/env python3
"""
Freya Glacier Analysis Level 0 (FG0) - Preprocess Data.

Data preprocessing pipeline for Maximum Covariance Analysis (MCA) using Singular Value Decomposition method.

This script is designed to preprocess the data for the MCA using SVD method to investigate
the relationship between the Greenland Ice Sheet (GrIS) surface mass balance (smb), sea ice (ci), and
geopotential height at 500 hPa (z500).

This script consists of two classes


Author: Sonika Shahi
Date: 4 August 2025

Environment: xeofs_env (conda activate xeofs_env on walrus server)
"""
# %%
import os
import sys
import glob
from networkx import radius
import xarray as xr
from xarray.coding.times import CFTimedeltaCoder
import xrft
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from scipy.spatial import cKDTree
from typing import Optional

# The following import is used for testing the codes.
from pyproj import Transformer, CRS
import matplotlib.pyplot as plt
import matplotlib as mpl

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Import modules from different directory
# Reference: https://www.geeksforgeeks.org/python-import-module-from-different-directory/
module_path = "/nas/home/sos/third_paper_work/codes/add_modules/"
if os.path.exists(module_path):
    sys.path.append(module_path)
    # from decorator_codes import logger
    from kdimensional_tree import kdtree_fast, kdtree_fast_batch
    import derive_and_correct_aws_variables as dcaws
    # from stats_calculation import scalar_stats
    from statistics_calculation import scalar_statistics as ss
    from statistics_calculation import vector_statistics as vs
    from figure_properties import FigureProperties as fp
    from racmo5km_rotated_grid_conversions import rotated_grid_transform
else:
    raise ImportError(f"Module path '{module_path}' does not exist. Ensure 'decorator_codes.py' is in the correct directory.")

# -------------------------
# Utility helpers
# -------------------------
os.chdir('/nas/home/sos/third_paper_work/codes/freyagl_analysis')
from mca_data_loader import variable_stats, load_coarsed_mjjas_data
from mca_utils import coarsen_matrix_3d, coarsen_coordinates_1d, coarsen_matrix_2d

# Reload a module
import importlib
# importlib.reload(dcaws)
# importlib.reload(statistics_calculation)

# def variable_stats(var_name: str) -> str:
#     """Return aggregation stat for a variable.

#     """
#     # mapping = {'smb': 'SUM', 'ci': 'MEDIAN', 'z0500': 'MEAN'}
#     # mapping = {'smb': 'MEDIAN', 'ci': 'MEDIAN', 'z0500': 'MEAN'}
#     mapping = {'smb': 'SUM', 'ci': 'MEAN', 'z0500': 'MEAN'}

#     if var_name not in mapping:
#         raise ValueError(f'Unknown variable name: {var_name}')
#     return mapping[var_name]

def _clean_fillvalues(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy()
    for v in ds.variables:
        for key in ["_FillValue", "missing_value"]:
            ds[v].attrs.pop(key, None)
            ds[v].encoding.pop(key, None)
    return ds

def _detrend_dim(da, dim='time'):
    """Detrend a DataArray along a specified dimension using linear regression."""

    p = da.polyfit(dim=dim, deg=1)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def _get_aggregation(var_name):
    """Return aggregation method for a variable.

    """
    mapping = {'smb': 'sum', 'ci': 'median', 'z0500': 'mean'}
    if var_name not in mapping:
        raise ValueError(f'Unknown variable name: {var_name}')
    return mapping[var_name]

def _get_greenland_mask_latitude_longitude(variable: str):
    if variable == 'smb':
        # racmo1km_lat_lon_coarsened_4 = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level4.nc')
        # racmo1km_latitude_coarsened_4  = racmo1km_lat_lon_coarsened_4['LAT']
        # racmo1km_longitude_coarsened_4  = racmo1km_lat_lon_coarsened_4['LON']

        gr_masks_1km = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
        land_sea_mask = gr_masks_1km['Promicemask']
        land_sea_mask.values[land_sea_mask.values > 1] = 1

        # Latitude and longitude
        racmo_latitude = gr_masks_1km['LAT']
        racmo_longitude = gr_masks_1km['LON']

        x = racmo_longitude.values
        y = racmo_latitude.values

    elif (variable == 'sif') | (variable == 'z0500'):
        # Coarsen Greenland mask and merge it with Svalbard and Iceland
        gr_masks_5km = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        land_sea_mask = gr_masks_5km['LSM_GR']
        # land_sea_mask = land_sea_mask.where(land_sea_mask != 0) # Ensure binary mask (1 for land, 0 for sea)
        # topo = gr_masks_5km['Topography']
        # topo_c = topo.copy()
        # # Mask Svalbard and Iceland
        # topo_c.values[(topo_c.values > 0) & (land_sea_mask.values == 0)] = 0
        # # Mask Greenland to seperate Iceland and Svalbard
        # topo.values[(topo.values == topo_c.values)] = 0
        # # Add Iceland and Svalbard in land sea mask
        # land_sea_mask.values[(topo.values > 100)] = 1

        # Latitude and longitude
        racmo_latitude = gr_masks_5km['lat']
        racmo_longitude = gr_masks_5km['lon']

    # if variable == 'sif':
    #     land_sea_mask = xr.where((land_sea_mask == 0), np.nan, land_sea_mask) # to avoid hiding the colors below
    #     colors_landseamask = [(255, 255, 255), (169, 169, 169)]
    # else:
        # colors_landseamask = [(169, 169, 169),(0, 0, 0)]
    colors_landseamask = [(0, 0, 0), (169, 169, 169)]
    cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
    bounds_landseamask = np.array([0, 0.5, 1.5])
    norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

    return land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask

def _get_latlon_meshgrid(variable: str, coarsen_level: Optional[int] = None):
    if variable == 'smb':
        if coarsen_level == 4:
            da = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level4.nc')
        else:
            da = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')
        x = da.coords['x'].values
        y = da.coords['y'].values
    elif (variable == 'sif') | (variable == 'z0500'):
        da = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        x = da.coords['rlon'].values
        y = da.coords['rlat'].values
    return np.meshgrid(x, y)

def _get_promice_mask_params(variable):
    if variable == 'sif':
        # Read the land-sea mask and Promice mask at 5 km resolution
        mask_5km_file = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        # Promice mask
        promice_mask_ramo_5km = mask_5km_file['Promicemask']
        # promice_mask_ramo_5km.values[(promice_mask_ramo_5km.values == 1) | (promice_mask_ramo_5km.values == 2)] = 3
        # promice_mask_ramo_5km = promice_mask_ramo_5km.where(promice_mask_ramo_5km != 0)

        land_sea_mask, _, _, _, _, _, _ = _get_greenland_mask_latitude_longitude(variable)
        # Remove iceland and svalbard from promice mask by applying land sea mask
        promice_mask_ramo_5km = promice_mask_ramo_5km.where(land_sea_mask==1)

        # colors_promice_mask = [(255, 255, 255), (65, 105, 225), (65, 105, 225), (135, 206, 235)]
        colors_promice_mask = [(244, 164, 96), (65, 105, 225), (65, 105, 225), (135, 206, 235)]
        cmap_promice_mask = fp.make_cmap(colors_promice_mask, bit=True)
        bounds_promice_mask = np.array([0, 1.5, 2.5, 3.5, 4.5])
        norm_promice_mask = mpl.colors.BoundaryNorm(bounds_promice_mask, cmap_promice_mask.N)

    return promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask

def _process_year(year, da, var_name, time_period):
    """Function for processing a single year.

    """
    aggregation = _get_aggregation(var_name)

    if time_period == 'JJAS':
        start = f"{year}-06-01"
        end   = f"{year}-09-27"
    elif time_period == 'JJ':
        start = f"{year}-06-01"
        end   = f"{year}-07-26"
    elif time_period == 'JA':
        start = f"{year}-07-01"
        end   = f"{year}-08-25"
    elif time_period == 'AS':
        start = f"{year}-08-01"
        end   = f"{year}-09-25"
    da_sub = da.sel(time=slice(start, end))

    if time_period == 'JJAS':
        expected_days = 119  # 17 weeks * 7 days
        if da_sub.sizes['time'] != expected_days:
            raise ValueError(f"{year} does not have exactly {expected_days} days between Jun 1 and Sep 27!")
    elif time_period in ['JJ', 'JA', 'AS']:
        expected_days = 56  # 8 weeks * 7 days
        if da_sub.sizes['time'] != expected_days:
            raise ValueError(f"{year} does not have exactly {expected_days} days for {time_period} period!")

    # Reshape into (17, 7, rlat, rlon)
    if aggregation == 'sum' and var_name == 'smb':
        if time_period == 'JJAS':
            # For smb, sum over each 7-day period
            da_reshaped = da_sub.data.reshape(17, 7, da_sub.sizes['y'], da_sub.sizes['x'])
        elif time_period in ['JJ', 'JA', 'AS']:
            da_reshaped = da_sub.data.reshape(8, 7, da_sub.sizes['y'], da_sub.sizes['x'])
    elif aggregation in ['mean', 'median'] and var_name in ['ci', 'z0500']:
        if time_period == 'JJAS':
            # For ci and z0500, mean or median over each 7-day period
            da_reshaped = da_sub.data.reshape(17, 7, da_sub.sizes['rlat'], da_sub.sizes['rlon'])
        elif time_period in ['JJ', 'JA', 'AS']:
            da_reshaped = da_sub.data.reshape(8, 7, da_sub.sizes['rlat'], da_sub.sizes['rlon'])

    # Median along 7-day axis
    if aggregation == 'mean':
        da_median = np.mean(da_reshaped, axis=1)
    elif aggregation == 'median':
        da_median = np.median(da_reshaped, axis=1)
    elif aggregation == 'sum':
        da_median = np.sum(da_reshaped, axis=1)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Wrap back into xarray
    if var_name == 'smb':
        if time_period == 'JJAS':
            weeks = np.arange(1, 18)  # 17 weeks
        elif time_period in ['JJ', 'JA', 'AS']:
            weeks = np.arange(1, 9)   # 8 weeks
        return xr.DataArray(da_median, dims=('week', 'y', 'x'), coords={'week': weeks, 'y': da_sub.y, 'x': da_sub.x, 'year': year})
    else:
        if time_period == 'JJAS':
            weeks = np.arange(1, 18)  # 17 weeks
        elif time_period in ['JJ', 'JA', 'AS']:
            weeks = np.arange(1, 9)   # 8 weeks
        return xr.DataArray(da_median, dims=('week', 'rlat', 'rlon'), coords={'week': weeks, 'rlat': da_sub.rlat, 'rlon': da_sub.rlon, 'year': year})

def _read_freya_glacier_aws_variables():
    """Read Freya AWS variables for quality check

    """
    # Read AWS data files
    freya_aws_path = '/nas/data/raw/igr/freya_2016_2020/'
    # Collect all .dat files
    allfiles = glob.glob(os.path.join(freya_aws_path, '*.dat'))

    frames = []
    for file_id in tqdm(allfiles, desc='Reading AWS files'):
        # Read each file (second row as header, whitespace delimiter)
        raw_data = pd.read_csv(file_id, sep=r'\s+', header=1, encoding='ISO-8859-1', na_values='NaN')
        # Combine Date + Time into single datetime index
        raw_data['Date'] = pd.to_datetime(raw_data['Date'] + ' ' + raw_data['Time[UTC]'])
        raw_data.set_index('Date', inplace=True)
        raw_data.drop(columns='Time[UTC]', inplace=True)

        # Ensure consistent sampling frequency
        freq = raw_data.index[1] - raw_data.index[0]
        raw_data = raw_data.resample(freq).mean()

        frames.append(raw_data)

    # Merge all AWS files into a single DataFrame
    data_freya_aws = pd.concat(frames).sort_index()

    # Resample again to enforce uniform frequency across entire dataset
    freq = data_freya_aws.index[1] - data_freya_aws.index[0]
    data_freya_aws = data_freya_aws.resample(freq).mean()

    # Add explicit datetime column for convenience
    data_freya_aws['dates'] = data_freya_aws.index

    # -------------------------------------------------------------------
    # 2. Known data gaps (metadata-based corrections) (see AWS Freya metadata PDF)
    data_freya_aws.loc['2018-02-15':'2018-04-24'] = np.nan   # AWS covered in snow
    data_freya_aws.loc['2022-04-24':] = np.nan               # Change of radiation & humidity sensors
    return data_freya_aws

def _subset_nearest(ds, rlon_bounds=None, rlat_bounds=None):
    """
    Subset a dataset or DataArray using rlon and rlat bounds,
    snapping to the nearest grid points to avoid floating-point issues.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        The input dataset or DataArray.
    rlon_bounds : tuple or list, optional
        (min, max) bounds for rlon coordinate.
    rlat_bounds : tuple or list, optional
        (min, max) bounds for rlat coordinate.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Subsetted dataset.
    """
    ds_subset = ds.copy(deep=True)

    if rlon_bounds is not None:
        rlon_start = float(ds.rlon.sel(rlon=rlon_bounds[0], method='nearest'))
        rlon_end   = float(ds.rlon.sel(rlon=rlon_bounds[1], method='nearest'))
        ds_subset = ds_subset.sel(rlon=slice(rlon_start, rlon_end))

    if rlat_bounds is not None:
        rlat_start = float(ds.rlat.sel(rlat=rlat_bounds[0], method='nearest'))
        rlat_end   = float(ds.rlat.sel(rlat=rlat_bounds[1], method='nearest'))
        ds_subset = ds_subset.sel(rlat=slice(rlat_start, rlat_end))

    return ds_subset
# %%
class SVDDataPreprocessor:
    """Class for preprocessing data for SVD analysis.

    This class contains methods to preprocess the data for SVD analysis, including calculating monthly
    and seasonal variables, detrending, and slicing the data.
    """

    @staticmethod
    def save_seperate_month_racmo(var_name):
        """ Read the NetCDF files, extract monthly data, and save one file for each month.

        """
        # %%
        config = {
            'ci': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/ci/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'ci',
                'pattern': 'ci.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'z0500': {
                'input_path': '/tempdata/knmi/racmo/pressure_levels/500hPa/z/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'z0500',
                'pattern': 'z0500.????.*_ERA5_3h_*.nc',
                'sel': {'pressure': 50000},
            },
            'smb': {
                'input_path': '/tempdata/knmi/racmo/variables_1km/smb/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'smb_rec',
                'pattern': 'smb_rec.????.*_ERA5_3h_*.nc',
                'sel': None,
            },
            'psurf': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/psurf/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'psurf',
                'pattern': 'psurf.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            't2m': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/t2m/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 't2m',
                'pattern': 't2m.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'tskin': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/tskin/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'tskin',
                'pattern': 'tskin.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'latf': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/latf/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'latf',
                'pattern': 'latf.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'q2m': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/q2m/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'q2m',
                'pattern': 'q2m.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'senf': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/senf/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'senf',
                'pattern': 'senf.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
            'alb': {
                'input_path': '/tempdata/knmi/racmo/variables_5km/alb/',
                'output_path': '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/',
                'dataset_var': 'alb',
                'pattern': 'alb.????.*_ERA5_3h_*.nc',
                'sel': {'height': 0},
            },
        }

        if var_name not in config:
            raise ValueError(f"Unknown variable: {var_name}")

        cfg = config[var_name]

        files = sorted(glob.glob(cfg['input_path'] + cfg['pattern']))

        for month in range(1, 13):

            print(f"Saving month {month} {var_name} data.")

            ds_list = []

            for f in files:

                ds = xr.open_dataset(f)[cfg['dataset_var']]

                if cfg['sel'] is not None:
                    ds = ds.sel(**cfg['sel'])

                monthly_data = ds.sel(time=ds['time.month'] == month)

                if monthly_data.sizes['time'] > 0:
                    ds_list.append(monthly_data)

            if not ds_list:
                print(f"No data found for month {month}.")
                continue

            ds_month = xr.concat(ds_list, dim='time')
            ds_month = ds_month.sortby('time')

            outfile = (
                cfg['output_path'] + f"{var_name}_{month}_trended.nc"
            )

            ds_month.to_netcdf(outfile, mode='w')

        print(f"Monthly {var_name} data saved.")
    # %%

    @staticmethod
    def calculate_monthly_racmo(var_name):
        """Calculate monthly racmo variables.

        Input variables in daily scale is calculated from FG0_rearrange_daily_into_monthly.py, function save_monthly_racmo(var_name).
        """
        # %%
        var_name = 'ci' # Test SMB 'MEAN' and 'MEDIAN' for testing
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
        write_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"

        if var_name == 'smb':
            read_file_path = f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile/"
        elif var_name == 'ci':
            read_file_path = f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"
        elif var_name == 'z0500':
            read_file_path = f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"

        # months = [5, 6, 7, 8, 9]
        # months = [6, 7, 8]
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        for month in months:
            da_month = xr.open_dataarray(f"{read_file_path}{var_name}_{month}_trended.nc")

            if var_name == 'smb':
                # da_monthly_mean = da_month.groupby('time.year').sum(dim='time')
                # stat = 'SUM'
                # -------- For testing ------------------------
                # da_monthly_mean = da_month.groupby('time.year').mean(dim='time') # Save 'MEAN' for testing
                # stat = 'MEAN'
                da_monthly_mean = da_month.groupby('time.year').median(dim='time') # Save 'MEDIAN' for testing
                stat = 'MEDIAN'
                # -------- For testing ------------------------
            elif var_name == 'ci':
                da_monthly_mean = da_month.groupby('time.year').median(dim='time')
                # stat = 'MEDIAN'
                stat = 'MEAN'
            elif var_name == 'z0500':
                da_monthly_mean = da_month.groupby('time.year').mean(dim='time')
                stat = 'MEAN'

            # Save the monthly mean, sum, or median
            print(f"Saving {var_name} monthly {stat} for month {month}")
            da_monthly_mean.to_netcdf(path=f"{write_file_path}{var_name}_{month}_{stat}_trend.nc", mode='w')
        # %%

    @staticmethod
    def slice_monthly_racmo(var_name, start_year, end_year):
        """Slice monthly racmo variables.

        """
        # %%
        var_name = 'ci' # Test ci for MEAN and MEDIAN
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        read_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/'

        # -------- For testing ------------------------
        # stat = 'MEAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        # stat = 'MEDIAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        start_year = 1990
        end_year = 2019
        # -------- For testing ------------------------

        # months = [5, 6, 7, 8, 9]
        months = [6, 7, 8]
        for month in months:
            da_month = xr.open_dataarray(f"{read_file_path}{var_name}/{var_name}_{month}_{stat}_trend.nc")

            # Extract the time slice
            da_month = da_month.sel(year=slice(start_year,  end_year))
            print(f"Slicing {var_name} for period {start_year}-{end_year}")
            # Save the sliced data
            da_month.to_netcdf(path=f"{read_file_path}{var_name}/{var_name}_{month}_{stat}_{start_year}to{end_year}_trend.nc", mode='w')
        # %%

    @staticmethod
    def calculate_monthly_detrended_racmo(var_name, start_year, end_year):
        """Calculate monthly detrended racmo variables.

        """
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        # %%
        # -------- For testing ------------------------
        var_name = 'smb' # Test SMB 'MEAN' and 'MEDIAN' for testing
        stat = 'MEAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        stat = 'MEDIAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        start_year = 1990
        end_year = 2019
        # -------- For testing ------------------------

        read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"

        # months = [5, 6, 7, 8, 9]
        months = [6, 7, 8]
        for month in months:
            da_month = xr.open_dataarray(f"{read_file_path}{var_name}_{month}_{stat}_{start_year}to{end_year}_trend.nc")

            # Extract the time slice
            da_month = da_month.sel(year=slice(start_year, end_year))

            # # Calculate anomaly
            # da_monthly_mean = da_month.mean(dim='year')
            # da_monthly_anomaly = da_month - da_monthly_mean

            # Detrend the anomalies
            print(f"Detrending {var_name} for period {start_year}-{end_year} and month {month}")
            da_month_detrend = xrft.detrend(da_month, dim='year', detrend_type='linear')
            # Save the detrended data
            da_month_detrend.to_netcdf(path=f"{read_file_path}{var_name}_{month}_{stat}_{start_year}to{end_year}_detrend.nc", mode='w')

            # %%
            # # Evaluate the detrended data
            # # Uncomment the following lines to visualize the data
            # import matplotlib.pyplot as plt
            # import pymannkendall as mk

            # # Define the Mann-Kendall test function
            # def mk_test_trend(df):
            #     # df = df.loc[:,col_name].copy(deep=True)
            #     result = mk.original_test(df)
            #     X = range(1,len(df)+1)
            #     prediction = result.slope * X + result.intercept
            #     return result, prediction

            # var_name = 'ci'/
            # month = 5
            # stat = 'MEDIAN'
            # start_year = 1990
            # end_year = 2019
            # read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/"
            # da_month = xr.open_dataarray(f"{read_file_path}{var_name}/{var_name}_{month}_{stat}_{start_year}to{end_year}_trend.nc")
            # data_detrend_xrft = xr.open_dataarray(f"{read_file_path}{var_name}/{var_name}_{month}_{stat}_{start_year}to{end_year}_detrend.nc")

            # x= 304; y = 50
            # data_detrend_xrft_point = data_detrend_xrft[:, x, y]

            # # Perform the Mann-Kendall test
            # result, da_month_trend = mk_test_trend(da_month[:, x, y].values)
            # # if result.p <= 0.1:
            # data_detrend_mk = da_month[:, x, y].values - da_month_trend

            # fig, ax = plt.subplots(1)
            # da_month[:, x, y].plot(ax=ax, color='black', label='Data')
            # ax.plot(da_month.coords['year'].values, da_month_trend, color='green', label='Trend Line')
            # # if result.p <= 0.1:
            # ax.plot(da_month.coords['year'].values, data_detrend_mk, color='blue', label='Detrended MK test')
            # data_detrend_xrft_point.plot(ax=ax, color='red', label='Detrended xrft')
            # fig.legend()

    @staticmethod
    def calculate_seasonal_racmo_with_single_dates(var_name, start_year, end_year, season_type):
        """Calculate seasonal racmo variables. (This function is replaced by calc_seasonal_racmo_with_single_dates_revised)

        Merge monthly data for June, July, and August to create a seasonal (JJA) dataset with 30 years of data.

        30 years (1990-2019) is used for MCA using SVD analysis.

        """
        # %%
        print('This function is replaced by calc_seasonal_racmo_with_single_dates_revised')
        # # -------- For testing ------------------------
        start_year = 1990
        end_year = 2019
        season_type = 'JJA'

        # var_name = 'ci' # Test SMB 'MEAN' and 'MEDIAN' for testing
        var_name = 'smb' # Test SMB 'MEAN' and 'MEDIAN' for testing
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        # # stat = 'MEAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        # stat = 'MEDIAN' # Test SMB 'MEAN' and 'MEDIAN' for testing

        # # -------- For testing ------------------------

        read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # Read monthly data for June, July, and August
        # June
        da_month_jun = xr.open_dataset(f"{read_file_path}{var_name}_6_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_jun = da_month_jun.rename_vars({list(da_month_jun.data_vars)[0]: 'jun'})
        # July
        da_month_jul = xr.open_dataset(f"{read_file_path}{var_name}_7_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_jul = da_month_jul.rename_vars({list(da_month_jul.data_vars)[0]: 'jul'})
        # August
        da_month_aug = xr.open_dataset(f"{read_file_path}{var_name}_8_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_aug = da_month_aug.rename_vars({list(da_month_aug.data_vars)[0]: 'aug'})

        if season_type == 'MJJAS':
            # May
            da_month_may = xr.open_dataset(f"{read_file_path}{var_name}_5_{stat}_{start_year}to{end_year}_trend.nc")
            da_month_may = da_month_may.rename_vars({list(da_month_may.data_vars)[0]: 'may'})

            # September
            da_month_sep = xr.open_dataset(f"{read_file_path}{var_name}_9_{stat}_{start_year}to{end_year}_trend.nc")
            da_month_sep = da_month_sep.rename_vars({list(da_month_sep.data_vars)[0]: 'sep'})

        if season_type == 'JJA':
            da_seasons = xr.merge([da_month_jun, da_month_jul, da_month_aug])
        elif season_type == 'MJJAS':
            da_seasons = xr.merge([da_month_may, da_month_jun, da_month_jul, da_month_aug, da_month_sep])

        # Merge the monthly data into a seasonal dataset
        if var_name == 'smb':
            # da_seasonal_mean = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).median('new')
            da_seasonal_mean = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).sum('new')
        elif var_name == 'z0500':
            da_seasonal_mean = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).mean('new')
        elif var_name == 'ci':
            # da_seasonal_mean = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).median('new')
            da_seasonal_mean = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).mean('new')

        da_seasonal_mean = da_seasonal_mean.rename(var_name)

        # Save the seasonal dataset with 30 years of data
        print(f"Saving {var_name} seasonal {season_type} {stat} for period {start_year}-{end_year}")
        # time_slice = da_seasonal_mean.coords['year'].shape[0]
        da_seasonal_mean.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_trend_single_dates.nc", mode='w')

        # %%
        # # Evaluate the difference between JJA and MJJAS  data
        # # Uncomment the following lines to visualize the data
        # import matplotlib.pyplot as plt
        # # var_name = 'ci'; stat = 'MEDIAN'; x = np.int64(304) + 1; y = np.int64(50)
        # # var_name = 'smb'; stat = 'SUM'; x = np.int64(1496 / 2); y = np.int64(2700 / 2)
        # var_name = 'z0500'; stat = 'MEAN'; x = np.int64(566 / 2); y = np.int64(438 / 2)
        # start_year = 1990
        # end_year = 2019
        # read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # da_season_jja = xr.open_dataarray(f"{read_file_path}{var_name}_JJA_{stat}_{start_year}to{end_year}_trend_single_dates.nc")
        # da_season_mjjas = xr.open_dataarray(f"{read_file_path}{var_name}_MJJAS_{stat}_{start_year}to{end_year}_trend_single_dates.nc")

        # fig, ax = plt.subplots(1)
        # da_season_jja[:, x, y].plot(ax=ax, color='black', label='Data JJA')
        # da_season_mjjas[:, x, y].plot(ax=ax, color='red', label='Data MJJAS')
        # fig.legend()
        # %%

    @staticmethod
    def calculate_seasonal_racmo_with_multiple_dates(var_name, start_year, end_year, season_type):
        """Calculate seasonal racmo variables with 30 years * multiple months of data.

        Merge monthly data for June, July, and August to create a seasonal (JJA) dataset with 30 years * multiple
        For JJA: 90 dates (30 years * 3 months)
        For MJJAS: 150 dates (30 years * 5 months)

        90 or 150 dates is used for MCA using SVD analysis.

        """
        # %%
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # Read monthly data for June, July, and August
        # Open and rename each month's dataset
        da_month_jun = xr.open_dataset(f"{read_file_path}{var_name}_6_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_jun = da_month_jun.rename_vars({list(da_month_jun.data_vars)[0]: 'jun'})

        da_month_jul = xr.open_dataset(f"{read_file_path}{var_name}_7_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_jul = da_month_jul.rename_vars({list(da_month_jul.data_vars)[0]: 'jul'})

        da_month_aug = xr.open_dataset(f"{read_file_path}{var_name}_8_{stat}_{start_year}to{end_year}_trend.nc")
        da_month_aug = da_month_aug.rename_vars({list(da_month_aug.data_vars)[0]: 'aug'})

        if season_type == 'MJJAS':
            # May
            da_month_may = xr.open_dataset(f"{read_file_path}{var_name}_5_{stat}_{start_year}to{end_year}_trend.nc")
            da_month_may = da_month_may.rename_vars({list(da_month_may.data_vars)[0]: 'may'})

            # September
            da_month_sep = xr.open_dataset(f"{read_file_path}{var_name}_9_{stat}_{start_year}to{end_year}_trend.nc")
            da_month_sep = da_month_sep.rename_vars({list(da_month_sep.data_vars)[0]: 'sep'})

        # Extract the DataArrays
        da_jun = da_month_jun['jun']
        da_jul = da_month_jul['jul']
        da_aug = da_month_aug['aug']

        if season_type == 'MJJAS':
            da_may = da_month_may['may']
            da_sep = da_month_sep['sep']

        # Stack all 3 months along a new time axis
        # First, expand dims to create a 'month' axis, then concatenate
        da_jun = da_jun.expand_dims(dim={'month': [6]})
        da_jul = da_jul.expand_dims(dim={'month': [7]})
        da_aug = da_aug.expand_dims(dim={'month': [8]})
        if season_type == 'MJJAS':
            da_may = da_may.expand_dims(dim={'month': [5]})
            da_sep = da_sep.expand_dims(dim={'month': [9]})

        # Concatenate the DataArrays along the 'month' dimension
        if season_type == 'JJA':
            da_all = xr.concat([da_jun, da_jul, da_aug], dim='month')
        elif season_type == 'MJJAS':
            # Concatenate over month including May and September
            da_all = xr.concat([da_may, da_jun, da_jul, da_aug, da_sep], dim='month')

        # Now reshape from (year=30, month=3, rlat, rlon) to (time=90, rlat, rlon)
        da_reshaped = da_all.stack(time=('year', 'month'))

        # Optionally sort time
        da_reshaped = da_reshaped.sortby('time')

        # You now have a DataArray with shape: (time=90, rlat, rlon)
        years = da_all.year.values
        months = da_all.month.values

        # Create a time variable with the correct format
        if season_type == 'JJA':
            # Broadcast into full list of 90 dates
            time_var = [f"{y}-{m:02d}" for y in years for m in [6, 7, 8]]
        elif season_type == 'MJJAS':
            # Broadcast into full list of 90 dates including May and September
            time_var = [f"{y}-{m:02d}" for y in years for m in [5, 6, 7, 8, 9]]

        # Remove old multiindex and the original 'year' and 'month' variables
        da_reshaped = da_reshaped.reset_index('time')
        da_reshaped = da_reshaped.drop_vars(['year', 'month'])

        # Now assign the proper datetime time coordinate
        da_reshaped['time'] = np.array(time_var, dtype='datetime64[M]')
        if var_name == 'smb':
            da_reshaped = da_reshaped.transpose('time', 'y', 'x')
        else:
            da_reshaped = da_reshaped.transpose('time', 'rlat', 'rlon')
        # Rename the DataArray to match the variable name
        da_reshaped = da_reshaped.rename(var_name)

        time_slice = da_reshaped.coords['time'].shape[0]
        # Save the seasonal dataset with 30 years * given months of data
        print(f"Saving {var_name} seasonal {season_type} {stat} for period {start_year}-{end_year}")
        da_reshaped.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_trend_multiple_dates.nc", mode='w')

        # %%
    @staticmethod
    def calculate_seasonal_racmo_with_single_dates_detrend(var_name, start_year, end_year, season_type):
        """Calculate seasonal detrended racmo variables. (This function is replaced by calc_seasonal_racmo_with_single_dates_detrend_revised)

        """
        # %%
        # -------- For testing ------------------------
        print('This function is replaced by calc_seasonal_racmo_with_single_dates_detrend_revised')
        start_year = 1990
        end_year = 2019
        season_type = 'JJA'

        # var_name = 'smb' # Test SMB 'MEAN' and 'MEDIAN' for testing
        var_name = 'smb' # Test SMB 'MEAN' and 'MEDIAN' for testing
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        # stat = 'MEAN' # Test SMB 'MEAN' and 'MEDIAN' for testing
        # stat = 'MEDIAN' # Test SMB 'MEAN' and 'MEDIAN' for testing

        # -------- For testing ------------------------

        # Read seasonal data with single dates
        read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        da_season = xr.open_dataarray(f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_trend_single_dates.nc")

        # Detrend the anomalies
        print(f"Detrending {var_name} for period {start_year}-{end_year} and season {season_type}")
        da_season_detrend_xrft = xrft.detrend(da_season, dim='year', detrend_type='linear')
        # Save the detrended data
        da_season_detrend_xrft.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_detrend_single_dates.nc", mode='w')

        # %%
        # # Evaluate the detrended data
        # # Uncomment the following lines to visualize the data
        # import matplotlib.pyplot as plt
        # import pymannkendall as mk

        # # Define the Mann-Kendall test function
        # def mk_test_trend(df):
        #     # df = df.loc[:,col_name].copy(deep=True)
        #     result = mk.original_test(df)
        #     X = range(1,len(df)+1)
        #     prediction = result.slope * X + result.intercept
        #     return result, prediction

        # var_name = 'ci'; stat = 'MEDIAN'; x = np.int64(304) + 1; y = np.int64(50)
        # var_name = 'smb'; stat = 'SUM'; x = np.int64(1496 / 2); y = np.int64(2700 / 2)
        # var_name = 'z0500'; stat = 'MEAN'; x = np.int64(566 / 2); y = np.int64(438 / 2)
        # season_type = 'MJJAS' #JJA' # 'MJJAS' # 'JJA
        # start_year = 1990
        # end_year = 2019

        # read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # da_season = xr.open_dataarray(f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_trend_single_dates.nc")
        # da_season_detrend_xrft = xr.open_dataarray(f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_detrend_single_dates.nc")
        # da_season_detrend_xrft = da_season_detrend_xrft[:, x, y]

        # # Perform the Mann-Kendall test
        # result, da_season_slope = mk_test_trend(da_season[:, x, y].values)
        # # if result.p <= 0.1:
        # da_season_detrend_mk = da_season[:, x, y].values - da_season_slope

        # fig, ax = plt.subplots(1)
        # # da_season[:, x, y].plot(ax=ax, color='black', label='Data')
        # # ax.plot(da_season.coords['year'].values, da_season_slope, color='green', label='Trend Line')
        # # if result.p <= 0.1:
        # da_season_detrend_xrft.plot(ax=ax, color='red', label='Detrended xrft')
        # ax.plot(da_season.coords['year'].values, da_season_detrend_xrft, color='blue', label='Detrended MK test')
        # fig.legend()
        # %%

    @staticmethod
    def calculate_seasonal_racmo_with_multiple_dates_detrend(var_name, start_year, end_year, season_type):
        """Calculate seasonal detrended racmo variables with 30 years * multiple months of data.

        """
        # Get the statistics for the variable
        stat = variable_stats(var_name)

        read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # Read monthly data for June, July, and August
        # Open and rename each month's dataset
        da_month_jun = xr.open_dataset(f"{read_file_path}{var_name}_6_{stat}_{start_year}to{end_year}_detrend.nc")
        da_month_jun = da_month_jun.rename_vars({list(da_month_jun.data_vars)[0]: 'jun'})

        da_month_jul = xr.open_dataset(f"{read_file_path}{var_name}_7_{stat}_{start_year}to{end_year}_detrend.nc")
        da_month_jul = da_month_jul.rename_vars({list(da_month_jul.data_vars)[0]: 'jul'})

        da_month_aug = xr.open_dataset(f"{read_file_path}{var_name}_8_{stat}_{start_year}to{end_year}_detrend.nc")
        da_month_aug = da_month_aug.rename_vars({list(da_month_aug.data_vars)[0]: 'aug'})

        if season_type == 'MJJAS':
            # May
            da_month_may = xr.open_dataset(f"{read_file_path}{var_name}_5_{stat}_{start_year}to{end_year}_detrend.nc")
            da_month_may = da_month_may.rename_vars({list(da_month_may.data_vars)[0]: 'may'})

            # September
            da_month_sep = xr.open_dataset(f"{read_file_path}{var_name}_9_{stat}_{start_year}to{end_year}_detrend.nc")
            da_month_sep = da_month_sep.rename_vars({list(da_month_sep.data_vars)[0]: 'sep'})

        # Extract the DataArrays
        da_jun = da_month_jun['jun']
        da_jul = da_month_jul['jul']
        da_aug = da_month_aug['aug']

        if season_type == 'MJJAS':
            da_may = da_month_may['may']
            da_sep = da_month_sep['sep']

        # Stack all 3 months along a new time axis
        # First, expand dims to create a 'month' axis, then concatenate
        da_jun = da_jun.expand_dims(dim={'month': [6]})
        da_jul = da_jul.expand_dims(dim={'month': [7]})
        da_aug = da_aug.expand_dims(dim={'month': [8]})
        if season_type == 'MJJAS':
            da_may = da_may.expand_dims(dim={'month': [5]})
            da_sep = da_sep.expand_dims(dim={'month': [9]})

        # Concatenate the DataArrays along the 'month' dimension
        if season_type == 'JJA':
            da_all = xr.concat([da_jun, da_jul, da_aug], dim='month')
        elif season_type == 'MJJAS':
            # Concatenate over month including May and September
            da_all = xr.concat([da_may, da_jun, da_jul, da_aug, da_sep], dim='month')

        # Now reshape from (year=30, month=3, rlat, rlon) to (time=90, rlat, rlon)
        da_reshaped = da_all.stack(time=('year', 'month'))

        # Optionally sort time
        da_reshaped = da_reshaped.sortby('time')

        # You now have a DataArray with shape: (time=90, rlat, rlon)
        years = da_all.year.values
        months = da_all.month.values

        # Create a time variable with the correct format
        if season_type == 'JJA':
            # Broadcast into full list of 90 dates
            time_var = [f"{y}-{m:02d}" for y in years for m in [6, 7, 8]]
        elif season_type == 'MJJAS':
            # Broadcast into full list of 90 dates including May and September
            time_var = [f"{y}-{m:02d}" for y in years for m in [5, 6, 7, 8, 9]]

        # Remove old multiindex and the original 'year' and 'month' variables
        da_reshaped = da_reshaped.reset_index('time')
        da_reshaped = da_reshaped.drop_vars(['year', 'month'])

        # Now assign the proper datetime time coordinate
        da_reshaped['time'] = np.array(time_var, dtype='datetime64[M]')
        if var_name == 'smb':
            da_reshaped = da_reshaped.transpose('time', 'y', 'x')
        else:
            da_reshaped = da_reshaped.transpose('time', 'rlat', 'rlon')
        # Rename the DataArray to match the variable name
        da_reshaped = da_reshaped.rename(var_name)

        time_slice = da_reshaped.coords['time'].shape[0]
        # Save the seasonal dataset with 30 years * given months of data
        print(f"Saving {var_name} seasonal {season_type} {stat} for period {start_year}-{end_year}")
        da_reshaped.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_detrend_multiple_dates.nc", mode='w')

        # %%
        # Evaluate the detrended multiple dates data
        # Uncomment the following lines to visualize the data
        # import matplotlib.pyplot as plt

        # var_name = 'ci'; stat = 'MEDIAN'; x = np.int64(304) + 1; y = np.int64(50)
        # var_name = 'smb'; stat = 'SUM'; x = np.int64(1496 / 2); y = np.int64(2700 / 2)
        # var_name = 'z0500'; stat = 'MEAN'; x = np.int64(566 / 2); y = np.int64(438 / 2)
        # season_type = 'JJA' #JJA' # 'MJJAS' # 'JJA
        # start_year = 1990
        # end_year = 2019

        # read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
        # da_season_detrend = xr.open_dataarray(f"{read_file_path}{var_name}_{season_type}_{stat}_{start_year}to{end_year}_detrend_multiple_dates.nc")
        # da_jun_detrend = xr.open_dataarray(f"{read_file_path}{var_name}_6_{stat}_{start_year}to{end_year}_detrend.nc")
        # da_jul_detrend = xr.open_dataarray(f"{read_file_path}{var_name}_7_{stat}_{start_year}to{end_year}_detrend.nc")
        # da_aug_detrend = xr.open_dataarray(f"{read_file_path}{var_name}_8_{stat}_{start_year}to{end_year}_detrend.nc")

        # fig, ax = plt.subplots(1)
        # # da_season[:, y, x].plot(ax=ax, color='black', label='Data')
        # da_season_detrend[:, y, x].plot(ax=ax, color='black', label='Data')
        # for j in range(0, 30):
        #     print(f"i={i}, j={j}")
        #     ax.plot(da_season_detrend.time[j * 3], da_jun_detrend[j, y, x].values, color='red', marker='o')
        #     ax.plot(da_season_detrend.time[j * 3 + 1], da_jul_detrend[j, y, x].values, color='red', marker='o')
        #     ax.plot(da_season_detrend.time[j * 3 + 2], da_aug_detrend[j, y, x].values, color='red', marker='o')

    @staticmethod
    def calculate_seasonal_racmo_with_single_dates_revised(var_name, start_year, end_year, season_type):
        """Calculate seasonal racmo variables.

        Merge monthly data for June, July, and August to create a seasonal (JJA) dataset with 30 years of data.

        30 years (1990-2019) is used for MCA using SVD analysis.

        """
        # %%
        # var_name = 'ci' # Test ci for MEDIAN and MEAN
        # var_name = 'smb' # Test SMB 'SUM', 'MEAN' and 'MEDIAN' for testing
        var_name = 'z0500' # Test z0500
        start_year = 1990
        end_year = 2019
        season_type = 'JJA'

        if var_name == 'smb':
            stat_loop = ['SUM', 'MEDIAN', 'MEAN']
        elif var_name == 'ci':
            stat_loop = ['MEAN', 'MEDIAN']
        elif var_name == 'z0500':
            stat_loop = ['MEAN']
        else:
            raise ValueError(f'Unknown variable name: {var_name}')

        for stat_month in stat_loop:
            read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
            # Read monthly data for June, July, and August
            # June
            da_month_jun = xr.open_dataset(f"{read_file_path}{var_name}_6_{stat_month}_{start_year}to{end_year}_trend.nc")
            da_month_jun = da_month_jun.rename_vars({list(da_month_jun.data_vars)[0]: 'jun'})
            # July
            da_month_jul = xr.open_dataset(f"{read_file_path}{var_name}_7_{stat_month}_{start_year}to{end_year}_trend.nc")
            da_month_jul = da_month_jul.rename_vars({list(da_month_jul.data_vars)[0]: 'jul'})
            # August
            da_month_aug = xr.open_dataset(f"{read_file_path}{var_name}_8_{stat_month}_{start_year}to{end_year}_trend.nc")
            da_month_aug = da_month_aug.rename_vars({list(da_month_aug.data_vars)[0]: 'aug'})

            da_seasons = xr.merge([da_month_jun, da_month_jul, da_month_aug])
            for stat_season in stat_loop:
                if stat_season == 'SUM':
                    da_seasonal_stat = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).sum('new')
                elif stat_season == 'MEDIAN':
                    da_seasonal_stat = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).median('new')
                elif stat_season == 'MEAN':
                    da_seasonal_stat = da_seasons.to_array(dim='new', name=list(da_month_aug.data_vars)[0]).mean('new')

                da_seasonal_stat = da_seasonal_stat.rename(var_name)
                # Save the seasonal dataset with 30 years of data
                print(f"Saving {var_name} seasonal {season_type} {stat_month} {stat_season} for period {start_year}-{end_year}")
                # time_slice = da_seasonal_stat.coords['year'].shape[0]
                da_seasonal_stat.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat_month}_{stat_season}_{start_year}to{end_year}_trend.nc", mode='w')
        # %%
    @staticmethod
    def calculate_seasonal_racmo_with_single_dates_detrend_revised(var_name, start_year, end_year, season_type):
        """Calculate seasonal detrended racmo variables.

        """
        # %%
        # var_name = 'ci' # Test ci for MEDIAN and MEAN
        # var_name = 'smb' # Test SMB 'SUM', 'MEAN' and 'MEDIAN' for testing
        var_name = 'z0500' # Test z0500
        # -------- For testing ------------------------
        start_year = 1990
        end_year = 2019
        season_type = 'JJA'

        if var_name == 'smb':
            stat_loop = ['SUM', 'MEDIAN', 'MEAN']
        elif var_name == 'ci':
            stat_loop = ['MEAN', 'MEDIAN']
        elif var_name == 'z0500':
            stat_loop = ['MEAN']
        else:
            raise ValueError(f'Unknown variable name: {var_name}')

        # -------- For testing ------------------------
        for stat_month in stat_loop:
            for stat_season in stat_loop:
                print(f"{var_name}: {stat_month}_{stat_season}")
                # Read seasonal data with single dates
                read_file_path = f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}/"
                da_season = xr.open_dataarray(f"{read_file_path}{var_name}_{season_type}_{stat_month}_{stat_season}_{start_year}to{end_year}_trend.nc")

                # Detrend the anomalies
                print(f"Detrending {var_name} for period {start_year}-{end_year} and season {season_type}")
                da_season_detrend_xrft = xrft.detrend(da_season, dim='year', detrend_type='linear')
                # Save the detrended data
                da_season_detrend_xrft.to_netcdf(path=f"{read_file_path}{var_name}_{season_type}_{stat_month}_{stat_season}_{start_year}to{end_year}_detrend.nc", mode='w')

    def calculate_weekly_racmo(start_year=1990, end_year=2019):
        """Calculate weekly racmo variables.

        """
        # %%
        for time_period in ['JJAS', 'JJ', 'JA', 'AS']: #['JJAS', 'JJ', 'JA', 'AS']
            # %%

            # --- Step 1: Define valid variable names ---
            var_names = ['ci', 'z0500', 'smb']
            # var_names = ['ci']
            for var_name in var_names:
                # Define file paths based on variable name
                read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
                if var_name == 'smb':
                    read_file_path = f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile/"
                elif var_name == 'ci':
                    read_file_path = f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"
                elif var_name == 'z0500':
                    read_file_path = f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"

                # --- Step 2: Load seasonal datasets ---
                if time_period == 'JJAS':
                    months = [6, 7, 8, 9]
                elif time_period == 'JJ':
                    months = [6, 7]
                elif time_period == 'JA':
                    months = [7, 8]
                elif time_period == 'AS':
                    months = [8, 9]
                dataarray = [xr.open_dataarray(f"{read_file_path}{var_name}_{m}_trended.nc") for m in months]

                # --- Step 3: Concatenate into a single dataset and sort ---
                da = xr.concat(dataarray, dim='time').sortby('time')

                aggregation = _get_aggregation(var_name)
                if aggregation is None:
                    raise ValueError(f"Aggregation method for {var_name} is not defined.")

                # --- Step 5: Calculate weekly statistics ---
                # Loop over all years and concatenate the results
                # all_years = [_process_year(year, da, aggregation) for year in range(1990, 2020)]
                # start_year = 1990
                # end_year = 2019 # Since z0500 data is available until 2019
                all_years = [_process_year(year, da, var_name, time_period) for year in range(start_year, end_year + 1)]
                da_week = xr.concat(all_years, dim='year')

                # Reorder if needed
                if var_name == 'smb':
                    da_week = da_week.transpose('year', 'week', 'y', 'x')
                elif var_name in ['ci', 'z0500']:
                    da_week = da_week.transpose('year', 'week', 'rlat', 'rlon')

                da_week = da_week.rename(var_name)
                # %%

                # # Stack year + week into single time axis ---
                # if var_name == 'smb':
                #     da_week = da_week.stack(total_time=('year', 'week')).transpose('total_time', 'y', 'x')
                # elif var_name in ['ci', 'z0500']:
                #     da_week = da_week.stack(total_time=('year', 'week')).transpose('total_time', 'rlat', 'rlon')

                # Save the seasonal dataset with 30 years of data
                print(f"Saving {var_name} weekly {aggregation} for period {start_year}-{end_year}")
                # time_slice = da_seasonal_stat.coords['year'].shape[0]
                write_file_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}")
                da_week.to_netcdf(path=f"{write_file_path}/{var_name}_{time_period}_{aggregation}_week_{start_year}to{end_year}_trend.nc", mode='w')

                # # Uncomment the following lines to perform additional analysis such as calculating weekly climatology, anomalies, and lags for SMB
                # # Convert MultiIndex into coordinate variables
                # da_week = da_week.reset_index('total_time')
                # da_week = da_week.rename(var_name)

                # da_weekly_clim = da_weekly.groupby("time.week").mean("time")
                # # Anomalies (remove climatology)
                # da_weekly_anom = da_weekly.groupby("time.week") - da_weekly_clim

                # lags = [1, 2, 3]  # weekly lags
                # smb_lags = []

                # for lag in lags:
                #     smb_lags.append(da_smb_weekly.shift(time=lag))
                # da_smb_lagged = xr.concat(smb_lags, dim="lag")

    def calculate_weekly_racmo_deseasonalize_detrend(start_year=1990, end_year=2019):
        """Deseasonalize and detrend weekly racmo variables.

        """
        # %%
        # --- Step 1: Define valid variable names ---
        var_names = ['ci', 'smb', 'z0500']
        for var_name in var_names:
            # var_name = 'ci'
            aggregation = _get_aggregation(var_name)

            read_file_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/{var_name}")
            da_week = xr.open_dataarray(f"{read_file_path}/{var_name}_JJAS_{aggregation}_week_{start_year}to{end_year}_trend.nc", mode='r')

            # 2) Deseasonalize: compute weekly climatology across years and subtract
            # Climatology has dims (week, y, x)
            da_climatology = da_week.mean(dim='year')
            print(f"Deseasonalizing {var_name} weekly {aggregation} for period {start_year}-{end_year}")
            da_week_deseasonalize = da_week - da_climatology  # dims: (year, week, y, x)

            # Save the deseasonalize dataset with 30 years of data
            print(f"Saving {var_name} weekly {aggregation} deseasonalized for period {start_year}-{end_year}")
            da_week_deseasonalize.to_netcdf(path=f"{read_file_path}/{var_name}_JJAS_{aggregation}_week_{start_year}to{end_year}_deseasonalized.nc", mode='w')

            # 3 Detrend the deseasonalized data
            da_week_deseasonalize_detrend = xrft.detrend(da_week_deseasonalize, dim=['year', 'week'], detrend_type='linear')
            print(f"Saving {var_name} weekly {aggregation} deseasonalized detrended for period {start_year}-{end_year}")
            da_week_deseasonalize_detrend.to_netcdf(path=f"{read_file_path}/{var_name}_JJAS_{aggregation}_week_{start_year}to{end_year}_deseasonalized_detrended.nc", mode='w')

            # 4 Detrend the original data
            da_week_detrend = xrft.detrend(da_week, dim=['year', 'week'], detrend_type='linear')
            print(f"Saving {var_name} weekly {aggregation} detrended for period {start_year}-{end_year}")
            da_week_detrend.to_netcdf(path=f"{read_file_path}/{var_name}_JJAS_{aggregation}_week_{start_year}to{end_year}_detrended.nc", mode='w')

            # %%
            # # Uncomment the following lines to visualize the data
            # year_dict = {year: pos for pos, year in enumerate(range(1990, 2020))}
            # year = year_dict[2001]
            # week = 16  # 16th week of JJAS
            # rlat_index = 500 # 75
            # rlon_index = 280 # 150

            # # Whole Greenland
            # fig, ax = plt.subplots(1, 1)
            # da_week[year, week, :, :].plot(ax=ax, cmap='RdBu')
            # ax.plot(da_week['rlon'][rlon_index].values, da_week['rlat'][rlat_index].values, marker='o', color='black', markersize=5)

            # # Point location
            # fig, ax = plt.subplots(1, 1)
            # # ax.plot(da_week[:, :, 75, 150], color='blue', label='Original') #[year, week, rlat, rlon]
            # # ax.plot(da_week[:, 2, 75, 150], color='blue', label='Original') #[year, week, rlat, rlon]
            # ax.plot(da_week[year, :, rlat_index, rlon_index], color='blue', label='Original') #[year, week, rlat, rlon]
            # ax.plot(da_climatology[:, rlat_index, rlon_index], color='g', label='Climatology') #[year, week, rlat, rlon]
            # ax.plot(da_deseasonalize[year, :, rlat_index, rlon_index], color='r', label='Anomaly') #[year, week, rlat, rlon]
            # fig.legend()

            # Stack to plot
            # da_week_deseasonalize_detrend_year = da_week_deseasonalize_detrend.stack(total_time=('year', 'week')).transpose('total_time', 'rlat', 'rlon')
            # # Use multiindex to create a single time axis
            # da_week_deseasonalize_year = da_week_deseasonalize.stack(total_time=('year', 'week')).transpose('total_time', 'rlat', 'rlon')
            # da_week_deseasonalize_year_detrend = xrft.detrend(da_week_deseasonalize_year, dim='total_time', detrend_type='linear')
            # da_week_year = da_week.stack(total_time=('year', 'week')).transpose('total_time', 'rlat', 'rlon')
            # rlat_index, rlon_index = 500, 280
            # fig, ax = plt.subplots(1, 1)
            # ax.plot(da_week_year[:, rlat_index, rlon_index], color='blue', label='Original') #[year, week, rlat, rlon]
            # ax.plot(da_week_deseasonalize_year[:, rlat_index, rlon_index], color='red', label='Deseasonalized') #[year, week, rlat, rlon]
            # ax.plot(da_week_deseasonalize_year_detrend[:, rlat_index, rlon_index], color='orange', label='Year Detrended') #[year, week, rlat, rlon]
            # ax.plot(da_week_deseasonalize_detrend_year[:, rlat_index, rlon_index], color='green', label='Detrended Year') #[year, week, rlat, rlon]

    @staticmethod
    def merge_one_year():
        """Merge one year of racmo variables.

        """
        # %%
        for var_name in ['smb']: # ['ci', 'smb', 'z0500']
            # Define file paths based on variable name
            read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
            if var_name == 'smb':
                read_file_path = f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile/"
            elif var_name == 'ci':
                read_file_path = f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"
            elif var_name == 'z0500':
                read_file_path = f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile/"

            for year in range(1990, 2020): #range(1990, 2020):
                dataarray = []
                months = list(range(1, 13))  # Months from January (1) to December (12)
                for m in months:
                    da_month = xr.open_dataarray(f"{read_file_path}{var_name}_{m}_trended.nc")
                    da_month_year = da_month.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
                    dataarray.append(da_month_year)

                # Concatenate into a single dataset and sort ---
                da_year = xr.concat(dataarray, dim='time').sortby('time')

                # Clean encodings and attributes
                da_year.attrs.pop('_FillValue', None)
                da_year.attrs.pop('missing_value', None)
                da_year.encoding.pop('_FillValue', None)
                da_year.encoding.pop('missing_value', None)

                # Define compression and encoding settings
                comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
                encoding = {da_year.name: {**comp, '_FillValue': np.nan}}

                # Saving the merged dataset with all years of data
                print(f"Saving {var_name} merged for year {year}")
                da_year.to_netcdf(path=f"{read_file_path}/{var_name}_{year}.nc", mode='w', encoding=encoding)

    def coarsen_data_month():
        """Coarse RACMO data from 1-km to 5-km resolution to reduce the memory usage.

        """
        # for var_name in ['ci', 'smb', 'z0500']:
        # for var_name in ['psurf', 't2m', 'tskin']: # 'smb', 'ci', 'z0500']:
        for var_name in ['latf', 'q2m', 'senf']: # 'smb', 'ci', 'z0500']:
            for coarsen_level in [2, 4, 6]:
                print(f"Coarsening monthly {var_name} at level {coarsen_level}")
                read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset')
                if var_name == 'smb':
                    read_file_path = Path(f"{read_file_path}/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
                elif var_name == 'z0500':
                    read_file_path = Path(f"{read_file_path}/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
                else:
                    read_file_path = Path(f"{read_file_path}/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

                months = list(range(1, 13))  # Months from January (1) to December (12)
                for m in months:
                    print(f"Month {m}")
                    da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended.nc")
                    # Coarsen the dataset
                    coarsened_array = coarsen_matrix_3d(array=da_month.values, level=coarsen_level)
                    # Coarsen coordinates
                    if var_name == 'smb':
                        x_coord_name, y_coord_name = 'x', 'y'
                    else:
                        x_coord_name, y_coord_name = 'rlon', 'rlat'
                    lon_coord_coarsened, lat_coord_coarsened = coarsen_coordinates_1d(X=da_month.coords[x_coord_name], Y=da_month.coords[y_coord_name], level=coarsen_level)

                    # --- Construct new DataArray with preserved attributes ---
                    da_coarsened = xr.DataArray(coarsened_array,
                                                dims=da_month.dims,
                                                coords={'time': da_month.coords['time'].values,
                                                        y_coord_name: lat_coord_coarsened,
                                                        x_coord_name: lon_coord_coarsened},
                                            attrs=da_month.attrs)  # preserve metadata from original DataArray
                    # Assign the variable name
                    da_coarsened.name = var_name

                    # Save the coarsened DataArray
                    # Define compression and encoding settings
                    comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
                    encoding = {da_coarsened.name: {**comp, '_FillValue': np.nan}}

                    coarsened_file_name = f"{var_name}_{m}_trended_coarsened_{coarsen_level}.nc"
                    da_coarsened.to_netcdf(f"{read_file_path}/{coarsened_file_name}", mode='w', encoding=encoding)

    def merge_MJJAS_coarse():
        """Merge coarsened data for the months May to September (MJJAS) into a single file.

        """
        # %%
        # for var_name in ['ci', 'smb', 'z0500']:
        # for var_name in ['psurf', 't2m', 'tskin']: # 'smb', 'ci', 'z0500']:
        for var_name in ['latf', 'q2m', 'senf']: # 'smb', 'ci', 'z0500']:
            read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset')
            if var_name == 'smb':
                read_file_path = Path(f"{read_file_path}/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
            elif var_name == 'z0500':
                read_file_path = Path(f"{read_file_path}/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
            else:
                read_file_path = Path(f"{read_file_path}/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

            # Define compression and encoding settings
            comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
            encoding = {var_name: {**comp, '_FillValue': np.nan}}

            # for coarsen_level in [0, 2, 4, 6]:
            for coarsen_level in [0,]:
                print(f"Merging MJJAS coarsened {var_name} at level {coarsen_level}")
                dataarray = []
                months = [5, 6, 7, 8, 9]  # Months from May (5) to September (9)
                for m in months:
                    print(f"Month {m}")
                    if coarsen_level == 0:
                        da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended.nc")
                    else:
                        da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended_coarsened_{coarsen_level}.nc")
                    dataarray.append(da_month)

                # Concatenate into a single dataset and sort ---
                da_mjjas = xr.concat(dataarray, dim='time').sortby('time')

                # Assign the variable name
                da_mjjas.name = var_name

                # Saving the merged dataset with all years of data
                print(f"Saving {var_name} MJJAS coarsened for level {coarsen_level}")
                coarsened_file_name = f"{var_name}_MJJAS_trended_coarsened_{coarsen_level}.nc"
                da_mjjas.to_netcdf(f"{read_file_path}/{coarsened_file_name}", mode='w', encoding=encoding)

    def merge_AMJJASO_coarse():
        """Merge coarsened data for the months April to October (AMJJASO) into a single file.

        """
        # %%
        for var_name in ['ci', 'smb', 'z0500']:
            read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset')
            if var_name == 'smb':
                read_file_path = Path(f"{read_file_path}/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
            elif var_name == 'ci':
                read_file_path = Path(f"{read_file_path}/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
            elif var_name == 'z0500':
                read_file_path = Path(f"{read_file_path}/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

            # Define compression and encoding settings
            comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
            encoding = {var_name: {**comp, '_FillValue': np.nan}}

            # for coarsen_level in [0, 2, 4, 6]:
            for coarsen_level in [2, 4, 6]:
                print(f"Merging AMJJASO coarsened {var_name} at level {coarsen_level}")
                dataarray = []
                months = [4, 5, 6, 7, 8, 9, 10]  # Months from April (4) to October (10)
                for m in months:
                    print(f"Month {m}")
                    if coarsen_level == 0:
                        da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended.nc")
                    else:
                        da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended_coarsened_{coarsen_level}.nc")
                    dataarray.append(da_month)

                # Concatenate into a single datasset and sort ---
                da_amjjaso = xr.concat(dataarray, dim='time').sortby('time')

                # Assign the variable name
                da_amjjaso.name = var_name

                # Saving the merged dataset with all years of data
                print(f"Saving {var_name} AMJJASO coarsened for level {coarsen_level}")
                coarsened_file_name = f"{var_name}_AMJJASO_trended_coarsened_{coarsen_level}.nc"
                da_amjjaso.to_netcdf(f"{read_file_path}/{coarsened_file_name}", mode='w', encoding=encoding)
                print(f"Saved {var_name} AMJJASO coarsened for level {coarsen_level}")

        # %%
    def coarsen_data_year():
        """Coarse RACMO data from 1-km to 5-km resolution to reduce the memory usage.

        """
        for var_name in ['ci', 'smb', 'z0500']:
            for coarsen_level in [2, 4, 6]:
                print(f"Coarsening yearly {var_name} at level {coarsen_level}")
                read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset')
                if var_name == 'smb':
                    read_file_path = Path(f"{read_file_path}/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
                elif var_name == 'ci':
                    read_file_path = Path(f"{read_file_path}/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
                elif var_name == 'z0500':
                    read_file_path = Path(f"{read_file_path}/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

                years = list(range(1990, 2020))  # Years from 1990 to 2019
                for year in years:
                    print(f"Year {year}")
                    da_year = xr.open_dataarray(f"{read_file_path}/{var_name}_{year}.nc")
                    coords_to_drop = ['height', 'lat', 'lon']
                    da_year = da_year.drop_vars([c for c in coords_to_drop if c in da_year.coords])

                    # Coarsen the dataset
                    coarsened_array = coarsen_matrix_3d(array=da_year.values, level=coarsen_level)
                    # Coarsen coordinates
                    if var_name == 'smb':
                        x_coord_name, y_coord_name = 'x', 'y'
                    elif var_name in ['ci', 'z0500']:
                        x_coord_name, y_coord_name = 'rlon', 'rlat'
                    lon_coord_coarsened, lat_coord_coarsened = coarsen_coordinates_1d(X=da_year.coords[x_coord_name], Y=da_year.coords[y_coord_name], level=coarsen_level)

                    # --- Construct new DataArray with preserved attributes ---
                    da_coarsened = xr.DataArray(coarsened_array,
                                                dims=da_year.dims,
                                                coords={'time': da_year.coords['time'].values,
                                                        y_coord_name: lat_coord_coarsened,
                                                        x_coord_name: lon_coord_coarsened},
                                            attrs=da_year.attrs)  # preserve metadata from original DataArray
                    # Assign the variable name
                    da_coarsened.name = var_name

                    # Save the coarsened DataArray
                    # Define compression and encoding settings
                    comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
                    encoding = {da_coarsened.name: {**comp, '_FillValue': np.nan}}

                    coarsened_file_name = f"{var_name}_{year}_trended_coarsened_{coarsen_level}.nc"
                    da_coarsened.to_netcdf(f"{read_file_path}/{coarsened_file_name}", mode='w', encoding=encoding)
                    # %%

    def coarsen_lat_lon_topo_promice_mask():
        # %%
        read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # for coarsen_level in [2, 4]:
        for coarsen_level in [4]:
            print(f"Coarsening lat, lon, topo, and promice mask at level {coarsen_level}")
            # ------------------------ Coarsen RACMO 1-km mask ------------------------
            # Coarsen RACMO 1km mask lat, lon, topo, and promice mask at level-4
            # Read the original RACMO mask at 1-km resolution
            mask1km_ds = xr.open_dataset(f"{read_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc")
            racmo_1km_mask_lat = mask1km_ds['LAT']
            racmo_1km_mask_lon = mask1km_ds['LON']
            racmo_1km_mask_promice = mask1km_ds['Promicemask']
            racmo_1km_mask_topo = mask1km_ds['Topography']

            # Coarsen the 2D lat, lon, topo, and promice mask arrays
            racmo_1km_mask_lat_coarsened_array = coarsen_matrix_2d(array=racmo_1km_mask_lat.values, level=coarsen_level)
            racmo_1km_mask_lon_coarsened_array = coarsen_matrix_2d(array=racmo_1km_mask_lon.values, level=coarsen_level)
            racmo_1km_mask_promice_coarsened_array = coarsen_matrix_2d(array=racmo_1km_mask_promice.values, level=coarsen_level)
            racmo_1km_mask_topo_coarsened_array = coarsen_matrix_2d(array=racmo_1km_mask_topo.values, level=coarsen_level)

            # Coarsen the 1D coordinates
            racmo_1km_mask_xcoord_coarsened_array, racmo_1km_mask_ycoord_coarsened_array = coarsen_coordinates_1d(X=racmo_1km_mask_lon.coords['x'], Y=racmo_1km_mask_lon.coords['y'], level=coarsen_level)

            # --- Construct new LON, LAT, TOPO, and PROMICE MASK dataArray with preserved attributes ---
            da_coarsened_1km_lon = xr.DataArray(racmo_1km_mask_lon_coarsened_array,
                                            dims=racmo_1km_mask_lon.dims,
                                            coords={'y': racmo_1km_mask_ycoord_coarsened_array, 'x': racmo_1km_mask_xcoord_coarsened_array},
                                            attrs=racmo_1km_mask_lon.attrs)  # preserve metadata from original DataArray
            da_coarsened_1km_lon.name = racmo_1km_mask_lon.name

            da_coarsened_1km_lat = xr.DataArray(racmo_1km_mask_lat_coarsened_array,
                                            dims=racmo_1km_mask_lat.dims,
                                            coords={'y': racmo_1km_mask_ycoord_coarsened_array, 'x': racmo_1km_mask_xcoord_coarsened_array},
                                            attrs=racmo_1km_mask_lat.attrs)  # preserve metadata from original DataArray
            da_coarsened_1km_lat.name = racmo_1km_mask_lat.name

            da_coarsened_1km_promice = xr.DataArray(racmo_1km_mask_promice_coarsened_array,
                                            dims=racmo_1km_mask_promice.dims,
                                            coords={'y': racmo_1km_mask_ycoord_coarsened_array, 'x': racmo_1km_mask_xcoord_coarsened_array},
                                            attrs=racmo_1km_mask_promice.attrs)  # preserve metadata from original DataArray
            da_coarsened_1km_promice.name = racmo_1km_mask_promice.name

            da_coarsened_1km_topo = xr.DataArray(racmo_1km_mask_topo_coarsened_array,
                                            dims=racmo_1km_mask_topo.dims,
                                            coords={'y': racmo_1km_mask_ycoord_coarsened_array, 'x': racmo_1km_mask_xcoord_coarsened_array},
                                            attrs=racmo_1km_mask_topo.attrs)  # preserve metadata from original DataArray
            da_coarsened_1km_topo.name = racmo_1km_mask_topo.name

            # --- Combine all four into a dataset ---
            ds_coarsened_1km = xr.Dataset({da_coarsened_1km_lon.name: da_coarsened_1km_lon,
                                           da_coarsened_1km_lat.name: da_coarsened_1km_lat,
                                           da_coarsened_1km_topo.name: da_coarsened_1km_topo,
                                           da_coarsened_1km_promice.name: da_coarsened_1km_promice})

            # Optionally, add global metadata
            ds_coarsened_1km.attrs = {"description": f"Coarsened at level-{coarsen_level} RACMO 1-km mask longitude, latitude, topo, and promice mask dataset",
                                    "source": "Derived from RACMO 1-km mask",}
            ds_coarsened_1km.to_netcdf(path=f"{read_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS_coarsened_level_{coarsen_level}.nc", mode='w')

            # %%
            # ------------------------ Coarsen RACMO 5-km mask ------------------------
            # Coarsen RACMO 5km mask lon and lat at level-{coarsen_level}
            # Read the original RACMO mask at 5-km resolution
            mask5km_ds = xr.open_dataset(f"{read_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc")
            racmo_5km_mask_lat = mask5km_ds['lat']
            racmo_5km_mask_lon = mask5km_ds['lon']

            # Coarsen the 2D lat and lon arrays
            racmo_5km_mask_lat_coarsened_array = coarsen_matrix_2d(array=racmo_5km_mask_lat.values, level=coarsen_level)
            racmo_5km_mask_lon_coarsened_array = coarsen_matrix_2d(array=racmo_5km_mask_lon.values, level=coarsen_level)

            # Coarsen the 1D coordinates
            racmo_5km_mask_xcoord_coarsened_array, racmo_5km_mask_ycoord_coarsened_array = coarsen_coordinates_1d(X=racmo_5km_mask_lon.coords['rlon'], Y=racmo_5km_mask_lon.coords['rlat'], level=coarsen_level)

            # --- Construct new LON and LAT dataArray with preserved attributes ---
            da_coarsened_5km_lon = xr.DataArray(racmo_5km_mask_lon_coarsened_array,
                                            dims=racmo_5km_mask_lon.dims,
                                            coords={'rlat': racmo_5km_mask_ycoord_coarsened_array, 'rlon': racmo_5km_mask_xcoord_coarsened_array},
                                            attrs=racmo_5km_mask_lon.attrs)  # preserve metadata from original DataArray
            da_coarsened_5km_lon.name = racmo_5km_mask_lon.name

            da_coarsened_5km_lat = xr.DataArray(racmo_5km_mask_lat_coarsened_array,
                                            dims=racmo_5km_mask_lat.dims,
                                            coords={'rlat': racmo_5km_mask_ycoord_coarsened_array, 'rlon': racmo_5km_mask_xcoord_coarsened_array},
                                            attrs=racmo_5km_mask_lat.attrs)  # preserve metadata from original DataArray
            da_coarsened_5km_lat.name = racmo_5km_mask_lat.name

            # --- Combine both into a dataset ---
            ds_coarsened_5km = xr.Dataset({da_coarsened_5km_lon.name: da_coarsened_5km_lon, da_coarsened_5km_lat.name: da_coarsened_5km_lat})

            # Optionally, add global metadata
            ds_coarsened_5km.attrs = {'description': f"Coarsened at level-{coarsen_level} RACMO 5-km mask longitude and latitude dataset",
                                    'source': 'Derived from RACMO 5-km mask',}
            ds_coarsened_5km.to_netcdf(path=f"{read_file_path}/FGRN055_Masks_5.5km_updated_coordinates_level_{coarsen_level}.nc", mode='w')

    def create_sea_ice_masks():
        # %%
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']

        # Coarsen the dataset
        coarsen_level = 2
        coarsened_array = coarsen_matrix_3d(array=sea_ice_buffer_da.values, level=coarsen_level)

        x_coord_name, y_coord_name = 'rlon', 'rlat'
        lon_coord_coarsened, lat_coord_coarsened = coarsen_coordinates_1d(X=sea_ice_buffer_da.coords[x_coord_name], Y=sea_ice_buffer_da.coords[y_coord_name], level=2)

        # --- Construct new DataArray with preserved attributes ---
        da_coarsened = xr.DataArray(coarsened_array,
                                    dims=sea_ice_buffer_da.dims,
                                    coords={'radius': sea_ice_buffer_da.coords['radius'].values,
                                            y_coord_name: lat_coord_coarsened,
                                            x_coord_name: lon_coord_coarsened},
                                attrs=sea_ice_buffer_da.attrs)  # preserve metadata from original DataArray

        # Assign the variable name
        da_coarsened.name = sea_ice_buffer_da.name

        # Save the coarsened DataArray
        # Define compression and encoding settings
        comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
        encoding = {da_coarsened.name: {**comp, '_FillValue': np.nan}}

        coarsened_file_name = f"ci_all_buffers_on_racmo5km_coarsened_{coarsen_level}.nc"
        da_coarsened.to_netcdf(f"{mask_file_path}/{coarsened_file_name}", mode='w', encoding=encoding)

    @staticmethod
    def calculate_mjjas_detrended_racmo():
        """Calculate MJJAS detrended racmo variables.

        Detrending is performed by grouping the data by day of year and applying a linear detrend along the time dimension for each group.
        """
        for var_name, coarsen_level in [('ci', 0), ('smb', 4), ('z0500', 0)]:
            var_da = load_coarsed_mjjas_data(var_name=var_name, start_year=1990, end_year=2019, coarsen_level=coarsen_level, clip=False)

            # Detrend the data by grouping by day of year and applying a linear detrend along the time dimension for each group
            print(f"Detrending variable: {var_name} .....")
            da_detrended = (var_da.groupby('time.dayofyear').map(lambda x: _detrend_dim(x, dim='time')))
            da_detrended.attrs['long_name'] = f"Detrended {var_da.attrs['long_name']}"

            # Test the detrending on a smaller slice of the data to ensure it works before applying to the whole dataset
            # var_da_slice = var_da.isel(time=slice(100, 150), rlat=slice(100, 150), rlon=slice(100, 150))
            # da_detrended = (var_da_slice.groupby('time.dayofyear').map(lambda x: _detrend_dim(x, dim='time')))

            read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
            if var_name == 'smb':
                read_file_path = Path(f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
            elif var_name == 'ci':
                read_file_path = Path(f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
            elif var_name == 'z0500':
                read_file_path = Path(f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
            coarsened_file_name = f"{var_name}_MJJAS_detrended_coarsened_{coarsen_level}.nc"
            da_detrended.to_netcdf(f"{read_file_path}/{coarsened_file_name}", mode='w')

            # # Test
            # x= 290; y = 200
            # var_da_point = var_da.sel(rlon=var_da.coords['rlon'][x], rlat=var_da.coords['rlat'][y], method='nearest')
            # da_detrended_point = da_detrended.sel(rlon=var_da.coords['rlon'][x], rlat=var_da.coords['rlat'][y], method='nearest')

            # fig, ax = plt.subplots(1)
            # var_da_point.plot(ax=ax, color='blue', lw=0, marker='o', ms=0.5, label='Original')
            # da_detrended_point.plot(ax=ax, color='red', lw=0, marker='o', ms=0.5, label='Detrended')
            # fig.legend()


            # %%
            # Test
            # da_detrend = xr.open_dataarray(f"{read_file_path}/{coarsened_file_name}")

    def calculate_monthly_mean_all_months():
        # %%
        # for var_name in ['ci', 'smb', 'z0500', 't2m', 'tskin', 't2m', 'q2m', 'latf', 'senf', 'psurf']:
        for var_name in ['smb', 'z0500', 't2m', 'tskin', 't2m', 'q2m', 'latf', 'senf', 'psurf']:
            # %%
            # var_name = 'smb'
            print(f"Calculating monthly mean for {var_name} .....")
            read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'

            if var_name == 'smb':
                read_file_path = f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile"
            elif var_name == 'z0500':
                read_file_path = f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile"
            else:
                read_file_path = f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile"

            # da_all = xr.open_dataset(f"{read_file_path}/{var_name}_all.nc")[var_name]
            # da_all_monthly_mean = da_all.resample(time='MS').mean()

            month_all = []
            months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Months from January (1) to December (12)
            for m in months:
                print(f"Month {m}")
                da_month = xr.open_dataarray(f"{read_file_path}/{var_name}_{m}_trended.nc")

                # Mean or sum for each year
                if var_name == 'smb':
                    da_month_mean = da_month.groupby('time.year').sum()
                else:
                    da_month_mean = da_month.groupby('time.year').mean()

                # Create monthly timestamps: 1990-01-01, 1991-01-01, ...
                years = da_month_mean['year'].values
                monthly_time = pd.to_datetime([f"{y}-{m:02d}-01" for y in years])

                # Replace the year coordinate with time
                da_month_mean = (da_month_mean.rename(year='time').assign_coords(time=monthly_time))

                # Case 1: if 'height' is a dimension -> drop it
                for var in ['height', 'lat', 'lon']:
                    if var in da_month_mean.dims:
                        da_month_mean = da_month_mean.drop_vars(var)
                    elif var in da_month_mean.coords:
                        da_month_mean = da_month_mean.reset_coords(var, drop=True)
                month_all.append(da_month_mean)

            # ---- Concatenate into a single dataset and sort ---- #
            print(f"Concatenating monthly means for {var_name} .....")
            all_year_monthly_mean = xr.concat(month_all, dim='time', join='outer').sortby('time')

            # Saving the merged dataset with all years of data
            print(f"Saving {var_name} monthly mean for all months")
            file_name = f"{var_name}_trended_monthly_mean_coarsened_0.nc"

             # Define compression and encoding settings
            comp = dict(zlib=True, complevel=4)  # adjust level (1–9)
            encoding = {all_year_monthly_mean.name: {**comp, '_FillValue': np.nan}}

            all_year_monthly_mean.to_netcdf(f"{read_file_path}/{file_name}", mode='w', encoding=encoding)

        # %%

class CreateRacmoMasks:
    """Class to create RACMO masks for Freya Glacier.

    This class contains methods to create and save RACMO masks for Freya Glacier at 1-km and 5-km resolutions.
    It also includes a method to add new coordinates to the RACMO mask at 5-km resolution.

    """
    @staticmethod
    def add_new_coordinates_on_racmo_mask_5km():
        # %%
        # Read RACMO files for metadata
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        read_file_path = f"{mask_file_path}FGRN055_Masks_5.5km.nc"
        racmo_5km_mask_topo = xr.open_dataset(read_file_path)

        # Variable file to copy the coordinates
        read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
        da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
        # Assign the coordinates from the variable file to the RACMO mask
        # The coordinates are assigned to the 'rlat' and 'rlon' dimensions of the RACMO mask
        racmo_5km_mask_topo = racmo_5km_mask_topo.assign_coords({
                                "rlat": da_rlat_rlon.rlat.values,
                                "rlon": da_rlat_rlon.rlon.values})

        # Drop both conflicting attributes from lon and lat
        cleaned = _clean_fillvalues(racmo_5km_mask_topo)

        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        cleaned.to_netcdf(path=f"{write_file_path}FGRN055_Masks_5.5km_updated_coordinates.nc", mode='w')
        # %%
        # racmo_5km_mask_topo_clavering_island = racmo_5km_mask_topo.sel(rlon=slice(0.9, 5.5), rlat=slice(2.35, 3.7))
        # racmo_5km_mask_topo_clavering_island.to_netcdf(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates_clavering_island.nc", mode='w')

    @staticmethod
    def add_new_coordinates_on_greenland_sea_mask_5km():
        # %%
        # Read RACMO files for metadata
        # mask_file_path = '/nas/data/processed/people/sos/composites_output/racmo/whole_domain_Zack_ci/timeseries/'
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        read_file_path = f"{mask_file_path}greenland_sea_mask.nc"
        greenland_sea_mask_5km = xr.open_dataarray(read_file_path)

        # Variable file to copy the coordinates
        read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
        da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
        # Assign the coordinates from the variable file to the RACMO mask
        # The coordinates are assigned to the 'rlat' and 'rlon' dimensions of the RACMO mask
        greenland_sea_mask_5km = greenland_sea_mask_5km.assign_coords({
                                "rlat": da_rlat_rlon.rlat.values,
                                "rlon": da_rlat_rlon.rlon.values})
        greenland_sea_mask_5km.name = 'Greenland_sea_on_5km_RACMO'
        greenland_sea_mask_5km.attrs = dict(title=f"Greenland sea on 5.5 km RACMO")
        greenland_sea_mask_5km.attrs['institution'] = "University of Graz (Sonika Shahi)"
        greenland_sea_mask_5km.attrs['history'] = f"Created on {datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} Python"
        greenland_sea_mask_5km.attrs['Conventions'] = "CF-1.8"

        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        greenland_sea_mask_5km.to_netcdf(path=f"{write_file_path}greenland_sea_mask_5km_updated_coordinates.nc", mode='w')

        # %%
        # racmo_5km_mask_topo_clavering_island = racmo_5km_mask_topo.sel(rlon=slice(0.9, 5.5), rlat=slice(2.35, 3.7))
        # racmo_5km_mask_topo_clavering_island.to_netcdf(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates_clavering_island.nc", mode='w')
        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        da = xr.open_dataarray(f"{write_file_path}greenland_sea_mask_5km_updated_coordinates.nc")
        fig, ax = plt.subplots(1, 1)
        da.plot(ax=ax)

        # %%
    @staticmethod
    def create_freya_glacier_on_qgis(resolution):
        """Convert and save Freya Glacier mask from shapefile to RACMO 1-km or 5-km grid.

        The input Freya Glacier's outline in shapefile format is preprocessed using QGIS.
        For more information about the preprocessing method, read 'README_shpfile_reproject' located in
        '/Users/sonika/Documents/Shahi_etal/3_paper_work/qgis_freya_map' path.

        The output is the Freya Glacier's outline in nc format of either RACMO 1-km or 5-km resolution, depending on the 'resolution' argument.

        Input
            resolution: str
                        '1km' for 1-km resolution or '5km' for 5-km resolution.

        References: https://gis.stackexchange.com/questions/238533/extracting-points-from-linestring-or-polygon-and-making-dictionary-out-of-them-i

        """
        # %%
        # Read freya glacier polygon shapefile and extract the lat and lon of the vertices
        gdf = gpd.read_file('/nas/data/processed/people/sos/third_paper_data/freya_gl_polygon_lat_lon/freya_gl_polygon_reprojected_to_EPSG4326_WGS84.shp')
        gdf_latlon = list(gdf.iloc[0]['geometry'].exterior.coords)
        # Alternative way to extract all vertices from the shapefile
        # for _, row in gdf.iterrows():
        #     gdf_latlon = list(row['geometry'].exterior.coords) OR
        # gdf_latlon = []
        # for _, row in gdf.iterrows():
        #     gdf_latlon.extend(list(row['geometry'].exterior.coords))

        # Create dataframe of lat and lon of each vertices of the shape file
        df_latlon = pd.DataFrame(columns=['lat', 'lon'], index=range(0, len(gdf_latlon)))
        df_latlon['lat'] = [coord[1] for coord in gdf_latlon]
        df_latlon['lon'] = [coord[0] for coord in gdf_latlon]
        # Alternative way to fill the dataframe manually; slower version
        # for i in range(0, len(df_latlon)):
        #     df_latlon.loc[i, 'lat'] = gdf_latlon[i][1]
        #     df_latlon.loc[i, 'lon'] = gdf_latlon[i][0]

        # Fill in with the nearest indices
        df_latlon['lat_i'] = np.nan
        df_latlon['lon_i'] = np.nan

        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        if resolution == '1km':
            # Read RACMO files for metadata
            read_file_path = f"{mask_file_path}Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc"
            racmo_1km_mask_topo = xr.open_dataset(read_file_path)['Topography']
            racmo_1km_mask_lat = xr.open_dataset(read_file_path)['LAT']
            racmo_1km_mask_lon = xr.open_dataset(read_file_path)['LON']

            # ifile_path = '/nas/data/processed/people/sos/racmo_zack_domain/zack_topo.nc'
            # racmo_1km_mask_topo = xr.open_dataset(ifile_path)['Topography']
            # racmo_1km_mask_lat = xr.open_dataset(ifile_path)['LAT']
            # racmo_1km_mask_lon = xr.open_dataset(ifile_path)['LON']

            # Create numpy arrays for latitude and longitude
            latvar = racmo_1km_mask_lat.values
            lonvar = racmo_1km_mask_lon.values

            # Find the nearest distance x and y polar coordinates on the nc file
            # Create an array to store the Freya Glacier outline
            freya_glacier_on_qgis = np.zeros_like(racmo_1km_mask_topo.values)
            # Alternative way to create an empty array; This works only if .values returns
            # a mutable array (which is usually okay, but not always guaranteed to be writable depending on memory layout).
            # freya_glacier_on_qgis_1km = racmo_1km_mask_topo.values
            # # Initialize the Freya Glacier outline array with zeros
            # freya_glacier_on_qgis_1km[:] = 0
        elif resolution == '5km':
            # Read RACMO files for metadata
            read_file_path = f"{mask_file_path}FGRN055_Masks_5.5km_updated_coordinates.nc"
            racmo_5km_mask_topo = xr.open_dataset(read_file_path)['Topography']
            racmo_5km_mask_lat = xr.open_dataset(read_file_path)['lat']
            racmo_5km_mask_lon = xr.open_dataset(read_file_path)['lon']

            # Create numpy arrays for latitude and longitude
            latvar = racmo_5km_mask_lat.values
            lonvar = racmo_5km_mask_lon.values

            # Find the nearest distance rlon and rlat rotated coordinates on the nc file
            # Create an array to store the Freya Glacier outline
            freya_glacier_on_qgis = np.zeros_like(racmo_5km_mask_topo.values)

        # Find the nearest indices for each x and y polar pair coordinates in the dataframe
        y_polar_i, x_polar_i = kdtree_fast_batch(latvar, lonvar, list(df_latlon.loc[:, 'lat']), list(df_latlon.loc[:, 'lon']))
        assert len(y_polar_i) == len(x_polar_i) == len(df_latlon), 'Lengths of input arrays and dataframe do not match.'

        # Fill the Freya Glacier outline array with 1s at the nearest indices
        # This is the outline of the Freya Glacier on the 1-km RACMO grid
        freya_glacier_on_qgis[y_polar_i, x_polar_i] = 1

        # Save the Freya Glacier outline in grid format
        # Use the x, y metadata from already existing RACMO topo file to create new netcdf file
        if resolution == '1km':
            dims = ['y', 'x']
            coords = {'y': racmo_1km_mask_topo.y.values, 'x': racmo_1km_mask_topo.x.values}
        elif resolution == '5km':
            dims = ['rlat', 'rlon']
            coords = {'rlat': racmo_5km_mask_topo.rlat.values, 'rlon': racmo_5km_mask_topo.rlon.values}

            # Note: The rlat and rlon are the coordinates in the 5-km resolution grid which is not present in the FGRN055_Masks_5.5km.nc.
            # Use the rlat, rlon metadata from already existing RACMO 5 km variables to create new netcdf file
            # read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
            # da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
            # coords = {'rlat': da_rlat_rlon.rlat.values, 'rlon': da_rlat_rlon.rlon.values}

        freya_glacier_on_qgis_da = xr.DataArray(
                                    data=freya_glacier_on_qgis,
                                    dims=dims,
                                    coords=coords,
                                    name=f"freya_glacier_on_qgis_{resolution}",
                                    attrs=dict(title=f"Freya Glacier {resolution} outline on QGIS",
                                    institution="University of Graz (Sonika Shahi)",
                                    history=f"Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python",
                    Conventions="CF-1.8"),)

        # Save the DataArray to a netCDF file
        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        freya_glacier_on_qgis_da.to_netcdf(path=f"{write_file_path}freya_glacier_on_qgis_{resolution}.nc", mode='w')

    @staticmethod
    def test_freya_glacier_on_qgis():
        """Test the Freya Glacier mask created by create_freya_glacier_on_qgis function.

        Check the saved files.
        """
        # %%
        # Uncomment the following lines to visualize the saved file
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        station = 'FRE2 (promice type, tripod)'

        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # --------------------------
        # # Testing 1-k resolution.
        # --------------------------
        # ----------------
        # Using kdtree_fast
        # ----------------
        # ifile_path = '/nas/data/processed/people/sos/racmo_zack_domain/zack_topo.nc' # Old path
        # ifile_path_topo = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/clavering_topo.nc'
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc'
        lat = xr.open_dataset(mask_file_path)['LAT']
        lon = xr.open_dataset(mask_file_path)['LON']
        # Extract latitude and longitude from nc files
        latvar = lat.values
        lonvar = lon.values
        iy_min, ix_min = kdtree_fast(latvar, lonvar, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # Extracting x and y coordinates
        ypoints = lat.coords['y'].values
        xpoints = lat.coords['x'].values
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(xpoints, ypoints)
        print(f"kdtf result: {X[iy_min, ix_min]} {Y[iy_min, ix_min]}")

        # -------------
        # Using pyproj
        # -------------
        # EPSG:4326 is the standard WGS84 lat/lon
        source_crs = CRS.from_epsg(4326)
        # Define the target CRS (Polar stereographic EPSG:3413)
        target_crs = CRS.from_epsg(3413)
        # Create the transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        # Perform the transformation
        x_polar, y_polar = transformer.transform(stations_data.loc['lon (W)', station], stations_data.loc['lat (N)', station])

        # Plot the Freya Glacier outline on the RACMO grid

        resolution = '1km'  # Change to '5km' for 5-km resolution
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        freya_glacier_on_qgis_da = xr.open_dataarray(f"{read_file_path}freya_glacier_on_qgis_{resolution}.nc")

        # Plot the Freya Glacier outline on the RACMO grid
        _, ax = plt.subplots(1)
        freya_glacier_on_qgis_da.plot(ax=ax, cmap='terrain')
        ax.plot(x_polar, y_polar, 'ro', markersize=2, label='Pyproj')
        ax.plot(X[iy_min, ix_min], Y[iy_min, ix_min], 'yo', markersize=2, label='kdtree_fast nearest point')
        xlims = [693044, 701544] # 702044-500; mid-way of 1-km
        ylims = [-1556596, -1548596]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend()

        # --------------------------
        # # Testing 5-km resolution.
        # --------------------------
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km.nc'
        racmo_5km_mask_lat = xr.open_dataset(read_file_path)['lat']
        racmo_5km_mask_lon = xr.open_dataset(read_file_path)['lon']
        # Create numpy arrays for latitude and longitude
        latvar = racmo_5km_mask_lat.values
        lonvar = racmo_5km_mask_lon.values
        iy_min, ix_min = kdtree_fast(latvar, lonvar, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # Extracting rlon and rlat coordinates
        read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
        da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
        ypoints = da_rlat_rlon.rlat.values
        xpoints = da_rlat_rlon.rlon.values
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(xpoints, ypoints)
        print(f"kdtf result: {X[iy_min, ix_min]} {Y[iy_min, ix_min]}")

        # -------------
        # Using pyproj
        # -------------
        # EPSG:4326 is the standard WGS84 lat/lon
        source_crs = CRS.from_epsg(4326)
        # Define the custom rotated pole CRS
        target_crs_rpolar = CRS.from_proj4("+proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +datum=WGS84")
        # Create the transformer
        transformer = Transformer.from_crs(source_crs, target_crs_rpolar, always_xy=True)
        # Perform the transformation
        rlon, rlat = transformer.transform(stations_data.loc['lon (W)', station], stations_data.loc['lat (N)', station])

        # Plot the Freya Glacier outline on the RACMO grid
        resolution = '5km'  # Change to '5km' for 5-km resolution
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        freya_glacier_on_qgis_da = xr.open_dataarray(f"{read_file_path}freya_glacier_on_qgis_{resolution}.nc")

        # Plot the Freya Glacier outline on the RACMO grid
        _, ax = plt.subplots(1)
        freya_glacier_on_qgis_da.plot(ax=ax, cmap='terrain')
        ax.plot(rlon, rlat, 'ro', markersize=2, label='Pyproj')
        ax.plot(X[iy_min, ix_min], Y[iy_min, ix_min], 'yo', markersize=2, label='kdtree_fast nearest point')

        xlims = [4.2, 4.6] # 702044-500; mid-way of 1-km
        ylims = [2.8, 3.2]
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.legend()
    # %%
    @staticmethod
    def create_freya_glacier_on_racmo(resolution):
        """Create Freya Glacier mask which overlaps with RACMO's Promice mask.

        This function creates a new Freya Glacier mask based on the overlapped section between
        'freya_glacier_on_qgis.nc' (generated by QGIS and function create_freya_glacier_on_qgis)
        and promice mask (from RACMO 1-km resolution, e.g., file 'Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc', variable 'Promicemask'
                                and RACMO 5-km solution, e.g., file 'FGRN055_Masks_5.5km.nc', variable 'Promicemask').

        The output is the Freya Glacier's outline within the Promice mask of RACMO 1-km and 5-km resolution.

        """
        # %%
        # Read the Freya Glacier outline
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        freya_glacier_on_qgis = xr.open_dataarray(f"{read_file_path}freya_glacier_on_qgis_{resolution}.nc")

        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        if resolution == '1km':
            # Read RACMO peripheral glacier mask
            read_promice_mask = f"{mask_file_path}Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc"
            promice_mask_racmo_1km = xr.open_dataset(read_promice_mask)['Promicemask']
            # Convert a masks representing Greenland Icesheet (3) into 0 and Peripheral glacier (2) into 1 which represents peripheral glacier.
            # 1 and 2 represents the peripheral glacier in 1 km topography.
            promice_mask_racmo_1km = xr.where(promice_mask_racmo_1km == 3, 0,
                                                        xr.where(promice_mask_racmo_1km == 2, 1, promice_mask_racmo_1km))

            # Ensure both datasets are aligned (same resolution and coordinates)
            freya_qgis, freya_racmo = xr.align(freya_glacier_on_qgis, promice_mask_racmo_1km, join="inner")

            # Calculate overlap: where both Freya mask and RACMO mask == 1
            freya_overlap = xr.where((freya_qgis == 1) & (freya_racmo == 1), 1, 0)

            # Save overlapped mask to NetCDF
            write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
            freya_overlap.name = f"freya_glacier_on_racmo_{resolution}"
            freya_overlap.attrs = {
                'title': f'Freya Glacier {resolution} outline on RACMO',
                'institution': 'University of Graz (Sonika Shahi)',
                'history': f'Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python',
                'Conventions': 'CF-1.8'}
            freya_overlap.to_netcdf(f"{write_file_path}/freya_glacier_on_racmo_1km.nc", mode='w')

        elif resolution == '5km':
            # Read RACMO peripheral glacier mask
            read_promice_mask = f"{mask_file_path}FGRN055_Masks_5.5km_updated_coordinates.nc"
            promice_mask_racmo_5km = xr.open_dataset(read_promice_mask)['Promicemask']
            # Convert a masks representing Greenland Icesheet (3) into 0 and Peripheral glacier (2) into 1 which represents peripheral glacier.
            # 1 and 2 represents the peripheral glacier in 5 km topography.
            promice_mask_racmo_5km = xr.where(promice_mask_racmo_5km == 4, 0,
                                    xr.where(promice_mask_racmo_5km == 3, 0,
                                    xr.where(promice_mask_racmo_5km == 2, 1, promice_mask_racmo_5km)))

            # Ensure both datasets are aligned (same resolution and coordinates)
            freya_qgis, freya_racmo = xr.align(freya_glacier_on_qgis, promice_mask_racmo_5km, join="inner")

            # Calculate overlap: where both Freya mask and RACMO mask == 1
            freya_overlap = xr.where((freya_qgis == 1) & (freya_racmo == 1), 1, 0)

            # Check whether the overlap is not empty
            if (freya_overlap.values == 1).any():
                print("At least one match found")
            else:
                print("No match found between Freya Glacier QGIS and RACMO 5 km.")

            # Since there is no overlap between Freya Glacier QGIS and RACMO 5 km, we will only save the RACMO 5-km mask over Clavering Island.
            # Create a mask for the Clavering Island area
            mask = (freya_racmo.rlon > 4.3) & (freya_racmo.rlon < 4.5) & \
                    (freya_racmo.rlat > 2.8) & (freya_racmo.rlat < 3.1)
            clavering_racmo = freya_racmo.where(mask, other=0)

            # Save the Clavering Island mask
            write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
            clavering_racmo.name = f"freya_glacier_on_racmo_{resolution}"
            clavering_racmo.attrs = {
                'title': f'Clavering Island {resolution} outline on RACMO',
                'institution': 'University of Graz (Sonika Shahi)',
                'history': f'Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python',
                'Conventions': 'CF-1.8'}

            clavering_racmo.to_netcdf(f"{write_file_path}/freya_glacier_on_racmo_5km.nc", mode='w')
        else:
            raise ValueError("Resolution must be either '1km' or '5km'.")

    @staticmethod
    def test_freya_glacier_on_racmo():
        """Test the Freya Glacier mask created by create_freya_glacier_on_racmo function.

        Check the saved files."""
        # %%
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        resolutions = ['1km', '5km']  # Change to '5km' for 5-km resolution
        for resolution in resolutions:
            print(f"Testing Freya Glacier mask for resolution: {resolution}")
            freya_glacier_on_racmo = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_racmo_{resolution}.nc")
            freya_glacier_on_qgis = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_qgis_{resolution}.nc")

            if resolution == '1km':
                xaxis, yaxis = freya_glacier_on_qgis.x.values, freya_glacier_on_qgis.y.values
                xlims = [693044, 701544] # 702044-500; mid-way of 1-km
                ylims = [-1556596, -1548596]
            elif resolution == '5km':
                xaxis, yaxis = freya_glacier_on_qgis.rlon.values, freya_glacier_on_qgis.rlat.values
                xlims = [4.2, 4.6]
                ylims = [2.8, 3.2]
            else:
                raise ValueError("Resolution must be either '1km' or '5km'.")
            # Get the grid points where the Freya Glacier is present
            y_qgis, x_qgis = np.where(freya_glacier_on_qgis.values == 1)
            y_racmo, x_racmo = np.where(freya_glacier_on_racmo.values == 1)
            # Plot the Freya Glacier outline on the RACMO and QGIS grid
            _, ax = plt.subplots(1)
            ax.contour(xaxis, yaxis, freya_glacier_on_qgis.values, levels=[0.5, 1.5], colors='red', alpha=0.5, label='FG on QGIS')
            ax.contour(xaxis, yaxis, freya_glacier_on_racmo.values, levels=[0.5, 1.5], colors='blue', alpha=0.5, label='FG on RACMO')
            # Plot the points where the Freya Glacier is present on both grids
            for y, x in zip(y_qgis, x_qgis):
                ax.plot(xaxis[x], yaxis[y], 'ro', markersize=2)
            for y, x in zip(y_racmo, x_racmo):
                ax.plot(xaxis[x], yaxis[y], 'bo', markersize=2)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            ax.legend()

    @staticmethod
    def create_accumulation_and_ablation_1km_mask_of_freya_glacier_on_racmo():
        # %%
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # Read Freya Glacier on RACMO mask
        freya_glacier_on_racmo = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_racmo_1km.nc")

        # Read Topography
        mask_1km = f"{mask_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc"
        racmo_1km_mask_topo = xr.open_dataset(mask_1km)['Topography']

        # Create Accumulation area mask of 1-km
        freya_glacier_on_racmo_acc = freya_glacier_on_racmo.copy()
        freya_glacier_on_racmo_acc = xr.where(racmo_1km_mask_topo < 950, 0, freya_glacier_on_racmo_acc)
        # Save overlapped mask to NetCDF
        freya_glacier_on_racmo_acc.name = f"freya_glacier_accumulation_area_on_racmo_1km"
        freya_glacier_on_racmo_acc.attrs = {
            'title': f'Freya Glacier accumulation area 1-km outline on RACMO',
            'institution': 'University of Graz (Sonika Shahi)',
            'history': f'Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python',
            'Conventions': 'CF-1.8'}
        freya_glacier_on_racmo_acc.to_netcdf(f"{mask_file_path}/freya_glacier_accumulation_area_on_racmo_1km.nc", mode='w')

        # Create Ablation area mask of 1-km
        freya_glacier_on_racmo_ab = freya_glacier_on_racmo.copy()
        freya_glacier_on_racmo_ab = xr.where(racmo_1km_mask_topo > 950, 0, freya_glacier_on_racmo_ab)
        # Save overlapped mask to NetCDF
        freya_glacier_on_racmo_ab.name = f"freya_glacier_ablation_area_on_racmo_1km"
        freya_glacier_on_racmo_ab.attrs = {
            'title': f'Freya Glacier ablation area 1-km outline on RACMO',
            'institution': 'University of Graz (Sonika Shahi)',
            'history': f'Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python',
            'Conventions': 'CF-1.8'}
        freya_glacier_on_racmo_ab.to_netcdf(f"{mask_file_path}/freya_glacier_ablation_area_on_racmo_1km.nc", mode='w')

        # %%
        # # # UNCOMMENT if you want to test the code
        # fig, ax = plt.subplots(1)
        # # freya_glacier_on_racmo_acc.plot(ax=ax)
        # freya_glacier_on_racmo_ab.plot(ax=ax)
        # # freya_glacier_on_racmo.plot(ax=ax)
        # xlims = [693044, 701544] # 702044-500; mid-way of 1-km
        # ylims = [-1556596, -1548596]
        # ax.set_xlim(xlims)
        # ax.set_ylim(ylims)


        # %%

    @staticmethod
    def create_freya_glacier_on_racmo_clavering_domain_1km():
        """Clip the Freya Glacier mask on RACMO to match the clipped input variables.

        """
        # %%
        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # Freya glacier mask from racmo at 1 km
        freya_glacier_on_racmo_greenland = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_racmo_1km.nc")
        freya_glacier_on_racmo_clavering_island = freya_glacier_on_racmo_greenland.sel(x=slice(579044, 801044), y=slice(-1592596, -1508596))
        freya_glacier_on_racmo_clavering_island.to_netcdf(f"{mask_file_path}/freya_glacier_on_racmo_1km_clavering_island.nc", mode='w')

        # RACMO 1-km topography mask
        racmo_mask_1km_greenland = xr.open_dataset(f"{mask_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc")
        racmo_mask_1km_clavering_island = racmo_mask_1km_greenland.sel(x=slice(579044, 801044), y=slice(-1592596, -1508596))
        racmo_mask_1km_clavering_island.to_netcdf(f"{mask_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_clavering_island.nc", mode='w')

        # Freya glacier accumulation area mask from racmo at 1 km
        freya_glacier_accumulation_area_on_racmo_1km_greenland = xr.open_dataarray(f"{mask_file_path}/freya_glacier_accumulation_area_on_racmo_1km.nc")
        freya_glacier_accumulation_area_on_racmo_1km_clavering_island = freya_glacier_accumulation_area_on_racmo_1km_greenland.sel(x=slice(579044, 801044), y=slice(-1592596, -1508596))
        freya_glacier_accumulation_area_on_racmo_1km_clavering_island.to_netcdf(f"{mask_file_path}/freya_glacier_accumulation_area_on_racmo_1km_clavering_island.nc", mode='w')

        # Freya glacier ablation area mask from racmo at 1 km
        freya_glacier_ablation_area_on_racmo_1km_greenland = xr.open_dataarray(f"{mask_file_path}/freya_glacier_ablation_area_on_racmo_1km.nc")
        freya_glacier_ablation_area_on_racmo_1km_clavering_island = freya_glacier_ablation_area_on_racmo_1km_greenland.sel(x=slice(579044, 801044), y=slice(-1592596, -1508596))
        freya_glacier_ablation_area_on_racmo_1km_clavering_island.to_netcdf(f"{mask_file_path}/freya_glacier_ablation_area_on_racmo_1km_clavering_island.nc", mode='w')

        # %%
    @staticmethod
    def create_freya_glacier_on_racmo_clavering_domain_5km():
        """Clip the Freya Glacier mask on RACMO to match the clipped input variables.

        """
        # %%
        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')

        # -------------- RACMO 5-km topography mask --------------
        racmo_mask_5km_greenland = xr.open_dataset(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc")
        racmo_mask_5km_clavering_island = _subset_nearest(racmo_mask_5km_greenland, rlon_bounds=(0.9, 5.5), rlat_bounds=(2.35, 3.7))
        racmo_mask_5km_clavering_island.to_netcdf(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates_clavering_island.nc", mode='w')

        # -------------- Freya glacier mask on racmo at 5 km ---------------
        freya_glacier_on_racmo_greenland = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_racmo_5km.nc")
        freya_glacier_on_racmo_clavering_island = _subset_nearest(freya_glacier_on_racmo_greenland, rlon_bounds=(0.9, 5.5), rlat_bounds=(2.35, 3.7))
        freya_glacier_on_racmo_clavering_island.to_netcdf(f"{mask_file_path}/freya_glacier_on_racmo_5km_clavering_island.nc", mode='w')

        # -------------- Point mask of closest Freya glacier mask racmo at 5 km ---------------
        # Extracting rlon and rlat coordinates
        y_values = freya_glacier_on_racmo_clavering_island.coords['rlat'].values
        x_values = freya_glacier_on_racmo_clavering_island.coords['rlon'].values
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(x_values, y_values)

        # Find the closest values of Promice mask of RACMO from station.
        yindexs_promice, xindexs_promice = np.where(freya_glacier_on_racmo_clavering_island == 1)
        y_values_promice = []
        x_values_promice = []
        for xindex_promice, yindex_promice in zip(xindexs_promice, yindexs_promice):
            x_values_promice.append(float(x_values[xindex_promice]))
            y_values_promice.append(float(y_values[yindex_promice]))

        # Read Freya Glacier's Station location
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'

        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        station = 'FRE2 (promice type, tripod)'

        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Find the nearest indices
        iy_min, ix_min = kdtree_fast(racmo_mask_5km_clavering_island.lat.values, racmo_mask_5km_clavering_island.lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])

        # Compute Euclidean distances d=(xi​−xfreya​)2+(yi​−yfreya​)2
        freya_point = [X[iy_min, ix_min], Y[iy_min, ix_min]]
        # Find the closest point
        distances = [ np.sqrt((x - freya_point[0])**2 + (y - freya_point[1])**2)
            for x, y in zip(x_values_promice, y_values_promice)]

        # Get index of closest point
        min_idx = int(np.argmin(distances))
        closest_point = (x_values_promice[min_idx], y_values_promice[min_idx])

        df_point_masks_5km = pd.DataFrame(columns=['rlat', 'rlon'], index=['freya_point', 'closest_point_to_freya_point'])
        df_point_masks_5km.loc['freya_point', 'rlon'] =  freya_point[0]
        df_point_masks_5km.loc['freya_point', 'rlat'] =  freya_point[-1]
        df_point_masks_5km.loc['closest_point_to_freya_point', 'rlon'] =  closest_point[0]
        df_point_masks_5km.loc['closest_point_to_freya_point', 'rlat'] =  closest_point[-1]

        df_point_masks_5km.to_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv")

        # -------------- Freya glacier mask on qgis at 5 km ---------------
        freya_glacier_on_qgis_greenland = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_qgis_5km.nc")
        freya_glacier_on_qgis_clavering_island = _subset_nearest(freya_glacier_on_qgis_greenland, rlon_bounds=(0.9, 5.5), rlat_bounds=(2.35, 3.7))
        freya_glacier_on_qgis_clavering_island.to_netcdf(f"{mask_file_path}/freya_glacier_on_qgis_5km_clavering_island.nc", mode='w')


        # %%
        df_point_masks_5km.to_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv")
        df_point_masks_5km = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_qgis_5km.nc")





        # %%

    @staticmethod
    def convert_sea_ice_buffer_from_shp_to_nc():
        """Convert sea ice buffer from shape file to nc file.

        """
        # %%
        # Read land-sea mask of RACMO 5-km to mask the sea-ice buffer over land.
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        land_sea_mask_ramo_5km = xr.open_dataset(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc")['LSM_GR']

        # Read RACMO files for metadata
        racmo_5km_mask_lat = xr.open_dataset(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc")['lat'].values
        racmo_5km_mask_lon = xr.open_dataset(f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc")['lon'].values

        # Use sea ice metadata to save the array as nc file
        ci_racmo_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        ci_racmo = xr.open_dataset(f"{ci_racmo_path}/ci_all.nc")['ci']
        # Create a mesh grid
        X, Y = np.meshgrid(ci_racmo.coords['rlon'].values, ci_racmo.coords['rlat'].values)

        # Empty list to store all sea-ice buffers and their corresponding radius values
        all_buffers = []
        radius_values = []

        radii = ['100km', '200km', '300km', '400km', '500km', '600km', '700km', '800km']
        for radius in radii:
            # Read sea-ice buffer polygon shapefile and extract the lat and lon of the vertices
            shape_file_path = Path('/nas/data/processed/people/sos/third_paper_data/sea_ice_extent_lat_lon')
            gdf = gpd.read_file(f"{shape_file_path}/FRE2_reprojected_EPSG_32627_buffer_{radius}_densify_5km_reprojected_EPSG_4326.shp")
            gdf_latlon = list(gdf.iloc[0]['geometry'].exterior.coords)

            # for _, row in gdf.iterrows():
            #     gdf_latlon = list(row['geometry'].exterior.coords)

            # Create dataframe of lat lon of each vertices of the shape file
            df_latlon = pd.DataFrame(columns=['lat', 'lon'],index=range(0, len(gdf_latlon)))
            df_latlon['lat'] = [coord[1] for coord in gdf_latlon]
            df_latlon['lon'] = [coord[0] for coord in gdf_latlon]
            # for i in range(0,len(df_latlon)):
            #     df_latlon.loc[i,'lat'] = gdf_latlon[i][1]
            #     df_latlon.loc[i,'lon'] = gdf_latlon[i][0]

            # ------------------------------
            # Vectorized nearest-neighbor lookup
            # ------------------------------
            iy, ix = kdtree_fast_batch(racmo_5km_mask_lat, racmo_5km_mask_lon, list(df_latlon.loc[:, 'lat']), list(df_latlon.loc[:, 'lon']))

            # Assign back to DataFrame
            df_latlon['iy'] = iy
            df_latlon['ix'] = ix
            df_latlon['X'] = X[iy, ix]
            df_latlon['Y'] = Y[iy, ix]

            # ------------------------------
            # Create sea ice mask
            # ------------------------------
            sea_ice_buffer = np.zeros_like(racmo_5km_mask_lat, dtype=np.int8)
            sea_ice_buffer[iy, ix] = 1

            # ------------------------------
            # Fill mask between break points (x-axis interpolation)
            # ------------------------------
            for iypoint in np.unique(df_latlon['iy'].astype(int)):
                points_on_line = df_latlon[df_latlon['iy'] == iypoint]
                ix_points = np.unique(points_on_line['ix'].astype(int))
                break_points = []
                for i in range(len(ix_points) - 1):
                    diff = ix_points[i+1] - ix_points[i]
                    if diff > 1:
                        break_points.append(ix_points[i] + 1)
                        break_points.append(ix_points[i+1] - 1)
                if break_points:
                    sea_ice_buffer[iypoint, break_points[0]:break_points[-1] + 1] = 1

            # Mask all sea-ice-mask over land
            sea_ice_buffer = xr.where((land_sea_mask_ramo_5km == 1), 0, sea_ice_buffer)
            all_buffers.append(sea_ice_buffer)
            radius_values.append(int(radius.replace('km', '')))

        buffer_3d = xr.DataArray(data=np.stack(all_buffers),
                                 dims=['radius', 'rlat', 'rlon'],
                                 coords={'radius': radius_values,
                                         'rlat': ci_racmo.rlat.values,
                                         'rlon': ci_racmo.rlon.values,},
                                 name='sea_ice_buffer',)
        buffer_3d.attrs = dict(title='Sea ice buffers on 5.5 km RACMO grid',
                               institution='University of Graz (Sonika Shahi)',
                               history=f"Created on {datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} Python",
                               Conventions='CF-1.8',)
        buffer_3d.to_netcdf(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc", mode='w')
        # %%
        # # UNCOMMENT to test the script
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # ds = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        # fig, ax = plt.subplots(1)
        # ds.sel(radius=100).plot(ax=ax)
        # %%

    def create_young_sound_mask():
        # %%
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        ds = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        sif_100 = ds.sel(radius=100)
        # Remove the sea-ice buffer over land and the area around Freya Glacier (Clavering Island) to create a Young Sound mask.
        sif_100.loc[dict(rlat=slice(3.15, 4))] = 0
        sif_100.loc[dict(rlat=slice(2, 2.9))] = 0
        sif_100.loc[dict(rlon=slice(4, 4.4))] = 0
        sif_100.loc[dict(rlon=slice(4.8, 5.5))] = 0

        # Change the name of the mask
        sif_100.name = 'young_sound_mask'
        sif_100.attrs = dict(title='Young Sound mask on 5.5 km RACMO grid',
                             institution='University of Graz (Sonika Shahi)',
                             history=f"Created on {datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} Python",
                             Conventions='CF-1.8',)
        sif_100.to_netcdf(f"{mask_file_path}/young_sound_mask.nc", mode='w')

        # %%
        # UNCOMMENT to test the script
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # ds = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        # zoom_xlim = (4, 5.5)
        # zoom_ylim = (2, 4)

        # fig, ax = plt.subplots(1)
        # sif_100 = ds.sel(radius=100)
        # sif_100.plot(ax=ax)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)

        # # Read the Young Sound mask to check the masked area
        # ys_mask = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        # fig, ax = plt.subplots(1)
        # ys_mask.plot(ax=ax)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)

        # # Plot zoomed-in map
        # fig, ax = plt.subplots(1)
        # variable = 'sif'
        # lonlat = _get_latlon_meshgrid(variable=variable)

        # # Promice mask
        # promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        # ax.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=5)

        # # Plot station
        # # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        # freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        # freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        # freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        # station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # ax.plot(station_rlon, station_rlat, marker='x', color='red', markersize=5, zorder=12, label='Closest point to Freya')

        # # Plot sea ice buffer mask at 100-km radius
        # colors_buffer = [(169, 169, 169), (169, 169, 169)]
        # cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        # bounds_buffer = np.array([-0.01, 0.01, 1.5])
        # norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # ax.contour(lonlat[0], lonlat[1], ys_mask.data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)
        # %%

    def create_z500hPa_mask():
        """ Create z500hPa mask combining land-sea mask and Greenland Sea mask.
        Applies geographic constraints used in wave breaking analysis.

        """
        # %%
        lsm_topo_mask = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km.nc'
        # --- Load datasets safely ---
        with xr.open_dataset(lsm_topo_mask) as ds:
            lsm = ds['LSM_GR'].copy()
            topo = ds['Topography'].copy()

        gsea_file = '/nas/data/processed/people/sos/composites_output/racmo/whole_domain_Zack_ci/timeseries/greenland_sea_mask.nc'
        gsea_mask = xr.open_dataarray(gsea_file)

        # --- Mask Svalbard and Iceland ---
        topo_c = xr.where((topo > 0) & (lsm == 0), 0, topo)

        # --- Mask Greenland to separate Iceland and Svalbard ---
        topo = xr.where(topo == topo_c, 0, topo)

        # --- Add Iceland and Svalbard into land-sea mask ---
        lsm = xr.where(topo > 100, 1, lsm)

        # --- Combine masks ---
        z500_mask = lsm + gsea_mask
        z500_mask = xr.where(z500_mask > 0, 1, 0)

        # --- Mask western region ---
        # Mask all the western part based on Figure 2 of https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021JD035489
        # which indicates a cyclonic wave breaking patterns producing more melt in northeast Greenland
        z500_mask.loc[dict(rlon=slice(None, 140))] = 0  # assumes last dim is longitude-like

        # --- Fill gap between Greenland Sea and Greenland outline ---
        z500_mask.loc[dict(rlon=slice(225, 324), rlat=slice(225, 499))] = 1

        # --- Rename and save ---
        z500_mask = z500_mask.rename('z500hPa_mask')

        # Change coordinates of rlat and rlon
        # Variable file to copy the coordinates
        read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
        da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
        # Assign the coordinates from the variable file to the RACMO mask
        # The coordinates are assigned to the 'rlat' and 'rlon' dimensions of the RACMO mask
        z500_mask = z500_mask.assign_coords({
                    'rlat': da_rlat_rlon.rlat.values,
                    'rlon': da_rlat_rlon.rlon.values})

        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        output_path = Path(mask_file_path)
        output_path.mkdir(parents=True, exist_ok=True)
        z500_mask.to_netcdf(output_path / 'z500hPa_mask.nc', mode='w')

        # %%
    def convert_greenland_sea_mask_from_shp_to_nc_NEEDS_CORRECTION():
        """Convert Greenland Sea mask from shape file to nc file.

        The original script to create Greenland Sea mask from shape file to nc file is
        on the folder /nas/home/sos/2_paper_work_ku/Codes/final_paper_code
        as create_Greenland_Sea_mask.py

        """
        # %%
        # Input files
        mask_file = '/tempdata/knmi/racmo/FGRN055_Masks_5.5km.nc'

        # Load NetCDF with xarray
        ds = xr.open_dataset(mask_file)

        topo = ds["Topography"].values
        latvar = ds["lat"].values
        lonvar = ds["lon"].values
        land_sea = ds["LSM_GR"].values

        rlat = ds["rlat"].values
        rlon = ds["rlon"].values

        # Topography mask
        topo_mask = (topo > 0).astype(int)

        # Remove Jan Mayen
        topo_mask[301:322, 376:390] = 0

        # Load shapefile (Greenland Sea)
        shapefile = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/GreenlandSea.shp'
        gdf = gpd.read_file(shapefile)

        # Extract polygon coordinates
        coords = np.array(gdf.geometry.iloc[0].exterior.coords)
        poly_lon = coords[:, 0]
        poly_lat = coords[:, 1]

        # Fast nearest-grid lookup using KDTree
        points = np.column_stack([latvar.ravel(), lonvar.ravel()])
        tree = cKDTree(points)
        query_points = np.column_stack([poly_lat, poly_lon])
        _, idx = tree.query(query_points)
        iy, ix = np.unravel_index(idx, latvar.shape)

        # Greenland Sea outline grid
        gsea_outline = np.zeros_like(latvar, dtype=np.int8)
        gsea_outline[iy, ix] = 1

        # Fill rows (vectorized)
        # Fill vertical gaps of size 1
        rows, cols = np.where(gsea_outline == 1)
        for c in np.unique(cols):
            r = rows[cols == c]
            if len(r) > 1:
                diff = np.diff(r)
                fill = r[:-1][diff == 2] + 1
                gsea_outline[fill, c] = 1

        # Fill columns (vectorized)
        for y in range(gsea_outline.shape[0]):
            cols = np.where(gsea_outline[y])[0]
            if len(cols):
                start = cols[0]
                end = cols[-1]
                fill_range = np.arange(start, end)
                valid = topo_mask[y, fill_range] != 1
                gsea_outline[y, fill_range[valid]] = 1

        # East Greenland coast correction
        east_coast = land_sea.copy()
        east_coast[:, :225] = np.nan
        rows_with_land = np.where(np.any(east_coast == 1, axis=1))[0]

        for y in rows_with_land[1:-1]:
            cols = np.where(gsea_outline[y])[0]
            if not len(cols):
                continue
            start = cols[0]
            end = cols[-1]
            above = np.where(gsea_outline[y + 1])[0]
            below = np.where(gsea_outline[y - 1])[0]
            if not len(above) or not len(below):
                continue
            end_above = above[-1]
            end_below = below[-1]
            if (
                abs(end_above - end_below) < 2
                and abs(end - end_below) > 2
                and abs(end - end_above) > 2
            ):
                grid_fill = int((end_above + end_below) / 2)
                gsea_outline[y, start : grid_fill + 1] = 1

        # Remove land overlap
        gsea_outline[topo_mask == 1] = 0

        # Variable file to copy the coordinates
        read_rlat_rlon_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile/ci_all.nc'
        da_rlat_rlon = xr.open_dataset(read_rlat_rlon_path)['ci']
        # Assign the coordinates from the variable file to the RACMO mask

        greenland_sea_mask_5km_da = xr.DataArray(
                            data=gsea_outline,
                            dims= ['rlat', 'rlon'],
                            coords={'rlat': da_rlat_rlon.rlat.values, 'rlon': da_rlat_rlon.rlon.values},
                            name=f"Greenland_sea_on_5km_RACMO",
                            attrs=dict(title=f"Greenland sea on 5.5 km RACMO",
                            institution="University of Graz (Sonika Shahi)",
                            history=f"Created on {datetime.now(timezone.utc).strftime("%B %d, %Y %H:%M:%S")} Python",
            Conventions="CF-1.8"),)

        # Save the DataArray to a netCDF file
        print(f"Saving Greenland sea mask to NetCDF...")
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        greenland_sea_mask_5km_da.to_netcdf(path=f"{mask_file_path}/greenland_sea_mask_5km.nc", mode='w')

        # %%

        # UNCOMMENT to test the script
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        greenland_sea_da = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km.nc")
        fig, ax = plt.subplots(1, 1)
        greenland_sea_da.plot(ax=ax)

        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        da = xr.open_dataarray(f"{write_file_path}greenland_sea_mask_5km_updated_coordinates.nc")
        fig, ax = plt.subplots(1, 1)
        da.plot(ax=ax)


    def slice_east_greenland_on_racmo_mask_5km():
        """ This is used for slicing the new coordinates on Greenland Sea mask at 5 km resolution after correcting the coordinates of the RACMO mask.

        It is used to plot the phenology of Greenland Sea sea ice.
        """
        # %%
        gr_masks_5km = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        # Subset to the region of interest (Greenland Sea)
        gr_masks_5km = gr_masks_5km.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        write_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        gr_masks_5km.to_netcdf(path=f"{write_file_path}FGRN055_Masks_5.5km_updated_coordinates_east_greenland.nc", mode='w')

        # %%
    def create_mask_for_variables_on_sea_with_sif_phenology_trend():
        """ This is used for creating a mask for variables on sea with SIF phenology trend.

        """
        # %%
        print("Creating a mask for variables on sea with SIF phenology trend ...")
        main_path = Path('/nas/data/processed/people/sos/third_paper_data')

        # Sea ice mask for different radius of sea ice buffer on RACMO 5-km grid
        mask_file_path = main_path / 'freyagl_svd_analysis/input_dataset/masks'
        sea_ice_buffer = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        # Slice the racmo_sif to the same spatial extent as the phenology dataset to speed up the following calculations
        sea_ice_buffer = sea_ice_buffer.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Phenology dataset of Greenland Sea sea ice
        phenology_file_path = main_path / 'freyagl_svd_analysis/sea_ice_smb_melt_indicator'
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{phenology_file_path}/greenland_sea_sif_phenology_{phenology}_trend_theilsen.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            phenology_slope_significance_mask = xr.where(ds[f'{phenology}_trend_pvalue'] < 0.05, 1, 0).squeeze(drop=True)

            # Create sea ice buffer masks within the significant phenology trend mask and
            # within the sea ice buffer at different radius values (100, 200, 300, 400, 500, 600, 700, 800 km) on RACMO 5-km grid.
            # Empty list to store all sea-ice buffers and their corresponding radius values
            all_buffers = []
            radius_values = []
            for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                print(f"Extracting variables within {radius} km sea ice buffer at 5-km resolution ...")
                # Read the sea ice buffer
                ci_buffer_within = sea_ice_buffer.sel(radius=radius)

                phenology_slope_significance_mask_within_ci_buffer = phenology_slope_significance_mask.where(ci_buffer_within != 0)
                fig, ax = plt.subplots(1, 1)
                phenology_slope_significance_mask_within_ci_buffer.plot(ax=ax)
                ax.set_title(f"Phenology slope significance mask within {radius} km sea ice buffer at 5-km resolution")

                all_buffers.append(phenology_slope_significance_mask_within_ci_buffer)
                radius_values.append(radius)
            buffer_3d = xr.DataArray(data=np.stack(all_buffers),
                                    dims=['radius', 'rlat', 'rlon'],
                                    coords={'radius': radius_values,
                                            'rlat': sea_ice_buffer.rlat.values,
                                            'rlon': sea_ice_buffer.rlon.values,},
                                    name='sea_ice_buffer',)
            buffer_3d.attrs = dict(title=f'Sea ice buffers (within) for significant {phenology} trend on 5.5 km RACMO grid',
                                institution='University of Graz (Sonika Shahi)',
                                history=f"Created on {datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} Python",
                                Conventions='CF-1.8',)
            buffer_3d.to_netcdf(f"{mask_file_path}/ci_all_within_buffers_{phenology}_trend_racmo5km.nc", mode='w')

            # Create sea ice buffer masks within the significant phenology trend mask and
            # inbetween the sea ice buffer at different radius values (100, 200, 300, 400, 500, 600, 700, 800 km) on RACMO 5-km grid.
            # Empty list to store all sea-ice buffers and their corresponding radius values
            all_buffers = []
            radius_values = []
            for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                print(f"Extracting variables inbetween {radius} and {radius+100} km sea ice buffer at 5-km resolution ...")
                # Read the sea ice buffer
                if radius == 100:
                    # Add young sound fjord mask
                    ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
                    ci_buffer_after = sea_ice_buffer.sel(radius=radius)
                    ci_buffer_inbetween = ci_buffer_after - ci_buffer_before
                else:
                    ci_buffer_before = sea_ice_buffer.sel(radius=radius-100)
                    ci_buffer_after = sea_ice_buffer.sel(radius=radius)
                    ci_buffer_inbetween = ci_buffer_after - ci_buffer_before

                phenology_slope_significance_mask_inbetween_ci_buffer = phenology_slope_significance_mask.where(ci_buffer_inbetween != 0)
                fig, ax = plt.subplots(1, 1)
                phenology_slope_significance_mask_inbetween_ci_buffer.plot(ax=ax)
                ax.set_title(f"Phenology slope significance mask inbetween {radius} km sea ice buffer at 5-km resolution")

                all_buffers.append(phenology_slope_significance_mask_inbetween_ci_buffer)
                radius_values.append(radius)
            buffer_3d = xr.DataArray(data=np.stack(all_buffers),
                                    dims=['radius', 'rlat', 'rlon'],
                                    coords={'radius': radius_values,
                                            'rlat': sea_ice_buffer.rlat.values,
                                            'rlon': sea_ice_buffer.rlon.values,},
                                    name='sea_ice_buffer',)
            buffer_3d.attrs = dict(title=f'Sea ice buffers (inbetween) for significant {phenology} trend on 5.5 km RACMO grid',
                                institution='University of Graz (Sonika Shahi)',
                                history=f"Created on {datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} Python",
                                Conventions='CF-1.8',)
            buffer_3d.to_netcdf(f"{mask_file_path}/ci_all_inbetween_buffers_{phenology}_trend_racmo5km.nc", mode='w')

    def create_iceland_svalbard_mask():
        # %%
        # Add Iceland and Svalbard
        lsm_topo_mask = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc'
        # --- Load datasets safely ---
        with xr.open_dataset(lsm_topo_mask) as ds:
            lsm = ds['LSM_GR'].copy()
            topo = ds['Topography'].copy()

        # --- Mask Greenland ---
        topo_c = xr.where((topo > 0) & (lsm == 0), 1, np.nan)
        topo_c.loc[dict(rlon=slice(None, 5))] = np.nan  # assumes last dim is longitude-like
        topo_c.loc[dict(rlat=slice(-3, 5), rlon=slice(2.5, 7.5))] = np.nan  # assumes last dim is longitude-like
        # topo_c.plot()

        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # output_path = Path(mask_file_path)
        # output_path.mkdir(parents=True, exist_ok=True)
        topo_c.to_netcdf(mask_file_path / 'iceland_svalbard_mask.nc', mode='w')
    # %%
class ExtractVariablesFromRacmo:
    """Class to extract Freya Glacier variables using prepared RACMO masks.

    This class contains methods to extract and save Freya Glacier variables using RACMO masks at 1-km and 5-km resolutions.
    """
    @staticmethod
    def extract_variable_freya_1km():
        """ Extract and process RACMO variables over whole the Freya Glacier outline at 1 km resolution.

        """
        # %%
        # ---------- Freya on whole mask ----------
        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')

        # Freya glacier mask from racmo
        freya_glacier_on_racmo_clavering_island = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_racmo_1km_clavering_island.nc")

        # ---------- Freya on point scale ----------
        # Read metadata to extract RACMO variables at point scale
        # Extract Freya Glacier's Station location
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)
        station = 'FRE2 (promice type, tripod)'

        # Read RACMO files for metadata
        mask_1km = f"{mask_file_path}/Icemask_Topo_Iceclasses_lon_lat_average_1km_clavering_island.nc"
        racmo_1km_mask_lat = xr.open_dataset(mask_1km)['LAT']
        racmo_1km_mask_lon = xr.open_dataset(mask_1km)['LON']

        # Find the nearest indices
        iy_min, ix_min = kdtree_fast(racmo_1km_mask_lat.values, racmo_1km_mask_lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(racmo_1km_mask_lat.coords['x'].values, racmo_1km_mask_lat.coords['y'].values)

        # ---------- Freya on Accumulation Area ----------
        freya_glacier_accumulation_area_on_racmo_1km_clavering_island = xr.open_dataarray(f"{mask_file_path}/freya_glacier_accumulation_area_on_racmo_1km_clavering_island.nc")

        # ---------- Freya on Ablation Area ----------
        freya_glacier_ablation_area_on_racmo_1km_clavering_island = xr.open_dataarray(f"{mask_file_path}/freya_glacier_ablation_area_on_racmo_1km_clavering_island.nc")

        # %%
        # List variables to mask with freya outline in racmo
        var_names = ['snowfall', 'precip', 't2m', 'runoff', 'smb']
        for var_name in var_names:
            # %%
            var_name = 'smb'
            print(f"Extracting variable: {var_name} at 1-km resolution ...")
            # Create empty DataFrame with this index and columns
            time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D', name='time')
            columns = [f"{var_name}_whole_racmo", f"{var_name}_point_racmo", f"{var_name}_acc_racmo", f"{var_name}_ab_racmo"]
            var_da_masked_all_df = pd.DataFrame(index=time_index, columns=columns)

            # Read variables to be masked
            var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_1km/racmo_clavering_domain/rearranged/onefile')
            var_da = xr.open_dataarray(f"{var_file_path}/{var_name}_all.nc")

            # ---------- Freya on whole mask ----------
            # Mask all values outside of the freya outline
            # # Broadcast mask into 3D (same shape as var_da)
            # field3d_mask = xr.broadcast(var_da, freya_glacier_on_racmo_clavering_island == 0)[1].astype(bool)
            # # Apply mask without modifying var_da in place
            # var_da_masked = var_da.where(~field3d_mask)
            var_da_masked = var_da.where(freya_glacier_on_racmo_clavering_island != 0)
            var_da_masked_mean = var_da_masked.mean(dim=['x', 'y'])
            var_da_masked_df = var_da_masked_mean.to_dataframe()
            var_da_masked_df = var_da_masked_df.resample('D').mean()
            var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} whole extracted from 1-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_whole_racmo"] = var_da_masked_df[var_name]

            # ---------- Freya on point scale ----------
            var_da_point = var_da.sel(x=X[iy_min, ix_min], y=Y[iy_min, ix_min])
            var_da_point_df = var_da_point.to_dataframe()
            var_da_point_df.drop(['x', 'y'], axis=1, inplace=True)
            var_da_point_df.rename(columns={var_da_point_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} point extracted from 1-km RACMO ...")
            # Assign the column
            var_da_masked_all_df[f"{var_name}_point_racmo"] = var_da_point_df[var_name]

            # ---------- Freya on Accumulation Area ----------
            # Mask all values outside of the freya accumulation outline
            # field3d_mask = xr.broadcast(var_da, freya_glacier_accumulation_area_on_racmo_1km_clavering_island == 0)[1]
            # # Apply mask without modifying var_da in place
            # var_da_masked_acc = var_da.where(~field3d_mask)
            var_da_masked_acc = var_da.where(freya_glacier_accumulation_area_on_racmo_1km_clavering_island != 0)
            var_da_masked_acc_mean = var_da_masked_acc.mean(dim=['x', 'y'])
            var_da_masked_acc_df = var_da_masked_acc_mean.to_dataframe()
            var_da_masked_acc_df = var_da_masked_acc_df.resample('D').mean()
            var_da_masked_acc_df.rename(columns={var_da_masked_acc_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} accumulation area extracted from 1-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_acc_racmo"] = var_da_masked_acc_df[var_name]

            # ---------- Freya on ablation Area ----------
            # # Mask all values outside of the freya ablation outline
            # field3d_mask = xr.broadcast(var_da, freya_glacier_ablation_area_on_racmo_1km_clavering_island == 0)[1]
            # # Apply mask without modifying var_da in place
            # var_da_masked_ab = var_da.where(~field3d_mask)
            var_da_masked_ab = var_da.where(freya_glacier_ablation_area_on_racmo_1km_clavering_island != 0)
            var_da_masked_ab_mean = var_da_masked_ab.mean(dim=['x', 'y'])
            var_da_masked_ab_df = var_da_masked_ab_mean.to_dataframe()
            var_da_masked_ab_df = var_da_masked_ab_df.resample('D').mean()
            var_da_masked_ab_df.rename(columns={var_da_masked_ab_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} ablation area extracted from 1-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_ab_racmo"] = var_da_masked_ab_df[var_name]

            base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
            var_da_masked_all_df.to_csv(f"{base_output_path}/{var_name}_masked_freya_1km.csv")

            # # UNCOMMENT if you want to test the code
            # fig, ax = plt.subplots(1)
            # # freya_glacier_on_racmo_acc.plot(ax=ax)
            # # var_da_masked[180, :, :].plot(ax=ax)
            # var_da_masked_acc[180, :, :].plot(ax=ax)
            # # freya_glacier_on_racmo.plot(ax=ax)
            # xlims = [693044, 701544] # 702044-500; mid-way of 1-km
            # ylims = [-1556596, -1548596]
            # ax.set_xlim(xlims)
            # ax.set_ylim(ylims)

            # %%
    # %%
    @staticmethod
    def extract_variable_whole_freya_5km():
        """ Extract and process RACMO variables over whole the Freya Glacier outline at 5 km resolution.

        """
        # %%
        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')

        # -------------- Freya glacier mask on racmo at 5 km ---------------
        # Freya glacier mask from racmo
        freya_glacier_on_racmo_clavering_island = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_racmo_5km_clavering_island.nc")

        # -------------- Point mask of closest Freya glacier mask racmo at 5 km ---------------
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)

        # -------------- Freya glacier mask on qgis at 5 km ---------------
        freya_glacier_on_qgis_clavering_island = xr.open_dataarray(f"{mask_file_path}/freya_glacier_on_qgis_5km_clavering_island.nc")

        # var_names = ['t2m', 'swsd', 'q2m', 'rh2m', 'latf', 'senf', 'ff10m', 'lwsd', 'lwsu']
        # var_names = ['tskin']
        # var_names = ['lwsd', 'swsu', 'clcov']
        var_names = ['alb']
        for var_name in var_names:
            print(f"Extracting variable: {var_name} at 5-km resolution ...")
            # Create empty DataFrame with this index and columns
            time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D', name='time')
            columns = [f"{var_name}_whole_racmo", f"{var_name}_freya_point_racmo", f"{var_name}_closest_point_racmo", f"{var_name}_whole_qgis"]
            var_da_masked_all_df = pd.DataFrame(index=time_index, columns=columns)

            # Read variables to be masked
            var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_clavering_domain/rearranged/onefile')
            var_da = xr.open_dataset(f"{var_file_path}/{var_name}_all.nc")[var_name]
            var_da = var_da.sel(height=0)

            # -------------- Freya glacier mask on racmo at 5 km ---------------
            field3d_mask = xr.broadcast(var_da, freya_glacier_on_racmo_clavering_island == 0)[1]
            # Apply mask without modifying var_da in place
            var_da_masked = var_da.where(~field3d_mask)
            var_da_masked_mean = var_da_masked.mean(dim=['rlon', 'rlat']).drop_vars('height')
            # var_da_masked_df = var_da_masked_mean.to_series().to_frame(name=var_name)
            var_da_masked_df = var_da_masked_mean.to_dataframe()
            var_da_masked_df = var_da_masked_df.resample('D').mean()
            var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} whole RACMO extracted from 5-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_whole_racmo"] = var_da_masked_df[var_name]

            # -------------- Point mask of Freya Glacier AWS in racmo at 5 km ---------------
            # At freya_aws_point
            var_da_freya_point = var_da.sel(rlon=freya_glacier_point_mask_5km_clavering_island.loc['freya_point', 'rlon'], rlat=freya_glacier_point_mask_5km_clavering_island.loc['freya_point', 'rlat'])
            var_da_freya_point_df = var_da_freya_point.to_dataframe()
            var_da_freya_point_df = pd.DataFrame(var_da_freya_point_df[var_name])
            var_da_freya_point_df.rename(columns={var_da_freya_point_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} point extracted from 5-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_freya_point_racmo"] = var_da_freya_point_df[var_name]

            # -------------- Point mask of closest point to Freya glacier in racmo at 5 km ---------------
            # At closest point to freya_aws_point
            var_da_closest_freya_point = var_da.sel(rlon=freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', 'rlon'], rlat=freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', 'rlat'])
            var_da_closest_freya_point_df = var_da_closest_freya_point.to_dataframe()
            var_da_closest_freya_point_df = pd.DataFrame(var_da_closest_freya_point_df[var_name])
            var_da_closest_freya_point_df.rename(columns={var_da_closest_freya_point_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} closest point extracted from 5-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_closest_point_racmo"] = var_da_closest_freya_point_df[var_name]

            # -------------- Freya glacier mask on QGIS at 5 km ---------------
            field3d_mask = xr.broadcast(var_da, freya_glacier_on_qgis_clavering_island == 0)[1]
            # Apply mask without modifying var_da in place
            var_da_masked = var_da.where(~field3d_mask)
            var_da_masked_mean = var_da_masked.mean(dim=['rlon', 'rlat']).drop_vars('height')
            # var_da_masked_df = var_da_masked_mean.to_series().to_frame(name=var_name)
            var_da_masked_df = var_da_masked_mean.to_dataframe()
            var_da_masked_df = var_da_masked_df.resample('D').mean()
            var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: var_name}, inplace=True)

            logger.info(f"Copying {var_name} whole QGIS extracted from 5-km RACMO ...")
            var_da_masked_all_df[f"{var_name}_whole_qgis"] = var_da_masked_df[var_name]

            # Save the masked variables
            base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
            var_da_masked_all_df.to_csv(f"{base_output_path}/{var_name}_masked_freya_5km.csv")

            # %%
            # # UNCOMMENT if you want to test the code
            # fig, ax = plt.subplots(1)
            # # # freya_glacier_on_racmo_acc.plot(ax=ax)
            # var_da_masked[180, :, :].plot(ax=ax)
            # # freya_glacier_on_racmo_clavering_island[180, :, :].plot(ax=ax)
            # # freya_glacier_on_racmo_clavering_island.plot(ax=ax)
            # # freya_glacier_on_qgis_clavering_island.plot(ax=ax)
            # # # freya_glacier_on_racmo.plot(ax=ax)
            # # Set x and y limits
            # xlims = [4.25 + 0.05, 4.6 - 0.05]
            # ylims = [2.75 + 0.05, 3.05]

            # ax.set_xlim(xlims)
            # ax.set_ylim(ylims)


    @staticmethod
    def extract_5km_variable_within_greenland_sea():
        """Extract the 5-km surface and atmospheric variables within the Greenland Sea.

        """
        # %%
        # Read Greenland Sea mask
        gsea_file = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/greenland_sea_mask_5km_updated_coordinates.nc'
        gsea_mask = xr.open_dataarray(gsea_file)

        # Create empty DataFrame with this index and columns
        # variables = ['ci', 'tskin', 'q2m', 'latf', 'senf', 't2m', 'ff10m', 'lwsd', 'lwsu']
        variables = ['ci',]
        time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D', name='time')
        var_da_masked_all_df = pd.DataFrame(index=time_index, columns=variables)

        for variable in variables:
            # variable = 'ci'
            # Read variables to be masked
            logger.info(f"Masking {variable} within the Greenland Sea...")

            var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
            var_da = xr.open_dataset(f"{var_file_path}/{variable}_all.nc")[variable]

            # -------------- sea ice buffer mask on racmo at 5 km ---------------
            field3d_mask = xr.broadcast(var_da, gsea_mask == 0)[1]
            # Apply mask without modifying var_da in place
            var_da_masked = var_da.where(~field3d_mask)

            if variable == 'ci':
                var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon'])
            else:
                var_da_masked_mean = var_da_masked.mean(dim=['rlat', 'rlon'])

            # var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon']).squeeze(drop=True) #.reset_coords('height', drop=True)
            # Case 1: if 'height' is a dimension -> drop it
            if 'height' in var_da_masked_mean.dims:
                var_da_masked_mean = var_da_masked_mean.squeeze('height', drop=True)

            # Case 2: if 'height' is only a coordinate -> remove it
            elif 'height' in var_da_masked_mean.coords:
                var_da_masked_mean = var_da_masked_mean.reset_coords('height', drop=True)

            var_da_masked_df = var_da_masked_mean.to_dataframe()
            var_da_masked_df = var_da_masked_df.resample('D').mean()
            var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: variable}, inplace=True)
            var_da_masked_all_df[variable] = var_da_masked_df[variable]

        # Save the masked variables
        logger.info(f"Saving all the 5-km masked variables within the Greenland Sea...")
        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_within_greenland_sea_sea_ice.csv")

        # %%
    @staticmethod
    def extract_5km_variable_within_sea_ice_buffer():
        """Extract the 5-km surface and atmospheric variables on the sea ice.

        """
        # %%
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']

        for radius in ['young_sound', 100, 200, 300, 400, 500, 600, 700, 800]:
            print(f"Extracting variables within {radius} km sea ice buffer at 5-km resolution ...")
            if radius == 'young_sound':
                # Add young sound fjord mask
                ci_buffer = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
            else:
                # Read the sea ice buffer
                ci_buffer = sea_ice_buffer_da.sel(radius=radius)

            # Create empty DataFrame with this index and columns
            variables = ['ci', 'tskin', 'q2m', 'latf', 'senf', 't2m', 'ff10m', 'lwsd', 'lwsu']
            time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D', name='time')
            var_da_masked_all_df = pd.DataFrame(index=time_index, columns=variables)

            for variable in variables:
                # variable = 'ci'
                # Read variables to be masked
                logger.info(f"Masking {variable} within {radius} sea ice...")

                var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
                var_da = xr.open_dataset(f"{var_file_path}/{variable}_all.nc")[variable]

                # -------------- sea ice buffer mask on racmo at 5 km ---------------
                field3d_mask = xr.broadcast(var_da, ci_buffer == 0)[1]
                # Apply mask without modifying var_da in place
                var_da_masked = var_da.where(~field3d_mask)

                if variable == 'ci':
                    var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon'])
                else:
                    var_da_masked_mean = var_da_masked.mean(dim=['rlat', 'rlon'])

                # var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon']).squeeze(drop=True) #.reset_coords('height', drop=True)
                # Case 1: if 'height' is a dimension -> drop it
                if 'height' in var_da_masked_mean.dims:
                    var_da_masked_mean = var_da_masked_mean.squeeze('height', drop=True)

                # Case 2: if 'height' is only a coordinate -> remove it
                elif 'height' in var_da_masked_mean.coords:
                    var_da_masked_mean = var_da_masked_mean.reset_coords('height', drop=True)

                if 'radius' in var_da_masked_mean.dims:
                    var_da_masked_mean = var_da_masked_mean.squeeze('radius', drop=True)
                elif 'radius' in var_da_masked_mean.coords:
                    var_da_masked_mean = var_da_masked_mean.reset_coords('radius', drop=True)

                var_da_masked_df = var_da_masked_mean.to_dataframe()
                var_da_masked_df = var_da_masked_df.resample('D').mean()
                var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: variable}, inplace=True)
                var_da_masked_all_df[variable] = var_da_masked_df[variable]

            # Save the masked variables
            logger.info(f"Saving all the 5-km masked variables within {radius} sea ice...")
            base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
            var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_within_{radius}_sea_ice.csv")

    @staticmethod
    def extract_5km_variable_inbetween_sea_ice_buffers():
        """Extract the 5-km surface and atmospheric variables on the sea ice inbetween the buffers.

        """
        # %%
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']

        for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
            if radius == 100:
                # Add young sound fjord mask
                ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
                ci_buffer_after = sea_ice_buffer_da.sel(radius=radius)
                ci_buffer = ci_buffer_after - ci_buffer_before
            else:
                ci_buffer_before = sea_ice_buffer_da.sel(radius=radius-100)
                ci_buffer_after = sea_ice_buffer_da.sel(radius=radius)
                ci_buffer = ci_buffer_after - ci_buffer_before

            # Create empty DataFrame with this index and columns
            variables = ['ci', 'tskin', 'q2m', 'latf', 'senf', 't2m', 'ff10m', 'lwsd', 'lwsu']
            time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D', name='time')
            var_da_masked_all_df = pd.DataFrame(index=time_index, columns=variables)

            for variable in variables:
                # variable = 'ci'
                # Read variables to be masked
                if radius == 100:
                    logger.info(f"Masking {variable} between YS - {radius} sea ice...")
                else:
                    logger.info(f"Masking {variable} between {radius-100} - {radius} sea ice...")

                var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
                var_da = xr.open_dataset(f"{var_file_path}/{variable}_all.nc")[variable]

                # -------------- sea ice buffer mask on racmo at 5 km ---------------
                field3d_mask = xr.broadcast(var_da, ci_buffer == 0)[1]
                # Apply mask without modifying var_da in place
                var_da_masked = var_da.where(~field3d_mask)

                if variable == 'ci':
                    var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon'])
                else:
                    var_da_masked_mean = var_da_masked.mean(dim=['rlat', 'rlon'])

                # var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon']).squeeze(drop=True) #.reset_coords('height', drop=True)
                # Case 1: if 'height' is a dimension -> drop it
                if 'height' in var_da_masked_mean.dims:
                    var_da_masked_mean = var_da_masked_mean.squeeze('height', drop=True)

                # Case 2: if 'height' is only a coordinate -> remove it
                elif 'height' in var_da_masked_mean.coords:
                    var_da_masked_mean = var_da_masked_mean.reset_coords('height', drop=True)

                if 'radius' in var_da_masked_mean.dims:
                    var_da_masked_mean = var_da_masked_mean.squeeze('radius', drop=True)
                elif 'radius' in var_da_masked_mean.coords:
                    var_da_masked_mean = var_da_masked_mean.reset_coords('radius', drop=True)

                var_da_masked_df = var_da_masked_mean.to_dataframe()
                var_da_masked_df = var_da_masked_df.resample('D').mean()
                var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: variable}, inplace=True)
                var_da_masked_all_df[variable] = var_da_masked_df[variable]

            # Save the masked variables
            logger.info(f"Saving all the 5-km masked variables within {radius} sea ice...")
            base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
            if radius == 100:
                var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_between_YS_{radius}_sea_ice.csv")
            else:
                var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_between_{radius-100}_{radius}_sea_ice.csv")

        # %%
        # # UNCOMMENT to test the script
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # ds = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        # zoom_xlim = (4, 5.5 + 1.5)
        # zoom_ylim = (1, 4 + 1.5)

        # # ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        # # ci_buffer_after = sea_ice_buffer_da.sel(radius=100)
        # # ci_inbetween = ci_buffer_after - ci_buffer_before

        # radius = 300
        # ci_buffer_before = sea_ice_buffer_da.sel(radius=radius-100)
        # ci_buffer_after = sea_ice_buffer_da.sel(radius=radius)
        # ci_buffer = ci_buffer_after - ci_buffer_before

        # fig, ax = plt.subplots(1)
        # ci_buffer_before.plot(ax=ax)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)

        # fig, ax = plt.subplots(1)
        # ci_buffer_after.plot(ax=ax)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)

        # fig, ax = plt.subplots(1)
        # ci_buffer.plot(ax=ax)
        # ax.set_xlim(zoom_xlim)
        # ax.set_ylim(zoom_ylim)

    def extract_5km_variable_within_inbetween_significant_sif_phenology_trend():
        """Extract the 5-km surface and atmospheric variables within and inbetween significant sea ice phenology trend.

        """
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            logger.info(f"Extracting variables within significant {phenology} trend at 5-km resolution ...")
            for mask_area in ['within_buffers', 'inbetween_buffers']:
                sea_ice_mask = xr.open_dataset(f"{main_path}/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                    sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)['sea_ice_buffer']
                    # Create empty DataFrame with this index and columns
                    variables = ['ci', 'tskin', 'q2m', 'latf', 'senf', 't2m', 'ff10m', 'lwsd', 'lwsu']
                    time_index = pd.date_range(start='1990-01-01', end='2020-12-31', freq='D')
                    var_da_masked_all_df = pd.DataFrame(index=time_index, columns=variables)
                    for variable in variables:
                        logger.info(f"Masking {variable} {mask_area} {radius} km sea ice for significant {phenology} trend...")
                        # Read variables to be masked
                        var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
                        var_da = xr.open_dataset(f"{var_file_path}/{variable}_all.nc")[variable]
                        var_da = var_da.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
                        # Mask the area with non-significant phenology trend on racmo at 5 km ---------------
                        var_da_masked = var_da.where(sea_ice_mask_radius == 1)

                        if variable == 'ci':
                            var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon'])
                        else:
                            var_da_masked_mean = var_da_masked.mean(dim=['rlat', 'rlon'])

                        # var_da_masked_mean = var_da_masked.median(dim=['rlat', 'rlon']).squeeze(drop=True) #.reset_coords('height', drop=True)

                        # Case 1: if 'height' is a dimension -> drop it
                        if 'height' in var_da_masked_mean.dims:
                            var_da_masked_mean = var_da_masked_mean.squeeze('height', drop=True)

                        # Case 2: if 'height' is only a coordinate -> remove it
                        elif 'height' in var_da_masked_mean.coords:
                            var_da_masked_mean = var_da_masked_mean.reset_coords('height', drop=True)

                        if 'radius' in var_da_masked_mean.dims:
                            var_da_masked_mean = var_da_masked_mean.squeeze('radius', drop=True)
                        elif 'radius' in var_da_masked_mean.coords:
                            var_da_masked_mean = var_da_masked_mean.reset_coords('radius', drop=True)

                        var_da_masked_df = var_da_masked_mean.to_dataframe()
                        var_da_masked_df = var_da_masked_df.resample('D').mean()
                        var_da_masked_df.rename(columns={var_da_masked_df.columns[0]: variable}, inplace=True)
                        var_da_masked_all_df[variable] = var_da_masked_df[variable]

                    # Save the masked variables
                    logger.info(f"Saving all the 5-km masked variables {mask_area} {radius} sea ice...")
                    base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
                    if mask_area == 'within_buffers':
                        var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_within_{radius}_{phenology}_sea_ice.csv")
                    elif mask_area == 'inbetween_buffers':
                        if radius == 100:
                            var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_between_YS_{radius}_{phenology}_sea_ice.csv")
                        else:
                            var_da_masked_all_df.to_csv(f"{base_output_path}/masked_5km_variables_between_{radius-100}_{radius}_{phenology}_sea_ice.csv")

    # %%
    def extract_5km_variable_at_zac():
        """Extract the 5-km surface and atmospheric variables at Zackenberg Station.

        """
        # %%
        # Read metadata to extract RACMO variables at point scale
        # Extract Zackenberg Station location
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Read RACMO files for metadata
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        mask_5km = f"{mask_file_path}/FGRN055_Masks_5.5km_updated_coordinates.nc"
        racmo_5km_mask_lat = xr.open_dataset(mask_5km)['lat']
        racmo_5km_mask_lon = xr.open_dataset(mask_5km)['lon']

        # Find the nearest indices
        station = 'ZAC'
        iy_min, ix_min = kdtree_fast(racmo_5km_mask_lat.values, racmo_5km_mask_lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(racmo_5km_mask_lat.coords['rlon'].values, racmo_5km_mask_lat.coords['rlat'].values)
        zac_rlon, zac_rlat = X[iy_min, ix_min], Y[iy_min, ix_min]

        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_zac')
        var_names = ['q2m', 't2m', 'tskin', 'ff10m', 'u10m', 'v10m', 'swsd', 'rh2m', 'latf', 'senf', 'lwsd', 'lwsu']
        for var_name in var_names:
            # Read variables to be masked
            var_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_clavering_domain/rearranged/onefile')
            var_da = xr.open_dataset(f"{var_file_path}/{var_name}_all.nc")[var_name]
            var_da = var_da.sel(height=0)

            # ---------- ZAC on point scale ----------
            var_da_point = var_da.sel(rlat=zac_rlat, rlon=zac_rlon)
            var_da_point_df = var_da_point.to_dataframe()
            var_da_point_df.drop(['height', 'rlat', 'rlon', 'lat', 'lon'], axis=1, inplace=True)
            var_da_point_df.rename(columns={var_da_point_df.columns[0]: var_name}, inplace=True)
            var_da_point_df.to_csv(f"{base_output_path}/{var_name}_point_zac_5km.csv")

        # %%
        # # UNCOMMENT if you want to test the zac location in RACMO
        # Find the nearest indices
        # station = 'FRE2 (promice type, tripod)'
        # iy_min, ix_min = kdtree_fast(racmo_5km_mask_lat.values, racmo_5km_mask_lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # # Create a mesh grid for plotting
        # X, Y = np.meshgrid(racmo_5km_mask_lat.coords['rlon'].values, racmo_5km_mask_lat.coords['rlat'].values)
        # fre_rlon, fre_rlat = X[iy_min, ix_min], Y[iy_min, ix_min]

        # racmo_5km_mask_lsm = xr.open_dataset(mask_5km)['LSM_GR']
        # racmo_5km_mask_ice = xr.open_dataset(mask_5km)['Promicemask']
        # racmo_5km_mask_topo = xr.open_dataset(mask_5km)['Topography']

        # x, y = racmo_5km_mask_lsm.coords['rlon'], racmo_5km_mask_lsm.coords['rlat']
        # xx, yy = np.meshgrid(x, y)
        # fig, ax = plt.subplots(1)
        # # ax.contourf(xx, yy, racmo_5km_mask_lsm.values, levels=[-0.5, 0.5, 1.5], colors=['lightblue', 'lightgrey'], alpha=0.5)
        # ax.contourf(xx, yy, racmo_5km_mask_lsm.values)
        # ax.contour(xx, yy, racmo_5km_mask_ice.values, alpha=0.5)
        # ax.plot(zac_rlon, zac_rlat, marker='o', color='red', markersize=3, label='ZAC Station')
        # ax.plot(fre_rlon, fre_rlat, marker='^', color='blue', markersize=3, label='FRE2 Station')

        # ax.set_xlim([3, 6])
        # ax.set_ylim([2.75, 3.5])

        # print(racmo_5km_mask_topo.sel(rlon=zac_rlon, rlat=zac_rlat).values)
        # print(racmo_5km_mask_topo.sel(rlon=fre_rlon, rlat=fre_rlat).values)

        # # station = 'ZAC'
        # # zac_rlon, zac_rlat = rotated_grid_transform(lon=stations_data.loc['lon (W)', station], lat=stations_data.loc['lat (N)', station], option='reg2rot', SP_coor=(-37.5,18))
        # # print(zac_rlon, zac_rlat)
        # # print(racmo_5km_mask_topo.interp(rlon=zac_rlon, rlat=zac_rlat).values)
        # # station = 'FRE2 (promice type, tripod)'
        # # fre_rlon, fre_rlat = rotated_grid_transform(lon=stations_data.loc['lon (W)', station], lat=stations_data.loc['lat (N)', station], option='reg2rot', SP_coor=(-37.5,18))
        # # print(fre_rlon, fre_rlat)
        # # print(racmo_5km_mask_topo.interp(rlon=fre_rlon, rlat=fre_rlat).values)
        # %%

    def extract_smb_within_significant_insignificance_amplitude():
        """ Extract SMB within significant and insignificant amplitude regions of the right component of the SVD analysis.

        The results are based on the analytical SVD analysis of the 'ci' and 'smb' variables, where the right component
        amplitude and phase are used to identify significant and insignificant regions.
        The extracted SMB data is then averaged over the spatial dimensions and saved for further analysis.

        """
        # %%
        # Set parameters for the SVD analysis
        total_modes = 3
        rotation_modes = 3
        mode = 1
        rotation = 'varimax'
        left_var_name = 'ci'
        coarsen_level_left = 0 # Left variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        right_var_name = 'smb'
        coarsen_level_right = 4 # Right variables are at original resolution; 'ci' and 'z0500' for MJJAS, but coarsened to level 2 for AMJJASO to speed up plotting.
        print(f"Extracting extract_smb_within_significant_insignificance_amplitude for {left_var_name} vs {right_var_name} with coarsen level {coarsen_level_left} and {coarsen_level_right} ...")

        main_path = Path(f"/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/{left_var_name}_vs_{right_var_name}")
        # Read component amplitude
        right_component_amplitude = xr.open_dataarray(f"{main_path}/right_component_amplitudes_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_{rotation}_total_mode_{total_modes}_rotation_mode_{rotation_modes}.nc")
        right_component_amplitude = right_component_amplitude.sel(mode=1)
        right_component_amplitude_sig = right_component_amplitude/right_component_amplitude.max()

        right_component_phase = xr.open_dataarray(f"{main_path}/right_component_phases_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_1990to2019_xeofs_{rotation}_total_mode_{total_modes}_rotation_mode_{rotation_modes}.nc")
        right_component_phase = right_component_phase.sel(mode=1)
        right_component_phase = right_component_phase.where(right_component_phase == 0, np.nan)
        right_component_phase_sig = xr.where(right_component_amplitude_sig > 0.25, 1, 0)

        # Read the original SMB dataset
        read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile'

        smb_masked_sig = pd.DataFrame()
        smb_masked_insig = pd.DataFrame()
        for year in range(1990, 2020):
            # year = 1990
            da_month = xr.open_dataarray(f"{read_file_path}/smb_{year}_trended_coarsened_4.nc")
            da_month_mask_1 = da_month.where(right_component_phase_sig == 1, np.nan)
            da_month_mask_1_mean = da_month_mask_1.mean(dim=['x', 'y'])
            da_month_mask_1_mean_df = da_month_mask_1_mean.to_dataframe(name='smb')
            da_month_mask_1_mean_df = pd.DataFrame(da_month_mask_1_mean_df['smb'])

            da_month = xr.open_dataarray(f"{read_file_path}/smb_{year}_trended_coarsened_4.nc")
            da_month_mask_0 = da_month.where(right_component_phase_sig == 0, np.nan)
            da_month_mask_0_mean = da_month_mask_0.mean(dim=['x', 'y'])
            da_month_mask_0_mean_df = da_month_mask_0_mean.to_dataframe(name='smb')

            smb_masked_sig = pd.concat([smb_masked_sig, da_month_mask_1_mean_df])
            smb_masked_insig = pd.concat([smb_masked_insig, da_month_mask_0_mean_df])

        output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        smb_masked_sig.to_csv(f"{output_path}/masked_smb_within_significant_amplitude.csv")
        smb_masked_insig.to_csv(f"{output_path}/masked_smb_within_insignificant_amplitude.csv")

        # %%
    def mask_sea_ice_with_greenland_sea():
        """ Mask the sea ice within the Greenland Sea and calculate 7 days running mean using the RACMO 5-km resolution dataset to
        increase the runtime eficiency of the code on FG1_data_analysis.py to plot the
        sea ice phenology.

        """
        # %%
        input_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        racmo_sif = xr.open_dataset(f'{input_file_path}/ci_all.nc')['ci'] # Try without chunking
        racmo_sif = racmo_sif.squeeze('height', drop=True)
        # Drop coordinates from the DataArray
        coords_to_drop = ['height', 'lat', 'lon', 'pressure']
        racmo_sif = racmo_sif.drop_vars([c for c in coords_to_drop if c in racmo_sif.coords])

        # Apply mask
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        greenland_sea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        racmo_sif = racmo_sif.where(greenland_sea_mask != 0)

        # Subset to the region of interest (Greenland Sea)
        racmo_sif = racmo_sif.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
        racmo_sif[0, :, :].plot()

        # 7-day running mean (current day + previous 6)
        racmo_sif_7d = racmo_sif.rolling(time=7, min_periods=7).mean()

        # Define Sea-ice year: Jan–Feb belong to previous year
        sif_year = (racmo_sif.time.dt.year - (racmo_sif.time.dt.month <= 2))
        # Add coordinates to the 7-day running mean DataArray
        racmo_sif_7d = racmo_sif_7d.assign_coords(sif_year=('time', sif_year.data))
        # racmo_sif = racmo_sif.assign_coords(year=xr.DataArray(racmo_sif.time.dt.year.data, dims='time'), doy=xr.DataArray(racmo_sif.time.dt.dayofyear.data, dims='time'))

        # Keep only requested years
        racmo_sif_7d = racmo_sif_7d.where(racmo_sif_7d.sif_year.isin(range(1990, 2021)), drop=True)
        # racmo_sif_7d[0, :, :].plot()

        # Choose only those SI years with above 0.15 SIF at least once during the year,
        # to avoid calculating phenology metrics for grid cells that are always ice-free.
        # True where SIC is below 0.15 for ALL days in the sif_year
        always_below_015 = (racmo_sif_7d < 0.15).groupby('sif_year').all(dim='time')
        racmo_sif_7d = racmo_sif_7d.groupby('sif_year').where(~always_below_015)
        racmo_sif_7d.to_netcdf(f"{input_file_path}/ci_all_masked_greenland_sea_7d.nc")

        # %%
        input_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        # Read original Sea ice data
        racmo_sif = xr.open_dataset(f'{input_file_path}/ci_all.nc')['ci'] # Try without chunking
        racmo_sif = racmo_sif.squeeze('height', drop=True)
        racmo_sif = racmo_sif.where(greenland_sea_mask != 0)
        racmo_sif.to_netcdf(f"{input_file_path}/ci_all_masked_greenland_sea.nc")
        # %%
class ExtractZ500VariablesFromRacmo:
    """Class to extract Geopotential height at 500 hPa using prepared RACMO masks.

    This class contains methods to extract and save Geopotential height at 500 hPa using RACMO masks at 5-km resolutions.
    """
    @staticmethod
    def extract_z500hpa_5km():
        """ Extract and process Geopotential height at 500 hPa over whole the Freya Glacier outline at 5 km resolution.

        """
        # %%
        # Read z500hPa mask
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        z500_mask = xr.open_dataarray(f"{mask_file_path}/z500hPa_mask.nc")

        # Read MJJAS z500hPa
        read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        # da_z500 = xr.open_dataset(read_file_path/'z0500_MJJAS_trended_coarsened_0.nc')['z0500']
        da_z500 = xr.open_dataset(read_file_path/'z0500_all.nc')['z0500']

        # Case 2: if 'height' is only a coordinate -> remove it
        if 'pressure' in da_z500.coords:
            da_z500 = da_z500.reset_coords('pressure', drop=True)

        # Mask z500 hPa with z500hPa mask on racmo at 5 km ---------------
        field3d_mask = xr.broadcast(da_z500, z500_mask == 0)[1]
        # Apply mask without modifying da_z500 in place
        da_z500_masked = da_z500.where(~field3d_mask)

        # Calculate mean z500 hPa over the masked area
        da_z500_masked_mean = da_z500_masked.mean(dim=['rlat', 'rlon'])
        da_z500_masked_mean_df = da_z500_masked_mean.to_dataframe()
        da_z500_masked_mean_df = da_z500_masked_mean_df.resample('D').mean()
        da_z500_masked_mean_df.rename(columns={da_z500_masked_mean_df.columns[0]: 'z500'}, inplace=True)

        # Save the masked variables
        logger.info(f"Saving all the 5-km z500 hPa masked over eastern part of Greenland and Greenland Sea...")
        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_z500')
        da_z500_masked_mean_df.to_csv(f"{base_output_path}/masked_z500hPa_east_greenland.csv")

        # # UNCOMMENT if you want to test the code
        # fig, ax = plt.subplots(1)
        # da_z500_masked[180, :, :].plot(ax=ax)
        # %%

    def extract_z500hpa_5km_rlon_slice():
        """ Extract and process Geopotential height at 500 hPa from the eastern part of Greenland and Greenland Sea at 5 km resolution.

        """
        # %%
        # Read MJJAS z500hPa
        read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        # da_z500 = xr.open_dataset(read_file_path/'z0500_MJJAS_trended_coarsened_0.nc')['z0500']
        da_z500 = xr.open_dataset(read_file_path/'z0500_all.nc')['z0500']

        # Case 2: if 'height' is only a coordinate -> remove it
        if 'pressure' in da_z500.coords:
            da_z500 = da_z500.reset_coords('pressure', drop=True)
        da_z500.loc[dict(rlon=slice(None, -2.5))] = np.nan  # assumes last dim is longitude-like

        # Calculate mean z500 hPa over the masked area
        da_z500_masked_mean = da_z500.mean(dim=['rlat', 'rlon'])
        da_z500_masked_mean_df = da_z500_masked_mean.to_dataframe()
        da_z500_masked_mean_df = da_z500_masked_mean_df.resample('D').mean()
        da_z500_masked_mean_df.rename(columns={da_z500_masked_mean_df.columns[0]: 'z500'}, inplace=True)

        # Save the masked variables
        logger.info(f"Saving all the 5-km z500 hPa masked over eastern part of Greenland and Greenland Sea...")
        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_z500')
        da_z500_masked_mean_df.to_csv(f"{base_output_path}/masked_z500hPa_east_greenland_using_rlon_slice.csv")

        # fig, ax = plt.subplots(1)
        # da_z500[0, :, :].plot(ax=ax)

        # lsm_topo_mask = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc'
        # # --- Load datasets safely ---
        # with xr.open_dataset(lsm_topo_mask) as ds:
        #     lsm = ds['LSM_GR'].copy()
        #     topo = ds['Topography'].copy()
        # lsm.loc[dict(rlon=slice(None, -2.5))] = 0  # assumes last dim is longitude-like
        # fig, ax = plt.subplots(1)
        # lsm.plot(ax=ax)

    def extract_z500hpa_5km_in_greenland_sea():
        """ Extract and process Geopotential height at 500 hPa from Greenland Sea at 5 km resolution.

        """
        # %%
        # Read Greenland Sea mask
        gsea_file = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/greenland_sea_mask_5km_updated_coordinates.nc'
        gsea_mask = xr.open_dataarray(gsea_file)

        # Read MJJAS z500hPa
        read_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        # da_z500 = xr.open_dataset(read_file_path/'z0500_MJJAS_trended_coarsened_0.nc')['z0500']
        da_z500 = xr.open_dataset(read_file_path/'z0500_all.nc')['z0500']

        # Case 2: if 'height' is only a coordinate -> remove it
        if 'pressure' in da_z500.coords:
            da_z500 = da_z500.reset_coords('pressure', drop=True)

        # Mask z500 hPa with Greenland Sea mask
        mask = xr.broadcast(da_z500, gsea_mask != 0)[1]
        da_z500_masked = da_z500.where(mask)

        # Calculate mean z500 hPa over the masked area
        da_z500_masked_mean = da_z500_masked.mean(dim=['rlat', 'rlon'])
        da_z500_masked_mean_df = da_z500_masked_mean.to_dataframe()
        da_z500_masked_mean_df = da_z500_masked_mean_df.resample('D').mean()
        da_z500_masked_mean_df.rename(columns={da_z500_masked_mean_df.columns[0]: 'z500'}, inplace=True)

        # Save the masked variables
        logger.info(f"Saving all the 5-km z500 hPa masked over Greenland Sea...")
        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_z500')
        da_z500_masked_mean_df.to_csv(f"{base_output_path}/masked_z500hPa_greenland_sea.csv")

# %%
class QualityCheckFreyaGlacierAWSVariables:
    def preprocess_incoming_shortwave_raditaion():
        """Tilt correct the SWRin of the Freya AWS.

        """
        # %%

        data_freya_aws = _read_freya_glacier_aws_variables()

        logger.info(f"Correcting incoming shortwave raditaion...")
        # ---------- Freya Glacier AWS Metadata ----------
        # Read metadata to extract RACMO variables at point scale
        # Extract Freya Glacier's Station location
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Extract Freya Glacier AWS lat and lon
        station = 'FRE2 (promice type, tripod)'
        freya_lat, freya_lon = stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station]

        # -------------------------------------------------------------------
        # Tilt correction for radiation measurements
        # Correct incoming shortwave radiation (SWRin) for tilt radiation
        # Net Radiometer CNR4 calibration constant from Benhard
        # Serial no: 140439 in microV/W/m²
        # s2_SWin110458=10.59; s2_SWout110458=10.82; s2_LWin110458=11.52; s2_LWout110458=9.57
        # Make sure the radiation data in microV/W/m2 are converted to the desired units (W/m2) by dividing with the calibration constant.
        # For more information see Zackenberg_freya_stations_data.py in 2_paper_work_ku folder.
        # Reference: https://doi.org/10.5194/essd-2021-80
        X = data_freya_aws['tiltx[(vx10^-2]'] * 1e2
        Y = data_freya_aws['tilty[(vx10^-2]'] * 1e2

        # Convert tilt sensor readings (voltage → degrees)
        poly = lambda v: 21.1 * abs(v) - 10.4 * abs(v)**2 + 3.6 * abs(v)**3 - 0.49 * abs(v)**4
        data_freya_aws['tiltX_deg'] = poly(X)
        data_freya_aws['tiltY_deg'] = poly(Y)

        data_freya_aws['dsr_cor'], data_freya_aws['usr_cor'], data_freya_aws['albedo_cor'], data_freya_aws['OKalbedos'] = dcaws.correct_dsr_for_tilt(
                dsr=data_freya_aws['SWin[W/m²]'],
                usr=data_freya_aws['SWout[W/m²]'],
                dlr=data_freya_aws['LWin[W/m²]'],
                Tair=data_freya_aws['AirTemp[°C]'],
                tiltx=data_freya_aws['tiltX_deg'],
                tilty=data_freya_aws['tiltY_deg'],
                dates=data_freya_aws['dates'],
                lat=freya_lat)

        # Convert correction SWRin to nan when raw SWRin is nan
        data_freya_aws.loc[np.isnan(data_freya_aws['SWin[W/m²]']), 'dsr_cor'] = np.nan
        data_freya_aws[['SWin[W/m²]', 'dsr_cor']].plot()

        # Calculate hourly SWRin
        df_swin_hour = ss.hourly_statistics(data=data_freya_aws[['SWin[W/m²]', 'dsr_cor']], method='mean')

        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        df_swin_hour.to_csv(f"{save_path}/freya_swin_hour.csv")

        # Calculate daily SWRin
        df_swin_day = ss.daily_statistics(data=data_freya_aws[['SWin[W/m²]', 'dsr_cor']], method='mean')
        # Save daily corrected dataframe
        df_swin_day.to_csv(f"{save_path}/freya_swin_day.csv")

        # # -------------------------------------------------------------------
        # # 4. Albedo correction
        # data_freya_aws['Okalbedos_value'] = data_freya_aws['albedo_cor']
        # data_freya_aws.loc[~data_freya_aws['OKalbedos'], 'Okalbedos_value'] = np.nan
        # %%
    def preprocess_outgoing_shortwave_raditaion():
        """ Tilt correct the SWRout of the Freya AWS.

        """
        # %%
        data_freya_aws = _read_freya_glacier_aws_variables()

        logger.info(f"Correcting incoming shortwave raditaion...")
        # ---------- Freya Glacier AWS Metadata ----------
        # Read metadata to extract RACMO variables at point scale
        # Extract Freya Glacier's Station location
        # station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        station_metadata_path = '/nas/data/processed/people/sos/third_paper_data/processed_aws/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Extract Freya Glacier AWS lat and lon
        station = 'FRE2 (promice type, tripod)'
        freya_lat, freya_lon = stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station]

        # -------------------------------------------------------------------
        # Tilt correction for radiation measurements
        # Correct incoming shortwave radiation (SWRin) for tilt radiation
        # Net Radiometer CNR4 calibration constant from Benhard
        # Serial no: 140439 in microV/W/m²
        # s2_SWin110458=10.59; s2_SWout110458=10.82; s2_LWin110458=11.52; s2_LWout110458=9.57
        # Make sure the radiation data in microV/W/m2 are converted to the desired units (W/m2) by dividing with the calibration constant.
        # For more information see Zackenberg_freya_stations_data.py in 2_paper_work_ku folder.
        # Reference: https://doi.org/10.5194/essd-2021-80
        X = data_freya_aws['tiltx[(vx10^-2]'] * 1e2
        Y = data_freya_aws['tilty[(vx10^-2]'] * 1e2

        # Convert tilt sensor readings (voltage → degrees)
        poly = lambda v: 21.1 * abs(v) - 10.4 * abs(v)**2 + 3.6 * abs(v)**3 - 0.49 * abs(v)**4
        data_freya_aws['tiltX_deg'] = poly(X)
        data_freya_aws['tiltY_deg'] = poly(Y)

        data_freya_aws['dsr_cor'], data_freya_aws['usr_cor'], data_freya_aws['albedo_cor'], data_freya_aws['OKalbedos'] = dcaws.correct_dsr_for_tilt(
                dsr=data_freya_aws['SWin[W/m²]'],
                usr=data_freya_aws['SWout[W/m²]'],
                dlr=data_freya_aws['LWin[W/m²]'],
                Tair=data_freya_aws['AirTemp[°C]'],
                tiltx=data_freya_aws['tiltX_deg'],
                tilty=data_freya_aws['tiltY_deg'],
                dates=data_freya_aws['dates'],
                lat=freya_lat)

        # Convert correction SWRout to nan when raw SWRout is nan
        data_freya_aws.loc[np.isnan(data_freya_aws['SWout[W/m²]']), 'usr_cor'] = np.nan
        data_freya_aws[['SWout[W/m²]', 'usr_cor']].plot()

        # Calculate hourly SWRout
        df_swout_hour = ss.hourly_statistics(data=data_freya_aws[['SWout[W/m²]', 'usr_cor']], method='mean')

        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        df_swout_hour.to_csv(f"{save_path}/freya_swout_hour.csv")

        # Calculate daily SWRout
        df_swout_day = ss.daily_statistics(data=data_freya_aws[['SWout[W/m²]', 'usr_cor']], method='mean')
        # Save daily corrected dataframe
        df_swout_day.to_csv(f"{save_path}/freya_swout_day.csv")

        # %%

    def preprocess_relative_humidity():
        """Correct relative humidity for below freezing point to get value with respect to ice. Also bias correct the shift in the relative humidity after 2018-02-15.

        """
        # %%
        data_freya_aws = _read_freya_glacier_aws_variables()

        logger.info(f"Correcting relative humidity...")
        # Relative humidity correction
        # Correcting relative humidity readings for T below 0 to give value with respect to ice.
        data_freya_aws['RH_corr'] = dcaws.correct_relative_humidity(RH=data_freya_aws.loc[:, 'RH[/100]'], Tair=data_freya_aws.loc[:, 'AirTemp[°C]']) #relative humidity
        data_freya_aws[['RH_corr', 'RH[/100]']].plot()

        # Adjust Relative humidity after '2018-02-15' using relative humidity from AP1
        # Calculate hourly mean
        fre_rh = ss.hourly_statistics(data=data_freya_aws[['RH_corr', 'RH[/100]']], method='mean')
        fre_rh.rename(columns={'RH_corr': 'fre_rh_corr'}, inplace=True)

        # Read and correct AP Oslen relative humidity for T below 0 to give value with respect to ice.
        ap_l_path = Path('/nas/data/raw/gem/glaciobasis/apolsen_all_dataset/data/aws_data_from_Signe')
        ap_l_rh = pd.read_csv(f"{ap_l_path}/zac_l_hour_relative_humidity.csv")
        ap_l_rh['time'] = pd.to_datetime(ap_l_rh['time'])
        ap_l_rh.set_index('time', inplace=True)
        ap_l_rh = ap_l_rh.asfreq('h')

        ap_l_temp = pd.read_csv(f"{ap_l_path}/zac_l_hour_temperature.csv")
        ap_l_temp['time'] = pd.to_datetime(ap_l_temp['time'])
        ap_l_temp.set_index('time', inplace=True)
        ap_l_temp = ap_l_temp.asfreq('h')

        ap_l_rh['rh_corr'] = dcaws.correct_relative_humidity(RH=ap_l_rh.loc[:, 'rh'], Tair=ap_l_temp.loc[:, 't_1']) #relative humidity
        ap_l_rh[['rh_corr', 'rh']].plot()
        ap_l_rh.rename(columns={'rh_corr': 'ap1_rh_corr'}, inplace=True)

        # Concatenate Freya and AP Oslen Relative humidity
        merged_rh = pd.concat([fre_rh['fre_rh_corr'], ap_l_rh['ap1_rh_corr']], axis=1, join="outer")
        merged_rh = merged_rh.dropna()

        # Use the linear relationship between RH of Freya and AP1 before the shift period to adjust the after shift period RH of Freya
        mean_before = merged_rh.loc[:'2018-02-14']
        slope_b, intercept_b, r_value_b, _, _ = stats.linregress(mean_before.ap1_rh_corr, mean_before.fre_rh_corr)
        x_new_b = np.arange(mean_before.ap1_rh_corr.min(), mean_before.ap1_rh_corr.max(), 1)
        y_pred_b = slope_b * x_new_b + intercept_b

        mean_after = merged_rh.loc['2018-02-15':]
        slope_a, intercept_a, r_value_a, _, _ = stats.linregress(mean_after.ap1_rh_corr, mean_after.fre_rh_corr)
        x_new_a = np.arange(mean_after.ap1_rh_corr.min(), mean_after.ap1_rh_corr.max(), 1)
        y_pred_a = slope_a * x_new_a + intercept_a

        fig, ax = plt.subplots(1)
        ax.plot(mean_before.ap1_rh_corr, mean_before.fre_rh_corr, 'o', ms=0.1, color='b')
        ax.plot(mean_after.ap1_rh_corr, mean_after.fre_rh_corr, 'o', ms=0.1, color='r')
        ax.plot(x_new_b, y_pred_b, lw=1, color='b', dashes=[6, 10], label='Before slope')
        ax.plot(x_new_a, y_pred_a, lw=1, color='r', dashes=[6, 10], label='After slope')

        x_dummy = np.arange(0, 100, 1)
        y_dummy = np.arange(0, 100, 1)
        ax.plot(x_dummy, y_dummy, lw=0.5, color='k', dashes=[6, 10])
        slope_dummy, intercept, r_value_d, _, _ = stats.linregress(x_dummy, y_dummy)
        ax.set_xlim([0, 100])
        ax.set_ylim([0, 100])
        ax.legend()
        print(f"before slope = {slope_b}")
        print(f"after slope = {slope_a}")
        print(f"perfect slope = {slope_dummy}")

        # -----------------------------
        # Quantile Mapping Adjustment
        # -----------------------------
        # --- Step 1: Split dataset into reference (before) and adjustment (after) periods ---
        reference_period = merged_rh.loc[:'2018-02-14', ['ap1_rh_corr', 'fre_rh_corr']].dropna()
        adjustment_period = merged_rh.loc['2018-02-15':, ['ap1_rh_corr', 'fre_rh_corr']].dropna(subset=['ap1_rh_corr'])

        # --- Step 2: Define Quantile Mapping transformer ---
        output_distribution = 'uniform'
        qm_ap1 = QuantileTransformer(output_distribution=output_distribution, n_quantiles=100, random_state=0)
        qm_freya = QuantileTransformer(output_distribution=output_distribution, n_quantiles=100, random_state=0)

        # Fit on reference period distributions
        qm_ap1.fit(reference_period[['ap1_rh_corr']])
        qm_freya.fit(reference_period[['fre_rh_corr']])

        # --- Step 3: Transform adjustment AP1 values into quantile space ---
        ap1_adj_q = qm_ap1.transform(adjustment_period[['ap1_rh_corr']])

        # --- Step 4: Map AP1 quantiles into Freya reference distribution ---
        freya_adj_corrected = qm_freya.inverse_transform(ap1_adj_q).flatten()

        # Copy Freya series and insert corrected values
        merged_rh['fre_rh_corr_adjusted_qm'] = merged_rh['fre_rh_corr'].copy()
        merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm'] = freya_adj_corrected

        # Clip to physical limits [0,100]
        merged_rh['fre_rh_corr_adjusted_qm'] = np.clip(merged_rh['fre_rh_corr_adjusted_qm'], 0, 100)
        # merged_rh['fre_rh_corr_adjusted_qm'] = np.where(merged_rh['fre_rh_corr_adjusted_qm'] > 100, 100, merged_rh['fre_rh_corr_adjusted_qm'])

        merged_rh[['fre_rh_corr', 'fre_rh_corr_adjusted_qm']].plot(figsize=(10, 5), alpha=0.8)

        # Apply that fixed bias (shift) to the adjustment-period Freya values.
        # That gives you a simpler correction: you retain the actual day-to-day Freya variation,
        # but shift the whole series to be consistent with the QM adjustment on average.
        # --- Step 1: Compute bias between QM-corrected and raw Freya in adjustment period ---
        bias_mean = (
            merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm'].mean()
            - merged_rh.loc[adjustment_period.index, 'fre_rh_corr'].mean()
        )

        print(f"Fixed bias (QM-corrected - raw Freya) = {bias_mean:.2f} %RH")

        # --- Step 2: Apply fixed bias shift to Freya's adjustment period ---
        freya_after_shifted = merged_rh.loc[adjustment_period.index, 'fre_rh_corr'] + bias_mean

        # --- Step 3: Insert into dataframe ---
        merged_rh['fre_rh_corr_adjusted_qm_shift'] = merged_rh['fre_rh_corr'].copy()
        merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm_shift'] = freya_after_shifted

        # --- Step 4: Clip to realistic bounds [0, 100] ---
        merged_rh['fre_rh_corr_adjusted_qm_shift'] = np.clip(merged_rh['fre_rh_corr_adjusted_qm_shift'], 0, 100)

        # Extend QM-derived correction so that instead of just a mean shift,
        # the variance (spread) of Freya’s adjustment-period values is corrected
        # Compute a scaling factor (variance adjustment) in addition to the shift,
        # so we can have both mean and variance correction from QM, but still preserve Freya’s raw shape
        # --- Step 1: Compute stats for adjustment period ---
        mean_raw = merged_rh.loc[adjustment_period.index, 'fre_rh_corr'].mean()
        std_raw  = merged_rh.loc[adjustment_period.index, 'fre_rh_corr'].std()

        mean_qm  = merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm'].mean()
        std_qm   = merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm'].std()

        print(f"Raw mean={mean_raw:.2f}, Raw std={std_raw:.2f}")
        print(f"QM mean={mean_qm:.2f}, QM std={std_qm:.2f}")

        # --- Step 2: Compute linear correction (shift + scale) ---
        freya_after_shift_scale = ((merged_rh.loc[adjustment_period.index, 'fre_rh_corr'] - mean_raw) / std_raw) * std_qm + mean_qm

        # --- Step 3: Insert into dataframe ---
        merged_rh['fre_rh_corr_adjusted_qm_mv'] = merged_rh['fre_rh_corr'].copy()
        merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm_mv'] = freya_after_shift_scale

        # --- Step 4: Clip to realistic bounds [0,100] ---
        merged_rh['fre_rh_corr_adjusted_qm_mv'] = np.clip(merged_rh['fre_rh_corr_adjusted_qm_mv'], 0, 100)

        # --- Step 5: Plot all versions ---
        merged_rh[['fre_rh_corr',
                'fre_rh_corr_adjusted_qm',       # full QM (quantile mapping) (distribution reshaping).
                'fre_rh_corr_adjusted_qm_shift', # QM-derived mean shift only (simple correction).
                'fre_rh_corr_adjusted_qm_mv'     # QM-derived mean + variance scaling (keeps Freya’s variability shape but aligns both mean and spread to QM).
                ]].plot(figsize=(12, 6), alpha=0.8)

        # Evaluation step
        # --- Step 1: Select comparison period (adjustment) ---
        # AP1 is anchor station, we’ll compare each Freya-corrected version against AP1’s adjustment-period values.
        ap1_adj = merged_rh.loc[adjustment_period.index, 'ap1_rh_corr']

        # Different Freya variants
        freya_raw   = merged_rh.loc[adjustment_period.index, 'fre_rh_corr']
        freya_qm    = merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm']
        freya_shift = merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm_shift']
        freya_mv    = merged_rh.loc[adjustment_period.index, 'fre_rh_corr_adjusted_qm_mv']

        # --- Step 2: Define metrics ---
        def evaluate(y_true, y_pred):
            mbe = (y_pred - y_true).mean()  # Mean Bias Error
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            corr = np.corrcoef(y_true, y_pred)[0, 1]
            return mbe, rmse, corr

        # --- Step 3: Collect results ---
        results = pd.DataFrame({
            "Raw":    evaluate(ap1_adj, freya_raw),
            "QM":     evaluate(ap1_adj, freya_qm),
            "Shift":  evaluate(ap1_adj, freya_shift),
            "Shift+Scale": evaluate(ap1_adj, freya_mv),
        }, index=["MBE", "RMSE", "Correlation"]).T

        print(results)

        # Test the adjustments with reference period statistics
        for corr in ['fre_rh_corr', 'fre_rh_corr_adjusted_qm', 'fre_rh_corr_adjusted_qm_shift', 'fre_rh_corr_adjusted_qm_mv']:
            if corr == 'fre_rh_corr':
                print(f"{corr}: {merged_rh.loc[reference_period.index, corr].mean()}")
                print(f"{corr}: {merged_rh.loc[reference_period.index, corr].std()} \n")
            else:
                print(f"{corr}: {merged_rh.loc[adjustment_period.index, corr].mean()}")
                print(f"{corr}: {merged_rh.loc[adjustment_period.index, corr].std()} \n")

        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        merged_rh.to_csv(f"{save_path}/freya_rh2m_hour.csv")

        # Calculate daily RH
        merged_rh_day = ss.daily_statistics(data=merged_rh, method='mean')
        # Save daily corrected dataframe
        merged_rh_day.to_csv(f"{save_path}/freya_rh2m_day.csv")

    # %%
    def preprocess_air_temperature():
        """Preprocess air temperature.

        """
        # %%

        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()

        # Read air temperature
        # Select and rename columns
        freya_t2m = pd.DataFrame(data_freya_aws.loc[:, 'AirTemp[°C]']).rename(columns={'AirTemp[°C]': 't2m'})
        freya_t2m.t2m.plot()

        # Calculate hourly mean
        freya_t2m_hour = ss.hourly_statistics(data=freya_t2m['t2m'], method='mean')

        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        freya_t2m_hour.to_csv(f"{save_path}/freya_t2m_hour.csv")

        # Calculate daily T2m
        merged_t2m_day = ss.daily_statistics(data=freya_t2m['t2m'], method='mean')
        # Save daily corrected dataframe
        merged_t2m_day.to_csv(f"{save_path}/freya_t2m_day.csv")

        # %%

    def calculate_specific_humidity():
        """Calculate specific humidity.

        """
        # %%

        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()

        # Read air pressure and temperature
        # Select and rename columns
        freya_press_t2m = (data_freya_aws[['p[hPa]', 'AirTemp[°C]']].rename(columns={'p[hPa]': 'press', 'AirTemp[°C]': 't2m'}))
        freya_press_t2m.press.plot()
        freya_press_t2m['press'] = np.where(freya_press_t2m['press'] < 700, np.nan, freya_press_t2m['press'])
        freya_press_t2m.press.plot()
        # Calculate hourly mean
        freya_press_t2m_hour = ss.hourly_statistics(data=freya_press_t2m[['press', 't2m']], method='mean')

        # Read relative humidity
        rh_save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/freya_rh2m_hour.csv'
        # freya_rh_hour = pd.read_csv(rh_save_path) # encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_rh_hour = pd.read_csv(rh_save_path, parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_rh_hour.index = pd.to_datetime(freya_rh_hour.index)
        # Resample to uniform frequency
        freq = freya_rh_hour.index.to_series().diff().mode()[0]
        freya_rh_hour = freya_rh_hour.resample(freq).mean()

        # Merge RH, pressure, and temperature into a new DataFrame
        freya_df = pd.concat([freya_rh_hour, freya_press_t2m_hour[['press', 't2m']]], axis=1)

        logger.info('Calculate specific humidity....')
        freya_q_hour = pd.DataFrame(index=freya_df.index, columns=['fre_rh_corr_adjusted_qm_q', 'fre_rh_corr_adjusted_qm_shift_q', 'fre_rh_corr_adjusted_qm_mv_q'])
        # Calculate specific humidity
        for rh in ['fre_rh_corr_adjusted_qm', 'fre_rh_corr_adjusted_qm_shift', 'fre_rh_corr_adjusted_qm_mv']:
            freya_q_hour.loc[:, f"{rh}_q"] = dcaws.calculate_specific_humidity(
                                            T=freya_df.t2m, p=freya_df.press, RH_cor=freya_df[rh])

        freya_q_hour.plot()
        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        freya_q_hour.to_csv(f"{save_path}/freya_q2m_hour.csv")

        # Calculate daily q
        freya_q_day = ss.daily_statistics(data=freya_q_hour, method='mean')
        # Save daily corrected dataframe
        freya_q_day.to_csv(f"{save_path}/freya_q2m_day.csv")

        # %%
    def preprocess_incoming_longwave_raditaion():
        """Correct incoming longwave radiation.

        """
        # %%
        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()
        freya_lwin = pd.DataFrame(data_freya_aws.loc[:, 'LWin[W/m²]'])
        freya_lwin.rename(columns={'LWin[W/m²]': 'lwin'}, inplace=True)
        freya_lwin.plot()

        # Calculate hourly mean
        freya_lwin_hour = ss.hourly_statistics(data=freya_lwin['lwin'], method='mean')
        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        freya_lwin_hour.to_csv(f"{save_path}/freya_lwin_hour.csv")

        # Calculate daily lwin
        freya_lwin_day = ss.daily_statistics(data=freya_lwin['lwin'], method='mean')
        # Save daily corrected dataframe
        freya_lwin_day.to_csv(f"{save_path}/freya_lwin_day.csv")

    def preprocess_outgoing_longwave_raditaion():
        """Correct outgoing longwave radiation.

        """
        # %%
        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()
        freya_lwout = pd.DataFrame(data_freya_aws.loc[:, 'LWout[W/m²]'])
        freya_lwout.rename(columns={'LWout[W/m²]': 'lwout'}, inplace=True)
        freya_lwout.plot()

        # Calculate hourly mean
        freya_lwout_hour = ss.hourly_statistics(data=freya_lwout['lwout'], method='mean')
        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        freya_lwout_hour.to_csv(f"{save_path}/freya_lwout_hour.csv")

        # Calculate daily lwout
        freya_lwout_day = ss.daily_statistics(data=freya_lwout['lwout'], method='mean')
        # Save daily corrected dataframe
        freya_lwout_day.to_csv(f"{save_path}/freya_lwout_day.csv")

        # %%

    def preprocess_wind_speed_and_direction():
        """Preprocess wind speed and direction.

        """
        # %%
        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()
        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        freya_ws_wd = data_freya_aws[['ws[m/s]', 'wdir[deg]']].rename(columns={'ws[m/s]': 'ws', 'wdir[deg]': 'wdir'})

        # Calculate hourly mean
        freya_ws_hour = ss.hourly_statistics(data=freya_ws_wd['ws'], method='mean')
        freya_ws_hour.plot()
        freya_ws_hour.to_csv(f"{save_path}/freya_ws_hour.csv")

        # Calculate daily wind speed
        freya_ws_day = ss.daily_statistics(data=freya_ws_wd['ws'], method='mean')
        freya_ws_day.plot()
        freya_ws_day.to_csv(f"{save_path}/freya_ws_day.csv")

        # Read wind direction
        # Calculate hourly wind direction
        freya_wd_hour = vs.wind_direction_hourly_mean(ws=freya_ws_wd['ws'], wd=freya_ws_wd['wdir'])
        freya_wd_hour.plot()
        freya_wd_hour.to_csv(f"{save_path}/freya_wd_hour.csv")

        # Calculate daily wind direction
        freya_wd_day = vs.wind_direction_daily_mean(ws=freya_ws_wd['ws'], wd=freya_ws_wd['wdir'])
        freya_wd_day.plot()
        freya_wd_day.to_csv(f"{save_path}/freya_wd_day.csv")

    def preprocess_atmospheric_pressure():
        """Preprocess atmospheric pressure in hPa.

        """
        # %%
        # Save hourly corrected dataframe
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()
        freya_atm_pressure = data_freya_aws[['p[hPa]']].rename(columns={'p[hPa]': 'atm_pressure'})
        # freya_atm_pressure.atm_pressure.plot()
        freya_atm_pressure['atm_pressure'] = np.where(freya_atm_pressure['atm_pressure'] < 700, np.nan, freya_atm_pressure['atm_pressure'])
        freya_atm_pressure.atm_pressure.plot()

        # Calculate hourly mean
        freya_atm_pressure_hour = ss.hourly_statistics(data=freya_atm_pressure['atm_pressure'], method='mean')
        freya_atm_pressure_hour.plot()
        freya_atm_pressure_hour.to_csv(f"{save_path}/freya_atm_pressure_hour.csv")

        # Calculate daily atmospheric pressure
        freya_atm_pressure_day = ss.daily_statistics(data=freya_atm_pressure['atm_pressure'], method='mean')
        freya_atm_pressure_day.plot()
        freya_atm_pressure_day.to_csv(f"{save_path}/freya_atm_pressure_day.csv")

        # %%
    def calculate_surface_temperature():
        """Calculate surface temperature from incoming longwave radiation.

        """
        # %%
        # Read Freya AWS measurements
        data_freya_aws = _read_freya_glacier_aws_variables()
        freya_lwin = pd.DataFrame(data_freya_aws.loc[:, 'LWin[W/m²]'])
        freya_lwin.rename(columns={'LWin[W/m²]': 'lwin'}, inplace=True)

        freya_lwout = pd.DataFrame(data_freya_aws.loc[:, 'LWout[W/m²]'])
        freya_lwout.rename(columns={'LWout[W/m²]': 'lwout'}, inplace=True)

        logger.info('Calculating surface temperature from incoming longwave radiation....')
        # Calculate surface temperature
        freya_ts = dcaws.calculate_surface_temperature(ulr=freya_lwout['lwout'], dlr=freya_lwin['lwin'], surface_type='ice')
        freya_ts = pd.DataFrame(freya_ts, columns=['ts'])
        freya_ts_hour.plot()

        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Calculate hourly mean
        freya_ts_hour = ss.hourly_statistics(data=freya_ts['ts'], method='mean')
        freya_ts_hour.to_csv(f"{save_path}/freya_ts_hour.csv")

        # Calculate daily ts
        freya_ts_day = ss.daily_statistics(data=freya_ts_hour, method='mean')
        # Save daily corrected dataframe
        freya_ts_day.to_csv(f"{save_path}/freya_ts_day.csv")

        # %%

    def calculate_temperature_difference_between_t2m_tsurf():
        # %%
        """Calculate temperature difference between 2-m air temperature and surface temperature.

        """
        save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed'
        freya_t2m_hour = pd.read_csv(f"{save_path}/freya_t2m_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_t2m_hour.index = pd.to_datetime(freya_t2m_hour.index)

        freya_ts_hour = pd.read_csv(f"{save_path}/freya_ts_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_ts_hour.index = pd.to_datetime(freya_ts_hour.index)

        logger.info('Calculating temperature difference between 2-m air temperature and surface temperature....')
        # Calculate temperature difference between 2-m air temperature and surface temperature
        freya_tdiff_hour = pd.DataFrame(freya_t2m_hour['t2m'] - freya_ts_hour['ts'], columns=['t2m_tsurf_diff'])
        # Save daily corrected dataframe
        freya_tdiff_hour.to_csv(f"{save_path}/freya_t2m_tsurf_diff_hour.csv")

        # Calculate daily temperature difference
        freya_tdiff_day = ss.daily_statistics(data=freya_tdiff_hour, method='mean')
        # Save daily corrected dataframe
        freya_tdiff_day.to_csv(f"{save_path}/freya_t2m_tsurf_diff_day.csv")

    def preprocess_snow_ice_height():
        """Preprocess the Freya AWS data by correcting the snow height and ice height measurement with sonic ranger installed on the floating AWS.

        Based on metadata in AWS Freya_Metadata_for_user.pdf

        """
        # %%
        # Elevation 688 m asl
        path = '/nas/data/raw/igr/freya_2016_2020/'
        data_freya_aws = {}
        # shows the order of files imported
        allfiles = glob.glob(os.path.join(path, '*.dat'))

        frame = []
        for file_id in tqdm(allfiles):
            raw_data = pd.read_csv(file_id, sep=r'\s+', header=1, encoding='ISO-8859-1', na_values='NaN')#,skiprows=2
            raw_data['Date'] = pd.to_datetime(raw_data['Date'] + ' ' + raw_data['Time[UTC]'])
            raw_data.set_index('Date', inplace=True)
            raw_data.drop('Time[UTC]', axis=1, inplace=True)
            timedelta = raw_data.index[1] - raw_data.index[0]
            raw_data = raw_data.resample(timedelta).mean()
            frame.append(raw_data)

        data_freya_aws = pd.concat(frame)
        data_freya_aws.sort_index(inplace=True)
        timedelta = data_freya_aws.index[1] - data_freya_aws.index[0]
        data_freya_aws = data_freya_aws.resample(timedelta).mean()
        data_freya_aws['dates'] = data_freya_aws.index.values
        data_freya_aws = data_freya_aws[['snowh[m]', 'iceh[m]']]

        plot_params = dict(marker='o', lw=0, ms=0.9)
        data_freya_aws.plot(**plot_params)
        plt.title('Freya Glacier Snow Height and Ice Height Measurement with Sonic Ranger Installed on the Floating AWS')

        # NOTE Start correcting the snow and ice height measurement chronologically by checking the 10 minutes change in snow and ice height measurement.
        # Make copy of original data to correct the snow height and ice height measurement with sonic ranger installed on the floating AWS
        data_freya_aws['snowh_corr[m]'] = data_freya_aws['snowh[m]'].copy(deep=True)
        data_freya_aws['iceh_corr[m]'] = data_freya_aws['iceh[m]'].copy(deep=True)
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)

        # NOTE Correction 1
        # Correction 2017 ice height measurement
        dficeh_index = data_freya_aws[data_freya_aws['diceh_corr[m]'] >= 0.1].index

        for index in dficeh_index:
            iceh_before_error = data_freya_aws.loc[index - pd.Timedelta(minutes=10), 'iceh_corr[m]']
            current_wrong_iceh = data_freya_aws.loc[index, 'iceh_corr[m]']
            data_freya_aws.loc[(data_freya_aws.index >= index) & (data_freya_aws['iceh_corr[m]'] == current_wrong_iceh), 'iceh_corr[m]'] = iceh_before_error

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['iceh[m]','iceh_corr[m]','snowh[m]','snowh_corr[m]']].plot(**plot_params, ax=ax)
        # Recalculate the change in ice height measurement after correction
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)

        # NOTE Correction 2
        # Correct 2018 snow height:
        # More than 2-m snow melted within 10 minutes which is physically not possible on the glacier
        # # Uncomment to visualize --------------------
        data_freya_aws.loc['2018-09-01 23:40:00':'2018-09-02 00:20:00', ['snowh[m]', 'dsnowh_corr[m]']].plot(marker='o', lw=0, ms=1.5)
        plt.title('Freya Glacier Snow Height Correction due to anomalous snow height change on 1-2 September 2018')
        # # Uncomment to visualize --------------------
        # From 2018-09-02 00:00:00 shift the snow height by adding 2.35 m i.e., 2018-09-01 23:50:00 snow height

        # Time when more than 30 cm snow falled or snow/ice melted aways in 10 minutes
        # Cross-checked with ZAC and APO snow height dataset
        notOksnowheightchange_30cm = data_freya_aws.loc[(data_freya_aws['dsnowh_corr[m]']).abs() >= 0.3, 'dsnowh_corr[m]'] # First index is 2018-09-02 00:00:00
        # Time when fixed USR is measuring either snow accumulation or snow/ice ablation
        changeiceheight = data_freya_aws.loc[(data_freya_aws['diceh_corr[m]'] != 0), ['diceh_corr[m]', 'iceh[m]', 'iceh_corr[m]']]
        # # Uncomment to visualize --------------------
        data_freya_aws.loc[:, ['diceh_corr[m]', 'iceh[m]','iceh_corr[m]']].plot(**plot_params)
        plt.title('Freya Glacier Ice Height Change Measurement with Fixed USR')
        # # Uncomment to visualize --------------------

        # Correcting snow height only after 2020 since station was dug out in 2018 from the snow and relocated on the new surface.
        # Shifting floating USR snow height measurement by 2.35 m
        shift_start_date = notOksnowheightchange_30cm.index[0] # '2018-09-02 00:00:00'
        shift_end_date = changeiceheight.loc['2019'].index[0] - pd.Timedelta(minutes=10) # '2019-06-23 23:50:00' # Found out manually when ice starts to melt as shown as USR in fixed stake

        # Change in 2018 ice height is due to extreme precipitation as snow event
        data_freya_aws.loc[shift_start_date:shift_end_date, 'snowh_corr[m]'] = \
            data_freya_aws.loc[shift_start_date:shift_end_date, 'snowh_corr[m]'] + \
                data_freya_aws.loc[shift_start_date - pd.Timedelta(minutes=10), 'snowh[m]'] # i.e., 2.35 m shift up; previous time snow height plus
        # Uncomment to visualize --------------------
        data_freya_aws.loc[:, ['snowh[m]', 'snowh_corr[m]']].plot(**plot_params)
        plt.title('Freya Glacier Snow Height Correction due to anomalous snow height change on 1-2 September 2018')
        # Uncomment to visualize --------------------

        # Fill up snow height measured by floating USR in AWS with the ice height measured by fixed USR
        fill_start_date = changeiceheight.loc['2019'].index[0] # '2019-06-24 00:00:00'
        fill_end_date = '2019-07-30 16:40:00' # Found out manually when ice starts to melt as shown as USR in fixed stake
        data_freya_aws.loc[fill_start_date:fill_end_date,'snowh_corr[m]'] = \
            data_freya_aws.loc[fill_start_date:fill_end_date,'iceh[m]']
        # Uncomment to visualize --------------------
        data_freya_aws.loc[:, ['snowh[m]', 'snowh_corr[m]']].plot(**plot_params)
        # Uncomment to visualize --------------------

        # Recalculate the change in snow height measurement after correction
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        # NOTE Correction 3
        # Extrapolate snow height measurement during the gap period when station was covered in snow and fixed USR was measuring ice height
        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws.loc[:, ['snowh[m]', 'snowh_corr[m]']].plot(**plot_params, ax=ax)
        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws.loc['2019-06-24 00:10:00':'2019-07-30 14:40:00', ['snowh[m]', 'snowh_corr[m]']].plot(**plot_params, ax=ax)

        snowh_to_extrapolate = pd.DataFrame(data_freya_aws.loc['2019-06-24 00:10:00':'2019-07-30 12:00:00', ['snowh[m]', 'snowh_corr[m]']])
        snowh_to_extrapolate['roll_mean'] = (snowh_to_extrapolate['snowh_corr[m]'].rolling(window='1h', min_periods=6).mean()) # 1 hour = 60 minutes
        snowh_to_extrapolate['linear_regression_indicator'] = np.array([i for i in range(len(snowh_to_extrapolate))])
        snowh_to_extrapolate.dropna(inplace=True)
        # snowh_to_extrapolate = snowh_to_extrapolate.resample('10min').mean() # Resample to 10 minutes frequency

        fig, ax = plt.subplots(figsize=(10, 6))
        snowh_to_extrapolate[['snowh_corr[m]', 'roll_mean']].plot(**plot_params, ax=ax)

        # Fit the linear regression model to the snow height data to extrapolate the snow height during the gap period
        x = snowh_to_extrapolate['linear_regression_indicator'].values
        y = snowh_to_extrapolate['roll_mean'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        x_extrapolate = np.array(range(6 * 24 * 31)) # Extrapolate for 31 days (one hour equal to 6 data points at 10 minutes frequency)
        x_extrapolate = x_extrapolate + x[-1] + 1 # Start from the last index of the known data to extrapolate forward
        snowh_extrapolated = slope * x_extrapolate + intercept

        gap_index = pd.date_range(start=snowh_to_extrapolate.index[-1] + pd.Timedelta(minutes=10),  # first missing timestamp
                                periods=len(snowh_extrapolated),
                                freq='10min')

        snowht_extrapolated = pd.DataFrame({'snowh_corr[m]': snowh_extrapolated}, index=gap_index)
        # Remove negative snow height values if any after extrapolation
        snowht_extrapolated = snowht_extrapolated[snowht_extrapolated['snowh_corr[m]'] >= 0]
        # Fill the extrapolated snow height values in the original dataframe
        # data_freya_aws.update(snowht_extrapolated)

        # Add a new column to indicate the extrapolated snow height values for visualization
        data_freya_aws['lr_extrapolated'] = np.nan
        data_freya_aws.loc[snowht_extrapolated.index, 'lr_extrapolated'] = snowht_extrapolated['snowh_corr[m]']

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['snowh_corr[m]', 'snowh[m]', 'lr_extrapolated']].plot(**plot_params, ax=ax)

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['snowh_corr[m]', 'snowh[m]', 'iceh_corr[m]', 'iceh[m]']].plot(**plot_params, ax=ax)

        # NOTE Correction 4
        # Correct ice height from Correction 2.
        # Replace snow height measurement by ice height measurement during the period when station was covered in snow and fixed USR was measuring ice height.
        _, ax = plt.subplots(figsize=(10, 6))
        # data_freya_aws[['iceh[m]', 'iceh_corr[m]', 'diceh_corr[m]']].plot(**plot_params, ax=ax)
        data_freya_aws[['iceh[m]', 'iceh_corr[m]']].plot(**plot_params, ax=ax)

        # Fill up ice height measured by floating USR in AWS with the ice height measured by fixed USR
        fill_start_date = changeiceheight.loc['2019'].index[0] # '2019-06-24 00:00:00'
        fill_end_date = pd.Timestamp('2019-07-30 16:40:00') # Found out manually when ice starts to melt as shown as USR in fixed stake
        data_freya_aws.loc[fill_start_date:fill_end_date, 'iceh_corr[m]'] = data_freya_aws.loc[fill_start_date - pd.Timedelta(minutes=10), 'iceh_corr[m]']

        _, ax = plt.subplots(figsize=(10, 6))
        # data_freya_aws[['iceh[m]', 'iceh_corr[m]', 'diceh_corr[m]']].plot(**plot_params, ax=ax)
        data_freya_aws[['iceh[m]', 'iceh_corr[m]']].plot(**plot_params, ax=ax)

        # Update snow accumulation and snow/ice melt
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['iceh[m]','iceh_corr[m]','snowh[m]','snowh_corr[m]']].plot(**plot_params, ax=ax)

        # NOTE Correction 5
        # Move the ice height from 2019-07-30 16:40:00 to 2019-08-01 00:00:00 by adding the change in ice height measurement at 2019-07-30 16:40:00
        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['iceh_corr[m]', 'diceh_corr[m]']].plot(**plot_params, ax=ax)

        previous_iceh = data_freya_aws.loc[fill_end_date, 'iceh_corr[m]']
        iceh_to_correct = ((data_freya_aws.loc[fill_end_date + pd.Timedelta(minutes=10):, 'iceh_corr[m]']).dropna()).iloc[0]
        correction_iceh = iceh_to_correct - previous_iceh

        # Use fill_end_date from Note Correction 4 to shift ice height measurement by adding the change in ice height measurement at fill_end_date
        # This assumes that there was no ice ablation in the autumn of 2019.
        data_freya_aws.loc[fill_end_date + pd.Timedelta(minutes=10):, 'iceh_corr[m]'] = data_freya_aws.loc[fill_end_date + pd.Timedelta(minutes=10):, 'iceh_corr[m]'] - correction_iceh
        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws['iceh_corr[m]'].plot(**plot_params, ax=ax)

        # Update snow accumulation and snow/ice melt
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['iceh[m]','iceh_corr[m]','snowh[m]','snowh_corr[m]']].plot(**plot_params, ax=ax)

        # NOTE Correction 6
        # Removing spikes/outliers from snow height data using moving average
        dfsnowh = pd.DataFrame(data_freya_aws[['snowh_corr[m]']].copy(deep=True))
        dfsnowh = dfsnowh.dropna()
        dfsnowh['rollmean'] = dfsnowh['snowh_corr[m]'].rolling(6).mean() # 1 hour = 60 minutes
        # Uncomment to visualize --------------------
        dfsnowh[['snowh_corr[m]', 'rollmean']].plot(**plot_params)
        # Uncomment to visualize --------------------

        dfsnowh['diff_snowh_rollmean'] = (dfsnowh['snowh_corr[m]'] - dfsnowh['rollmean']).abs()
        dfsnowh[['diff_snowh_rollmean']].plot(**plot_params)

        # Snow height already corrected before 2020 for station relocation on the new surface.
        dfsnowh.loc[:'2019', 'diff_snowh_rollmean'] = np.nan # already corrected for station relocation
        dfsnowh_index = dfsnowh[(dfsnowh['diff_snowh_rollmean']).abs() >= 0.2].index # TODO Check if it is correct time of 30 cm snow fall
        data_freya_aws.loc[dfsnowh_index, 'snowh_corr[m]'] = np.nan
        # Uncomment to visualize --------------------
        data_freya_aws[['snowh[m]', 'snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        # data_freya_aws[['snowh[m]', 'snowh_corr[m]']].plot(**plot_params)
        # Uncomment to visualize --------------------

        # Recalculate the change in snow height measurement after correction
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        # NOTE Correction 7
        # Correct ice height to nan for anomalous values of ablation
        data_freya_aws[['iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        # Manually found dates
        nan_start_date = '2021-07-27 00:00:00'
        nan_end_date = '2021-07-27 23:50:00'
        data_freya_aws.loc[nan_start_date:nan_end_date, 'iceh_corr[m]'] = np.nan
        # # Uncomment to visualize --------------------
        data_freya_aws.loc['2021-7-25':'2021-7-29', ['iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        plt.title('Freya Glacier Ice Height Correction due to anomalous ablation on 27 July 2021')
        # # Uncomment to visualize --------------------

        # Recalculate the change in ice height measurement after correction
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)

        # NOTE Correction 8
        # Correct snow height to 0 when ice is melting i.e., when fixed sonic ranger height is decreasing.
        # For 2020
        data_freya_aws.loc['2020':, ['snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        # data_freya_aws.loc[:, ['snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        # Find index when ice starts to melt i.e. where value is nan
        changeiceheight = data_freya_aws.loc[(data_freya_aws.index.year == 2020) & (data_freya_aws['iceh_corr[m]'].isna() == True), ['iceh[m]', 'iceh_corr[m]']]
        zero_start_date = changeiceheight.index[0] #'2020-07-11 00:00:00'
        zero_end_date = changeiceheight.index[-1] #'2020-09-08 00:00:00'
        data_freya_aws.loc[zero_start_date:zero_end_date, 'snowh_corr[m]'] = 0
        data_freya_aws.loc['2020':, ['snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)

        # Recalculate the change in snow and ice height measurement after correction
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        # # For 2021
        data_freya_aws.loc['2021':, ['snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)
        # Find index when ice starts to melt i.e. where value is nan
        changeiceheight = (data_freya_aws.loc[(data_freya_aws.index.year == 2021) & (data_freya_aws['diceh_corr[m]'] != 0), ['iceh[m]', 'iceh_corr[m]', 'diceh_corr[m]']]).dropna()
        zero_start_date = changeiceheight.index[0] #'2021-07-13 00:00:00'
        zero_end_date = changeiceheight.index[-1] #'2021-09-06 12:00:00'
        data_freya_aws.loc[zero_start_date:zero_end_date, 'snowh_corr[m]'] = 0
        data_freya_aws.loc['2021':, ['snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params)

        # Recalculate the change in snow and ice height measurement after correction
        data_freya_aws['diceh_corr[m]'] = data_freya_aws['iceh_corr[m]'].diff(1)
        data_freya_aws['dsnowh_corr[m]'] = data_freya_aws['snowh_corr[m]'].diff(1)

        # NOTE Correction 9
        # Refine snow and ice height
        data_freya_aws.loc['2021':, ['iceh_corr[m]', 'iceh[m]']].plot(**plot_params)
        iceh_min_2021 = np.min(data_freya_aws.loc[data_freya_aws.index.year == 2021, 'iceh_corr[m]'])
        data_freya_aws.loc[(data_freya_aws.index.year == 2021) & (data_freya_aws['iceh_corr[m]'] ==  iceh_min_2021), 'iceh_corr[m]'] = np.nan
        data_freya_aws.loc['2021':, ['iceh_corr[m]', 'iceh[m]']].plot(**plot_params)

        # NOTE Correction 10
        # Refine snow and ice height with rolling mean
        data_freya_aws['iceh_corr_roll_mean'] = data_freya_aws['iceh_corr[m]'].rolling(6).mean() # 1 hour = 60 minutes
        data_freya_aws['snowh_corr_roll_mean'] = data_freya_aws['snowh_corr[m]'].rolling(6).mean() # 1 hour = 60 minutes
        data_freya_aws[['snowh_corr[m]', 'snowh_corr_roll_mean']].plot(**plot_params)
        data_freya_aws[['iceh_corr[m]', 'iceh_corr_roll_mean']].plot(**plot_params)

        # NOTE Correction 11 - No additional correction but just filling in the snow height measurement during the gap period when station was covered in snow and fixed USR was measuring ice height by extrapolating from the known snow height data using linear regression.
        fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
        data_freya_aws.loc['2019-06-10 00:00:00':'2019-07-31 12:00:00', ['snowh[m]', 'snowh_corr[m]']].plot(**plot_params, ax=ax)

        snowh_to_fillin = pd.DataFrame(data_freya_aws.loc['2019-06-10 00:00:00':'2019-07-30 12:00:00', ['snowh[m]', 'snowh_corr[m]']])
        snowh_to_fillin['roll_mean'] = (snowh_to_fillin['snowh_corr[m]'].rolling(window='1h', min_periods=6).mean()) # 1 hour = 60 minutes
        snowh_to_fillin['linear_regression_indicator'] = np.array([i for i in range(len(snowh_to_fillin))])
        snowh_to_fillin_na = np.where(snowh_to_fillin['snowh_corr[m]'].isna())
        snowh_to_fillin.dropna(inplace=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        snowh_to_fillin[['snowh_corr[m]', 'roll_mean']].plot(**plot_params, ax=ax)

        # Fit the linear regression model to the snow height data to extrapolate the snow height during the gap period
        x = snowh_to_fillin['linear_regression_indicator'].values
        y = snowh_to_fillin['roll_mean'].values
        slope, intercept, r_value, p_value, std_err = linregress(x, y)

        x_to_fillin = np.array(range(snowh_to_fillin['linear_regression_indicator'].iloc[0], snowh_to_fillin['linear_regression_indicator'].iloc[-1] + 1)) # Fill in for the whole period of 2019-06-10 00:00:00 to 2019-07-30 12:00:00
        snowh_filledin = slope * x_to_fillin + intercept

        gap_index = pd.date_range(start=snowh_to_fillin.index[0],
                                end=snowh_to_fillin.index[-1],  # first and last missing timestamp
                                freq='10min')
                                #   periods=len(snowh_filledin),
        snowh_filledin = pd.DataFrame({'snowh_corr[m]': snowh_filledin}, index=gap_index)

        # Add a new column to indicate the extrapolated snow height values for visualization
        data_freya_aws['lr_filled_in'] = np.nan
        data_freya_aws.loc[snowh_filledin.index, 'lr_filled_in'] = snowh_filledin['snowh_corr[m]']

        fig, ax = plt.subplots(figsize=(10, 6))
        data_freya_aws[['snowh_corr[m]', 'snowh[m]', 'lr_filled_in']].plot(**plot_params, ax=ax)

        fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
        ax.plot(snowh_filledin.index, snowh_filledin['snowh_corr[m]'], marker='o', lw=0, ms=0.9, label='Snow Height Filled in by Linear Regression')
        data_freya_aws.loc[:, ['snowh[m]','snowh_corr[m]', 'iceh[m]', 'iceh_corr[m]']].plot(**plot_params, ax=ax)
        ax.set_xlim('2019-06-09 00:00:00', '2019-07-31 12:00:00')
        ax.legend()

        # Final correction: Set snow height to 0 when ice is melting i.e., when fixed sonic ranger height is decreasing.
        fig, ax = plt.subplots(figsize=(10, 6), dpi=600)
        ax.plot(data_freya_aws.index, data_freya_aws['iceh[m]'], marker='o', lw=0, ms=2, color='blue', label='Raw Ice Height')
        ax.plot(data_freya_aws.index, data_freya_aws['iceh_corr[m]'], marker='o', lw=0, ms=0.9, color='cyan', label='Ice Height Corrected')
        ax.plot(data_freya_aws.index, data_freya_aws['snowh[m]'], marker='o', lw=0, ms=2, color='orange', label='Raw Snow Height')
        ax.plot(data_freya_aws.index, data_freya_aws['snowh_corr[m]'], marker='o', lw=0, ms=0.9, color='red', label='Snow Height Corrected')
        # ax.plot(data_freya_aws.index, data_freya_aws['lr_filled_in'], marker='o', lw=0, ms=0.9, color='green', label='Snow Height Filled in')
        ax.plot(data_freya_aws.index, data_freya_aws['lr_extrapolated'], marker='o', lw=0, ms=0.9, color='green', label='Snow Height Extrapolated')
        # ax.plot(snowh_filledin.index, snowh_filledin['snowh_corr[m]'], marker='o', lw=0, ms=0.9, color='purple', label='Snow Height Filled in')
        plt.legend()
        # plt.title('Freya Glacier Snow Height and Ice Height Measurement with Sonic Ranger Installed on the Floating AWS after Correction')

        # # -----------------------------------------------------
        # # Correction from here -------------------------------
        # # -----------------------------------------------------
        # # Step 1 -------------
        # # NOTE: In order to connect from '2019-06-23 20:40:00' to '2019-06-24 00:00:00' in snow height, we need to fill up 0.43 m snow melt in 3 hours which is not possible on the glacier in the midnight sun.
        # # data_freya_aws.loc['2019-06-23 20:40:00', 'snowh_corr[m]'] - data_freya_aws.loc['2019-06-24 00:00:00', 'snowh_corr[m]']
        # # Result: np.float64(0.42999999999999994)
        # fig, ax = plt.subplots(figsize=(10, 6))
        # data_freya_aws.loc['2019-5':'2019-9', ['snowh[m]','snowh_corr[m]', 'iceh[m]','iceh_corr[m]',]].plot(**plot_params, ax=ax)

        # # Step 2 --------------
        # fig, ax = plt.subplots(figsize=(10, 6))
        # data_freya_aws.loc['2019-7-1':'2019-7-29', 'snowh_corr[m]'].plot(**plot_params, ax=ax)
        # snow_section_corr = pd.DataFrame(data_freya_aws.loc['2019-7-1':'2019-7-29', 'snowh_corr[m]'])
        # snow_section_corr['rollmean'] = snow_section_corr['snowh_corr[m]'].rolling(6).mean() # 1 hour = 60 minutes
        # # snow_section_corr.plot(**plot_params, ax=ax)
        # snow_section_corr['diff_snowh_rollmean'] = (snow_section_corr['snowh_corr[m]'] - snow_section_corr['rollmean']).abs()
        # fig, ax = plt.subplots(figsize=(10, 6))
        # snow_section_corr['diff_snowh_rollmean'].plot(**plot_params, ax=ax)
        # snow_section_corr_index = snow_section_corr[(snow_section_corr['diff_snowh_rollmean']).abs() >= 0.2].index
        # data_freya_aws.loc[snow_section_corr_index, 'snowh_corr[m]'] = np.nan

        # fig, ax = plt.subplots(figsize=(10, 6))
        # data_freya_aws.loc['2019-5':'2019-9', ['iceh[m]', 'iceh_corr[m]', 'snowh[m]','snowh_corr[m]', ]].plot(**plot_params, ax=ax)
        save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed'

        # Calculate hourly mean snow and ice height
        variable_names = ['snowh_corr[m]', 'iceh_corr[m]', 'lr_extrapolated', 'lr_filled_in']
        freya_snow_ice_height = pd.DataFrame(data_freya_aws[variable_names])
        freya_snow_ice_height_hour =  ss.hourly_statistics(data=freya_snow_ice_height[variable_names], method='mean')
        # Save hourly corrected dataframe
        freya_snow_ice_height_hour.to_csv(f"{save_path}/freya_snow_ice_height_hour.csv")

        # Calculate daily mean snow and ice height
        freya_snow_ice_height_day =  ss.daily_statistics(data=freya_snow_ice_height[variable_names], method='mean')
        # Save daily corrected dataframe
        freya_snow_ice_height_day.to_csv(f"{save_path}/freya_snow_ice_height_day.csv")

        # %%
class QualityCheckZackenbergAWSVariables:
    def preprocess_wind_speed_and_direction():
        """Preprocess wind speed and direction.
        Deprecated: This function is not used in the final paper since we found out that the wind speed and direction data from Zackenberg AWS have many missing values and are not reliable for the analysis. We decided to use the ERA5 reanalysis data for wind speed and direction instead. However, we keep this function here for reference and potential future use when more reliable wind data from Zackenberg AWS are available.

        """
        # %%
        print("This function is deprecated. Use the function 'preprocessed_and_combine_zac_variables' instead.")
        # Read data path
        raw_aws_data_path = Path('/nas/data/raw/gem/climatebasis')
        # Save data path
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read 10 minutes wind speed
        zac_ws = pd.read_csv(f"{raw_aws_data_path}/Wind_speed_200cm_10min_average/Wind speed @ 200cm - 10min average (m_s)_10.17897_M0Q9-S706_data.txt", sep='\t', na_values='-9999')
        zac_ws['Date'] = zac_ws['Date'] + ' ' + zac_ws['Time']
        zac_ws.index = pd.to_datetime(zac_ws.Date)
        zac_ws.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_ws.index.to_series().diff().mode()[0]
        zac_ws = zac_ws.resample(freq).mean()
        zac_ws.rename(columns={'WS (m/s)': 'ws'}, inplace=True)

        # Calculate hourly mean
        zac_ws_hour = ss.hourly_calculation(data=zac_ws['ws'], method='mean')
        zac_ws_hour.plot()
        zac_ws_hour.to_csv(f"{save_path}/zac_ws_hour.csv")

        # Calculate daily wind speed
        zac_ws_day = ss.daily_calculation(data=zac_ws['ws'], method='mean')
        zac_ws_day.to_csv(f"{save_path}/zac_ws_day.csv")

        # Read 10 minutes wind direction
        zac_wd = pd.read_csv(f"{raw_aws_data_path}/Wind_direction_750cm_10min_average/Wind direction @ 750cm - 10min average (°)_10.17897_DT8E-G216_data.txt", sep='\t', na_values='-9999')
        zac_wd['Date'] = zac_wd['Date'] + ' ' + zac_wd['Time']
        zac_wd.index = pd.to_datetime(zac_wd.Date)
        zac_wd.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_wd.index.to_series().diff().mode()[0]
        zac_wd = zac_wd.resample(freq).mean()
        zac_wd.rename(columns={'WD (°)': 'wd'}, inplace=True)

        # Calculate hourly wind direction
        zac_wd_hour = vs.wind_direction_hourly_mean(ws=zac_ws['ws'], wd=zac_wd['wd'])
        zac_wd_hour.to_csv(f"{save_path}/zac_wd_hour.csv")

        # Calculate daily wind direction
        zac_wd_day = vs.wind_direction_daily_mean(ws=zac_ws['ws'], wd=zac_wd['wd'])
        zac_wd_day.to_csv(f"{save_path}/zac_wd_day.csv")

    # %%
    def preprocess_air_temperature():
        """Preprocess 2-m air temperature.

        """
        # %%
        print("This function is deprecated and not used in the final paper since we found out that the wind speed and direction data from Zackenberg AWS have many missing values and are not reliable for the analysis. We decided to use the ERA5 reanalysis data for wind speed and direction instead. However, we keep this function here for reference and potential future use when more reliable wind data from Zackenberg AWS are available.")
        # Data path
        raw_aws_data_path = Path('/nas/data/raw/gem/climatebasis')
        # Save data path
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read 2-m air temperature
        zac_t2m_200 = pd.read_csv(f"{raw_aws_data_path}/Air_temperature_200cm/Air temperature, 200cm @ 60_30min sample (°C)_10.17897_G5WS-0W04_data.txt", sep='\t', na_values='-9999')
        zac_t2m_200['Date'] = zac_t2m_200['Date'] + ' ' + zac_t2m_200['Time']
        zac_t2m_200.index = pd.to_datetime(zac_t2m_200.Date)
        zac_t2m_200.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_t2m_200.index.to_series().diff().mode()[0]
        zac_t2m_200 = zac_t2m_200.resample(freq).mean()
        zac_t2m_200.rename(columns={'AT (°C)': 't2m'}, inplace=True)

        # Calculate hourly mean
        zac_t2m_hour = ss.hourly_statistics(data=zac_t2m_200['t2m'], method='mean')
        zac_t2m_hour.plot()
        zac_t2m_hour.to_csv(f"{save_path}/zac_t2m_hour.csv")

        # Calculate daily air temperature
        zac_t2m_day = ss.daily_statistics(data=zac_t2m_200['t2m'], method='mean')
        zac_t2m_day.to_csv(f"{save_path}/zac_t2m_day.csv")

        # Read 7-m air temperature
        zac_t2m_750 = pd.read_csv(f"{raw_aws_data_path}/Air_temperature_750cm/Air temperature, 750cm @ 60_30min sample (°C)_10.17897_9V7J-Z845_data.txt", sep='\t', na_values='-9999')
        zac_t2m_750['Date'] = zac_t2m_750['Date'] + ' ' + zac_t2m_750['Time']
        zac_t2m_750.index = pd.to_datetime(zac_t2m_750.Date)
        zac_t2m_750.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_t2m_750.index.to_series().diff().mode()[0]
        zac_t2m_750 = zac_t2m_750.resample(freq).mean()
        zac_t2m_750.rename(columns={'AT (°C)': 't7m'}, inplace=True)

        # Calculate hourly mean
        zac_t2m_hour = ss.hourly_statistics(data=zac_t2m_750['t7m'], method='mean')
        zac_t2m_hour.plot()
        zac_t2m_hour.to_csv(f"{save_path}/zac_t7m_hour.csv")

        # Calculate daily air temperature
        zac_t2m_day = ss.daily_statistics(data=zac_t2m_750['t7m'], method='mean')
        zac_t2m_day.to_csv(f"{save_path}/zac_t7m_day.csv")

    def  preprocess_relative_humidity():
        """Preprocess relative humidity.

        """
        # %%
        print("This function is deprecated. Use the function 'preprocessed_and_combine_zac_variables' instead.")

        # Read data path
        raw_aws_data_path = Path('/nas/data/raw/gem/climatebasis')
        # Save data path
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read 2-m relative humidity``
        zac_rh = pd.read_csv(f"{raw_aws_data_path}/Relative_humidity_200cm/Relative humidity @ 200cm - 60_30min sample (%)_10.17897_CSHM-R138_data.txt", sep='\t', na_values='-9999')
        zac_rh['Date'] = zac_rh['Date'] + ' ' + zac_rh['Time']
        zac_rh.index = pd.to_datetime(zac_rh.Date)
        zac_rh.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_rh.index.to_series().diff().mode()[0]
        zac_rh = zac_rh.resample(freq).mean()
        zac_rh.rename(columns={'RH (%)': 'rh'}, inplace=True)

        # Calculate hourly relative humidity
        zac_rh_hour = ss.hourly_statistics(data=zac_rh['rh'], method='mean')
        zac_rh_hour.plot()
        zac_rh_hour.to_csv(f"{save_path}/zac_rh_hour.csv")

        # Calculate daily relative humidity
        zac_rh_day = ss.daily_statistics(data=zac_rh['rh'], method='mean')
        zac_rh_day.to_csv(f"{save_path}/zac_rh_day.csv")

    def preprocess_atmospheric_pressure():
        """Preprocess atmospheric pressure in hPa.

        """
        # %%
        print("This function is deprecated. Use the function 'preprocessed_and_combine_zac_variables' instead.")

        # Read data path
        raw_aws_data_path = Path('/nas/data/raw/gem/climatebasis')
        # Save data path
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read 2-m relative humidity
        zac_atm_pressure = pd.read_csv(f"{raw_aws_data_path}/Air_pressure_200cm/Air pressure @ 200cm - 60min sample (hPa)_10.17897_WG9V-3J16_data.txt", sep='\t', na_values='-9999')
        zac_atm_pressure['Date'] = zac_atm_pressure['Date'] + ' ' + zac_atm_pressure['Time']
        zac_atm_pressure.index = pd.to_datetime(zac_atm_pressure.Date)
        zac_atm_pressure.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        freq = zac_atm_pressure.index.to_series().diff().mode()[0]
        zac_atm_pressure = zac_atm_pressure.resample(freq).mean()
        zac_atm_pressure.rename(columns={'Pressure (hPa)': 'atm_pressure'}, inplace=True)

        # Calculate hourly atmospheric pressure
        zac_atm_pressure_hour = ss.hourly_statistics(data=zac_atm_pressure['atm_pressure'], method='mean')
        zac_atm_pressure_hour.plot()
        zac_atm_pressure_hour.to_csv(f"{save_path}/zac_atm_pressure_hour.csv")

        # Calculate daily atmospheric pressure
        zac_atm_pressure_day = ss.daily_statistics(data=zac_atm_pressure['atm_pressure'], method='mean')
        zac_atm_pressure_day.to_csv(f"{save_path}/zac_atm_pressure_day.csv")

    def calculate_specific_humidity():
        """Calculate specific humidity.

        """
        # %%
        print("This function is deprecated. Use the function 'preprocessed_and_combine_zac_variables' instead.")

        # Read T2m, RH, and pressure
        # Data path
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read 2-m air temperature
        zac_t2m_hour = pd.read_csv(f"{input_path}/zac_t2m_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_t2m_hour.index = pd.to_datetime(zac_t2m_hour.index)
        # Resample to uniform frequency
        freq = zac_t2m_hour.index.to_series().diff().mode()[0]
        zac_t2m_hour = zac_t2m_hour.resample(freq).mean()

        # Read relative humidity
        zac_rh_hour = pd.read_csv(f"{input_path}/zac_rh_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_rh_hour.index = pd.to_datetime(zac_rh_hour.index)
        # Resample to uniform frequency
        freq = zac_rh_hour.index.to_series().diff().mode()[0]
        zac_rh_hour = zac_rh_hour.resample(freq).mean()

        # Read atmospheric pressure
        zac_atm_pressure_hour = pd.read_csv(f"{input_path}/zac_atm_pressure_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_atm_pressure_hour.index = pd.to_datetime(zac_atm_pressure_hour.index)
        # Resample to uniform frequency
        freq = zac_atm_pressure_hour.index.to_series().diff().mode()[0]
        zac_atm_pressure_hour = zac_atm_pressure_hour.resample(freq).mean()

        # Merge RH, pressure, and temperature into a new DataFrame
        variables_to_calculate_specific_humidity = pd.concat([zac_rh_hour, zac_atm_pressure_hour, zac_t2m_hour], axis=1)
        zac_q_hour = dcaws.calculate_specific_humidity(
                        T=variables_to_calculate_specific_humidity.t2m, p=variables_to_calculate_specific_humidity.atm_pressure, RH_cor=variables_to_calculate_specific_humidity.rh)
        zac_q_hour = pd.DataFrame(zac_q_hour, columns=['q2m'], index=variables_to_calculate_specific_humidity.index)
        # Save hourly q
        zac_q_hour.to_csv(f"{input_path}/zac_q2m_hour.csv")

        # Calculate daily q
        zac_q_day = ss.daily_statistics(data=zac_q_hour, method='mean')
        # Save daily corrected dataframe
        zac_q_day.to_csv(f"{input_path}/zac_q2m_day.csv")

    def preprocessed_and_combine_zac_variables():
        """Preprocess and combine Zackenberg AWS variables.

        """
        # %%
        # Read data path
        raw_aws_data_path = Path('/nas/data/raw/gem/climatebasis')

        # ------------------------------------
        # Wind Speed
        # Read 10 minutes wind speed
        zac_ws = pd.read_csv(f"{raw_aws_data_path}/Wind_speed_200cm_10min_average/Wind speed @ 200cm - 10min average (m_s)_10.17897_M0Q9-S706_data.txt", sep='\t', na_values='-9999')
        zac_ws['Date'] = zac_ws['Date'] + ' ' + zac_ws['Time']
        zac_ws.index = pd.to_datetime(zac_ws.Date)
        zac_ws.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        # dt = zac_ws.index.to_series().diff()
        # mask = (dt.shift(1) > pd.Timedelta('10min'))
        # change_points = zac_ws.index[mask]
        # print(change_points)
        freq = zac_ws.index.to_series().diff().mode()[0]
        zac_ws = zac_ws.resample(freq).mean()
        zac_ws.rename(columns={'WS (m/s)': 'ws'}, inplace=True)
        # Calculate hourly mean for 10 minutes data frequency.
        zac_ws_hour = ss.hourly_calculation(data=zac_ws['ws'], method='mean', max_values_per_hour=6)

        fig, ax = plt.subplots(figsize=(10, 6))
        zac_ws.plot(ax=ax)
        zac_ws_hour.plot(ax=ax)

        # ------------------------------------
        # Wind Direction
        # Read 10 minutes wind direction
        zac_wd = pd.read_csv(f"{raw_aws_data_path}/Wind_direction_750cm_10min_average/Wind direction @ 750cm - 10min average (°)_10.17897_DT8E-G216_data.txt", sep='\t', na_values='-9999')
        zac_wd['Date'] = zac_wd['Date'] + ' ' + zac_wd['Time']
        zac_wd.index = pd.to_datetime(zac_wd.Date)
        zac_wd.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        # dt = zac_wd.index.to_series().diff()
        # mask = (dt.shift(1) > pd.Timedelta('10min'))
        # change_points = zac_wd.index[mask]
        # print(change_points)
        freq = zac_wd.index.to_series().diff().mode()[0]
        zac_wd = zac_wd.resample(freq).mean()
        zac_wd.rename(columns={'WD (°)': 'wd'}, inplace=True)
        # Calculate hourly wind direction
        zac_wd_hour = vs.calculate_wind_direction_hourly_mean(ws=zac_ws['ws'], wd=zac_wd['wd'])

        fig, ax = plt.subplots(figsize=(10, 6))
        zac_wd.plot(ax=ax)
        zac_wd_hour.plot(ax=ax)

        # ------------------------------------
        # Air temperature
        # Read 2-m air temperature
        zac_t2m = pd.read_csv(f"{raw_aws_data_path}/Air_temperature_200cm/Air temperature, 200cm @ 60_30min sample (°C)_10.17897_G5WS-0W04_data.txt", sep='\t', na_values='-9999')
        zac_t2m['Date'] = zac_t2m['Date'] + ' ' + zac_t2m['Time']
        zac_t2m.index = pd.to_datetime(zac_t2m.Date)
        zac_t2m.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        zac_t2m.dropna(inplace=True)

        # ZAC T2m data has some irregular frequency, so resample to uniform frequency
        # Frequency of the data before and after 2009 is different, so resample separately and then combine
        # 1 hour frequency before 2009 and 30-min frequency after 2009
        dt = zac_t2m.index.to_series().diff()
        # mask = (dt == pd.Timedelta('30min')) & (dt.shift(1) != pd.Timedelta('30min'))
        # mask = (dt == pd.Timedelta('30min')) & (dt.shift(1) > pd.Timedelta('1day'))
        mask = (dt.shift(1) == pd.Timedelta('1hr')) & (dt > pd.Timedelta('30day'))
        change_points = zac_t2m.index[mask]
        print(change_points)
        change_point = zac_t2m.index[mask][0]

        # Before 2009
        zac_t2m_before = zac_t2m.loc[zac_t2m.index < change_point]
        # zac_t2m_before = zac_t2m.loc[:'2008-12-31 23:50:00']
        freq = zac_t2m_before.index.to_series().diff().mode()[0]
        zac_t2m_before = zac_t2m_before.resample(freq).mean()

        # After 2009
        zac_t2m_after = zac_t2m.loc[change_point: ]
        freq = zac_t2m_after.index.to_series().diff().mode()[0]
        zac_t2m_after = zac_t2m_after.resample(freq).mean()
        # Calculate hourly mean for after 2009 since it has 30 mins frequency.
        zac_t2m_after = ss.hourly_calculation(data=zac_t2m_after['AT (°C)'], method='mean', max_values_per_hour=2)

        # Combine before and after 2009 data
        zac_t2m_hour = pd.concat([zac_t2m_before, zac_t2m_after])
        zac_t2m_hour.rename(columns={'AT (°C)': 't2m'}, inplace=True)
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        zac_t2m.plot(ax=ax)
        zac_t2m_hour.plot(ax=ax)

        # ------------------------------------
        # Relative Humidity
        # Read 2-m relative humidity
        zac_rh = pd.read_csv(f"{raw_aws_data_path}/Relative_humidity_200cm/Relative humidity @ 200cm - 60_30min sample (%)_10.17897_CSHM-R138_data.txt", sep='\t', na_values='-9999')
        zac_rh['Date'] = zac_rh['Date'] + ' ' + zac_rh['Time']
        zac_rh.index = pd.to_datetime(zac_rh.Date)
        zac_rh.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)

        dt = zac_rh.index.to_series().diff()
        mask = (dt == pd.Timedelta('30min')) & (dt.shift(1) == pd.Timedelta('1h'))
        change_points = zac_rh.index[mask]
        print(change_points)

        # Before
        zac_rh_before = zac_rh.loc[:'2014-06-13 23:00:00']
        freq = zac_rh_before.index.to_series().diff().mode()[0]
        zac_rh_before = zac_rh_before.resample(freq).mean()

        # After
        zac_rh_after = zac_rh.loc['2014-06-14 00:00:00': ]
        freq = zac_rh_after.index.to_series().diff().mode()[0]
        zac_rh_after = zac_rh_after.resample(freq).mean()
        # Calculate hourly mean for after 2014 since it has 30 mins frequency.
        zac_rh_after = ss.hourly_calculation(data=zac_rh_after['RH (%)'], method='mean', max_values_per_hour=2)

        # Combine before and after data
        zac_rh_hour = pd.concat([zac_rh_before, zac_rh_after])
        zac_rh_hour.rename(columns={'RH (%)': 'rh'}, inplace=True)
        # Calculate hourly mean
        fig, ax = plt.subplots(figsize=(10, 6))
        zac_rh.plot(ax=ax)
        zac_rh_hour.plot(ax=ax)

        # ------------------------------------
        # Atmospheric Pressure
        zac_atm_pressure = pd.read_csv(f"{raw_aws_data_path}/Air_pressure_200cm/Air pressure @ 200cm - 60min sample (hPa)_10.17897_WG9V-3J16_data.txt", sep='\t', na_values='-9999')
        zac_atm_pressure['Date'] = zac_atm_pressure['Date'] + ' ' + zac_atm_pressure['Time']
        zac_atm_pressure.index = pd.to_datetime(zac_atm_pressure.Date)
        zac_atm_pressure.drop(columns=['Date', 'Time', 'Quality Flag'], inplace=True)
        # dt = zac_atm_pressure.index.to_series().diff()
        # mask = (dt > pd.Timedelta('1hr'))
        # change_points = zac_atm_pressure.index[mask]
        # print(change_points)
        freq = zac_atm_pressure.index.to_series().diff().mode()[0]
        zac_atm_pressure = zac_atm_pressure.resample(freq).mean()
        zac_atm_pressure.rename(columns={'Pressure (hPa)': 'atm_pressure'}, inplace=True)

        # Calculate hourly atmospheric pressure
        zac_atm_pressure_hour = ss.hourly_calculation(data=zac_atm_pressure['atm_pressure'], method='mean', max_values_per_hour=1)
        zac_atm_pressure_hour.plot()

        # ------------------------------------
        # Specific Humidity
        # Merge RH, pressure, and temperature into a new DataFrame
        variables_to_calculate_specific_humidity = pd.concat([zac_t2m_hour, zac_rh_hour, zac_atm_pressure_hour], axis=1)
        zac_q_hour = dcaws.calculate_specific_humidity(
                        T=variables_to_calculate_specific_humidity.t2m, p=variables_to_calculate_specific_humidity.atm_pressure, RH_cor=variables_to_calculate_specific_humidity.rh)
        zac_q_hour = pd.DataFrame(zac_q_hour, columns=['q2m'], index=variables_to_calculate_specific_humidity.index)
        zac_q_hour.plot()

        # Combine all variables into one DataFrame
        # Save data path
        save_path = Path('/nas/data/processed/people/sos/third_paper_data/processed_aws')
        zac_combined = pd.concat([zac_ws_hour, zac_wd_hour, zac_t2m_hour, zac_rh_hour, zac_atm_pressure_hour, zac_q_hour], axis=1)
        zac_combined.to_csv(f"{save_path}/zac_aws_hour.csv")

        # %%

class RacmoWindSpeedDirectionPreprocessor:
    def calculate_wind_speed_direction_from_racmo_u_v():
        """Calculate wind speed and direction from RACMO u (zonal) and v (meridional) wind components.

        """
        # %%
        # Read u and v wind components from RACMO
        # ZAC data path
        base_input_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_zac')
        zac_u = pd.read_csv(f"{base_input_path}/u10m_point_zac_5km.csv", parse_dates=['time'], index_col='time')
        zac_v = pd.read_csv(f"{base_input_path}/v10m_point_zac_5km.csv", parse_dates=['time'], index_col='time')
        # Calculate wind speed and direction from u and v wind components
        zac_ws_racmo_calc, zac_wd_racmo = vs.calculate_wind_speed_direction_from_u_v(zac_u, zac_v)
        # Save calculated wind speed and direction
        zac_wd_racmo.to_csv(f"{base_input_path}/wd_point_zac_5km.csv")

        # # Test ff10m_point_zac_5km.csv with calculated wind speed
        # zac_ff10m = pd.read_csv(f"{base_input_path}/ff10m_point_zac_5km.csv", parse_dates=['time'], index_col='time')
        # ff10m_minus_calculated_ws = zac_ff10m['ff10m'] - zac_ws_racmo_calc['ws']
        # ff10m_minus_calculated_ws.plot()

        # Freya Glacier data path
        base_input_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        fre_u = pd.read_csv(f"{base_input_path}/u10m_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
        fre_v = pd.read_csv(f"{base_input_path}/v10m_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
        # Calculate wind speed and direction from u and v wind components
        fre_ws_racmo_calc, fre_wd_racmo = vs.calculate_wind_speed_direction_from_u_v(fre_u['u10m_closest_point_racmo'], fre_v['v10m_closest_point_racmo'])
        # Save calculated wind speed and direction
        fre_wd_racmo.to_csv(f"{base_input_path}/wd_masked_freya_5km.csv")
        fre_ws_racmo_calc.to_csv(f"{base_input_path}/ws_masked_freya_5km.csv")

        # # # Test ff10m_masked_freya_5km.csv with calculated wind speed
        # fre_ff10m = pd.read_csv(f"{base_input_path}/ff10m_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
        # ff10m_minus_calculated_ws = fre_ff10m['ff10m_closest_point_racmo'] - fre_ws_racmo_calc['ws']
        # ff10m_minus_calculated_ws.plot()



        # %%
if __name__ == '__main__':
    # SVDDataPreprocessor.calculate_weekly_racmo()
    # SVDDataPreprocessor.calculate_weekly_racmo_deseasonalize_detrend()
    # SVDDataPreprocessor.merge_one_year()
    # SVDDataPreprocessor.coarsen_data_month()
    # SVDDataPreprocessor.coarsen_data_year()
    # SVDDataPreprocessor.merge_MJJAS_coarse()
    # ExtractVariablesFromRacmo.extract_5km_variable_within_sea_ice_buffer()
    # ExtractVariablesFromRacmo.extract_variable_freya_1km()
    # ExtractVariablesFromRacmo.extract_variable_whole_freya_5km()
    # CreateRacmoMasks.convert_sea_ice_buffer_from_shp_to_nc()
    # SVDDataPreprocessor.merge_AMJJASO_coarse()
    # SVDDataPreprocessor.coarsen_lat_lon()
    # ExtractVariablesFromRacmo.extract_5km_variable_within_sea_ice_buffer()
    # ExtractVariablesFromRacmo.extract_5km_variable_inbetween_sea_ice_buffers()
    # ExtractVariablesFromRacmo.extract_5km_variable_at_zac()
    # SVDDataPreprocessor.calculate_mjjas_detrended_racmo()

    for var_name in ['alb']: #['psurf', 't2m', 'tskin']:
        SVDDataPreprocessor.save_seperate_month_racmo(var_name=var_name)
    # SVDDataPreprocessor.coarsen_data_month()
    # SVDDataPreprocessor.merge_MJJAS_coarse()
    # CreateRacmoMasks.create_mask_for_variables_on_sea_with_sif_phenology_trend()
    # ExtractVariablesFromRacmo.extract_5km_variable_within_inbetween_significant_sif_phenology_trend()
    # SVDDataPreprocessor.calculate_monthly_mean_all_months()
