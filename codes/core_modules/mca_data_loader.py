#!/usr/bin/env python3
# %%
from __future__ import annotations
import xarray as xr
from typing import Literal, Optional, Iterable, Union
from pathlib import Path
import xrft

# %%
def variable_stats(var_name: str) -> str:
    # stats_map = {'smb': 'SUM', 'ci': 'MEDIAN', 'z0500': 'MEAN'}
    # stats_map = {'smb': 'MEDIAN', 'ci': 'MEDIAN', 'z0500': 'MEAN'}
    stats_map = {'smb': 'SUM', 'ci': 'MEAN', 'z0500': 'MEAN'}

    try:
        return stats_map[var_name]
    except KeyError as e:
        raise ValueError(f"Unknown variable name: {var_name!r}. Expected one of {list(stats_map)}") from e

def rename_latlon_coords(
    da: xr.DataArray | xr.Dataset,
    da_original: Optional[xr.DataArray | xr.Dataset] = None,
    direction: Literal['forward', 'backward', 'auto'] = 'forward'
):
    """Rename coordinate names between (rlat, rlon, y, x) and (lat, lon).
    direction='auto' tries to infer based on present coords.
    """
    # Decide direction automatically, if requested
    if direction == 'auto':
        has_rl = {'rlat' in da.coords or 'y' in da.coords,
                  'rlon' in da.coords or 'x' in da.coords}
        has_ll = {'lat' in da.coords, 'lon' in da.coords}
        direction = 'forward' if any(has_rl) else 'backward' if any(has_ll) else 'forward'

    rename_dict = {}

    if direction == 'forward':
        # Drop existing lat/lon to avoid conflicts
        for coord in ('lat', 'lon'):
            if coord in da.coords:
                da = da.drop_vars(coord)

        if 'rlat' in da.coords: rename_dict['rlat'] = 'lat'
        if 'rlon' in da.coords: rename_dict['rlon'] = 'lon'
        if 'y'    in da.coords: rename_dict['y']    = 'lat'
        if 'x'    in da.coords: rename_dict['x']    = 'lon'

    elif direction == 'backward':
        if da_original is None:
            raise ValueError('da_original must be provided for backward renaming.')
        if 'rlat' in da_original.coords: rename_dict['lat'] = 'rlat'
        if 'rlon' in da_original.coords: rename_dict['lon'] = 'rlon'
        if 'y'    in da_original.coords: rename_dict['lat'] = 'y'
        if 'x'    in da_original.coords: rename_dict['lon'] = 'x'

    return da.rename(rename_dict) if rename_dict else da

def load_data_original(
    var_name: str,
    month_or_season: Literal['month', 'season'],
    start_year: int,
    end_year: int,
    trend_or_detrend: str,
    trend_date: Optional[str] = None,
    month: Optional[int] = None,
    season: Optional[str] = None,
    clip: Optional[str] = False,
) -> xr.DataArray:
    """Load monthly or seasonal variable data with consistent 'time' coordinate."""

    base_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset')
    stat = variable_stats(var_name)

    if month_or_season == 'month':
        if month is None:
            raise ValueError("month must be specified when month_or_season='month'")
        path = base_path / f"{var_name}/{var_name}_{month}_{stat}_{start_year}to{end_year}_{trend_or_detrend}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        da = xr.open_dataarray(path).rename({"year": "time"}) if "year" in xr.open_dataarray(path).coords else xr.open_dataarray(path)

    elif month_or_season == 'season':
        if season is None or trend_date is None:
            raise ValueError("season and trend_date must be specified when month_or_season='season'")
        path = base_path / f"{var_name}/{var_name}_{season}_{stat}_{start_year}to{end_year}_{trend_or_detrend}_{trend_date}_dates.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        da = xr.open_dataarray(path)
        if trend_date == 'single' and 'year' in da.coords:
            da = da.rename({'year': 'time'})
    else:
        raise ValueError("month_or_season must be 'month' or 'season'")

    # Ensure time is the first dimension for consistency
    if da.dims[0] != "time":
        da = da.transpose("time", ...)

    if clip == True:
        if var_name == 'smb':
            da = da.where(da.x >= 100000, drop=True)
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da = da.where(da.rlon >= -3.5, drop=True)
    return da

def load_data(
    var_name: str,
    stat_month: str,
    stat_season: str,
    month_or_season: Literal['month', 'season'],
    start_year: int,
    end_year: int,
    trend_or_detrend: str,
    month: Optional[int] = None,
    season: Optional[str] = None,
    clip: Optional[str] = False,
) -> xr.DataArray:
    """Load monthly or seasonal variable data with consistent 'time' coordinate."""

    base_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset')
    # stat = variable_stats(var_name)

    if month_or_season == 'month':
        print('Needs to revise load_data function for month stat')
        # if month is None:
        #     raise ValueError("month must be specified when month_or_season='month'")
        # path = base_path / f"{var_name}/{var_name}_{month}_{stat}_{start_year}to{end_year}_{trend_or_detrend}.nc"
        # if not path.exists():
        #     raise FileNotFoundError(path)
        # da = xr.open_dataarray(path).rename({"year": "time"}) if "year" in xr.open_dataarray(path).coords else xr.open_dataarray(path)
        raise NotImplementedError("Monthly data loading with specific stat not implemented yet.")
    elif month_or_season == 'season':
        if season is None:
            raise ValueError("season and trend_date must be specified when month_or_season='season'")
        path = base_path / f"{var_name}/{var_name}_{season}_{stat_month}_{stat_season}_{start_year}to{end_year}_{trend_or_detrend}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        da = xr.open_dataarray(path)
        if 'year' in da.coords:
            da = da.rename({'year': 'time'})
    else:
        raise ValueError("month_or_season must be 'month' or 'season'")

    # Ensure time is the first dimension for consistency
    if da.dims[0] != "time":
        da = da.transpose("time", ...)

    if clip == True:
        if var_name == 'smb':
            da = da.where(da.x >= 100000, drop=True)
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da = da.where(da.rlon >= -3.5, drop=True)
    return da

# %%
def get_aggregation(var_name):
    """Return aggregation method for a variable.

    """
    mapping = {'smb': 'sum', 'ci': 'median', 'z0500': 'mean'}
    if var_name not in mapping:
        raise ValueError(f'Unknown variable name: {var_name}')
    return mapping[var_name]

def load_weekly_data(
    var_name: str,
    pattern: Literal['left', 'right'],
    lag: int,
    decomposition_type: Literal['trend', 'detrended', 'detrended_deseasonalized'],
    aggregation: Literal['sum', 'mean', 'median'],
    time_period: Literal['JJAS', 'JJ', 'JA', 'AS'],
    start_year: int = 1990,
    end_year: int = 2019,
    clip: Optional[str] = False,
    ) -> xr.DataArray:
    """Load weekly variable data with consistent 'time' coordinate."""

    # %%
    # var_name = 'z0500'
    # pattern = 'left'  # 'left' or 'right'
    # lag = 1
    # decomposition_type = 'trend'
    # aggregation = get_aggregation(var_name)
    # time_period = 'JJA'  # 'JJAS', 'JJ', 'JA', 'AS'
    # start_year = 1990
    # end_year = 2019
    # clip = False

    base_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset')
    path = base_path / f"{var_name}/{var_name}_{time_period}_{aggregation}_week_{start_year}to{end_year}_trend.nc"
    da_week = xr.open_dataarray(path)

    if clip == True:
        if var_name == 'smb':
            da_week = da_week.where(da_week.x >= 100000, drop=True)
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da_week = da_week.where(da_week.rlon >= -3.5, drop=True)

    if time_period == 'JJAS':
        # Slice weeks from 1 June to 30 August (weeks 1 to 13) ---
        first_week = 1 # 1 June - 7 June
        last_week = 13 # 24 August - 30 August
    elif time_period in ['JJ', 'JA', 'AS']:
        first_week = 1  # 1 June - 7 June
        last_week = 4   # 22 June - 28 June

    # Lag only applies to the right pattern
    if pattern == 'left':
        da_week = da_week.sel(week=slice(first_week, last_week))
    elif pattern == 'right':
        # Slice weeks based on lag for right pattern
        da_week = da_week.sel(week=slice(first_week + lag, last_week + lag))  # Apply lag by slicing

    # Apply decomposition
    if decomposition_type  == 'trend':
        print(f"Using trend {var_name} weekly {aggregation} for period {start_year}-{end_year}")
        da_decomposition = da_week.copy()
    elif decomposition_type  == 'detrended':
        # Detrend the original data
        print(f"Detrending {var_name} weekly {aggregation} detrended for period {start_year}-{end_year}")
        da_decomposition = xrft.detrend(da_week, dim=['year', 'week'], detrend_type='linear')
    elif decomposition_type  == 'detrended_deseasonalized':
        print(f"Detrending and deseasonalizing {var_name} weekly {aggregation} for period {start_year}-{end_year}")
        # Detrend first
        da_week_detrended = xrft.detrend(da_week, dim=['year', 'week'], detrend_type='linear')
        # The deseasonalize the detrended data first
        da_climatology = da_week_detrended.mean(dim='year')
        da_decomposition = da_week_detrended - da_climatology  # dims: (year, week, y, x)
    else:
        raise ValueError("decomposition_type must be one of 'trend', 'detrended', or 'detrended_deseasonalized'") #'deseasonalized', or ")

    # Uncomment if needed in the future
    # elif decomposition_type  == 'deseasonalized':
    #     # Deseasonalize: compute weekly climatology across years and subtract
    #     # Climatology has dims (week, y, x)
    #     da_climatology = da_week.mean(dim='year')
    #     print(f"Deseasonalizing {var_name} weekly {aggregation} for period {start_year}-{end_year}")
    #     da_decomposition = da_week - da_climatology  # dims: (year, week, y, x)
    # %%
    return da_decomposition

def load_coarsed_mjjas_data(var_name, start_year=1990, end_year=2019, coarsen_level=0, clip=False):
    """Load coarsened data for a specific variable and time range.

    """
    print(f"Processing variable: {var_name}")
    # Define file paths based on variable name
    read_file_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/complex_mca_input'
    # if var_name == 'smb':
    #     read_file_path = Path(f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
    # elif var_name == 'z0500':
    #     read_file_path = Path(f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
    # else:
    #     read_file_path = Path(f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

    da = xr.open_dataarray(f"{read_file_path}/{var_name}_MJJAS_trended_coarsened_{coarsen_level}.nc")
    da = da.sel(time=da['time'].dt.year.isin(range(start_year, end_year + 1)))

    if clip == True:
        if var_name == 'smb':
            da = da.sel(x=slice(-638956 + 1000 * 1000, -638956 + 1000 * 1500), y=slice(-3354596. + 1000 * 1000, -3354596. + 1000 * 1200))
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da = da.sel(rlon=slice(da.rlon.values[0] + 0.05 * 250, da.rlon.values[0] + 0.05 * 350), rlat=slice(da.rlat.values[0] + 0.05 * 290, da.rlat.values[0] + 0.05 * 490))

    # Remove unnecessary coordinates
    coords_to_drop = ['height', 'lat', 'lon', 'pressure']
    da = da.drop_vars([c for c in coords_to_drop if c in da.coords])

    return da

def load_coarsed_monthly_mean_data(var_name, start_year=1990, end_year=2019, coarsen_level=0, clip=False):
    """Load coarsened data for a specific variable and time range.

    """
    print(f"Processing variable: {var_name}")
    # Define file paths based on variable name
    read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
    if var_name == 'smb':
        read_file_path = Path(f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
    elif var_name == 'z0500':
        read_file_path = Path(f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
    else:
        read_file_path = Path(f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

    da = xr.open_dataarray(f"{read_file_path}/{var_name}_trended_monthly_mean_coarsened_{coarsen_level}.nc")
    da = da.sel(time=da['time'].dt.year.isin(range(start_year, end_year + 1)))

    if clip == True:
        if var_name == 'smb':
            da = da.sel(x=slice(-638956 + 1000 * 1000, -638956 + 1000 * 1500), y=slice(-3354596. + 1000 * 1000, -3354596. + 1000 * 1200))
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da = da.sel(rlon=slice(da.rlon.values[0] + 0.05 * 250, da.rlon.values[0] + 0.05 * 350), rlat=slice(da.rlat.values[0] + 0.05 * 290, da.rlat.values[0] + 0.05 * 490))

    # Remove unnecessary coordinates
    coords_to_drop = ['height', 'lat', 'lon', 'pressure']
    da = da.drop_vars([c for c in coords_to_drop if c in da.coords])

    return da

def load_coarsed_mjjas_data_detrended(var_name, start_year=1990, end_year=2019, coarsen_level=0, clip=False):
    """Load coarsened detrended data for a specific variable and time range.

    """
    print(f"Processing variable: {var_name}")
    # Define file paths based on variable name
    read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
    if var_name == 'smb':
        read_file_path = Path(f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
    elif var_name == 'ci':
        read_file_path = Path(f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
    elif var_name == 'z0500':
        read_file_path = Path(f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

    da = xr.open_dataarray(f"{read_file_path}/{var_name}_MJJAS_detrended_coarsened_{coarsen_level}.nc")
    da = da.sel(time=da['time'].dt.year.isin(range(start_year, end_year + 1)))
    if clip == True:
        if var_name == 'smb':
            da = da.sel(x=slice(-638956 + 1000 * 1000, -638956 + 1000 * 1500), y=slice(-3354596. + 1000 * 1000, -3354596. + 1000 * 1200))
        elif (var_name == 'ci') | (var_name == 'z0500') | (var_name == 't2m') | (var_name == 'tskin'):
            da = da.sel(rlon=slice(da.rlon.values[0] + 0.05 * 250, da.rlon.values[0] + 0.05 * 350), rlat=slice(da.rlat.values[0] + 0.05 * 290, da.rlat.values[0] + 0.05 * 490))

    # Remove unnecessary coordinates
    coords_to_drop = ['height', 'lat', 'lon', 'pressure']
    da = da.drop_vars([c for c in coords_to_drop if c in da.coords])

    return da

def load_coarsed_amjjaso_data(var_name, start_year=1990, end_year=2019, coarsen_level=0, clip=False):
    """Load coarsened data for a specific variable and time range.

    """
    print(f"Processing variable: {var_name}")
    # Define file paths based on variable name
    read_file_path = '/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/'
    if var_name == 'smb':
        read_file_path = Path(f"{read_file_path}surface_levels/variable_1km/racmo_whole_domain/rearranged/onefile")
    elif var_name == 'ci':
        read_file_path = Path(f"{read_file_path}surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
    elif var_name == 'z0500':
        read_file_path = Path(f"{read_file_path}pressure_levels/variable_5km/racmo_whole_domain/rearranged/onefile")

    da = xr.open_dataarray(f"{read_file_path}/{var_name}_AMJJASO_trended_coarsened_{coarsen_level}.nc")
    da = da.sel(time=da['time'].dt.year.isin(range(start_year, end_year + 1)))
    if clip == True:
        if var_name == 'smb':
            da = da.sel(x=slice(-638956 + 1000 * 1000, -638956 + 1000 * 1500), y=slice(-3354596. + 1000 * 1000, -3354596. + 1000 * 1200))
        elif (var_name == 'ci') | (var_name == 'z0500'):
            da = da.sel(rlon=slice(da.rlon.values[0] + 0.05 * 250, da.rlon.values[0] + 0.05 * 350), rlat=slice(da.rlat.values[0] + 0.05 * 290, da.rlat.values[0] + 0.05 * 490))

    # Remove unnecessary coordinates
    coords_to_drop = ['height', 'lat', 'lon', 'pressure']
    da = da.drop_vars([c for c in coords_to_drop if c in da.coords])

    return da

# %%

# decomposition_type = 'trend'
# start_year = 1990
# end_year = 2019
# lag = 1
# clip = True
# #
# left_var_name = 'ci'
# left_aggregation = get_aggregation(var_name=left_var_name)
# left_var_da = load_weekly_data(var_name=left_var_name, pattern='left', lag=lag, decomposition_type=decomposition_type, aggregation=left_aggregation, start_year=start_year, end_year=end_year, clip=clip)
# # left_var_da_sliced = left_var_da.isel(time=slice(0, 5))  # Apply lag by slicing

# right_var_name = 'z0500'
# right_aggregation = get_aggregation(var_name=right_var_name)
# right_var_da = load_weekly_data(var_name=right_var_name, pattern='right', lag=lag, decomposition_type=decomposition_type, aggregation=right_aggregation, start_year=start_year, end_year=end_year, clip=clip)
# # right_var_da_sliced = right_var_da.isel(time=slice(0, 5))  # Apply lag by slicing

# # %%
# import xeofs as xe
# total_mode = 1
# # Fit MCA (xeofs can standardize internally)
# mca = xe.cross.MCA(n_modes=total_mode, standardize=True)
# mca.fit(left_var_da_sliced, right_var_da_sliced, dim='time')

# # # %%
# scores = mca.scores(normalized=True)                               # [left_scores, right_scores]s
# scores[0].name = 'left_scores'
# if 'time' in scores[0].indexes:
#     print('yes, time is in indexes')
#     if isinstance(scores[0].indexes['time'], pd.MultiIndex):
#         print("yes, time is a MultiIndex")
#         da_reindexed = scores[0].unstack('time')
#         da_reindexed.to_netcdf('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_xeofs_weekly/test_left_scores.nc')

