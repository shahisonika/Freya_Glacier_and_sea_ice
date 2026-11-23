#!/usr/bin/env python3
"""
Freya Glacier Analysis Level 1 (FG1) - Analyze Freya Glacier preprocessed dataset.

Author: Sonika Shahi
Updated: 28 Aug 2025

Environment: xeofs_env2 (conda activate xeofs_env2 on walrus server)
"""
# %%
import os
import sys
import glob
import numpy as np
from numpy.random import randint
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.patches import Patch, Rectangle, Polygon, Circle, ConnectionPatch
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
from matplotlib.text import Text
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy import signal
from scipy.stats import spearmanr, linregress, shapiro, normaltest, anderson, mannwhitneyu, rankdata, theilslopes, kendalltau, wilcoxon
from scipy.stats import norm
import seaborn as sns
from datetime import timedelta, datetime, timezone
import time
from tqdm import tqdm
plt.rc('font', family='Helvetica')
import xarray as xr
from xarray.coding.times import CFTimedeltaCoder
from typing import Optional
from pyproj import Transformer, CRS
# import pywt # For individual CWT if needed, but WTC functions usually handle CWT internally.
# import pycwt as wavelet
from statsmodels.tsa.stattools import adfuller, kpss, acf
import calendar
# from scipy.signal import filtfilt, butter
import ruptures as rpt
from arch.unitroot import ADF, KPSS, PhillipsPerron
import pymannkendall as mk
import pingouin as pg
import pandas as pd
from pathlib import Path
import re

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utility helpers
# -------------------------
os.chdir('/nas/home/sos/third_paper_work/codes/freyagl_analysis')
from mca_data_loader import load_data, get_aggregation
from station_data_loader import read_station_metadata, read_freya_aws_data, read_fog_classification, read_sif_data_from_racmo, read_station_data_from_racmo, read_zackenberg_aws_data

# -------------------------
# Import modules from different directory
# -------------------------
# Reference: https://www.geeksforgeeks.org/python-import-module-from-different-directory/
module_path = "/nas/home/sos/third_paper_work/codes/add_modules/"
if os.path.exists(module_path):
    sys.path.append(module_path)
    from figure_properties import FigureProperties as fp
    from statistics_calculation import scalar_statistics as ss
    from statistics_calculation import  vector_statistics as vs
    # from derive_and_correct_aws_variables import _calculate_saturation_vapor_pressure as csvp
    # import derive_and_correct_aws_variables as dcav
    from derive_and_correct_aws_variables import calculate_declination, calculate_hour_angle, calculate_zenith
else:
    raise ImportError(f"Module path '{module_path}' does not exist.")

class HandlerCircle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        # Center the circle
        center = (xdescent + width / 2, ydescent + height / 2)

        # 🔑 Increase radius to make dashes more visible!
        # Default 0.4 may be too small → try 0.5–0.6 for clarity
        radius = min(width, height) * 0.55  # ← increased from 0.4 → clearer dashes

        circle = Circle(
            center,
            radius=radius,
            edgecolor=orig_handle.get_edgecolor(),
            facecolor='none',
            linewidth=orig_handle.get_linewidth(),
            linestyle=orig_handle.get_linestyle(),  # e.g., '--', ':', '-.'
            # ✅ No 'dashes=' — not supported for Circle
        )
        return [circle]

class HandlerHistogramBars(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        bars = []
        n_bars = 5
        bar_width = width / n_bars

        # Heights that resemble a distribution shape
        heights = [0.3, 0.7, 1.0, 0.7, 0.3]

        for i, h in enumerate(heights):
            rect = Rectangle((xdescent + i * bar_width, ydescent),
                bar_width * 0.9, h * height,
                facecolor=orig_handle.get_facecolor(),
                edgecolor=orig_handle.get_edgecolor(),
                lw=orig_handle.get_linewidth(), transform=trans)
            bars.append(rect)
        return bars

class HandlerVerticalBoxPlot(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Geometry parameters (relative)
        box_y0 = ydescent + 0.2 * height
        box_y1 = ydescent + 0.8 * height
        box_x0 = xdescent + 0.3 * width
        box_x1 = xdescent + 0.7 * width

        x_mid = 0.5 * (box_x0 + box_x1)
        # Box (IQR)
        box = Rectangle((box_x0, box_y0), box_x1 - box_x0, box_y1 - box_y0, facecolor=orig_handle.get_facecolor(), edgecolor='black', lw=0.5, transform=trans)
        # Median line (horizontal across box)
        median = Line2D([box_x0, box_x1], [ydescent + 0.5 * height, ydescent + 0.5 * height], color='black', lw=1, transform=trans)
        # Whiskers (vertical)
        whisker_bottom = Line2D([x_mid, x_mid], [ydescent, box_y0], color='black', lw=1, transform=trans)
        whisker_top = Line2D([x_mid, x_mid], [box_y1, ydescent + height], color='black', lw=1, transform=trans)
        # Whisker caps
        cap_width = 0.2 * width
        cap_bottom = Line2D([x_mid - cap_width/2, x_mid + cap_width/2], [ydescent, ydescent], color='black', lw=0.5, transform=trans)
        cap_top = Line2D([x_mid - cap_width/2, x_mid + cap_width/2], [ydescent + height, ydescent + height], color='black', lw=0.5, transform=trans)

        return [whisker_bottom, whisker_top, cap_bottom, cap_top, box, median]

class HandlerVerticalBoxPlot2(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):

        # --- geometry ---
        box_y0 = ydescent + 0.2 * height
        box_y1 = ydescent + 0.8 * height
        box_x0 = xdescent + 0.3 * width
        box_x1 = xdescent + 0.7 * width

        x_mid = 0.5 * (box_x0 + box_x1)
        y_mid = ydescent + 0.5 * height

        # --- box (IQR) ---
        box = Rectangle(
            (box_x0, box_y0),
            box_x1 - box_x0,
            box_y1 - box_y0,
            facecolor=orig_handle.get_facecolor(),
            edgecolor='black',
            lw=0.5,
            transform=trans
        )

        # --- median line ---
        median = Line2D(
            [box_x0, box_x1],
            [y_mid, y_mid],
            color='black',
            lw=1,
            transform=trans
        )

        # --- whiskers ---
        whisker_bottom = Line2D([x_mid, x_mid], [ydescent, box_y0],
                                color='black', lw=1, transform=trans)
        whisker_top = Line2D([x_mid, x_mid], [box_y1, ydescent + height],
                             color='black', lw=1, transform=trans)

        cap_width = 0.2 * width
        cap_bottom = Line2D(
            [x_mid - cap_width/2, x_mid + cap_width/2],
            [ydescent, ydescent],
            color='black',
            lw=0.5,
            transform=trans
        )

        cap_top = Line2D(
            [x_mid - cap_width/2, x_mid + cap_width/2],
            [ydescent + height, ydescent + height],
            color='black',
            lw=0.5,
            transform=trans
        )

        # --- ⭐ TEXT centered exactly on median ---
        label_text = orig_handle.get_label()  # use handle label

        text = Text(
            # x=x_mid + 0.25 * width,   # <-- SHIFT RIGHT
            x=box_x1 + 0.05 * width,   # <-- SHIFT RIGHT
            y=y_mid,
            text=label_text,
            ha='left',   # <-- ALIGN LEFT
            va='center',
            fontsize=fontsize,
            transform=trans
        )

        return [whisker_bottom, whisker_top, cap_bottom, cap_top, box, median, text]

def _align_secondary_yaxis(ax, ax_twinx, left_labels, right_labels):
    """
    Align the positions of the ticks on ax_twinx (right y-axis)
    so they match the vertical screen positions of ax's ticks (left y-axis).
    Tick *values* remain independent; only positions are aligned.

    Parameters
    ----------
    ax : matplotlib Axes
        Primary axis (left y-axis).
    ax_twinx : matplotlib Axes
        Secondary axis created via ax.twinx().
    left_labels : list of str
        Labels to apply to the primary axis ticks.
    right_labels : list of str
        Labels to apply to the secondary axis ticks.
    """

    # ---- Get tick positions in data coordinates (primary axis) ----
    left_ticklocs = ax.get_yticks()

    # ---- Transform primary tick locations into display (pixel) coords ----
    left_display = ax.transData.transform(
        np.column_stack([np.zeros_like(left_ticklocs), left_ticklocs])
    )[:, 1]   # take Y coordinate

    # ---- Convert display coords → data coords for twin axis ----
    right_ticklocs = ax_twinx.transData.inverted().transform(
        np.column_stack([np.zeros_like(left_display), left_display])
    )[:, 1]

    # ---- Apply these aligned tick positions ----
    ax_twinx.set_yticks(right_ticklocs)

    # ---- Apply labels you want ----
    ax.set_yticklabels(left_labels)
    ax_twinx.set_yticklabels(right_labels)

def _align_secondary_yaxis(ax, ax_twinx, left_labels, right_labels):
    """
    Align the positions of the ticks on ax_twinx so that they match the
    vertical screen positions of ax's ticks.
    The numeric values displayed on the ticks remain independent.
    """

    # ---- Get tick positions in data coords (primary axis) ----
    left_ticklocs = ax.get_yticks()

    # ---- Transform left tick locations into display coordinates ----
    left_display = ax.transData.transform(
        np.column_stack([np.zeros_like(left_ticklocs), left_ticklocs])
    )[:, 1]

    # ---- Convert display coords → data coords for twin axis ----
    right_ticklocs = ax_twinx.transData.inverted().transform(
        np.column_stack([np.zeros_like(left_display), left_display])
    )[:, 1]

    # ---- Apply these "aligned" tick positions ----
    ax_twinx.set_yticks(right_ticklocs)

    # ---- Apply labels that YOU want ---
    ax.set_yticklabels(left_labels)
    ax_twinx.set_yticklabels(right_labels)

def _bootstrap_median_diff_ci(x, y, n_boot=10000, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Guard: empty input
    if len(x) == 0 or len(y) == 0:
        return np.nan, (np.nan, np.nan), np.array([])

    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        x_sample = rng.choice(x, size=len(x), replace=True)
        y_sample = rng.choice(y, size=len(y), replace=True)

        boot_diffs[i] = np.median(x_sample) - np.median(y_sample)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_diffs, alpha)
    upper = np.percentile(boot_diffs, 100 - alpha)

    obs = np.median(x) - np.median(y)

    return obs, (lower, upper), boot_diffs

def _bootstrap_median_diff_percentage_ci(x, y, n_boot=10000, ci=95, random_state=42):
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaNs
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    # Guard: empty input
    if len(x) == 0 or len(y) == 0:
        return np.nan, (np.nan, np.nan), np.array([])

    boot_diffs = np.empty(n_boot)

    for i in range(n_boot):
        x_sample = rng.choice(x, size=len(x), replace=True)
        y_sample = rng.choice(y, size=len(y), replace=True)

        boot_diffs[i] = ((np.median(y_sample) - np.median(x_sample)) / np.median(y_sample)) * 100  # Percentage change

    alpha = (100 - ci) / 2
    lower = np.percentile(boot_diffs, alpha)
    upper = np.percentile(boot_diffs, 100 - alpha)

    obs = ((np.median(y) - np.median(x)) / np.median(y)) * 100  # Percentage change

    return obs, (lower, upper), boot_diffs

def _calculate_7day_smooth_365days(df, col_name):
    s = df[col_name]
    s_ext = pd.concat([s.iloc[-3:], s, s.iloc[:3]], ignore_index=True)
    # Compute centered 7-day mean
    smooth = s_ext.rolling(window=7, center=True, min_periods=7).mean()
    # Extract the original 365 days
    df[col_name + '_7day'] = smooth.iloc[3:-3].to_numpy()
    return df

def _calculate_edof(x, y):
    """ Calculate Effective Degrees of Freedom using the lag-1 autocorrelation.

    Based on Bretherton et al. (1999) approximation.
    """
    # Lag-1 autocorrelation
    r1_x = acf(x, nlags=1, fft=True)[1]
    r1_y = acf(y, nlags=1, fft=True)[1]

    N = len(x)
    # EDOF formula
    N_eff = N * (1 - r1_x * r1_y) / (1 + r1_x * r1_y)
    return max(N_eff, 3) # Ensure at least 3 degrees of freedom

def _calculate_percentage_of_wind_blowing_from_certain_direction_per_month(wind_direction_in_cardinal_direction, column_name):
    """Monthly percentage of wind blowing from certain direction.

    """
    if column_name not in wind_direction_in_cardinal_direction.columns:
        raise KeyError(f"Station '{column_name}' not found in wind_direction_in_cardinal_direction")

    # Create DataFrame
    df = pd.DataFrame(wind_direction_in_cardinal_direction[column_name])
    # Count occurrences per month and direction
    df = (df.groupby(df.index.month).value_counts().unstack(fill_value=0))
    # 16-point compass order
    directions_16 = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
    # Ensure all directions exist and are in correct order
    df = df.reindex(columns=directions_16, fill_value=0)

    # Monthly totals
    total = df.sum(axis=1)
    # Convert to percentage
    percentage = df.div(total, axis=0).mul(100)

    return pd.DataFrame(percentage), pd.DataFrame(total)

def _cliffs_delta_fast(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    nx = len(x)
    ny = len(y)

    combined = np.concatenate([x, y])
    ranks = rankdata(combined)

    rx = np.sum(ranks[:nx])
    delta = (2 * rx - nx * (nx + 1)) / (nx * ny) - 1

    # Caculate the probabilities
    p_x_greater_y = (delta + 1) / 2
    p_y_greater_x = (1 - delta) / 2

    px_r, py_r = _round_two_percentages_to_100(
        p_x_greater_y * 100, p_y_greater_x * 100)
    return delta, px_r, py_r

def _cliffs_delta_interpretation(delta):
    """Interpret Cliff's delta value.

    Probability that a randomly selected value from one group exceeds a
    randomly selected value from the other group.

    The magnitude |δ| quantifies the strength of this dominance.
    The sign means the direction of dominance, not the size.

    δ > 0 → Katabatic values tend to be larger than non-katabatic values
    δ < 0 → Non-katabatic values tend to be larger than katabatic values
    δ = 0 → No systematic dominance (≈ 50/50)

    Example:
    # Effect size magnitude
        # Using standard thresholds:
        # | |δ| | Interpretation |
        # |---|---|
        # | 0–0.147 | negligible |
        # | 0.147–0.33 | small |
        # | 0.33–0.474 | medium |
        # | ≥ 0.474 | large |
        # Your value: ∣𝛿∣=0.268⇒small-to-moderate effect

    """
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return 'negligible'
    elif abs_delta < 0.33:
        return 'small'
    elif abs_delta < 0.474:
        return 'medium'
    else:
        return 'large'

def _correct_pvalue(r, n_eff, k=1):
    """Recalculate p-value based on effective degrees of freedom.

    k = number of covariates (Partial Correlation controls)
    """
    # t-statistic
    if abs(r) >= 1: return 0.0
    t_stat = r * np.sqrt((n_eff - 2 - k) / (1 - r**2))
    # Two-tailed p-value
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_eff - 2 - k))
    return p_val

def _curved_circle_text(ax, x0, y0, radius, text, scale, tune=250, angle_center=-45, fontsize=6):
    # desired arc length in plot units
    target_arc = scale * tune   # controls text width (tune if needed)

    # compute angle span from arc length
    angle_span = np.degrees(target_arc / (scale * radius))

    chars = list(text)
    angles = np.linspace(angle_center - angle_span/2, angle_center + angle_span/2, len(chars))

    for char, ang in zip(chars, angles):
        theta = np.radians(ang)
        x = x0 + scale * radius * np.cos(theta)
        y = y0 + scale * radius * np.sin(theta)
        rot = ang + 90
        ax.text(x, y, char, rotation=rot, ha='center', va='center', fontsize=fontsize, color='k', rotation_mode='anchor',
                transform_rotates_text=True, path_effects=[pe.withStroke(linewidth=1, foreground='white')], zorder=12)

def _doa_per_year(da):
    """Calculate the Day of Advance (DOA) for one year.

    Rules:
    1. Normally, select DOA where delta_days > 1.
    2. If delta_days == 1, select only if the summer minimum SIC < 0.15.
    """
    # Last minimum SIF time for this year
    t_min = da['time'].where(da == da.min('time')).max('time', skipna=True)

    # Times after the minimum
    after_min = da['time'] > t_min

    # First day SIC > 15% after the final summer minimum
    doa_time = (da['time'].where(after_min & (da > 0.15)).min('time', skipna=True))

    # Compute difference in days
    delta_days = (doa_time - t_min).dt.days

    # Identify valid DOA:
    # - delta_days > 1 → valid
    # - delta_days == 1 → only valid if minimum SIC < 0.15
    valid = (delta_days > 1) | ((delta_days == 1) & (da.min('time') < 0.15))

    # Apply mask
    doa_time = doa_time.where(valid)

    # Convert to day-of-year and allow NaN
    # doa_doy = doa_time.dt.dayofyear.astype('float')
    # doa_doy = doa_time.dt.dayofyear

    # Fill missing values with flag (e.g., 9999)
    # doa_doy = doa_doy.fillna(9999)

    return doa_time

def _doc_per_year(da):
    """ Calculate the day of closing (DOC) first day SIC increases above 80% following the final summer minimum.

    da: DataArray with dims (time, rlat, rlon) for ONE sif_year
    returns: (rlat, rlon) DOC as day-of-year.
    """
    # Last minimum sif time for this year
    t_min = da['time'].where(da == da.min('time')).max('time', skipna=True)

    # Times after the minimum
    after_min = da['time'] > t_min

    # First day SIC increases above 80% following the final summer minimum
    doc_time = (da['time'].where(after_min & (da > 0.8)).min('time', skipna=True))

    # return (doc_time).dt.dayofyear
    return doc_time

def _doo_per_year(da):
    """Calculate the Day of opening (DOO) last day SIC <80% before the first summer minimum.

    da: DataArray with dims (time, rlat, rlon) for ONE sif_year
    returns: (rlat, rlon) DOO as day-of-year.
    """
    # First minimum time for this year
    t_min = da['time'].where(da == da.min('time')).min('time', skipna=True)

    # Times before the minimum
    before_min = da['time'] < t_min

    # Last day before minimum where SIC > 0.8
    doo_time = (da['time'].where(before_min & (da > 0.8)).max('time', skipna=True))

    # +1 day, convert to DOY to doo
    # return (doo_time + np.timedelta64(1, 'D')).dt.dayofyear
    return doo_time + np.timedelta64(1, 'D')

def _dor_per_year(da):
    """Calculate the Day of retreat (DOR) last day SIC <15% before the first summer minimum.

    da: DataArray with dims (time, rlat, rlon) for ONE sif_year
    returns: (rlat, rlon) DOR as day-of-year.
    """
    # First minimum time for this year
    t_min = da['time'].where(da == da.min('time')).min('time', skipna=True)

    # Times before the minimum
    before_min = da['time'] < t_min

    # Last time SIC < 0.15 before minimum
    dor_time = (da['time'].where(before_min & (da > 0.15)).max('time', skipna=True))

    # Compute difference in days
    delta_days = (t_min - dor_time).dt.days
    valid = (delta_days > 1) | ((delta_days == 1) & (da.min('time') < 0.15))

    # Apply mask
    dor_time = dor_time.where(valid)

    return dor_time

def _filter_fog_events():
    """ Filter fog events when fog in present within the study period but outside of the main event,
    which may confound the analysis of the change in the atmospheric variables during the main event.

    """
    # Use 'dsr_cor' as an example variable to define the fog mask, but the same process can be applied to other variables if needed.
    variable = ['dsr_cor']
    freya_variable = read_freya_aws_data(variable=variable)
    freya_variable.index = pd.to_datetime(freya_variable.index)

    # Define fog mask
    results_fog_all = read_fog_classification()
    freya_variable['fog_cover_aws'] = np.nan
    for event in results_fog_all.events_fixed.unique():
        event_loop = results_fog_all.loc[results_fog_all.events_fixed == event]
        for _, row in event_loop.iterrows():
            freya_variable.loc[(freya_variable.index >= row['start_time']) & (freya_variable.index <= row['end_time']), 'fog_cover_aws'] = 1
    freya_variable['fog_mask'] = freya_variable['fog_cover_aws'] == 1

    valid_events = []
    for event in results_fog_all.events_fixed.unique():
        # Define the fog events that overlap with the current interval
        fog_events = pd.DataFrame(results_fog_all.loc[results_fog_all.events_fixed == event])

        start = fog_events['start_interval_buffer'].values[0]
        end = fog_events['end_interval_buffer'].values[0]
        subset = pd.DataFrame(freya_variable.loc[start:end]).sort_index()

        # Select interested foggy days for the given events.
        # Discard other foggy days that do not overlap with the current interval
        # to ensure the consistency of the analysis of the change in net radiation.
        subset_fog_event = []
        for _, row in fog_events.iterrows():
            subset_fog_event.append(subset.loc[row['start_time']:row['end_time']])
        subset_fog_event = pd.concat(subset_fog_event)

        subset['fog_on_aws_not_in_fog_events'] = False
        subset['fog_on_aws_not_in_fog_events'] = subset['fog_mask'] & ~subset.index.isin(subset_fog_event.index)
        if subset['fog_on_aws_not_in_fog_events'].sum() == 0:
            valid_events.append(event)
    return valid_events

def _filter_fog_events_2():
    """ Filter fog events when fog in present within the study period but outside of the main event,
    which may confound the analysis of the change in the atmospheric variables during the main event.

    """
    # Use 'dsr_cor' as an example variable to define the fog mask, but the same process can be applied to other variables if needed.
    variable = ['dsr_cor']
    freya_variable = read_freya_aws_data(variable=variable)
    freya_variable.index = pd.to_datetime(freya_variable.index)

    # Define fog mask
    results_fog_all = read_fog_classification()
    freya_variable['fog_cover_aws'] = np.nan
    for event in results_fog_all.events_fixed.unique():
        event_loop = results_fog_all.loc[results_fog_all.events_fixed == event]
        for _, row in event_loop.iterrows():
            freya_variable.loc[(freya_variable.index >= row['start_time']) & (freya_variable.index <= row['end_time']), 'fog_cover_aws'] = 1
    freya_variable['fog_mask'] = freya_variable['fog_cover_aws'] == 1

    valid_events = []
    for event in results_fog_all.events_fixed.unique():
        valid_events.append(event)
    return valid_events

def _get_effective_sample_size(variable, day_time, significant_events):
    """ Calculate effective sample size for significant events to inform the reliability of the results and guide the visualization (e.g., marker size in the plot)."""

    input_path = Path('/nas/data/processed/people/sos/third_paper_data/processed_aws/fog_effects/statistical_diff_results')

    significant_effective_ns = []
    fog_events_impacts_on_variable = pd.read_csv(f"{input_path}/fog_events_impacts_on_{variable}.csv")

    for significant_event in significant_events:
        subset = fog_events_impacts_on_variable.loc[fog_events_impacts_on_variable['events_fixed'] == significant_event]
        n_x = subset[f"{variable}_{day_time}_fog_vs_clear_sky_n_x"].unique()[0]
        n_y = subset[f"{variable}_{day_time}_fog_vs_clear_sky_n_y"].unique()[0]

        if np.isnan(n_x) or np.isnan(n_y):
            n_eff = np.nan
        elif n_x == 0 or n_y == 0:
            n_eff = np.nan  # Effective sample size is undefined if either group has zero samples
        else:
            n_eff = (2 * n_x * n_y) / (n_x + n_y)
        significant_effective_ns.append(n_eff)

    significant_effective_ns = np.array(significant_effective_ns)
    min_neff = np.nanmin(significant_effective_ns)
    max_neff = np.nanmax(significant_effective_ns)
    return min_neff, max_neff

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

def _get_greenland_mask_latitude_longitude_east_greenland():
    # Coarsen Greenland mask and merge it with Svalbard and Iceland
    mask_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
    gr_masks_5km = xr.open_dataset(f"{mask_path}/FGRN055_Masks_5.5km_updated_coordinates_east_greenland.nc")

    land_sea_mask = gr_masks_5km['LSM_GR']
    # land_sea_mask = land_sea_mask.where(land_sea_mask != 0) # Ensure binary mask (1 for land, 0 for sea)
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

    land_sea_mask = xr.where((land_sea_mask == 0), np.nan, land_sea_mask) # to avoid hiding the colors below
    colors_landseamask = [(255, 255, 255), (169, 169, 169)]
    cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
    bounds_landseamask = np.array([-0.01, 0.01, 1.5])
    norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

    return land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask

def _get_high_low_sea_ice_days(month_name, change_point_1, period, change_point_2=None):
    """ Get high and low sea ice days for function plot_violinplot_freya_composites_based_on_high_low_ci.

    The high and low sea ice days are calculated for each month (May to September) and day (1 to 31) separately from 1990 to 2020.
    So, the high sea ice days are those days when the sea ice concentration is above the 95th percentile for that month and day.
    The low sea ice days are those days when the sea ice concentration is below the 5th percentile for that month and day.

    For example, for May 1st, the high sea ice days are those days when the sea ice concentration is above the 95th percentile of all May 1st days from 1990 to 2020.
    Similarly, for May 1st, the low sea ice days are those days when the sea ice concentration is below the 5th percentile of all May 1st days from 1990 to 2020.

    Parameters
    ----------
    month_name : str
        Month name (e.g., 'May', 'June', etc.)

    change_point_1 : int
        First change-point year.

    period : {'whole', 'before', 'between', 'after'}
        Time period relative to the change point(s):
        – 'whole'   : years between change_point_1 and change_point_2 (inclusive)
        - 'before'  : years < change_point_1
        - 'between' : change_point_1 ≤ years < change_point_2 (requires change_point_2)
        - 'after'   : years ≥ change_point_1

    change_point_2 : int, optional
        Second change-point year, required when period='whole' or 'between'.

    """
    # Read the masked 5km racmo variables within 100km sea ice concentration
    masked_5km_variables_100km_sif_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
    masked_5km_variables_100km_sif = pd.read_csv(f"{masked_5km_variables_100km_sif_path}/masked_5km_variables_within_100km_sea_ice.csv", parse_dates=['time'], index_col='time')
    masked_5km_variables_100km_sif.index = pd.to_datetime(masked_5km_variables_100km_sif.index)
    # Resample to uniform frequency
    freq = masked_5km_variables_100km_sif.index.to_series().diff().mode()[0]
    masked_5km_variables_100km_sif = masked_5km_variables_100km_sif.resample(freq).mean()
    sif_in_100km_sif = pd.DataFrame(masked_5km_variables_100km_sif['ci'])
    if period == 'whole':
        if change_point_2 is None:
            raise ValueError("change_point_2 must be provided when period='whole'")
        sif_in_100km_sif = sif_in_100km_sif[(sif_in_100km_sif.index.year >= change_point_1) & (sif_in_100km_sif.index.year <= change_point_2)]
    elif period == 'before':
        sif_in_100km_sif = sif_in_100km_sif[sif_in_100km_sif.index.year < change_point_1]
    elif period == 'between':
        if change_point_2 is None:
            raise ValueError("change_point_2 must be provided when period='between'")
        sif_in_100km_sif = sif_in_100km_sif[(sif_in_100km_sif.index.year >= change_point_1) & (sif_in_100km_sif.index.year < change_point_2)]
    elif period == 'after':
        sif_in_100km_sif = sif_in_100km_sif[sif_in_100km_sif.index.year >= change_point_1]
    else:
        raise ValueError("period must be one of {'whole', 'before', 'between', 'after'}")

    if month_name == 'May':
        month_id = [5]
    elif month_name == 'June':
        month_id = [6]
    elif month_name == 'July':
        month_id = [7]
    elif month_name == 'August':
        month_id = [8]
    elif month_name == 'September':
        month_id = [9]
    elif month_name == 'MJJAS':
        month_id = [5, 6, 7, 8, 9]

    # Select high and low sea ice days for each month and day (doy) separately
    sif_in_100km_sif_month = pd.DataFrame(sif_in_100km_sif[sif_in_100km_sif.index.month.isin(month_id)])
    sif_in_100km_sif_month['high_or_low_sif'] = None
    sif_in_100km_sif_month['sif_param'] = None

    # Calculate high and low sea ice parameters (e.g., 75th and 25th percentiles) for that month
    high_sif_param = sif_in_100km_sif_month.quantile(0.75).iloc[0]
    low_sif_param = sif_in_100km_sif_month.quantile(0.25).iloc[0]

    # Select high sea ice years
    sif_df_high = sif_in_100km_sif_month.loc[sif_in_100km_sif_month['ci'] > high_sif_param]
    sif_in_100km_sif_month.loc[sif_df_high.index, 'high_or_low_sif'] = 'high_sif'
    sif_in_100km_sif_month.loc[sif_df_high.index, 'sif_param'] = high_sif_param

    # Select low sea ice years
    if low_sif_param == 0:
        sif_df_low = sif_in_100km_sif_month.loc[(sif_in_100km_sif_month['ci'] == 0)]
    else:
        sif_df_low = sif_in_100km_sif_month.loc[(sif_in_100km_sif_month['ci'] < low_sif_param)]
    sif_in_100km_sif_month.loc[sif_df_low.index, 'high_or_low_sif'] = 'low_sif'
    sif_in_100km_sif_month.loc[sif_df_low.index, 'sif_param'] = low_sif_param

    return sif_in_100km_sif_month

def _get_high_low_sea_ice_days_based_on_doy(month_name, change_point_1, period, change_point_2=None):
    """ Get high and low sea ice days for function plot_violinplot_freya_composites_based_on_high_low_ci.

    The high and low sea ice days are calculated for each month (May to September) and day (1 to 31) separately from 1990 to 2020.
    So, the high sea ice days are those days when the sea ice concentration is above the 95th percentile for that month and day.
    The low sea ice days are those days when the sea ice concentration is below the 5th percentile for that month and day.

    For example, for May 1st, the high sea ice days are those days when the sea ice concentration is above the 95th percentile of all May 1st days from 1990 to 2020.
    Similarly, for May 1st, the low sea ice days are those days when the sea ice concentration is below the 5th percentile of all May 1st days from 1990 to 2020.

    Parameters
    ----------
    month_name : str
        Month name (e.g., 'May', 'June', etc.)

    change_point_1 : int
        First change-point year.

    period : {'whole', 'before', 'between', 'after'}
        Time period relative to the change point(s):
        – 'whole'   : years between change_point_1 and change_point_2 (inclusive)
        - 'before'  : years < change_point_1
        - 'between' : change_point_1 ≤ years < change_point_2 (requires change_point_2)
        - 'after'   : years ≥ change_point_1

    change_point_2 : int, optional
        Second change-point year, required when period='whole' or 'between'.

    """
    # Read the masked 5km racmo variables within 100km sea ice concentration
    masked_5km_variables_100km_sif_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
    masked_5km_variables_100km_sif = pd.read_csv(f"{masked_5km_variables_100km_sif_path}/masked_5km_variables_within_100km_sea_ice.csv", parse_dates=['time'], index_col='time')
    masked_5km_variables_100km_sif.index = pd.to_datetime(masked_5km_variables_100km_sif.index)
    # Resample to uniform frequency
    freq = masked_5km_variables_100km_sif.index.to_series().diff().mode()[0]
    masked_5km_variables_100km_sif = masked_5km_variables_100km_sif.resample(freq).mean()
    sif_in_100km_sif = pd.DataFrame(masked_5km_variables_100km_sif['ci'])
    if period == 'whole':
        if change_point_2 is None:
            raise ValueError("change_point_2 must be provided when period='whole'")
        sif_in_100km_sif = sif_in_100km_sif[(sif_in_100km_sif.index.year >= change_point_1) & (sif_in_100km_sif.index.year <= change_point_2)]
    elif period == 'before':
        sif_in_100km_sif = sif_in_100km_sif[sif_in_100km_sif.index.year < change_point_1]
    elif period == 'between':
        if change_point_2 is None:
            raise ValueError("change_point_2 must be provided when period='between'")
        sif_in_100km_sif = sif_in_100km_sif[(sif_in_100km_sif.index.year >= change_point_1) & (sif_in_100km_sif.index.year < change_point_2)]
    elif period == 'after':
        sif_in_100km_sif = sif_in_100km_sif[sif_in_100km_sif.index.year >= change_point_1]
    else:
        raise ValueError("period must be one of {'whole', 'before', 'between', 'after'}")

    if month_name == 'May':
        month_id = [5]
    elif month_name == 'June':
        month_id = [6]
    elif month_name == 'July':
        month_id = [7]
    elif month_name == 'August':
        month_id = [8]
    elif month_name == 'September':
        month_id = [9]
    elif month_name == 'MJJAS':
        month_id = [5, 6, 7, 8, 9]

    # Select high and low sea ice days for each month and day (doy) separately
    sif_in_100km_sif_month = pd.DataFrame(sif_in_100km_sif[sif_in_100km_sif.index.month.isin(month_id)])
    sif_in_100km_sif_month['high_or_low_sif'] = None
    sif_in_100km_sif_month['sif_param'] = None
    sif_in_100km_sif_month['doy'] = sif_in_100km_sif_month.index.dayofyear
    doy_in_month = np.unique(sif_in_100km_sif_month['doy'])
    for doy in doy_in_month:
        sif_in_100km_sif_month_doy = sif_in_100km_sif_month[sif_in_100km_sif_month['doy'] == doy]
        # Calculate high and low sea ice parameters (e.g., 75th and 25th percentiles) for that month and day
        high_sif_param = sif_in_100km_sif_month_doy.quantile(0.75).iloc[0]
        low_sif_param = sif_in_100km_sif_month_doy.quantile(0.25).iloc[0]

        # Select high sea ice years
        sif_df_high = sif_in_100km_sif_month_doy.loc[sif_in_100km_sif_month_doy['ci'] > high_sif_param]
        sif_in_100km_sif_month.loc[sif_df_high.index, 'high_or_low_sif'] = 'high_sif'
        sif_in_100km_sif_month.loc[sif_df_high.index, 'sif_param'] = high_sif_param

        # Select low sea ice years
        if low_sif_param == 0:
            sif_df_low = sif_in_100km_sif_month_doy.loc[(sif_in_100km_sif_month_doy['ci'] == 0)]
        else:
            sif_df_low = sif_in_100km_sif_month_doy.loc[(sif_in_100km_sif_month_doy['ci'] < low_sif_param)]
        sif_in_100km_sif_month.loc[sif_df_low.index, 'high_or_low_sif'] = 'low_sif'
        sif_in_100km_sif_month.loc[sif_df_low.index, 'sif_param'] = low_sif_param

    return sif_in_100km_sif_month

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

def _get_latlon_meshgrid_east_greenland():
    # Read Greenland mask at 5 km resolution to plot the land and sea
    mask_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
    gr_masks_5km = xr.open_dataset(f"{mask_path}/FGRN055_Masks_5.5km_updated_coordinates_east_greenland.nc")
    x = gr_masks_5km.coords['rlon'].values
    y = gr_masks_5km.coords['rlat'].values
    return np.meshgrid(x, y)

def _get_lonlat_params(variable):
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

def _get_lonlat_params_east_greenland():
    # Color longitude
    bounds_lon = [-30, -20, -10, 0]
    colors_lon = [(190, 190, 190), (190, 190, 190), (190, 190, 190), (190, 190, 190)]
    cmap_lon = fp.make_cmap(colors_lon, bit=True)
    norm_lon = mpl.colors.BoundaryNorm(bounds_lon, cmap_lon.N)

    # Color latitude
    bounds_lat = [70, 75, 80]
    colors_lat = [(190, 190, 190), (190, 190, 190), (190, 190, 190), (190, 190, 190)]
    cmap_lat = fp.make_cmap(colors_lat, bit=True)
    norm_lat = mpl.colors.BoundaryNorm(bounds_lat, cmap_lat.N)
    return bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat

def _get_lonlat_ticks(variable, bounds_lon, bounds_lat):
    mask_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
    if (variable == 'sif') | (variable == 'z0500'):
        xcoord = 'rlon'
        ycoord = 'rlat'
        da = xr.open_dataset(mask_path / 'FGRN055_Masks_5.5km_updated_coordinates.nc')
    elif variable == 'smb':
        xcoord = 'x'
        ycoord = 'y'
        da = xr.open_dataset(mask_path / 'Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc')

    xcoord_min = np.min(da.coords[xcoord].values)
    xcoord_max = np.max(da.coords[xcoord].values)

    ycoord_min = np.min(da.coords[ycoord].values)
    ycoord_max = np.max(da.coords[ycoord].values)

    if (variable == 'sif') | (variable == 'z0500'):
        # Convert rlat rlon into lat lon for x-y tick label
        xlims = [xcoord_min, xcoord_max]
        ylims = [ycoord_min + 2, xcoord_max]
        bounds_lat_label = bounds_lat[:-1]   # The 80 N latitude goes beyond map frame, so exclude it.
    else:
        # Convert x,y into lat lon for x-y tick label
        xlims = [xcoord_min, xcoord_max]
        ylims = [ycoord_min, ycoord_max]
        bounds_lat_label = bounds_lat
    bounds_lon_label = bounds_lon[1:-2] # The longitude goes beyond map frame, so exclude it.

    # Convert the lowest lower left corner lat and lon from y, ylims[0] and x, xlims[0]
    # Define Polar stereographic EPSG:3413
    crs_xy = CRS.from_epsg(3413)
    # Define the custom rotated pole CRS
    crs_rpolar = CRS.from_proj4('+proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +datum=WGS84')
    # EPSG:4326 is the standard WGS84 lat/lon
    crs_lonlat = CRS.from_epsg(4326)

    # Create the transformer
    if (variable == 'sif') | (variable == 'z0500'):
        # Create the transformer from rotated polar to latlon (WGS84 lat/lon)
        transformer_to_lonlat = Transformer.from_crs(crs_rpolar, crs_lonlat, always_xy=True)
        # Create the transformer from latlon (WGS84 lat/lon) to rotated polar
        transformer_from_lonlat = Transformer.from_crs(crs_lonlat, crs_rpolar, always_xy=True)
    else:
        # Create the transformer from xy (Polar stereographic EPSG:3413) to latlon (WGS84 lat/lon)
        transformer_to_lonlat = Transformer.from_crs(crs_xy, crs_lonlat, always_xy=True)
        # Create the transformer from latlon (WGS84 lat/lon) to xy (Polar stereographic EPSG:3413)
        transformer_from_lonlat = Transformer.from_crs(crs_lonlat, crs_xy, always_xy=True)

    # Perform the transformation from given coordinate to lonlat
    lon_low, lat_low = transformer_to_lonlat.transform(xlims[0], ylims[0])

    # Longitude labels
    xticks = []
    xticklabels = []
    for bound_lon_label in bounds_lon_label:
        x_tick, _ = transformer_from_lonlat.transform(bound_lon_label, lat_low + np.cos(lat_low))
        xticks.append(x_tick)
        # xticklabels.append(f"{abs(bound_lon_label)}$^\\circ$W")
        xticklabels.append(f"{abs(bound_lon_label)}°N")

    # Latitude labels
    yticks = []
    yticklabels = []
    for bound_lat_label in bounds_lat_label:
        _, y_tick = transformer_from_lonlat.transform(lon_low, bound_lat_label)
        yticks.append(y_tick)
        # yticklabels.append(f"{bound_lat_label}$^\\circ$N")
        yticklabels.append(f"{bound_lat_label}°N")

    # Rescale the xticks
    if (variable == 'sif') | (variable == 'z0500'):
        # Rescale the yticks
        scaler = 0.09999999999999964
        for i, _ in enumerate(yticks):
            if i == 0:
                yticks[i] = yticks[i] + 2 * scaler
            if i == 2:
                yticks[i] = yticks[i] + i - 2 * scaler
            else:
                yticks[i] = yticks[i] + i
    else:
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

    return xlims, xticks, xticklabels, ylims, yticks, yticklabels

def _get_lonlat_ticks_east_greenland(bounds_lon, bounds_lat):
    mask_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
    da = xr.open_dataset(f"{mask_path}/FGRN055_Masks_5.5km_updated_coordinates_east_greenland.nc")

    xcoord = 'rlon'
    ycoord = 'rlat'

    xcoord_min = np.min(da.coords[xcoord].values)
    xcoord_max = np.max(da.coords[xcoord].values)

    ycoord_min = np.min(da.coords[ycoord].values)
    ycoord_max = np.max(da.coords[ycoord].values)

    # Convert rlat rlon into lat lon for x-y tick label
    xlims = [xcoord_min, xcoord_max]
    ylims = [ycoord_min + 2, xcoord_max]
    bounds_lat_label = bounds_lat   # The 80 N latitude goes beyond map frame, so exclude it.
    # bounds_lon_label = bounds_lon # The longitude goes beyond map frame, so exclude it.
    # bounds_lat_label = bounds_lat[:-1]   # The 80 N latitude goes beyond map frame, so exclude it.
    bounds_lon_label = bounds_lon[:-1] # The longitude goes beyond map frame, so exclude it.

    # Convert the lowest lower left corner lat and lon from y, ylims[0] and x, xlims[0]
    # Define Polar stereographic EPSG:3413
    crs_xy = CRS.from_epsg(3413)
    # Define the custom rotated pole CRS
    crs_rpolar = CRS.from_proj4('+proj=ob_tran +o_proj=latlon +o_lat_p=18.0 +lon_0=-37.5 +datum=WGS84')
    # EPSG:4326 is the standard WGS84 lat/lon
    crs_lonlat = CRS.from_epsg(4326)

    # Create the transformer
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

    # Rescale the xticks
    for i, _ in enumerate(xticks):
        if i == 0:
            xticks[i] = xticks[i] + i + 0.1
        elif i == 1:
            # xticks[i] = xticks[i] + i - 2 * scaler
            xticks[i] = xticks[i] + 0.5
        elif i == 2:
            # xticks[i] = xticks[i] + i - 2 * scaler
            xticks[i] = xticks[i] + 1.5


    # Latitude labels
    yticks = []
    yticklabels = []
    for bound_lat_label in bounds_lat_label:
        _, y_tick = transformer_from_lonlat.transform(lon_low, bound_lat_label)
        yticks.append(y_tick)
        # yticklabels.append(f"{bound_lat_label}$^\\circ$N")
        yticklabels.append(f"{bound_lat_label}°N")

    # Rescale the yticks
    for i, _ in enumerate(yticks):
        if i == 1:
            # yticks[i] = yticks[i] + i - 2 * scaler
            yticks[i] = yticks[i] + i - 0.9
        elif i == 2:
            # yticks[i] = yticks[i] + i - 2 * scaler
            yticks[i] = yticks[i] + i - 1.9

    return xlims, xticks, xticklabels, ylims, yticks, yticklabels

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

        # colors_promice_mask = [(244, 164, 96), (95, 155, 230), (95, 155, 230), (135, 206, 235)]
        colors_promice_mask = [(178, 223, 138), (95, 155, 230), (95, 155, 230), (135, 206, 235)]
        cmap_promice_mask = fp.make_cmap(colors_promice_mask, bit=True)
        bounds_promice_mask = np.array([0, 1.5, 2.5, 3.5, 4.5])
        norm_promice_mask = mpl.colors.BoundaryNorm(bounds_promice_mask, cmap_promice_mask.N)

    return promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask

def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def _hl_bootstrap_ci(x, y, n_boot=10000, ci=95, random_state=None):
    """
    Bootstrap confidence interval for the Hodges–Lehmann location shift estimator.

    This function estimates the uncertainty of the Hodges–Lehmann estimator of
    location shift between two independent samples using nonparametric bootstrap
    resampling. For each bootstrap iteration, both samples are resampled with
    replacement and the Hodges–Lehmann estimator is recomputed.

    Parameters
    ----------
    x : array-like
        First sample (e.g., fog period observations).
    y : array-like
        Second sample (e.g., clear-sky or reference period observations).
    n_boot : int, optional
        Number of bootstrap resamples used to approximate the sampling distribution.
    ci : float, optional
        Confidence level (default is 95 for a 95% confidence interval).
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    theta_hat : float
        Hodges–Lehmann point estimate of the location shift.
    ci_interval : tuple of float
        Lower and upper bounds of the bootstrap percentile confidence interval.
    boot : ndarray
        Bootstrap distribution of the estimator.

    Notes
    -----
    - This is a nonparametric percentile bootstrap method; no distributional
      assumptions (e.g., normality) are required.
    - The approach is appropriate for skewed, heteroscedastic, or non-normal data.
    - It inherits robustness from the Hodges–Lehmann estimator, which is based
      on the median of pairwise differences.
    - The method assumes independence between samples.

    Methodological background
    -------------------------
    The estimator is based on:

        theta = median(x_i - y_j)

    and uncertainty is quantified via bootstrap resampling of the original data.

    References
    ----------
    Hodges, J. L., & Lehmann, E. L. (1963).
    "Estimates of location based on rank tests."
    Annals of Mathematical Statistics, 34(2), 598–611.

    Lehmann, E. L. (2006).
    Nonparametrics: Statistical Methods Based on Ranks.
    Springer.

    Efron, B., & Tibshirani, R. J. (1993).
    An Introduction to the Bootstrap.
    Chapman & Hall/CRC.

    Davison, A. C., & Hinkley, D. V. (1997).
    Bootstrap Methods and Their Application.
    Cambridge University Press.
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) == 0 or len(y) == 0:
        return np.nan, (np.nan, np.nan), np.array([])

    theta_hat = _hodges_lehmann(x, y, rng)

    boot = np.empty(n_boot)

    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boot[i] = _hodges_lehmann(xb, yb, rng)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot, alpha)
    upper = np.percentile(boot, 100 - alpha)

    return theta_hat, (lower, upper), boot

def _hodges_lehmann(x, y, rng, max_pairs=50000):
    """
    Hodges–Lehmann estimator of location shift between two independent samples.

    This function computes a robust estimate of the central tendency of the
    difference between two distributions, defined as the median of all pairwise
    differences x_i - y_j. For large samples, a Monte Carlo approximation is used
    to reduce computational cost.

    Parameters
    ----------
    x : array-like
        First sample (e.g., fog period observations).
    y : array-like
        Second sample (e.g., clear-sky or reference period observations).
    rng : np.random.Generator
        Random number generator for reproducible subsampling.
    max_pairs : int, optional
        Threshold for exact vs approximate computation. If n*m exceeds this value,
        a Monte Carlo approximation using `max_pairs` sampled differences is used.

    Returns
    -------
    float
        Hodges–Lehmann estimator of the location shift (median of pairwise differences).
        Returns NaN if either input is empty.

    Notes
    -----
    - The Hodges–Lehmann estimator is a robust measure of central tendency for
      two-sample location shift problems.
    - It is particularly useful for skewed or non-normal distributions and is
      more efficient than simple median differences in many cases.
    - The estimator is consistent with the Wilcoxon rank-sum framework under
      location shift assumptions.

    References
    ----------
    Hodges, J. L., & Lehmann, E. L. (1963).
    "Estimates of location based on rank tests."
    Annals of Mathematical Statistics, 34(2), 598–611.

    Lehmann, E. L. (2006).
    Nonparametrics: Statistical Methods Based on Ranks.
    Springer.

    Hollander, M., Wolfe, D. A., & Chicken, E. (2013).
    Nonparametric Statistical Methods (3rd ed.).
    Wiley.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n, m = len(x), len(y)

    if n == 0 or m == 0:
        return np.nan

    # exact for small samples
    if n * m <= max_pairs:
        return np.median(x[:, None] - y[None, :])

    # approximation for large samples
    xi = rng.choice(x, size=max_pairs, replace=True)
    yj = rng.choice(y, size=max_pairs, replace=True)

    return np.median(xi - yj)

def _identify_significant_events(variable):
    """Identify significant events based on the permutation test results for the impact of fog on the given variable during different day times (day, twilight, night).

    An event is considered significant if the permutation p-value is less than or equal to 0.05, indicating a statistically
    significant difference between fog and clear sky conditions for that variable during the event.
    The function returns three lists of significant events for day, twilight, and night time periods, respectively."""

    input_path = Path('/nas/data/processed/people/sos/third_paper_data/processed_aws/fog_effects/statistical_diff_results')

    significant_events_day = []
    significant_events_twilight = []
    significant_events_night = []
    for day_time in ['day', 'twilight', 'night']:
        p_value_column = f"{variable}_{day_time}_fog_vs_clear_sky_permutation_p_value"
        variables_to_nan = [f"{variable}_{day_time}_fog_vs_clear_sky_n_x",
                            f"{variable}_{day_time}_fog_vs_clear_sky_n_y",
                            f"{variable}_{day_time}_fog_vs_clear_sky_bootstrap_median_diff",
                            f"{variable}_{day_time}_fog_vs_clear_sky_bootstrap_median_diff_ci_lower",
                            f"{variable}_{day_time}_fog_vs_clear_sky_bootstrap_median_diff_ci_upper"]

        fog_events_impacts_on_variable = pd.read_csv(f"{input_path}/fog_events_impacts_on_{variable}.csv")

        # Remove events if fog is present within the analysis period but not in the main event.
        # valid_events = _filter_fog_events()
        valid_events = _filter_fog_events_2()
        fog_events_impacts_on_variable.loc[~fog_events_impacts_on_variable['events_fixed'].isin(valid_events), variables_to_nan] = np.nan
        # fog_events_impacts_on_variable = fog_events_impacts_on_variable[fog_events_impacts_on_variable['events_fixed'].isin(valid_events)]

        # If the sample size is too small (n <= 2), set the corresponding median difference and confidence interval to NaN to avoid
        # unreliable results due to insufficient data.
        sample_size_x = fog_events_impacts_on_variable[f"{variable}_{day_time}_fog_vs_clear_sky_n_x"]
        sample_size_y = fog_events_impacts_on_variable[f"{variable}_{day_time}_fog_vs_clear_sky_n_y"]
        # size_threshold = 2
        size_threshold = 6
        small_sample_size = (sample_size_x  <= size_threshold) | (sample_size_y <= size_threshold)
        fog_events_impacts_on_variable.loc[small_sample_size, variables_to_nan] = np.nan

        # If p-value > 0.05, consider the impact of fog on the variable during the event as statistically insignificant and
        # set the corresponding median difference and confidence interval to NaN to avoid misleading.            interpretations.
        # insignificant_value = fog_events_impacts_on_variable[p_value_column] > 0.05
        insignificant_value = fog_events_impacts_on_variable[p_value_column] > 0.01
        fog_events_impacts_on_variable.loc[insignificant_value, variables_to_nan] = np.nan

        # ----------------- Test -----------------------
        # fog_events_impacts_on_variable['start_time'] = pd.to_datetime(fog_events_impacts_on_variable['start_time'])
        # fog_events_impacts_on_variable['end_time'] = pd.to_datetime(fog_events_impacts_on_variable['end_time'])
        # fog_events_impacts_on_variable.loc[fog_events_impacts_on_variable['start_time'].dt.month == 9, variables_to_nan] = np.nan

        # # Remove event 16; it starts in August but ends in September, and the fog event itself is likely not significant as it only lasts for 1 day in August. The modelled clear-sky conditions during this event are likely unreliable, which leads to an unreliable estimation of the impact of fog on the variable during this event. Therefore, we set the results for this event to NaN to avoid misleading interpretations.
        # fog_events_impacts_on_variable.loc[fog_events_impacts_on_variable['events_fixed'] == 16, variables_to_nan] = np.nan
        # # ----------------- Test -----------------------
#
        # if (variable == 'dsr_cor') & (day_time == 'night'):
        #     # Manual correction for event 35 of SWRin.
        #     # The modelled clear-sky SWRin during the night time (18-06 UTC) of event 35 is likely unreliable,
        #     # which leads to an unreliable estimation of the impact of fog on SWRin during this event.
        #     # Therefore, we set the results for this event to NaN to avoid misleading interpretations.
        #     fog_events_impacts_on_variable.loc[fog_events_impacts_on_variable.events_fixed == 35, variables_to_nan] = np.nan

        # if variable == 'albedo':
        #     # Only valid if the SWRin is significant to ensure the reliability
        #     # of the analysis of the change in albedo during fog events.
        #     reference_var_1 = pd.read_csv(f"{input_path}/fog_events_impacts_on_dsr_cor.csv")
        #     # Drop events which are insignificant or without any data
        #     insignificant_value = reference_var_1[f"dsr_cor_{day_time}_fog_vs_clear_sky_permutation_p_value"] > 0.05
        #     reference_var_1.loc[insignificant_value, variables_to_nan] = np.nan
        #     if day_time == 'night':
        #         reference_var_1.loc[reference_var_1.events_fixed == 35, variables_to_nan] = np.nan

        #     reference_var_1 = pd.DataFrame(reference_var_1.dropna(subset=[f"dsr_cor_{day_time}_fog_vs_clear_sky_bootstrap_median_diff"],
        #                                                                         how='all'))
        #     # Keep only events also present in reference_var_1
        #     valid_events = reference_var_1['events_fixed'].unique()
        #     fog_events_impacts_on_variable = (fog_events_impacts_on_variable[fog_events_impacts_on_variable['events_fixed'].isin(valid_events)])

        if variable in ['dshf_u', 'dlhf_u']:
            reference_var_1 = pd.read_csv(f"{input_path}/fog_events_impacts_on_wspd_u.csv")
            # Drop events which are insignificant or without any data
            insignificant_value = reference_var_1[f"wspd_u_{day_time}_fog_vs_clear_sky_permutation_p_value"] > 0.05
            reference_var_1.loc[insignificant_value, variables_to_nan] = np.nan
            reference_var_1 = pd.DataFrame(reference_var_1.dropna(subset=[f"wspd_u_{day_time}_fog_vs_clear_sky_bootstrap_median_diff"], how='all'))

            if variable == 'dshf_u':
                reference_var_2 = pd.read_csv(f"{input_path}/fog_events_impacts_on_t_u.csv")
                # Drop events which are insignificant or without any data
                insignificant_value = reference_var_2[f"t_u_{day_time}_fog_vs_clear_sky_permutation_p_value"] > 0.05
                reference_var_2.loc[insignificant_value, variables_to_nan] = np.nan
                reference_var_2 = pd.DataFrame(reference_var_2.dropna(subset=[f"t_u_{day_time}_fog_vs_clear_sky_bootstrap_median_diff"], how='all'))

            elif variable == 'dlhf_u':
                reference_var_2 = pd.read_csv(f"{input_path}/fog_events_impacts_on_qh_u.csv")
                # Drop events which are insignificant or without any data
                insignificant_value = reference_var_2[f"qh_u_{day_time}_fog_vs_clear_sky_permutation_p_value"] > 0.05
                reference_var_2.loc[insignificant_value, variables_to_nan] = np.nan
                reference_var_2 = pd.DataFrame(reference_var_2.dropna(subset=[f"qh_u_{day_time}_fog_vs_clear_sky_bootstrap_median_diff"], how='all'))

            # Keep only events also present in both reference variables
            valid_events = set(reference_var_1['events_fixed'].unique()) | set(reference_var_2['events_fixed'].unique())
            fog_events_impacts_on_variable = (fog_events_impacts_on_variable[fog_events_impacts_on_variable['events_fixed'].isin(valid_events)])

        # Drop events which are insignificant or without any data
        fog_events_impacts_on_variable = pd.DataFrame(fog_events_impacts_on_variable.dropna(subset=[f"{variable}_{day_time}_fog_vs_clear_sky_bootstrap_median_diff"], how='all'))

        if day_time == 'day':
            significant_events_day.append(fog_events_impacts_on_variable.events_fixed.unique())
        elif day_time == 'twilight':
            significant_events_twilight.append(fog_events_impacts_on_variable.events_fixed.unique())
        elif day_time == 'night':
            significant_events_night.append(fog_events_impacts_on_variable.events_fixed.unique())

    # Combine significant events across different day times to inform the visualization
    # (e.g., color coding in the plot) and interpretation of the results.
    significant_events_all = np.unique(np.concatenate(significant_events_day + significant_events_twilight + significant_events_night))
    significant_events_day = significant_events_day[0] if significant_events_day else np.array([])
    significant_events_twilight = significant_events_twilight[0] if significant_events_twilight else np.array([])
    significant_events_night = significant_events_night[0] if significant_events_night else np.array([])
    return significant_events_all, significant_events_day, significant_events_twilight, significant_events_night

def _katabatic_anabatic_wind_direction_based_on_fog_events_at_freya(df, fog_type, variable, interested_wind_directions):
    """Get the count and percentage of different wind directions during different fog events at Freya Glacier.

    """
    if fog_type == 'no_fog':
        df_fog_event = pd.DataFrame(df.loc[df['fog_on_glacier'] == 0, variable])
    elif fog_type == 'fog_on_glacier':
        df_fog_event = pd.DataFrame(df.loc[df['fog_on_glacier'] == 1, variable])
    elif fog_type == 'fog_cover_aws':
        df_fog_event = pd.DataFrame(df.loc[df['fog_cover_aws'] == 1, variable])
    else:
        raise ValueError(f"Invalid fog_type: {fog_type}. Must be one of 'no_fog', 'fog_on_glacier', or 'fog_cover_aws'.")
    df_fog_event = df_fog_event.loc[df_fog_event[variable].isin(interested_wind_directions)]
    df_fog_event_count = df_fog_event[variable].value_counts().reset_index()
    df_fog_event_count['perc'] = (df_fog_event_count['count'] / df_fog_event_count['count'].sum()) * 100
    df_fog_event_count[variable] = pd.Categorical(df_fog_event_count[variable], categories=interested_wind_directions, ordered=True)
    df_fog_event_count = df_fog_event_count.sort_values(variable).reset_index(drop=True)
    df_fog_event_count.index = df_fog_event_count[variable]
    # Define grouping
    direction_map = {'NW': 'ANA', 'WNW': 'ANA', 'SE': 'KAT', 'SSE': 'KAT'}
    # Add sector column
    df_fog_event_count['sector'] = df_fog_event_count['wd_cardinal'].map(direction_map)
    # Aggregate
    result = (df_fog_event_count.groupby('sector').agg({'count': 'sum', 'perc': 'sum'}).reset_index())
    result = result.set_index('sector')

    return result

def _paired_bootstrap_ci(x, y, n_boot=10000, ci=95, random_state=None):
    """
    Nonparametric paired bootstrap confidence interval for median differences.

    This function estimates the distribution of a paired location shift by
    resampling paired differences and computing the median statistic on each
    bootstrap sample. It is designed for situations where observations in x and y
    are naturally aligned (e.g., time-matched measurements or counterfactual
    simulations).

    Parameters
    ----------
    x : array-like
        First sample (e.g., observed radiation under condition A).
    y : array-like
        Second sample (e.g., counterfactual or condition B at matched timestamps).
    n_boot : int, optional
        Number of bootstrap resamples used to approximate the sampling distribution.
    ci : float, optional
        Confidence level (default is 95 for a 95% confidence interval).
    random_state : int or np.random.Generator, optional
        Random seed or generator for reproducibility.

    Returns
    -------
    theta_hat : float
        Observed median of paired differences (x - y).
    ci_interval : tuple of float
        Lower and upper bounds of the bootstrap percentile confidence interval.
    boot : ndarray
        Bootstrap distribution of the median paired differences.

    Notes
    -----
    - This method assumes a *paired design*, where each x[i] corresponds exactly
      to y[i].
    - The estimator targets the median of the paired difference distribution,
      which is robust to outliers and non-normality.
    - The bootstrap procedure is nonparametric and does not assume any specific
      distribution of the differences.
    - This approach is appropriate for counterfactual or matched observational
      studies (e.g., meteorological reanalysis vs model output).

    Methodological background
    -------------------------
    The estimator is defined as:

        d_i = x_i - y_i
        theta = median(d_i)

    Uncertainty is estimated using the nonparametric bootstrap:

        d_i* ~ resample(d_i with replacement)
        theta* = median(d_i*)

    References
    ----------
    Efron, B., & Tibshirani, R. J. (1993).
    An Introduction to the Bootstrap.
    Chapman & Hall/CRC.

    Davison, A. C., & Hinkley, D. V. (1997).
    Bootstrap Methods and Their Application.
    Cambridge University Press.

    Wilcox, R. R. (2017).
    Introduction to Robust Estimation and Hypothesis Testing (4th ed.).
    Academic Press.

    Hesterberg, T. C. (2015).
    What Teachers Should Know About the Bootstrap.
    The American Statistician, 69(4), 371–386.
    """
    rng = np.random.default_rng(random_state)

    x = np.asarray(x)
    y = np.asarray(y)

    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]

    if len(x) != len(y):
        raise ValueError("Paired bootstrap requires equal-length aligned samples.")

    d = x - y  # paired differences

    if len(d) == 0:
        return np.nan, (np.nan, np.nan), np.array([])

    boot = np.empty(n_boot)

    for i in range(n_boot):
        sample = rng.choice(d, size=len(d), replace=True)
        boot[i] = np.median(sample)

    alpha = (100 - ci) / 2
    lower = np.percentile(boot, alpha)
    upper = np.percentile(boot, 100 - alpha)

    return np.median(d), (lower, upper), boot

def _paired_permutation_test(x, y, n_perm=10000, seed=42):
    """
    Perform a paired permutation (sign-flipping) test on two matched samples.

    This non-parametric test evaluates whether the paired differences
    between `x` and `y` are centered around zero. The test preserves the
    pairing structure by randomly flipping the sign of each paired
    difference during permutation.

    The test statistic used here is the median of the paired differences.

    Parameters
    ----------
    x : array-like
        First paired sample (e.g., observed foggy SWRin values).

    y : array-like
        Second paired sample with the same timestamps/order as `x`
        (e.g., hypothetical clear-sky SWRin values).

    n_perm : int, optional
        Number of random sign-flipping permutations used to estimate
        the null distribution. Default is 10000.

    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
        Default is 42.

    Returns
    -------
    observed : float
        Observed median difference between paired samples.

    p_value : float
        Two-sided permutation p-value estimated from the proportion
        of permuted statistics with absolute value greater than or
        equal to the observed statistic.

    perm_stats : ndarray
        Array containing the permutation distribution of the test
        statistic.

    Notes
    -----
    This test is appropriate for paired or matched observations where
    each value in `x` corresponds directly to a value in `y`
    (e.g., same timestamps under different conditions).

    Unlike an independent-sample permutation test, this method preserves
    the temporal pairing structure and is therefore more suitable for
    comparing observed and hypothetical clear-sky SWRin at identical
    timestamps.

    The method assumes paired differences are exchangeable under the
    null hypothesis.

    Examples
    --------
    >>> obs, p, perms = _paired_permutation_test(fog_swrin, clear_sky_swrin)
    >>> print(p)
    """

    rng = np.random.default_rng(seed)

    x = np.asarray(x)
    y = np.asarray(y)

    diffs = x - y

    observed = np.median(diffs)

    perm_stats = np.empty(n_perm)

    for i in range(n_perm):

        signs = rng.choice([-1, 1], size=len(diffs))

        perm_diffs = diffs * signs

        perm_stats[i] = np.median(perm_diffs)

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return observed, p_value, perm_stats

def _permutation_test(x, y, n_perm=10000, seed=42):
    rng = np.random.default_rng(seed)

    x = np.asarray(x)
    y = np.asarray(y)

    observed = _stat_diff_median(x, y)

    combined = np.concatenate([x, y])
    nx = len(x)

    perm_stats = np.empty(n_perm)

    for i in range(n_perm):
        perm = rng.permutation(combined)   # NEW array each time
        perm_x = perm[:nx]
        perm_y = perm[nx:]
        perm_stats[i] = _stat_diff_median(perm_x, perm_y)

    p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

    return observed, p_value, perm_stats

def _plot_idealized_violin(ax):
    """Generate an idealized violin plot to illustrate the concept of violin plot.

    The boxplot inside the white violin plot shows the median, interquartile range, and 5th and 95th percentiles of the high sea ice days.

    """
    # Generate skewed data (right-skewed)
    # core = np.random.gamma(shape=1.0, scale=1.0, size=950)
    # tail = np.random.gamma(shape=2.0, scale=2.0, size=50)  # long tail
    # example_high = pd.DataFrame(np.concatenate([core, tail]), columns=['example_variable'])
    np.random.seed(42)

    example_high = pd.DataFrame(np.random.gamma(shape=1.2, scale=2.0, size=1000), columns=['example_variable'])
    example_high['types'] = 'type1'

    example_low = pd.DataFrame(np.random.gamma(shape=2.0, scale=2.0, size=1000), columns=['example_variable'])
    example_low['types'] = 'type2'

    example_data = pd.concat([example_high, example_low], ignore_index=True)
    example_data['time'] = 'idealized'

    example_data.loc[example_data['types'] == 'type2', 'example_variable'] = np.nan

    pos = 0.08
    boxplot_prop = dict(showcaps=False, widths=0.16, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
    boxplot_prop_ideal = dict(boxprops=dict(facecolor='w', color='k', linewidth=0.5))
    ax.boxplot(example_data.loc[example_data['types']=='type1', 'example_variable'], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_ideal, **boxplot_prop)
    sns.violinplot(x='time', y='example_variable', hue='types', hue_order=['type1', 'type2'], split=True, inner=None, palette={'type1':'w', 'type2':'royalblue'}, data=example_data, density_norm='area', linewidth=0.4, ax=ax)

    ax.get_legend().set_visible(False)
    [sp.set_visible(False) for sp in ax.spines.values()]
    ax.set_xlim([-0.4, 0.5])
    ax.set_ylim([-2, 9])
    ax.axis('off')

    pos = 0.2
    quantile_fontsize = 5

    ax.text(pos, example_high['example_variable'].quantile(0.75)+0.2, '75%', fontsize=quantile_fontsize)
    ax.text(pos, example_high['example_variable'].mean()-0.3,'Mean', fontsize=quantile_fontsize, fontweight='bold')
    ax.text(pos, example_high['example_variable'].quantile(0.50)-0.9,'50%', fontsize=quantile_fontsize)
    ax.text(pos, example_high['example_variable'].quantile(0.25)-1.2,'25%', fontsize=quantile_fontsize)

def _rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def _round_two_percentages_to_100(px, py):
    px_r = int(round(px))
    py_r = 100 - px_r
    return px_r, py_r

def _round_wind_direction_percentage_to_100_percent(wd_percentage_df):
    """Round wind direction percentage per month to 100 percent.

    """
    floors = np.floor(wd_percentage_df)

    # Compute remainder (scalar)
    remainder = int(100 - floors.sum(axis=1).iloc[0])

    # Fractional parts
    fractional = wd_percentage_df - floors

    # Add 1 to largest fractional columns
    cols_to_increment = (fractional.iloc[0].sort_values(ascending=False).index[:remainder])
    floors.loc[floors.index[0], cols_to_increment] += 1
    rounded_100 = floors.astype(int)
    rounded_100 = rounded_100.rename(columns={rounded_100.columns[0]: 'wd_perc'})

    return rounded_100

def _save_phenology_variable(time_da: xr.DataArray,
                             var_name: str,
                             long_name_doy: str,
                             description: str,
                             title: str,
                             output_path: Path,
                             filename: str,
                             doy_fillvalue: int = -9999,
                             compression_level: int = 4,
                             long_name_time: Optional[str] = None):

    """Save a sea-ice phenology variable with timestamp and day-of-year.


    Parameters
    ----------
    time_da : xr.DataArray
        Datetime DataArray (e.g., doo, dor, doa)
    var_name : str
        Base variable name (e.g., 'doo', 'dor', 'doa')
    long_name_doy : str
        Long name for day-of-year variable
    description : str
        Description for both variables
    title : str
        Dataset title
    output_path : Path
        Directory to save NetCDF file
    filename : str
        Output NetCDF filename
    doy_fillvalue : int, optional
        Fill value for DOY variable
    compression_level : int, optional
        NetCDF compression level
    long_name_time : str, optional
        Long name for timestamp variable. Only for DOO, DOR, DOA, DOC
    """

    # ---------------- Dataset ----------------
    if long_name_time:
        ds = xr.Dataset({f"{var_name}_time": time_da, f"{var_name}_doy": time_da.dt.dayofyear.astype(float), })
    else:
        ds = xr.Dataset({f"{var_name}_doy": time_da, })

    # ---------------- Variable attributes ----------------
    if long_name_time:
        ds[f"{var_name}_time"].attrs = {'long_name': long_name_time, 'description': description,}
    ds[f"{var_name}_doy"].attrs = {'long_name': long_name_doy, 'description': description, 'units': 'day_of_year',}

    # ---------------- Global attributes ----------------
    ds.attrs = {'title': title,
                'institution': 'University of Graz (Sonika Shahi)',
                'source': 'RACMO2.3p2 at 5 km resolution output',
                'history': ('Created to study sea ice phenology: ' f"{datetime.now(timezone.utc).strftime('%B %d, %Y %H:%M:%S')} UTC Python"),
                'Conventions': 'CF-1.8',}

    # ---------------- Encoding ----------------
    if long_name_time:
        encoding = {f"{var_name}_time": {'dtype': 'int64',       # Data type when read
                                        'units': 'days since 1970-01-01',  # CF-compliant
                                        'calendar': 'proleptic_gregorian',
                                        'zlib': True,           # Compression
                                        'complevel': compression_level},         # Compression level
                    f"{var_name}_doy": {'dtype': 'int16',
                                        'zlib': True,
                                        'complevel': compression_level,
                                        '_FillValue': doy_fillvalue}}
    else:
        encoding = {f"{var_name}_doy": {'dtype': 'int16',
                                        'zlib': True,
                                        'complevel': compression_level,
                                        '_FillValue': doy_fillvalue}}

    # ---------------- Write file ----------------
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    ds.to_netcdf(output_path / filename, mode='w', encoding=encoding, format='NETCDF4',)
    end = time.perf_counter()

    print(f"{filename}: saved in {(end - start) / 60:.2f} minutes")
    return ds

# def _scale_marker_size(n_eff, min_neff, max_neff, min_size=6, max_size=12):
def _scale_marker_size(n_eff, min_neff, max_neff, min_size=9, max_size=18):
    """The effective sample size penalizes imbalance.
    - If n_x and n_y are equal, n_eff = n_x = n_y, and the marker size will be max_size.
    - If one group has a much smaller sample size than the other, n_eff will be closer to the smaller sample size, and the marker size will be scaled down accordingly.
    “How many balanced observations does this comparison behave like?”

    Marker size represents effective sample size, which decreases when clear-sky and fog sample counts are highly imbalanced.

    """
    scaled = (np.sqrt(n_eff) - np.sqrt(min_neff)) / (np.sqrt(max_neff) - np.sqrt(min_neff))

    return min_size + scaled * (max_size - min_size)

def _scale_marker_size_median(neff, min_neff, max_neff, min_size=4, max_size=12):
    """Scale marker size linearly from neff."""
    if max_neff == min_neff:
        return (min_size + max_size) / 2

    return min_size + (neff - min_neff) * (max_size - min_size) / (max_neff - min_neff)

def _snap_backward(ts, idx):
    """Snap to previous available timestamp ≤ ts"""
    pos = idx.searchsorted(ts, side='right') - 1
    if pos < 0:
        return idx[0]
    return idx[pos]

def _snap_forward(ts, idx):
    """Snap to next available timestamp ≥ ts"""
    pos = idx.searchsorted(ts, side='left')
    if pos >= len(idx):
        return idx[-1]
    return idx[pos]

def _spearmanr_block_bootstrap(x, y, block_size=10, n_boot=1000, alpha=5):
    """Calculate Spearman correlation with confidence intervals using block bootstrap for autocorrelated time series.

    This function is designed for time series data where autocorrelation may
    inflate the apparent significance of correlations. It resamples contiguous
    blocks of data rather than individual points to preserve temporal structure.

    Parameters
    ----------
    x : array-like
        First time series (1D list or numpy array).
    y : array-like
        Second time series (same length as x).
    block_size : int, optional
        Length of each block for block bootstrap. Should be larger than
        the autocorrelation length of the series. Default is 10.
    n_boot : int, optional
        Number of bootstrap resamples. Default is 1000.
    alpha : float, optional
        Significance level for confidence intervals (in percent). Default is 5
        for a 95% confidence interval.

    Returns
    -------
    r_orig : float
        Spearman correlation coefficient of the original series.
    pval_orig : float
        Two-sided p-value of the original Spearman correlation.
    lower : float
        Lower bound of the bootstrap confidence interval.
    upper : float
        Upper bound of the bootstrap confidence interval.

    Notes
    -----
    - For autocorrelated time series, choose a block_size larger than the
      series' autocorrelation length to obtain accurate confidence intervals.
    - Confidence intervals are calculated using the percentile method.
    - This function does not remove trends or seasonality; preprocessing
      such as detrending or deseasonalization should be done prior to calling
      this function if required.
    """
    x = np.array(x)
    y = np.array(y)

    # Calculate original Spearman correlation and p-value
    r_orig, pval_orig = spearmanr(x, y)

    n = len(x)
    n_blocks = n // block_size

    # create blocks
    x_blocks = [x[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    y_blocks = [y[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    np.random.seed(42)  # Set a fixed seed for reproducibility
    r_boot = []
    p_val_boot = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, n_blocks)
        x_sample = np.concatenate([x_blocks[i] for i in idx])
        y_sample = np.concatenate([y_blocks[i] for i in idx])
        r_sample, p_val_sample = spearmanr(x_sample, y_sample)
        r_boot.append(r_sample)
        p_val_boot.append(p_val_sample)

    # mask all the non significant correlations
    r_boot_sig = [rval for rind, rval in enumerate(r_boot) if p_val_boot[rind] <= 0.05]

    # Calculate 95% confidence intervals (100 - alpha)
    lower = np.percentile(r_boot_sig, alpha/2) # Lower percentile (e.g. 2.5)
    upper = np.percentile(r_boot_sig, 100 - alpha/2) # Upper percentile (e.g. 97.5)
    return r_orig, pval_orig, lower, upper

def _spearmanr_block_bootstrap_partial(data, x_column, y_column, covar=None, block_size=10, n_boot=1000, alpha=5):
    """Calculate Partial Spearman correlation with confidence intervals using block bootstrap for autocorrelated time series.

    This function is designed for time series data where autocorrelation may
    inflate the apparent significance of correlations. It resamples contiguous
    blocks of data rather than individual points to preserve temporal structure.

    Parameters
    ----------
    DataFrame containing the time series data.
    x_column : str
        Name of the column for the first time series.
    y_column : str
        Name of the column for the second time series.
    covar : str, optional
        Name of the column for the third time series (for partial correlation). Default is None.
    block_size : int, optional
        Length of each block for block bootstrap. Should be larger than
        the autocorrelation length of the series. Default is 10.
    n_boot : int, optional
        Number of bootstrap resamples. Default is 1000.
    alpha : float, optional
        Significance level for confidence intervals (in percent). Default is 5
        for a 95% confidence interval.

    Returns
    -------
    r_orig : float
        Spearman correlation coefficient of the original series.
    pval_orig : float
        Two-sided p-value of the original Spearman correlation.
    lower : float
        Lower bound of the bootstrap confidence interval.
    upper : float
        Upper bound of the bootstrap confidence interval.

    Notes
    -----
    - For autocorrelated time series, choose a block_size larger than the
      series' autocorrelation length to obtain accurate confidence intervals.
    - Confidence intervals are calculated using the percentile method.
    - This function does not remove trends or seasonality; preprocessing
      such as detrending or deseasonalization should be done prior to calling
      this function if required.
    """
    stats = pg.partial_corr(data=data, x=x_column, y=y_column, covar=covar, method='spearman')

    # Calculate original Spearman correlation and p-value
    r_orig, pval_orig = stats['r'].values[0], stats['p-val'].values[0]

    n = len(data)
    n_blocks = n // block_size

    # create blocks
    x_blocks = [data[x_column].values[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    y_blocks = [data[y_column].values[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    z_blocks = [data[covar].values[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

    np.random.seed(42)  # Set a fixed seed for reproducibility
    r_boot = []
    p_val_boot = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, n_blocks)
        x_sample = np.concatenate([x_blocks[i] for i in idx])
        y_sample = np.concatenate([y_blocks[i] for i in idx])
        z_sample = np.concatenate([z_blocks[i] for i in idx])

        stats_sample = pg.partial_corr(data=pd.DataFrame({x_column: x_sample, y_column: y_sample, covar: z_sample}),
                                       x=x_column, y=y_column, covar=covar, method='spearman')
        r_sample, p_val_sample = stats_sample['r'].values[0], stats_sample['p-val'].values[0]
        r_boot.append(r_sample)
        p_val_boot.append(p_val_sample)

    # mask all the non significant correlations
    r_boot_sig = [rval for rind, rval in enumerate(r_boot) if p_val_boot[rind] <= 0.05]

    # Calculate 95% confidence intervals (100 - alpha)
    lower = np.percentile(r_boot_sig, alpha/2) # Lower percentile (e.g. 2.5)
    upper = np.percentile(r_boot_sig, 100 - alpha/2) # Upper percentile (e.g. 97.5)
    return r_orig, pval_orig, lower, upper

def _stat_diff_median(x, y):
    return np.median(x) - np.median(y)

def _theil_sen_and_pvalue(y, x):
    """
    Compute Theil-Sen slope and Kendall tau p-value.
    """
    mask = np.isfinite(y)
    if mask.sum() < 3:
        return np.nan, np.nan

    slope, _, _, _ = theilslopes(y[mask], x[mask])
    _, pval = kendalltau(x[mask], y[mask])

    return slope, pval


# %%
class StationDataAnalysis:
    """Evaluate RACMO SMB with SMB from Hynek et al.(2024).

    """
    def plot_cloud_cover_fraction():
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

        # Constants declaration
        eps_overcast = 1   #overcast sky emissivity
        eps_clear = 9.36508e-6 # clear sky emissivity
        sigma = 5.67e-8 # 5.67*10**(-8) #STEFAN_BOLTZMAN_CONSTANT
        eps = 0.622 # ratio of molecular weights of water and dry air
        T_0_abs = -273.15 # Absolute zero degree Celsius
        T_0 = 273.15 # Freezing point temperature in K

        # overcast conditions assuming the black-body radiation
        LR_overcast = eps_overcast * sigma * (aws_t_u + T_0)**4
        # theoretical downward longwave radiation flux corresponding to clear sky conditions using the equation from Swinbank (1963)
        LR_clear = eps_clear * sigma * (aws_t_u + T_0)**6

        CloudCov = (aws_dlr - LR_clear) / (LR_overcast - LR_clear)
        """
        # %%
        # Read cloud cover, air temperature, and downward longwave radiation data from AWS
        variables_to_load = ['cc', 't_u', 'dlr']
        freya_aws = read_freya_aws_data(variable=variables_to_load)

        # Calculate overcast and clear sky longwave radiation fluxes
        eps_overcast = 1   # Overcast sky emissivity
        eps_clear = 9.36508e-6 # Clear sky emissivity
        sigma = 5.67e-8 # Stefan-Boltzmann constant
        T_0 = 273.15 # Freezing point temperature in K

        # Overcast conditions assuming the black-body radiation
        freya_aws['dlr_overcast'] = eps_overcast * sigma * (freya_aws.t_u + T_0)**4
        # Theoretical downward longwave radiation flux corresponding to clear sky conditions
        # using the equation from Swinbank (1963)
        freya_aws['dlr_clear'] = eps_clear * sigma * (freya_aws.t_u + T_0)**6
        years = [2016, 2019, 2022]
        months = [5, 6, 7, 8, 9]
        freya_aws = freya_aws.loc[(freya_aws.index.year.isin(years)) & (freya_aws.index.month.isin(months))]

        colors_rgb = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163), (255, 127, 0)]
        colors = [(c[0]/255, c[1]/255, c[2]/255) for c in colors_rgb]  # Normalize RGB values to [0, 1]
        # colors = _rgb_to_hex(np.array(colors))
        colors_dict = dict(zip(months, colors))

        fig = plt.figure(figsize=(3, 3), dpi=600)
        # Outer GridSpec: controls large row gap (applies to all columns)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.15], hspace=0.2)   # large spacing between row groups
        gs_top = outer[0].subgridspec(1, 1, hspace=0, wspace=0)      # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)           # small internal spacing

        ax = fig.add_subplot(gs_top[0, 0])

        for month in [5, 6, 7, 8, 9]:
            freya_aws_month = freya_aws.loc[freya_aws.index.month == month]
            ax.plot(freya_aws_month.t_u, freya_aws_month.dlr, color=colors_dict[month], ls='None', marker='.', markersize=0.5)
        ax.plot(freya_aws.t_u, freya_aws.dlr_overcast, color='black', marker='None', ls='-', lw=0.5)
        ax.plot(freya_aws.t_u, freya_aws.dlr_clear, color='royalblue', marker='None', ls='-', lw=0.5)

        ax.set_xlim([-20, 20])
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_xticklabels(['-20', '-10', '0', '10', '20'])
        ax.set_xticks([-15, -5, 5, 15], minor=True)
        ax.set_xlabel(r'T$_\mathdefault{air}$ (°C)', fontsize=7) # labelpad=0.1
        ax.xaxis.set_label_coords(0.5, -0.09)

        ax.set_ylim([150, 400])
        ax.set_yticks([150, 200, 250, 300, 350, 400])
        ax.set_yticklabels(['150', '200', '250', '300', '350', '400'])
        ax.set_yticks([175, 225, 275, 325, 375], minor=True)
        ax.set_ylabel(r'LW$_{\downarrow}$ (W m$^\mathdefault{-2}$)', fontsize=7)
        ax.yaxis.set_label_coords(-0.120, 0.55)
        # ax.yaxis.set_label_coords(-0.125, 0.55)

        [ax.spines[i].set_visible(False) for i in ['top', 'right']]
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        ax.spines['left'].set_bounds(150, 400)
        ax_spine_shift = 3 # Adjust this value to control the outward shift of the spines
        ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax.spines['bottom'].set_bounds(-20, 20)
        ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        cmap = fp.make_cmap(colors_rgb, bit=True)
        bounds = np.array([1, 2, 3, 4, 5, 6])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [1.5, 2.5, 3.5, 4.5, 5.5]
        cbticklabel = ['May', 'June', 'Jul', 'Aug', 'Sep']

        tick_params_major = dict(direction='out', length=5, pad=1, labelsize=7, width=0.5, color='k')
        tick_params_minor = dict(direction='out', length=2.5, width=0.5, color='k')
        ax.tick_params(axis='both', which='major', **tick_params_major)
        ax.tick_params(axis='both', which='minor', **tick_params_minor)

        # Draw grid
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
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

        # Set legends
        ax = fig.add_subplot(gs_bot[0, 0])
        ax.axis('off')

        cbar_ax = ax.inset_axes([0.275, 0.3, 0.7, 0.33]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm)
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=7, pad=0.5, length=2, direction='inout', width=0.5)
        cb.outline.set_linewidth(0.1)

        legend_line = dict(linestyle='-', marker='None', lw=0.9)
        overcast = Line2D([], [], label='Overcast sky', color='black', **legend_line)
        clear = Line2D([], [], label='Clear sky', color='royalblue', **legend_line)

        legend_param = dict(loc='center', fontsize=7, frameon=False, handletextpad=0.5, alignment='center')
        ax.legend(handles=[overcast, clear], bbox_to_anchor=(0.1, 0.1), ncol=1, handlelength=1, labelspacing=0.1, markerscale=1, **legend_param)
        # ax.legend(handles=[overcast, clear], bbox_to_anchor=(0.1, 0.1), ncol=2, columnspacing=1, handlelength=1, labelspacing=1.3, markerscale=1, **legend_param)

        targets = dict(left=250, right=35, top=50, bottom=50, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_cloud_cover_fraction'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_cloud_cover_fraction.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)


        # %%
class RacmoEvaluation:
    """Evaluate RACMO SMB with SMB from Hynek et al.(2024).

    """
    def evaluate_and_plot_smb():
        """Evaluate RACMO SMB with SMB from Hynek et al.(2024).
        Updated version of the function to evaluate and plot SMB.

        """
        # %%
        # File path
        smb_file = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/station_data/exp_Freya_MB_data_v_202610.csv'
        # Load CSV, skipping the first row if it's extra header info
        smb_obs_annual = pd.read_csv(smb_file, sep=';', encoding='utf-8', index_col=0)
        # Transpose the DataFrame so years (or observations) become the index
        smb_obs_annual = smb_obs_annual.T
        smb_obs_annual.index = pd.Index([2000 + int(year.split('/')[1]) for year in smb_obs_annual.index], name='Year')

        # Rename columns to more consistent names
        smb_obs_annual.rename(columns={
            'bw  [ m w.e.]': 'bw_obs',
            'bw error est. +/-': 'bw_std_err_obs',
            'ba  [ m w.e.]': 'ba_obs',
            'ba error est. +/-': 'ba_std_err_obs'}, inplace=True)

        # Ensure the index is integer type
        smb_obs_annual.index = smb_obs_annual.index.astype(int)
        # Correct bw for accumulation (0.35 m w.e.) from avalanche in 2018 from Hynek et al. (2024)
        smb_obs_annual.loc[2018, 'bw_obs'] = 1.54
        smb_obs_annual = smb_obs_annual[['bw_obs', 'bw_std_err_obs', 'ba_obs', 'ba_std_err_obs']].astype(float)

        # File path and name for modeled Freya SMB from RACMO2.3 1 km
        base_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var/from_freya_glacier')
        smb_mod_all_daily = pd.read_csv(f"{base_file_path}/smb_masked_freya_1km.csv", header=0)
        smb_mod_all_daily['time'] = pd.to_datetime(smb_mod_all_daily['time'])
        smb_mod_all_daily.set_index('time', inplace=True)
        # Determine original time resolution
        time_delta = smb_mod_all_daily.index[1] - smb_mod_all_daily.index[0]
        smb_mod_all_daily = smb_mod_all_daily.resample(time_delta).mean()
        # Select only one-average
        smb_mod_daily = pd.DataFrame(smb_mod_all_daily.loc[:, 'smb_whole_racmo'])
        # Aggregate to monthly SMB (end-of-month sum)
        smb_mod_monthly = smb_mod_daily.resample('ME').sum()
        smb_mod_monthly['month'] = smb_mod_monthly.index.month
        # Assign mass balance type: 'bw' = winter, 'bs' = summer
        # Winter = Oct-Apr (months 10-4), Summer = May-Sep
        smb_mod_monthly['mb'] = np.where((smb_mod_monthly['month'] >= 10) | (smb_mod_monthly['month'] <= 4), 'bw', 'bs')
        # Year for seasonal mass balance: assign Oct-Apr to the next year
        smb_mod_monthly['year'] = smb_mod_monthly.index.year
        smb_mod_monthly['mb_year'] = np.where(smb_mod_monthly['month'] >= 10, smb_mod_monthly['year'] + 1, smb_mod_monthly['year'])
        # Group by mass balance year and type to calculate seasonal sums
        smb_mod_seasonal_mb = smb_mod_monthly.groupby(['mb_year', 'mb']).sum()
        # Extract winter (bw) and summer (bs) mass balances in m w.e.
        smb_mod_seasonal_bw = smb_mod_seasonal_mb.xs('bw', level='mb') / 1000
        smb_mod_seasonal_bs = smb_mod_seasonal_mb.xs('bs', level='mb') / 1000

        # Compute annual mass balance (ba = bw + bs) in m w.e.
        smb_mod_annual_ba = smb_mod_seasonal_bw + smb_mod_seasonal_bs
        # Ensure column names are clear
        smb_mod_seasonal_bw.rename(columns={'smb_whole_racmo': 'bw_mod'}, inplace=True)
        smb_mod_seasonal_bs.rename(columns={'smb_whole_racmo': 'bs_mod'}, inplace=True)
        smb_mod_annual_ba.rename(columns={'smb_whole_racmo': 'ba_mod'}, inplace=True)

        # Combine observed and modeled SMB
        smb_freya = pd.concat([smb_obs_annual, smb_mod_annual_ba, smb_mod_seasonal_bw, smb_mod_seasonal_bs], axis=1)

        # Reindex to specific years (2008-2020)
        smb_freya = smb_freya.reindex(range(2008, 2021))

        # Plots
        # fig, ax = plt.subplots(figsize=(6.2, 6), dpi=600)
        fig, ax = plt.subplots(figsize=(5.2, 5), dpi=600)

        # Plot annual observed and modelled SMB
        # ax.plot(smb_freya.ba_obs, smb_freya.ba_mod, marker='^', color='k', ms=5, lw=0, zorder=10)

        for year in smb_freya.index:
            # Asymmetric errors
            xerr = [[smb_freya.loc[year, 'ba_std_err_obs']],   # lower error
                        [smb_freya.loc[year, 'ba_std_err_obs']]]    # upper error
            ax.errorbar(smb_freya.loc[year, 'ba_obs'], smb_freya.loc[year, 'ba_mod'], xerr=xerr, fmt='o', color='k',
                        markersize=5, ecolor='k', elinewidth=1, capsize=4, zorder=10)

        # 1:1 line
        x_hyp = np.array(range(-3, 3, 1))
        y_hyp = np.array(range(-3, 3, 1))
        ax.plot(x_hyp, y_hyp, lw=0.3, color='gray', ls='dashed', dashes=(10, 5), alpha=0.8)

        # Predicted line
        slope_ba, intercept_ba, r_value_ba, pvalue_ba, _ = stats.linregress(smb_freya.ba_obs, smb_freya.ba_mod)
        y_pred = slope_ba * x_hyp + intercept_ba
        ax.plot(x_hyp, y_pred, lw=0.8, color='k', dashes=[6, 10], zorder=11)
        # Convert date string to text to plot
        date_list = np.array(smb_freya.index).astype('str')
        date_text = [list(datel)[2]+list(datel)[3] for datel in date_list]
        text_arg_years = dict(horizontalalignment='center', size=9, path_effects=[pe.withStroke(linewidth=0.5, foreground="white")])
        for i, j in enumerate(smb_freya.index):
            if j == 2018:
                # ax.text(smb_freya.loc[j, 'ba_obs'], smb_freya.loc[j, 'ba_mod'] - 0.09, rf"{date_text[i]}$^{{*}}$", color='k', zorder=12, **text_arg_years)
                ax.text(smb_freya.loc[j, 'ba_obs'], smb_freya.loc[j, 'ba_mod'] - 0.09, rf"{date_text[i]}$^{{\dagger}}$", color='k', zorder=12, **text_arg_years)
            else:
                ax.text(smb_freya.loc[j, 'ba_obs'], smb_freya.loc[j, 'ba_mod'] - 0.09, date_text[i], color='k', zorder=12, **text_arg_years)

        # Plot winter observed and modelled SMB
        # ax.plot(smb_freya.bw_obs, smb_freya.bw_mod, marker='^', color='b', ms=5, lw=0, zorder=10)
        for year in smb_freya.index:
            # Asymmetric errors
            xerr = [[smb_freya.loc[year, 'bw_std_err_obs']],   # lower error
                        [smb_freya.loc[year, 'bw_std_err_obs']]]    # upper error
            ax.errorbar(smb_freya.loc[year, 'bw_obs'], smb_freya.loc[year, 'bw_mod'], xerr=xerr, fmt='o', color='b',
                        markersize=5, ecolor='b', elinewidth=1, capsize=4, zorder=10)

        # Drop nan values to fit a line to the data set
        smb_freya_ = smb_freya.dropna()
        slope_bw, intercept_bw, r_value_bw, pvalue_bw, _ = linregress(smb_freya_.bw_obs, smb_freya_.bw_mod)
        y_pred = slope_bw * x_hyp + intercept_bw
        ax.plot(x_hyp, y_pred, lw=0.8, color='b', dashes=[6, 10], zorder=11)
        # Convert date string to text to plot
        date_list = np.array(smb_freya_.index).astype('str')
        date_text = [list(datel)[2] + list(datel)[3] for datel in date_list]
        for i, j in enumerate(smb_freya_.index):
            if j == 2018:
                # ax.text(smb_freya_.loc[j, 'bw_obs'], smb_freya_.loc[j, 'bw_mod'] - 0.09, rf"{date_text[i]}$^{{*}}$", color='b', zorder=12, **text_arg_years)
                ax.text(smb_freya_.loc[j, 'bw_obs'], smb_freya_.loc[j, 'bw_mod'] - 0.09, rf"{date_text[i]}$^{{\dagger}}$", color='b', zorder=12, **text_arg_years)
            else:
                ax.text(smb_freya_.loc[j, 'bw_obs'], smb_freya_.loc[j, 'bw_mod'] - 0.09, date_text[i], color='b', zorder=12, **text_arg_years)

        ax.axvline(lw=0.4, color='gray', ls='--', zorder=1)
        ax.hlines(0, -1.5, 1.5, lw=0.4, color='gray', ls='--', zorder=1)

        # Set axes properties
        ax.set_xlim([-1.5, 1.7])
        ax.set_ylim([-1.5, 1.5])
        ax.set_xticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_xticklabels([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_yticklabels([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_xlabel(r'SMB$_\mathdefault{FRG}$ (m w.e.)', labelpad=0.1, fontsize=11)
        ax.set_ylabel(r'SMB$_\mathdefault{RACMO}$ (m w.e.)', labelpad=0.1, fontsize=11)
        # ax.set_xlabel(r'SMB $\mathit{in}$ $\mathit{situ}$ (m w.e.)', labelpad=0.1, fontsize=11)
        # ax.set_ylabel('SMB RACMO2.3 1 km (m w.e.)', labelpad=0.1, fontsize=11)
        ax.yaxis.set_label_coords(-0.072, 0.5)

        [ax.spines[i].set_visible(False) for i in ['right', 'top']]
        [i.set_linewidth(0.6) for i in ax.spines.values()]

        ax.spines['bottom'].set_bounds(-1.5, 1.5)
        # ax.spines['bottom'].set_bounds(-1.5, 2)
        ax.spines['bottom'].set_position(('outward', 4))
        ax.spines['left'].set_bounds(-1.5, 1.5)
        # ax.spines['left'].set_bounds(-1.5, 2)
        ax.spines['left'].set_position(('outward', 4))

        tick_params_major = dict(direction='out', length=4, pad=1, labelsize=10, width=0.6, color='black')
        ax.tick_params(axis='both', which='major', **tick_params_major)

        # Draw grid
        ax.grid(which='major', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=0)
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

        # Define first legend
        marker_style_legend = dict(ms=4, lw=0, mew=0)
        handleba1 = Line2D([], [], label='Annual balance', marker='o', color='k', mfc='k', **marker_style_legend)
        handleba2 = Line2D([], [], label='N: ' + str(len(smb_freya.ba_obs.dropna())), **marker_style_legend)
        handleba3 = Line2D([], [], label='b0: ' + str(np.round(slope_ba, 2)), **marker_style_legend)
        handleba4 = Line2D([], [], label='b1: ' + str(np.round(intercept_ba, 2)), **marker_style_legend)
        # Assign stars depending on p-value
        if pvalue_ba <= 0.01:
            # stars = r"$^{**}$"
            stars = r"$^{*}$"
        elif pvalue_ba <= 0.05:
            stars = r"$^{*}$"
        else:
            stars = ""
        # # build LaTeX-friendly label
        r2_value = np.round(r_value_ba**2, 2)
        r2_plabel = fr"R$^\mathdefault{{2}}$: {r2_value}{stars}"
        handleba5= Line2D([], [], label=r2_plabel, **marker_style_legend)
        rmse = root_mean_squared_error(smb_freya.ba_obs, smb_freya.ba_mod)
        handleba6= Line2D([],[],label='RMSE: ' + str(np.round(rmse, 2)), **marker_style_legend)
        handleba = [handleba1, handleba2, handleba3, handleba4, handleba5, handleba6]
        # first_legend = ax.legend(bbox_to_anchor=(0.73, 0.005), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=11, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')
        first_legend = ax.legend(bbox_to_anchor=(0.73, 0.001), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=9, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')

        # Define second legend
        handlebw1 = Line2D([], [], label='Winter balance', marker='o', color='b', mfc='b', **marker_style_legend)
        handlebw2 = Line2D([], [], label='N: ' + str(len(smb_freya.bw_obs.dropna())), **marker_style_legend)
        handlebw3 = Line2D([], [], label='b0: ' + str(np.round(slope_bw, 2)), **marker_style_legend)
        handlebw4 = Line2D([], [], label='b1: ' + str(np.round(intercept_bw, 2)), **marker_style_legend)
        # Compute p-value stars for significance levels
        # Assign stars depending on p-value
        if pvalue_bw <= 0.01:
            # stars = r"$^{**}$"
            stars = r"$^{*}$"
        elif pvalue_bw <= 0.05:
            stars = r"$^{*}$"
        else:
            stars = ""
        # Build LaTeX-friendly label
        r2_value = np.round(r_value_bw**2, 2)
        r2_plabel = fr"R$^\mathdefault{{2}}$: {r2_value}{stars}"

        handlebw5= Line2D([], [], label=r2_plabel, **marker_style_legend)
        # handlebw5= Line2D([], [], label=r'R$^{\mathdefault{2}}$: ' + str(np.round(r_value**2, 2)), **marker_style_legend)
        rmse = root_mean_squared_error(smb_freya_.bw_obs, smb_freya_.bw_mod)
        handlebw6= Line2D([],[],label='RMSE: ' + str(np.round(rmse, 2)), **marker_style_legend)
        handlebw = [handlebw1, handlebw2, handlebw3, handlebw4, handlebw5, handlebw6]
        plt.legend(bbox_to_anchor=(0.99, 0.001), loc='lower right', handles=handlebw, ncol=1, borderaxespad=0, fontsize=9, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='b')

        # Manually add the first legend back to the plot
        ax.add_artist(first_legend)

        targets = dict(left=300, right=25, top=30, bottom=245, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Save the plot
        logger.info(f"Saving plot of 'Freya on RACMO mask'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'evaluate_and_plot_smb.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)


        # %%
        # Print mean bias
        # Align observations and modelled values by year
        obs = smb_freya.ba_obs
        mod = smb_freya.ba_mod

        # Mean bias
        mean_bias = (mod - obs).mean()

        print(f"Mean annual bias: {mean_bias:.3f} m w.e. yr-1")

        obs = smb_freya.bw_obs
        mod = smb_freya.bw_mod

        # Mean bias
        mean_bias = (mod - obs).mean()

        print(f"Mean winter bias: {mean_bias:.3f} m w.e. yr-1")

         # %%
    def evaluate_and_plot_wind_direction():
        """Evalaute racmo wind direction.

        """
        # %%
        # Freya Glacier wind components from AWS and RACMO
        # AWS
        freya_wind_hr = []
        # read_freya_aws_data() # Print the name of variables and their long name available in Freya AWS data
        for v in ['wspd_u', 'wdir_u']:
            df = read_freya_aws_data(variable=v)
            freya_wind_hr.append(df)
        freya_wind_hr = pd.concat(freya_wind_hr, axis=1).sort_index()
        # Calculate daily mean wind direction
        freya_wd_daily_obs = vs.calculate_wind_direction_daily_mean(freya_wind_hr['wspd_u'], freya_wind_hr['wdir_u'], max_values_per_day=24)
        freya_wd_daily_obs.rename(columns={'wd': 'wd_freya_obs'}, inplace=True)

        # RACMO
        # read_station_data_from_racmo(station='freya') # Print the name of variables available in Freya RACMO data
        freya_wd_daily_mod = read_station_data_from_racmo(station='freya', variable='wd')
        freya_wd_daily_mod.rename(columns={'wd': 'wd_freya_mod'}, inplace=True)

        # Zackenberg wind components from AWS and RACMO
        # read_zackenberg_aws_data()
        # AWS
        # Print the name of variables and their long name available in Zackenberg AWS data
        zac_wind_hr = []
        for v in ['ws', 'wd']:
            df = read_zackenberg_aws_data(variable=v)
            zac_wind_hr.append(df)
        zac_wind_hr = pd.concat(zac_wind_hr, axis=1).sort_index()
        # Calculate daily mean wind direction
        zac_wd_daily_obs = vs.calculate_wind_direction_daily_mean(zac_wind_hr['ws'], zac_wind_hr['wd'], max_values_per_day=24)
        zac_wd_daily_obs.rename(columns={'wd': 'wd_zac_obs'}, inplace=True)

        # RACMO
        # read_station_data_from_racmo(station='zac') # Print the name of variables available in Zackenberg RACMO data
        zac_wd_daily_mod = read_station_data_from_racmo(station='zac', variable='wd')
        zac_wd_daily_mod.rename(columns={'wd': 'wd_zac_mod'}, inplace=True)

        # Concatenate all wind direction data
        wd_combined = pd.concat([freya_wd_daily_obs, freya_wd_daily_mod, zac_wd_daily_obs, zac_wd_daily_mod], axis=1)

        # Classify wind direction into cardinal directions for each station daily average
        date_range = pd.date_range(start=wd_combined.index[0], end=wd_combined.index[-1], freq='d')
        cardinal_directions_daily = pd.DataFrame(columns=wd_combined.columns, index=date_range)
        for station in wd_combined.columns:
            cardinal_directions_daily.loc[:, station] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_combined, station)[f'{station}_class']

        # Calculate percentage of classified daily average wind direction per month
        # Select months May to September
        months = [5, 6, 7, 8, 9]
        miindex = pd.MultiIndex.from_product([wd_combined.columns, months])
        directions_16 = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        cardinal_directions_percentage  = pd.DataFrame(index=miindex, columns=directions_16)
        cardinal_directions_total  = pd.DataFrame(index=miindex, columns=['sample_size'])

        for station in wd_combined.columns:
            if station in ['wd_freya_obs', 'wd_freya_mod']:
                data = cardinal_directions_daily[['wd_freya_obs', 'wd_freya_mod']].dropna()
            elif station in ['wd_zac_obs', 'wd_zac_mod']:
                data = cardinal_directions_daily[['wd_zac_obs', 'wd_zac_mod']].dropna()
            wd_percentage, wd_total  = _calculate_percentage_of_wind_blowing_from_certain_direction_per_month(data, station)
            for month in months:
                cardinal_directions_percentage.loc[(station, month), :] = wd_percentage.loc[month, :]
                cardinal_directions_total.loc[(station, month), 'sample_size'] = wd_total.loc[month, :].values

        # Select stations and their colors
        stations = ['wd_freya_obs', 'wd_freya_mod', 'wd_zac_obs', 'wd_zac_mod']
        stations_colors = ['royalblue', 'royalblue','orangered', 'orangered']
        linestyles = ['-', '--', '-', '--']
        linewidths = [0.9, 0.5, 0.9, 0.5]
        # Combine everything
        var_styles = {var: {'color': color, 'linestyle': ls, 'linewidth': lw}
                      for var, color, ls, lw in zip(stations, stations_colors, linestyles, linewidths)}

        fig = plt.figure(figsize=(4.5, 3), dpi=600)
        gs = plt.GridSpec(2, 3, hspace=0.27, wspace=0.18)
        subplot_loops = [('May', 5, 0, 0), ('Jun', 6, 0, 1), ('Jul', 7, 0, 2), ('Aug', 8, 1, 0), ('Sep', 9, 1, 1)] # (month, column)

        cardinal_directions_number = len(directions_16)
        direction_8 = np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        for subplot_loop in subplot_loops:
            month_name = subplot_loop[0]
            month = subplot_loop[1]
            row = subplot_loop[2]
            col = subplot_loop[3]
            ax = fig.add_subplot(gs[row, col], polar=True)
            for station in stations:
                # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
                angles = [n / float(cardinal_directions_number) * 2 * np.pi for n in range(cardinal_directions_number)]
                angles += angles[:1]

                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)

                obs = (cardinal_directions_percentage.loc[(station, month), :]).tolist()
                obs += obs[:1]
                ax.plot(angles, obs, **var_styles[station])

            # Set xticks and labels
            ax.set_xticks([n / float(8) * 2 * np.pi for n in range(8)])
            ax.set_xticklabels(direction_8, color='grey', size=6)

            # Set minor ticks
            directions_8_minor = np.array(['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW'])
            ax.set_xticks([(n + 0.5) / 8 * 2 * np.pi for n in range(8)], minor=True)
            ax.set_xticklabels(directions_8_minor, color='grey', size=5, minor=True)

            # Grid styling
            ax.xaxis.grid(True, which='major', linewidth=0.3, linestyle='-')
            ax.xaxis.grid(True, which='minor', linewidth=0.3, linestyle='--', color='lightgrey')

            # Move labels closer to the circle
            for label in ax.get_xticklabels():
                label.set_clip_on(False)      # <-- critical
                # label.set_y(0.51)     # default ~1.1, lower = closer
                label.set_y(0.22)     # default ~1.1, lower = closer
                # label.set_zorder(12)
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')

            # Move labels closer to the circle
            for label in ax.get_xminorticklabels():
                label.set_clip_on(False)      # <-- critical
                # label.set_y(0.51)     # default ~1.1, lower = closer
                label.set_y(0.15)     # default ~1.1, lower = closer
                # label.set_zorder(12)
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')

            # Set yticks and labels
            ax.yaxis.grid(True, linewidth=0.3)
            ax.set_rlabel_position(15)
            ax.set_ylim(0, 50)
            ax.set_yticks([10, 20, 30, 40])
            ax.set_yticklabels(['10%', '20%', '30%', '40%'], color='grey', size=4, zorder=12)
            ax.tick_params(axis='y', which='major', width=0, length=0.1, pad=0.0001)
            for label in ax.get_yticklabels():
                label.set_zorder(12)
                label.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

            # Polar spine (outer circle)
            ax.spines['polar'].set_linewidth(0.4)
            ax.spines['polar'].set_zorder(1)
            ax.spines['polar'].set_color('k')

            # Add subplot letter
            if month == 5:
                subplot_number = '(a)'
            elif month == 6:
                subplot_number = '(b)'
            elif month == 7:
                subplot_number = '(c)'
            elif month == 8:
                subplot_number = '(d)'
            elif month == 9:
                subplot_number = '(e)'
            ax.text(-0.1, 1.03, f"{subplot_number} {month_name}", horizontalalignment='left', transform=ax.transAxes, size=6, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # Create legend
            value = int(cardinal_directions_total.loc[('wd_freya_obs', month), :].values[0])
            label_fre = rf"n$_\mathdefault{{AWS_\mathdefault{{FRG}}}}$: {value}"
            value = int(cardinal_directions_total.loc[('wd_zac_obs', month), :].values[0])
            label_zac = rf"n$_\mathdefault{{AWS_\mathdefault{{ZAC}}}}$: {value}"

            nan_patch_fre = Patch(color='none', label=label_fre)
            nan_patch_zac = Patch(color='none', label=label_zac)
            # station_legend = ax.legend(bbox_to_anchor= (1.1, 0.06), handles=[nan_patch_fre, nan_patch_zac], ncol=1, borderaxespad=0, fontsize=6, frameon=False, handletextpad=0.005, labelspacing=0.3)
            if month == 7:
                station_legend = ax.legend(bbox_to_anchor= (0.8, -0.06), handles=[nan_patch_fre, nan_patch_zac], ncol=1, borderaxespad=0, fontsize=6, frameon=False, handletextpad=0.005, labelspacing=0.3)
            else:
                station_legend = ax.legend(bbox_to_anchor= (1.2, 0.07), handles=[nan_patch_fre, nan_patch_zac], ncol=1, borderaxespad=0, fontsize=6, frameon=False, handletextpad=0.005, labelspacing=0.3)
            for station_legend_text in station_legend.get_texts():
                station_name = re.findall(r'\\mathdefault\{([^{}]+)\}', station_legend_text.get_text())[-1]
                station_legend_text.set_color('royalblue' if 'FRG' in station_name else 'orangered')
                station_legend_text.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
                station_legend_text.set_ha('center')

        # ------------------------------------
        # Plot idealized wind rose in new axis
        # ------------------------------------
        cardinal_directions = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        ax = fig.add_subplot(gs[1, 2], polar=True)

        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Each sector is 22.5 degrees wide
        # Katabatic wind sector (degrees)
        fre_obs_start = np.deg2rad(45 - 11.25)
        fre_obs_end   = np.deg2rad(45 + 11.25)

        fre_mod_start = np.deg2rad(135 - 11.25)
        fre_mod_end   = np.deg2rad(135 + 11.25)

        zac_mod_start = np.deg2rad(225 - 11.25)
        zac_mod_end   = np.deg2rad(225 + 11.25)

        zac_obs_start = np.deg2rad(315 - 11.25)
        zac_obs_end   = np.deg2rad(315 + 11.25)

        for wind_type in [(fre_obs_start, fre_obs_end, 'royalblue', '-'), (fre_mod_start, fre_mod_end,'royalblue', '--'), (zac_obs_start, zac_obs_end, 'orangered', '-'), (zac_mod_start, zac_mod_end, 'orangered', '--')]:
            # Sector properties
            theta_center = (wind_type[0] + wind_type[1]) / 2
            theta_width  = wind_type[1] - wind_type[0]
            radius = 100  # magnitude of katabatic wind
            # Plot filled wind-rose sector
            # ax.bar(theta_center, radius, width=theta_width, bottom=0, color=wind_type[2], linewidth=0, alpha=0.9, zorder=12)
            ax.bar(theta_center, radius, width=theta_width, bottom=0, color='none', ec=wind_type[2], linewidth=0.5, ls=wind_type[3], zorder=12)

        # angles_deg = [0, 90, 180, 270]
        # ax.set_xticks(np.deg2rad(angles_deg))
        # ax.set_xticklabels(['N', 'E', 'S', 'W'], color='grey', size=6)
        ax.set_xticks([])  # Remove default xticks and labels

        # Move labels closer to the circle
        for label in ax.get_xticklabels():
            label.set_clip_on(False)      # <-- critical
            label.set_y(0.35)     # default ~1.1, higher = closer
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            label.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])

        legend_params = dict(horizontalalignment='center', verticalalignment='center', size=6, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')], zorder=12)
        y_pos = 60
        ax.text(np.deg2rad(45), y_pos, rf"AWS$_\mathdefault{{FRG, obs}}$", **legend_params, color='royalblue')
        ax.text(np.deg2rad(135), y_pos, rf"AWS$_\mathdefault{{FRG, mod}}$", **legend_params, color='royalblue')
        ax.text(np.deg2rad(225), y_pos, rf"AWS$_\mathdefault{{ZAC, mod}}$", **legend_params, color='orangered')
        ax.text(np.deg2rad(315), y_pos, rf"AWS$_\mathdefault{{ZAC, obs}}$", **legend_params, color='orangered')

        ax.xaxis.grid(True, which='major', linewidth=0.2, linestyle='-', color='lightgrey')

        # Set yticks and labels
        ax.set_yticklabels([])
        ax.yaxis.grid(False)

        # Polar spine (outer circle)
        ax.spines['polar'].set_linewidth(0.2)
        ax.spines['polar'].set_zorder(1)
        ax.spines['polar'].set_color('k')

        targets = dict(left=45, right=30, top=70, bottom=90, hspace=300, wspace=500)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'evaluate_and_plot_wind_direction'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'evaluate_and_plot_wind_direction.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
    # %%
    def evaluate_and_plot_t2m_q2m_at_zac():
        # %%
        # Read Zackenberg AWS
        # read_zackenberg_aws_data()
        zac_variables = []
        for v in ['t2m', 'q2m']:
            df = read_zackenberg_aws_data(variable=v)
            zac_variables.append(df)
        zac_variables = pd.concat(zac_variables, axis=1).sort_index()

        # Calculate daily mean temperature and specific humidity at Zackenberg
        zac_variables_daily = ss.daily_calculation(zac_variables, method='mean', max_values_per_day=18)
        zac_variables_daily['q2m'] = zac_variables_daily['q2m'] * 1000 # convert from kg/kg to g/kg

        # Read RACMO dataset
        # read_station_data_from_racmo(station='zac') # Print the name of variables available in Freya RACMO data
        freya_t2m_daily_mod = read_station_data_from_racmo(station='zac', variable='t2m')
        freya_t2m_daily_mod.rename(columns={'t2m': 't2m_zac_mod'}, inplace=True)
        freya_t2m_daily_mod = freya_t2m_daily_mod - 273.15 # K into deg C

        freya_q2m_daily_mod = read_station_data_from_racmo(station='zac', variable='q2m')
        freya_q2m_daily_mod.rename(columns={'q2m': 'q2m_zac_mod'}, inplace=True)
        freya_q2m_daily_mod = freya_q2m_daily_mod * 1000 # convert from kg/kg to g/kg
        # Concatenate all variables
        t2m_combined = pd.concat([freya_t2m_daily_mod, zac_variables_daily.t2m], axis=1)
        t2m_combined = t2m_combined.loc[t2m_combined.index.month.isin([5, 6, 7, 8, 9])]
        t2m_combined = t2m_combined.dropna()

        q2m_combined = pd.concat([freya_q2m_daily_mod, zac_variables_daily.q2m], axis=1)
        q2m_combined = q2m_combined.loc[q2m_combined.index.month.isin([5, 6, 7, 8, 9])]
        q2m_combined = q2m_combined.dropna()

        # fig, axes = plt.subplots(1, 2, figsize=(2.5, 1.2), dpi=600)
        # fig, axes = plt.subplots(1, 2, figsize=(2.75, 1.35), dpi=600)
        fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.75), dpi=600)
        # for ax, data, col_name in zip(axes, [q2m_combined, t2m_combined], ['q2m', 't2m']):
        for ax, data, col_name in zip(axes, [t2m_combined, q2m_combined], ['t2m', 'q2m']):
            # if col_name == 'q2m':
            #     ax = fig.add_axes([-0.15, 0.15, 0.9, 0.9])   # [left, bottom, width, height] in figure coordinates (0 to 1)
            # elif col_name == 't2m':
            #     ax = fig.add_axes([0.45, 0.15, 0.9, 0.9])   # [left, bottom, width, height] in figure coordinates (0 to 1)

            ax.plot(data.iloc[:, 1], data.iloc[:, 0], 'ok', ms=0.05, alpha=0.5)

            # 1:1 line
            x_hyp = np.arange(data.min().min(), data.max().max(), 1)
            y_hyp = np.arange(data.min().min(), data.max().max(), 1)
            ax.plot(x_hyp, y_hyp, '--', color='gray', lw=0.5)

            # Predicted line
            slope, intercept, r_value, pvalue, _ = linregress(data[col_name], data[f"{col_name}_zac_mod"])
            y_pred = slope * x_hyp + intercept
            ax.plot(x_hyp, y_pred, lw=0.5, color='k', dashes=[10, 5])

            ax.set_xlim(data.min().min(), data.max().max())
            ax.set_ylim(data.min().min(), data.max().max())

            if col_name == 'q2m':
                subplot_label = '(b)'
                ticks = [0, 2, 4, 6]
                ticks_minor = [1, 3, 5]
                ticklabels = [0, 2, 4, 6]
                xlabel = r'q$_\mathdefault{air, AWS_\mathdefault{ZAC}}$ (g kg$^\mathdefault{-1}$)'
                ylabel = r'q$_\mathdefault{air, RACMO}$ (g kg$^\mathdefault{-1}$)'
                bounds = (0, 6)
                # x_label_coord = (0.5, -0.16)
                x_label_coord = (0.5, -0.12)
                # y_label_coord = (-0.14, 0.5)
                y_label_coord = (-0.12, 0.5)

            elif col_name == 't2m':
                subplot_label = '(a)'
                ticks = [-20, -10, 0, 10, 20]
                ticks_minor = [-15, -5, 5, 15]
                ticklabels = [-20, -10, 0, 10, 20]
                xlabel = r'T$_\mathdefault{air, AWS_\mathdefault{ZAC}}$ ($^\circ{\mathdefault{C}}$)'
                ylabel = r'T$_\mathdefault{air, RACMO}$ ($^\circ{\mathdefault{C}}$)'
                bounds = (-20, 20)
                # x_label_coord = (0.5, -0.16)
                x_label_coord = (0.5, -0.12)
                # y_label_coord = (-0.19, 0.5)
                y_label_coord = (-0.12, 0.5)

            ax.set_xticks(ticks)
            ax.set_xticks(ticks_minor, minor=True)
            ax.set_xticklabels(ticklabels)
            ax.set_xlabel(xlabel, labelpad=0.1, fontsize=6)
            ax.xaxis.set_label_coords(*x_label_coord)

            ax.set_yticks(ticks)
            ax.set_yticks(ticks_minor, minor=True)
            ax.set_yticklabels(ticklabels)
            ax.set_ylabel(ylabel, labelpad=0.1, fontsize=6)
            ax.yaxis.set_label_coords(*y_label_coord)

            [ax.spines[i].set_visible(False) for i in ['right', 'top']]
            [i.set_linewidth(0.4) for i in ax.spines.values()]

            ax.spines['bottom'].set_bounds(bounds[0], bounds[1])
            ax.spines['bottom'].set_position(('outward', 4))
            ax.spines['left'].set_bounds(bounds[0], bounds[1])
            # ax.spines['left'].set_bounds(-1.5, 2)
            ax.spines['left'].set_position(('outward', 4))

            tick_params_major = dict(direction='out', length=2, pad=1, labelsize=6, width=0.4, color='black')
            ax.tick_params(axis='both', which='major', **tick_params_major)
            ax.tick_params(axis='both', which='minor', length=1, width=0.2, color='black')

            # Draw grid
            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=0)
            # # Get spine bounds
            x0, x1 = ax.spines['bottom'].get_bounds()
            y0, y1 = ax.spines['left'].get_bounds()
            # Create clipping rectangle based on spine bounds
            grid_clip = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax.transData)
            # Clip gridlines
            for line in ax.get_xgridlines() + ax.get_ygridlines():
                line.set_clip_path(grid_clip)

            ax.text(-0.01, 0.945, f"{subplot_label}", transform=ax.transAxes, color='k', fontdict={'size': 7, 'weight': 'bold'}, zorder=12)

            ax.set_aspect('equal', adjustable='box')
            # Define legend
            marker_style_legend = dict(ms=5, lw=0, mew=0.5)
            handleba2 = Line2D([], [], label=f"N: {len(data)}", **marker_style_legend)
            handleba3 = Line2D([], [], label=f"b0: {np.round(slope, 2)}", **marker_style_legend)
            handleba4 = Line2D([], [], label=f"b1: {np.round(intercept, 2)}", **marker_style_legend)
            # Assign stars depending on p-value
            if pvalue <= 0.01:
                stars = r"$^{**}$"
            elif pvalue <= 0.05:
                stars = r"$^{*}$"
            else:
                stars = ""
            # # build LaTeX-friendly label
            r2_value = np.round(r_value**2, 2)
            r2_plabel = fr"R$^\mathdefault{{2}}$: {r2_value}{stars}"
            handleba5= Line2D([], [], label=r2_plabel, **marker_style_legend)

            rmse = root_mean_squared_error(data[col_name], data[f"{col_name}_zac_mod"])
            rmse = np.round(rmse, 2)

            handleba6= Line2D([],[],label=rf"RMSE: {rmse}", **marker_style_legend)
            handleba = [handleba2, handleba3, handleba4, handleba5, handleba6]
            ax.legend(bbox_to_anchor=(1., -0.05), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=5, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')

        targets = dict(left=160, right=0, top=15, bottom=160, hspace=0, wspace=350)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Save the plot
        logger.info(f"Saving plot of 'Freya on RACMO mask'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'evaluate_and_plot_t2m_q2m_at_zac.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)


        # %%

class YoungSoundFjordAnalaysis:
    def preprocess_young_sound_ice_break_freeze_duration():
        # %%
        input_path = Path('/nas/data/raw/gem/marinebasis')
        output_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')

        # Sea ice breakup
        sif_break = pd.read_csv(f"{input_path}/Sea_ice_breakup_10.17897_SM1Q-6A72_data.txt", sep='\t')
        sif_break['Start open water'] = pd.to_datetime(sif_break['Start open water'] )
        for i in range(len(sif_break)):
            sif_break.loc[i, 'sif_break_doy'] = (sif_break.loc[i, 'Start open water']).timetuple().tm_yday
        sif_break = pd.DataFrame(sif_break.loc[1:, :]) # Remove 2024 for consistency with other variables
        sif_break.index = sif_break['Year']

        # Sea ice formation
        sif_formation = pd.read_csv(f"{input_path}/Sea_ice_formation_10.17897_5MNP-KX83_data.txt", sep='\t')
        sif_formation['End open water'] = pd.to_datetime(sif_formation['End open water'] )
        for i in range(len(sif_formation)):
            sif_formation.loc[i, 'sif_formation_doy'] = (sif_formation.loc[i, 'End open water']).timetuple().tm_yday
        sif_formation.index = sif_formation['Year']

        # Open water duration
        sif_free_duration = pd.read_csv(f"{input_path}/Open_water_duration_10.17897_R3QB-7Q71_data.txt", sep='\t')
        sif_free_duration.rename(columns={'Open water duration (days)':'open_water_duration_days'}, inplace=True)
        # sif_free_duration['sif_break_minus_formation'] = sif_formation['sif_formation_doy'].values - sif_break['sif_break_doy'].values
        sif_free_duration.index = sif_free_duration['Year']

        # Combine all data
        sif_combined = pd.concat([sif_break['sif_break_doy'], sif_formation['sif_formation_doy'], sif_free_duration['open_water_duration_days']], axis=1)
        # Sort by Year in ascending order
        sif_combined = sif_combined.sort_values(by='Year', ascending=True)
        sif_combined.to_csv(f"{output_path}/young_sound_fjord_sea_ice_combined.csv")

    # %%
    def plot_time_series_young_sound_sea_ice_at_break_freeze_duration():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)

        # SMB and Sea ice data only till 2020
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])
        young_sound_fjord_sea_ice_combined['sif_break_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_break_doy'] - young_sound_fjord_sea_ice_combined['sif_break_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_break_doy'].std()
        young_sound_fjord_sea_ice_combined['sif_formation_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_formation_doy'] - young_sound_fjord_sea_ice_combined['sif_formation_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_formation_doy'].std()
        young_sound_fjord_sea_ice_combined['open_water_duration_days_norm'] = (young_sound_fjord_sea_ice_combined['open_water_duration_days'] - young_sound_fjord_sea_ice_combined['open_water_duration_days'].mean()) / young_sound_fjord_sea_ice_combined['open_water_duration_days'].std()

        years = young_sound_fjord_sea_ice_combined.index.astype(float).values
        date_range = np.arange(years[0], years[-1] + 0.5, 0.5)
        x_pred = (date_range - years[0]) / (years[1] - years[0])

        major_years = [2002, 2006, 2010, 2014, 2018]
        minor_years = [y for y in years if y not in major_years]
        xlim = (years[0], years[-1])

        fig = plt.figure(figsize=(3, 1.6), dpi=600)
        gs = plt.GridSpec(2, 2, hspace=0.5, wspace=0)   # small internal spacing
        ax = fig.add_subplot(gs[0, 0])

        # Plot
        young_sound_doy = pd.DataFrame(index=young_sound_fjord_sea_ice_combined.index, columns=np.arange(120, 360), data=0)
        for year in years:
            young_sound_doy.loc[year, young_sound_fjord_sea_ice_combined.loc[year, 'sif_break_doy']:young_sound_fjord_sea_ice_combined.loc[year, 'sif_formation_doy']] = 1

        # Define colormap
        colors = [(255, 0, 0), (255, 255, 255), (65, 105, 225),]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.0001, 0.0001, 1])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        x_min = young_sound_doy.columns.min()
        x_max = young_sound_doy.columns.max()
        y_min = young_sound_doy.index.min()
        y_max = young_sound_doy.index.max()

        im = ax.imshow(young_sound_doy, aspect='auto', cmap=cmap, norm=norm, origin='lower', extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5])
        ax.axvline(x=young_sound_fjord_sea_ice_combined['sif_break_doy'].median(), color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=young_sound_fjord_sea_ice_combined['sif_formation_doy'].median(), color='b', lw=0.5, ls='--', zorder=1)

        ax.set_xlim([x_min - 0.5, x_max + 0.5])
        # ax.set_xticks([120, 160, 200, 240, 280, 320, 360])
        ax.set_xticks([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticklabels([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticks([135, 165, 195, 225, 255, 285, 315, 345], minor=True)
        ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        major_years = [2002, 2005, 2008, 2011, 2014, 2017, 2020]
        minor_years = [y for y in years if y not in major_years]
        ylim = (years[0] - 0.5, years[-1] + 0.5)
        ax.set_ylim(ylim)
        ax.set_yticks(major_years)
        ax.set_yticklabels(major_years)
        ax.set_yticks(minor_years, minor=True)

        ax.tick_params(axis='both', which='major', direction='out', pad=0.8, length=4, width=0.5, labelsize=6)
        # ax.tick_params(axis='both', which='minor', direction='out', length=1.5, width=0.5)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=4)
        ax.tick_params(axis='y', which='minor', direction='out', length=1.5, width=0.5)

        ax.set_xlabel('DOY', horizontalalignment='center', verticalalignment='center', fontsize=6)
        # ax.yaxis.label.set_rotation(90)
        # ax.xaxis.set_label_coords(0, -)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(major_years[0], major_years[-1])
        ax.spines['bottom'].set_bounds(x_min, x_max)
        ax.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.3) for i in ax.spines.values()]

        ax.text(0.05, 1, 'a)', ha='center', va='center', fontsize=6, transform=ax.transAxes, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

        ax = fig.add_subplot(gs[1, 0:2])
        for young_sound_sif, color in ([('sif_break_doy_norm', 'r'), ('sif_formation_doy_norm', 'b'), ('open_water_duration_days_norm', 'g')]):
            # Plot sea ice breakup DOY with Mann-Kendall trend lines
            ax.plot(young_sound_fjord_sea_ice_combined.index, young_sound_fjord_sea_ice_combined[young_sound_sif], color=color, lw=0.5, marker='o', ms=0.5)
            # mk_result_sif_break = mk.original_test(young_sound_fjord_sea_ice_combined[young_sound_sif].values)
            # sif_break_pred = mk_result_sif_break.slope * x_pred + mk_result_sif_break.intercept
            # ax.plot(date_range, sif_break_pred, lw=0.5, color=color, ls='-.', zorder=1)
        ax.axhline(y=0, color='k', lw=0.5, ls='--', zorder=1)

        ax.set_xlim(xlim)
        ax.set_xticks(major_years)
        ax.set_xticklabels(major_years)
        ax.set_xticks(minor_years, minor=True)

        ax.set_ylim([-2.5, 2.5])
        ax.set_yticks(np.arange(-2, 3, 1))
        ax.set_yticklabels(np.arange(-2, 3, 1))

        ax.set_ylabel('Normalized', fontsize=6, horizontalalignment='center', verticalalignment='center')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.3) for i in ax.spines.values()]
        ax.spines['left'].set_bounds(-2, 2)

        ax.grid(which='major', axis='both', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)
        ax.tick_params(axis='both', which='major', direction='out', length=3, width=0.5, labelsize=6, pad=0.8)
        ax.tick_params(axis='both', which='minor', direction='out', length=1.5, width=0.5)

        ax.text(0.03, 1, 'b)', ha='center', va='center', fontsize=6, transform=ax.transAxes, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        # Plot legend
        ax = fig.add_subplot(gs[0, 1])
        [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'bottom', 'left']]
        ax.axis('off')

        marker_style_legend = dict(lw=0.5)
        open_water = Patch(color=mcolors.to_hex(tuple(x / 255 for x in (65, 105, 225))), label='Open water', linewidth=0)
        break_med_handle = Line2D([], [], label='Median Fjord-ice breakup DOY', color='r', ls='--', **marker_style_legend)
        freeze_med_handle = Line2D([], [], label='Median Fjord-ice formation DOY', color='b', ls='--', **marker_style_legend)
        break_handle = Line2D([], [], label='Fjord-ice breakup', color='r', marker='o', ms=1, **marker_style_legend)
        freeze_handle = Line2D([], [], label='Fjord-ice formation', color='b', marker='o', ms=1, **marker_style_legend)
        open_handle = Line2D([], [], label='Open water duration', color='g', marker='o', ms=1, **marker_style_legend)

        handles = [open_water, break_med_handle, freeze_med_handle, break_handle, freeze_handle, open_handle]
        ax.legend(bbox_to_anchor=(0.6, 0.25), loc='center', handles=handles, ncol=1, borderaxespad=0, fontsize=5, frameon=False, handletextpad=0.5, labelspacing=0.2, markerscale=1.5, labelcolor='k', handlelength=1, handleheight=1)

        targets = dict(left=150, right=55, top=25, bottom=105, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_time_series_young_sound_sea_ice_at_break_freeze_duration'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_time_series_young_sound_sea_ice_at_break_freeze_duration.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
    # %%

    def calculate_young_sound_ice_break_freeze_duration_zscores():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)

        for period in ['2002_2020', '2016_2021']:
            # SMB and Sea ice data only till 2020
            sif_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[period.split('_')[0]:period.split('_')[1], :])
            sif_combined['sif_break_doy_norm'] = (sif_combined['sif_break_doy'] - sif_combined['sif_break_doy'].mean()) / sif_combined['sif_break_doy'].std()
            sif_combined['sif_formation_doy_norm'] = (sif_combined['sif_formation_doy'] - sif_combined['sif_formation_doy'].mean()) / sif_combined['sif_formation_doy'].std()
            sif_combined['open_water_duration_days_norm'] = (sif_combined['open_water_duration_days'] - sif_combined['open_water_duration_days'].mean()) / sif_combined['open_water_duration_days'].std()

            # --------------------------------------
            # Find the years when sea ice breakup DOY, formation DOY, and duration days are more than 1 standard deviation above or below the mean (i.e., normalized value >= 1 or <= -1).
            # --------------------------------------
            sif_conditions = ['Break (DOY)', 'Freeze (DOY)', 'Open water (days)']  # May to September
            zscores = ['z > 1', 'z < -1']
            miindex = pd.MultiIndex.from_product([sif_conditions, zscores], names=['sif_condition', 'zscores'])
            sif_conditions_zscores = pd.DataFrame(index=range(0, 6), columns=miindex)

            # Break-up years
            sif_conditions_zscores[('Break (DOY)', 'z > 1')] = pd.Series(sif_combined.loc[sif_combined.sif_break_doy_norm >= 1].index.astype(str).values)
            sif_conditions_zscores[('Break (DOY)', 'z < -1')] = pd.Series(sif_combined.loc[sif_combined.sif_break_doy_norm <= -1].index.astype(str).values)
            # Formation years
            sif_conditions_zscores[('Freeze (DOY)', 'z > 1')] = pd.Series(sif_combined.loc[sif_combined.sif_formation_doy_norm >= 1].index.astype(str).values)
            sif_conditions_zscores[('Freeze (DOY)', 'z < -1')] = pd.Series(sif_combined.loc[sif_combined.sif_formation_doy_norm <= -1].index.astype(str).values)
            # Duration years
            sif_conditions_zscores[('Open water (days)', 'z > 1')] = pd.Series(sif_combined.loc[sif_combined.open_water_duration_days_norm >= 1].index.astype(str).values)
            sif_conditions_zscores[('Open water (days)', 'z < -1')] = pd.Series(sif_combined.loc[sif_combined.open_water_duration_days_norm <= -1].index.astype(str).values)

            # Drop rows where all values are NaN and fill remaining NaN values with "–"
            sif_conditions_zscores = (sif_conditions_zscores.dropna(how='all').fillna('–'))
            sif_conditions_zscores.to_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_zscores_{period}.csv")

    def table_plot_young_sound_sea_ice_break_freeze_duration_zscores():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')

        for period in ['2002_2020', '2016_2021']:
            sif_break_freeze_duration_table_zscores = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_zscores_{period}.csv", header=[0, 1], index_col=0)

            # Plot significance test results as a table
            fig, ax = plt.subplots(figsize=(2, 1), dpi=600)  # Adjust size as needed
            ax.axis('off')  # Hide axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Flatten columns for multicolumn plotting
            flat_cols = [f"{c[1]}" for c in sif_break_freeze_duration_table_zscores.columns]
            n_cols = len(flat_cols)
            # Create the percentage of wind direction and wind speed data table
            sif_break_freeze_table = ax.table(cellText=sif_break_freeze_duration_table_zscores.values,   # Round numbers to 2 decimals
                                    colLabels=flat_cols,
                                    # rowLabels=sif_break_freeze_duration_table_zscores.index,
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0, 0, 1, 0.9], # [x, y, width, height]
                                    colWidths=[1.0 / n_cols] * n_cols) # THIS removes left blank space

            sif_break_freeze_table.auto_set_font_size(False)
            sif_break_freeze_table.set_fontsize(4)
            sif_break_freeze_table.scale(0.4, 0.3) # (x_scale, y_scale)
            # n_rows = len(sif_break_freeze_duration_table_zscores) + 1
            # row_height = 0.9 / n_rows
            # for (row, col), cell in sif_break_freeze_table.get_celld().items():
            #     cell.set_height(row_height)


            # --- Set cell sizes and alignment ---
            for (row, col), cell in sif_break_freeze_table.get_celld().items():
                if row == 0:
                    cell.set_fontsize(5)
                # # Heights
                # cell.set_height(0.009 if row == 0 else 0.005)
                # cell.set_height(0.0001 if row == 0 else 0.0001)
                # Center text
                cell.get_text().set_horizontalalignment('center')
                cell.get_text().set_verticalalignment('center')
                # Padding
                # cell.PAD = 0.01
                # Default borders
                cell.visible_edges = ''
                cell.set_edgecolor('#666666' if row > 0 else 'black')
                cell.set_linewidth(0.2 if row > 0 else 0.8)

            # --- Header styling ---
            for col in range(len(sif_break_freeze_duration_table_zscores.columns)):
                cell = sif_break_freeze_table[(0, col)]
                cell.get_text().set_color('k')
                cell.visible_edges = 'B'  # top & bottom line
                cell.set_linewidth(0.4)

            # --- Add horizontal lines for the rest ---
            n_rows = len(sif_break_freeze_duration_table_zscores.index) + 1
            n_cols = len(sif_break_freeze_duration_table_zscores.columns)
            for row in range(1, n_rows):
                for col in range(n_cols):
                    sif_break_freeze_table[(row, col)].visible_edges = 'B'

            n_cols = len(sif_break_freeze_duration_table_zscores.columns)

            # --- Add group labels above the table ---
            group_labels = list(dict.fromkeys([col[0] for col in sif_break_freeze_duration_table_zscores.columns]))
            n_groups = len(group_labels)
            for i, label in enumerate(group_labels):
                x_center = (i + 0.5) / n_groups  # evenly distribute across axes
                ax.text(x_center, 0.9, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
            ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

            targets = dict(left=20, right=10, top=0, bottom=10, hspace=0, wspace=0)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'table_plot_young_sound_sea_ice_break_freeze_duration_zscores_{period}'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f'table_plot_young_sound_sea_ice_break_freeze_duration_zscores_{period}.png'
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    def calculate_young_sound_sea_ice_break_freeze_duration_quantiles():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)

        # SMB and Sea ice data only till 2020
        sif_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])

        # --------------------------------------
        # Table of the 25% and 75% quantiles of sea ice breakup DOY, formation DOY, and duration days, along with their anomalies.
        # --------------------------------------
        sif_conditions = ['Break (DOY)', 'Freeze (DOY)', 'Open water (days)']  # May to September
        sif_conditions_quantiles = ['25 %', '75 %']
        miindex = pd.MultiIndex.from_product([sif_conditions, sif_conditions_quantiles], names=['sif_condition', 'quantiles'])
        sif_break_freeze_duration_table_25_75 = pd.DataFrame(index=range(0, 6), columns=miindex)

        # Breakup DOY
        sif_break_25 = sif_combined.loc[sif_combined['sif_break_doy'] < sif_combined['sif_break_doy'].quantile(0.25)]
        sif_break_freeze_duration_table_25_75[('Break (DOY)', '25 %')] = pd.Series((sif_break_25['sif_break_doy'].round(1).astype(int).astype(str) + ' (' + sif_break_25.index.astype(str) + ')' + '\n' + '(' + sif_break_25['sif_break_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_break_75 = sif_combined.loc[sif_combined['sif_break_doy'] > sif_combined['sif_break_doy'].quantile(0.75)]
        sif_break_freeze_duration_table_25_75[('Break (DOY)', '75 %')] = pd.Series((sif_break_75['sif_break_doy'].round(1).astype(int).astype(str) + ' (' + sif_break_75.index.astype(str) + ')' + '\n' + '(' + sif_break_75['sif_break_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)

        # Formation DOY
        sif_formation_25 = sif_combined.loc[sif_combined['sif_formation_doy'] < sif_combined['sif_formation_doy'].quantile(0.25)]
        sif_break_freeze_duration_table_25_75[('Freeze (DOY)', '25 %')] = pd.Series((sif_formation_25['sif_formation_doy'].round(1).astype(int).astype(str) + ' (' + sif_formation_25.index.astype(str) + ')' + '\n' + '(' + sif_formation_25['sif_formation_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_formation_75 = sif_combined.loc[sif_combined['sif_formation_doy'] > sif_combined['sif_formation_doy'].quantile(0.75)]
        sif_break_freeze_duration_table_25_75[('Freeze (DOY)', '75 %')] = pd.Series((sif_formation_75['sif_formation_doy'].round(1).astype(int).astype(str) + ' (' + sif_formation_75.index.astype(str) + ')' + '\n' + '(' + sif_formation_75['sif_formation_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)

        # Duration days
        sif_duration_25 = sif_combined.loc[sif_combined['open_water_duration_days'] < sif_combined['open_water_duration_days'].quantile(0.25)]
        sif_break_freeze_duration_table_25_75[('Open water (days)', '25 %')] = pd.Series((sif_duration_25['open_water_duration_days'].round(1).astype(int).astype(str) + ' (' + sif_duration_25.index.astype(str) + ')' + '\n' + '(' + sif_duration_25['open_water_duration_days_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_duration_75 = sif_combined.loc[sif_combined['open_water_duration_days'] > sif_combined['open_water_duration_days'].quantile(0.75)]
        sif_break_freeze_duration_table_25_75[('Open water (days)', '75 %')] = pd.Series((sif_duration_75['open_water_duration_days'].round(1).astype(int).astype(str) + ' (' + sif_duration_75.index.astype(str) + ')' + '\n' + '(' + sif_duration_75['open_water_duration_days_anomalies'].round(0).astype(int).astype(str) + ')').values)
        # Drop rows where all values are NaN and fill remaining NaN values with "–"
        sif_break_freeze_duration_table_25_75 = (sif_break_freeze_duration_table_25_75.dropna(how='all').fillna('–'))
        sif_break_freeze_duration_table_25_75.to_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_25_75.csv")

        # --------------------------------------
        # Table of the 5% and 95% quantiles of sea ice breakup DOY, formation DOY, and duration days, along with their anomalies.
        # --------------------------------------
        sif_conditions_quantiles = ['5 %', '95 %']
        miindex = pd.MultiIndex.from_product([sif_conditions, sif_conditions_quantiles], names=['sif_condition', 'quantiles'])
        sif_break_freeze_duration_table_5_95 = pd.DataFrame(index=range(0, 6), columns=miindex)

        # Breakup DOY
        sif_break_5 = sif_combined.loc[sif_combined['sif_break_doy'] < sif_combined['sif_break_doy'].quantile(0.05)]
        sif_break_freeze_duration_table_5_95[('Break (DOY)', '5 %')] = pd.Series((sif_break_5['sif_break_doy'].round(1).astype(int).astype(str) + ' (' + sif_break_5.index.astype(str) + ')' + '\n' + '(' + sif_break_5['sif_break_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_break_95 = sif_combined.loc[sif_combined['sif_break_doy'] > sif_combined['sif_break_doy'].quantile(0.95)]
        sif_break_freeze_duration_table_5_95[('Break (DOY)', '95 %')] = pd.Series((sif_break_95['sif_break_doy'].round(1).astype(int).astype(str) + ' (' + sif_break_95.index.astype(str) + ')' + '\n' + '(' + sif_break_95['sif_break_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)

        # Formation DOY
        sif_formation_5 = sif_combined.loc[sif_combined['sif_formation_doy'] < sif_combined['sif_formation_doy'].quantile(0.05)]
        sif_break_freeze_duration_table_5_95[('Freeze (DOY)', '5 %')] = pd.Series((sif_formation_5['sif_formation_doy'].round(1).astype(int).astype(str) + ' (' + sif_formation_5.index.astype(str) + ')' + '\n' + '(' + sif_formation_5['sif_formation_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_formation_95 = sif_combined.loc[sif_combined['sif_formation_doy'] > sif_combined['sif_formation_doy'].quantile(0.95)]
        sif_break_freeze_duration_table_5_95[('Freeze (DOY)', '95 %')] = pd.Series((sif_formation_95['sif_formation_doy'].round(1).astype(int).astype(str) + ' (' + sif_formation_95.index.astype(str) + ')' + '\n' + '(' + sif_formation_95['sif_formation_doy_anomalies'].round(0).astype(int).astype(str) + ')').values)

        # Duration days
        sif_duration_5 = sif_combined.loc[sif_combined['open_water_duration_days'] < sif_combined['open_water_duration_days'].quantile(0.05)]
        sif_break_freeze_duration_table_5_95[('Open water (days)', '5 %')] = pd.Series((sif_duration_5['open_water_duration_days'].round(1).astype(int).astype(str) + ' (' + sif_duration_5.index.astype(str) + ')' + '\n' + '(' + sif_duration_5['open_water_duration_days_anomalies'].round(0).astype(int).astype(str) + ')').values)
        sif_duration_95 = sif_combined.loc[sif_combined['open_water_duration_days'] > sif_combined['open_water_duration_days'].quantile(0.95)]
        sif_break_freeze_duration_table_5_95[('Open water (days)', '95 %')] = pd.Series((sif_duration_95['open_water_duration_days'].round(1).astype(int).astype(str) + ' (' + sif_duration_95.index.astype(str) + ')' + '\n' + '(' + sif_duration_95['open_water_duration_days_anomalies'].round(0).astype(int).astype(str) + ')').values)
        # Drop rows where all values are NaN and fill remaining NaN values with "–"
        sif_break_freeze_duration_table_5_95 = (sif_break_freeze_duration_table_5_95.dropna(how='all').fillna('–'))
        sif_break_freeze_duration_table_5_95.to_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_5_95.csv")

            # %%


    def table_plot_young_sound_ice_break_freeze_duration_5_95():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        sif_break_freeze_duration_table_5_95 = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_5_95.csv", header=[0, 1], index_col=0)

        # Plot significance test results as a table
        fig, ax = plt.subplots(figsize=(2, 0.6), dpi=600)  # Adjust size as needed
        ax.axis('off')  # Hide axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Flatten columns for multicolumn plotting
        flat_cols = [f"{c[1]}" for c in sif_break_freeze_duration_table_5_95.columns]
        n_cols = len(flat_cols)
        # Create the percentage of wind direction and wind speed data table
        sif_break_freeze_table = ax.table(cellText=sif_break_freeze_duration_table_5_95.values,   # Round numbers to 2 decimals
                                colLabels=flat_cols,
                                # rowLabels=sif_break_freeze_duration_table_5_95.index,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 0.9], # [x, y, width, height]
                                colWidths=[1.0 / n_cols] * n_cols) # THIS removes left blank space

        sif_break_freeze_table.auto_set_font_size(False)
        sif_break_freeze_table.set_fontsize(4)
        sif_break_freeze_table.scale(0.4, 0.3) # (x_scale, y_scale)
        # n_rows = len(sif_break_freeze_duration_table_5_95) + 1
        # row_height = 0.9 / n_rows
        # for (row, col), cell in sif_break_freeze_table.get_celld().items():
        #     cell.set_height(row_height)


        # --- Set cell sizes and alignment ---
        for (row, col), cell in sif_break_freeze_table.get_celld().items():
            if row == 0:
                cell.set_fontsize(5)
            # # Heights
            # cell.set_height(0.009 if row == 0 else 0.005)
            # cell.set_height(0.0001 if row == 0 else 0.0001)
            # Center text
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
            # Padding
            # cell.PAD = 0.01
            # Default borders
            cell.visible_edges = ''
            cell.set_edgecolor('#666666' if row > 0 else 'black')
            cell.set_linewidth(0.2 if row > 0 else 0.8)

        # --- Header styling ---
        for col in range(len(sif_break_freeze_duration_table_5_95.columns)):
            cell = sif_break_freeze_table[(0, col)]
            cell.get_text().set_color('k')
            cell.visible_edges = 'B'  # top & bottom line
            cell.set_linewidth(0.4)

        # --- Add horizontal lines for the rest ---
        n_rows = len(sif_break_freeze_duration_table_5_95.index) + 1
        n_cols = len(sif_break_freeze_duration_table_5_95.columns)
        for row in range(1, n_rows):
            for col in range(n_cols):
                sif_break_freeze_table[(row, col)].visible_edges = 'B'

        n_cols = len(sif_break_freeze_duration_table_5_95.columns)

        # --- Add group labels above the table ---
        group_labels = list(dict.fromkeys([col[0] for col in sif_break_freeze_duration_table_5_95.columns]))
        n_groups = len(group_labels)
        for i, label in enumerate(group_labels):
            x_center = (i + 0.5) / n_groups  # evenly distribute across axes
            ax.text(x_center, 0.9, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
        ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

        targets = dict(left=20, right=10, top=0, bottom=10, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_young_sound_ice_break_freeze_duration_5_95'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'table_plot_young_sound_ice_break_freeze_duration_5_95.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def table_plot_young_sound_ice_break_freeze_duration_25_75():
        # %%
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        sif_break_freeze_duration_table_25_75 = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_break_freeze_duration_table_25_75.csv", header=[0, 1], index_col=0)

        # Plot significance test results as a table
        fig, ax = plt.subplots(figsize=(2, 1), dpi=600)  # Adjust size as needed
        ax.axis('off')  # Hide axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Flatten columns for multicolumn plotting
        flat_cols = [f"{c[1]}" for c in sif_break_freeze_duration_table_25_75.columns]
        n_cols = len(flat_cols)
        # Create the percentage of wind direction and wind speed data table
        sif_break_freeze_table = ax.table(cellText=sif_break_freeze_duration_table_25_75.values,   # Round numbers to 2 decimals
                                colLabels=flat_cols,
                                # rowLabels=sif_break_freeze_duration_table_25_75.index,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 0.9], # [x, y, width, height]
                                colWidths=[1.0 / n_cols] * n_cols) # THIS removes left blank space

        sif_break_freeze_table.auto_set_font_size(False)
        sif_break_freeze_table.set_fontsize(4)
        sif_break_freeze_table.scale(0.4, 0.3) # (x_scale, y_scale)
        # n_rows = len(sif_break_freeze_duration_table_25_75) + 1
        # row_height = 0.9 / n_rows
        # for (row, col), cell in sif_break_freeze_table.get_celld().items():
        #     cell.set_height(row_height)


        # --- Set cell sizes and alignment ---
        for (row, col), cell in sif_break_freeze_table.get_celld().items():
            if row == 0:
                cell.set_fontsize(5)
            # # Heights
            # cell.set_height(0.009 if row == 0 else 0.005)
            # cell.set_height(0.0001 if row == 0 else 0.0001)
            # Center text
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
            # Padding
            # cell.PAD = 0.01
            # Default borders
            cell.visible_edges = ''
            cell.set_edgecolor('#666666' if row > 0 else 'black')
            cell.set_linewidth(0.2 if row > 0 else 0.8)

        # --- Header styling ---
        for col in range(len(sif_break_freeze_duration_table_25_75.columns)):
            cell = sif_break_freeze_table[(0, col)]
            cell.get_text().set_color('k')
            cell.visible_edges = 'B'  # top & bottom line
            cell.set_linewidth(0.4)

        # --- Add horizontal lines for the rest ---
        n_rows = len(sif_break_freeze_duration_table_25_75.index) + 1
        n_cols = len(sif_break_freeze_duration_table_25_75.columns)
        for row in range(1, n_rows):
            for col in range(n_cols):
                sif_break_freeze_table[(row, col)].visible_edges = 'B'

        n_cols = len(sif_break_freeze_duration_table_25_75.columns)

        # --- Add group labels above the table ---
        group_labels = list(dict.fromkeys([col[0] for col in sif_break_freeze_duration_table_25_75.columns]))
        n_groups = len(group_labels)
        for i, label in enumerate(group_labels):
            x_center = (i + 0.5) / n_groups  # evenly distribute across axes
            ax.text(x_center, 0.9, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
        ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

        targets = dict(left=20, right=10, top=0, bottom=10, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'table_plot_young_sound_ice_break_freeze_duration_25_75'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'table_plot_young_sound_ice_break_freeze_duration_25_75.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

class SeaIceAnalysis:
    def plot_sea_ice_masks():
        # %%
        # Point mask of closest Freya glacier mask racmo at 5 km
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')

        # Plot
        # fig = plt.figure(figsize=(2.2, 1.5), dpi=600)
        fig = plt.figure(figsize=(2.1, 1.5), dpi=600)
        # gs = plt.GridSpec(2, 2, hspace=0, wspace=0.01)   # small internal spacing

        # ax = fig.add_subplot(gs[0:2, 0])
        # Greenland map
        ax = fig.add_axes([0.002, 0.05, 0.55, 0.9])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        variable = 'sif'
        lonlat = _get_latlon_meshgrid(variable=variable)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=5)

        # Land sea mask
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude(variable=variable)
        ax.contour(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, linewidths=0.05, zorder=9)

        # Plot station
        # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]
        ax.plot(station_rlon, station_rlat, marker='x', color='r', markeredgewidth=0.5, markersize=2, zorder=12, label='Closest point to Freya')

        # Read the sea ice buffer masks at different radii around the closest point to Freya in the glacier mask at 5 km resolution
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        radius_int_x = 10
        radius_int_y = 120
        for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
            binary_mask = xr.where(sea_ice_buffer_da.sel(radius=radius) > 0.5, 1, 0)
            cs = ax.contour(lonlat[0], lonlat[1], binary_mask, levels=[0.5], colors='grey', linewidths=0.5,  linestyles='dashed', zorder=1)
            # Add text labels for the radius of the circles
            circle_scale = 0.05 / 5.5
            if radius in [200, 400, 600, 800]:  # Add labels for every other radius to avoid clutter
                # if radius == 200:
                # _curved_circle_text(ax, station_rlon, station_rlat, 200, '200 km', scale=circle_scale, tune=330)
                # elif radius == 400:
                #     _curved_circle_text(ax, station_rlon, station_rlat, 400, '400 km', scale=circle_scale, tune=330)
                # elif radius == 600:
                #     _curved_circle_text(ax, station_rlon, station_rlat, 600, '600 km', scale=circle_scale, tune=330)
                # elif radius == 800:
                #     _curved_circle_text(ax, station_rlon, station_rlat, 800, '800 km', scale=circle_scale, tune=330)
                # else:
                _curved_circle_text(ax, station_rlon, station_rlat, radius, f"{radius} km", scale=circle_scale, tune=330)

        # Draw rectangle for zoomed-in map
        zoom_xlim = (4, 5.5)
        zoom_ylim = (2, 4)
        rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),                     # lower-left corner
                         zoom_xlim[1] - zoom_xlim[0],                      # width
                         zoom_ylim[1] - zoom_ylim[0],                      # height
                         linewidth=0.5, edgecolor='k', facecolor='none', zorder=15)
        ax.add_patch(rect)

        [i.set_visible(False) for i in ax.spines.values()]  # Set visible spines to have a linewidth of 0.3
        ax.set_xlim([-7.5, 12.5])
        # ax.set_ylim([-13, 13])
        ax.set_ylim([-12, 11.6])
        ax.axis('off')
        ax.set_aspect('equal', adjustable='box')

        # Plot zoomed-in map
        # ax = fig.add_subplot(gs[0, 1])
        ax = fig.add_axes([0.48, 0.53, 0.55, 0.45])   # [left, bottom, width, height] in figure coordinates (0 to 1)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=0)

        # Land sea mask
        # binary_mask = xr.where(land_sea_mask > 0.5, 1, 0)
        # ax.contour(lonlat[0], lonlat[1], binary_mask, levels=[0.5], colors='black', linewidths=0.5, zorder=9)

        # Plot station
        ax.plot(station_rlon, station_rlat, marker='x', color='red', markersize=3,  markeredgewidth=0.5, zorder=12, label='Closest point to Freya')

        # Plot sea ice buffer mask at 100-km radius
        colors_buffer = [(169, 169, 169), (169, 169, 169)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        ax.contour(lonlat[0], lonlat[1], sea_ice_buffer_da.sel(radius=100).data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)
        _curved_circle_text(ax, station_rlon, station_rlat, 100, '100 km', scale=circle_scale, tune=55, angle_center=0)

        radius_int_x = 10
        radius_int_y = 120
        rlat_rlon_1km = 0.05 / 5.5
        add_radius_x = rlat_rlon_1km * radius_int_x # rlon in 100 km, 200 km ...
        add_radius_y = rlat_rlon_1km * radius_int_y
        # ax.text(station_rlon + 0.7, station_rlat + 0.1, '100 km', size=4, color='k', path_effects=[pe.withStroke(linewidth=1, foreground='white')], zorder=12)

        # Add young sound fjord mask
        ys_mask = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        colors_buffer = [(169, 169, 169), (0, 0, 205)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        cs_young_sound = ax.contour(lonlat[0], lonlat[1], ys_mask, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)

        ax.set_xlim(zoom_xlim)
        ax.set_ylim(zoom_ylim)
        ax.set_aspect('equal', adjustable='box')

        xticks = [zoom_xlim[0]]
        xticklables_km = ['', 50, 100, 150, ]
        # xticklables_km = ['', 50, 100, 150, 200, 250]
        for distance in xticklables_km[1:]:
            xticks.append(zoom_xlim[0] + rlat_rlon_1km * distance)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklables_km)

        yticks = [zoom_ylim[0]]
        yticklables_km = ['', 50, 100, 150, 200]
        for distance in yticklables_km[1:]:
            yticks.append(zoom_ylim[0] + rlat_rlon_1km * distance)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklables_km)

        # Manually add zero at lower left
        ax.text(zoom_xlim[0] - 0.05, zoom_ylim[0] - 0.05, '0', ha='right', va='top', fontsize=5)
        ax.tick_params(axis='both', which='major', direction='inout', pad=0.8, length=4, width=0.5, labelsize=5)
        ax.text(0.5, -0.2, 'km', ha='center', va='center', fontsize=5, transform=ax.transAxes)
        ax.text(-0.4, 0.5, 'km', ha='center', va='center', fontsize=5, rotation=90, transform=ax.transAxes)

        [i.set_linewidth(0.3) for i in ax.spines.values()]

        # Plot legends
        # ax = fig.add_subplot(gs[1, 1])
        ax = fig.add_axes([0.47, 0.005, 0.55, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)

        [i.set_visible(False) for i in ax.spines.values()]
        ax.axis('off')

        marker_style_legend = dict(lw=0.5)
        greenland_icesheet_patch = Patch(color=mcolors.to_hex(tuple(x / 255 for x in (135, 206, 235))), label='GrIS', linewidth=0)
        peripheral_patch = Patch(color=mcolors.to_hex(tuple(x / 255 for x in (65, 105, 225))), label='PGIC', linewidth=0)
        tundra_patch = Patch(color=mcolors.to_hex(tuple(x / 255 for x in (244, 164, 96))), label='Tundra', linewidth=0)
        sif_buffer_patch = Circle((0.1, 0.1), radius=0.4, edgecolor='grey', facecolor='none', linestyle='--', linewidth=0.8,label='Sea ice buffer (km)')
        # Number of grid cells in the young sound fjord mask that are equal to 1 (indicating they are part of the fjord)
        num_cells_equal_1 = (ys_mask == 1).sum().item()
        young_sound_area = int(num_cells_equal_1 * (5.5 * 5.5))  # Area in km^2 (5 km x 5 km grid cells)
        young_sound_buffer_patch = Circle((0.1, 0.1), radius=0.4, edgecolor=mcolors.to_hex(tuple(x / 255 for x in (0, 0, 205))), facecolor='none', linestyle='--', linewidth=0.8,label=rf'Young Sound buffer ({young_sound_area} km$^\mathdefault{{2}}$)')
        station_patch = Line2D([0], [0], marker='x', color='red', linestyle='None', markeredgewidth=0.5, markersize=3, label='Closest point to Freya Glacier ')

        # --- Add to legend with custom handler ---
        handles = [greenland_icesheet_patch, peripheral_patch, tundra_patch, sif_buffer_patch, young_sound_buffer_patch, station_patch, ]
        ax.legend(bbox_to_anchor=(0.5, 0.5), loc='center', handles=handles, ncol=1, borderaxespad=0,
                  fontsize=5, frameon=False, handletextpad=0.5, labelspacing=0.4, markerscale=1.5,
                  labelcolor='k', handlelength=1, handleheight=1, handler_map={Circle: HandlerCircle()})

        # targets = dict(left=10, right=55, top=30, bottom=10, hspace=0, wspace=0)
        # fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        # fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_sea_ice_masks'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_sea_ice_masks.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def calculate_greenland_sea_sif_phenology():
        """
        Calculate Day of Opening (DOO), Day of Retreat (DOR), Day of Advance (DOA), and Day of Closing (DOC) based on RACMO sea ice concentration data in the Greenland Sea.

        The function reads RACMO sea ice concentration data, applies necessary masks to focus on the Greenland Sea, and then calculates the following metrics for each year:
        - Day of Opening (DOO): The last day when sea ice concentration is above 80% before the first summer minimum.
        - Day of Retreat (DOR): The last day when sea ice concentration drops below 15% before the first summer minimum. If it never drops below 15%, the DOR is set to the first summer minimum.
        - Day of Advance (DOA): The first day when sea ice concentration increases above 15% after the final summer minimum. If there is only one minimum, DOA is set to the final minimum.
        - Day of Closing (DOC): The first day when sea ice concentration increases above 80% after the final summer minimum.
        - Outer Ice-Free Period (OIFP): The difference between DOC and DOO, capturing both open ocean and ice transition periods.
        - Inner Ice-Free Period (IIFP): The difference between DOA and DOR, capturing the core period of open water between the retreat and advance of sea ice.

        References:
        https://doi.org/10.3390/rs10091328
        DOI 10.1088/1748-9326/aafb84
        """
        # %%
        # start_time = time.time()
        input_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/complex_mca_input')
        # racmo_sif = xr.open_dataset(f'{input_file_path}/ci_all.nc', chunks='auto')['ci']
        racmo_sif = xr.open_dataset(f'{input_file_path}/ci_all.nc')['ci'] # Try without chunking
        racmo_sif = racmo_sif.squeeze('height', drop=True)
        # Drop coordinates from the DataArray
        coords_to_drop = ['height', 'lat', 'lon', 'pressure']
        racmo_sif = racmo_sif.drop_vars([c for c in coords_to_drop if c in racmo_sif.coords])

        # Apply mask
        # ci_buffer = xr.open_dataarray(f"{mask_file_path}/ci_buffer_200km_on_racmo5km.nc")
        # racmo_sif = racmo_sif.where(ci_buffer != 0)
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        greenland_sea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        racmo_sif = racmo_sif.where(greenland_sea_mask != 0)

        # Subset to the region of interest (Greenland Sea)
        racmo_sif = racmo_sif.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
        # racmo_sif = racmo_sif.sel(rlon=slice(4, 5), rlat=slice(2, 4))
        racmo_sif[0, :, :].plot()

        # Chunk created task scheduling overhead, lazy graph building, chunk merging
        # racmo_sif = racmo_sif.chunk({'time':-1, 'rlat':100, 'rlon':100})

        # ------------------------ Preprocessing ------------------------
        # Remove sea ice concentration < 1% to avoid noise in the following calculations
        # racmo_sif = xr.where(racmo_sif < 0.01, 0, racmo_sif)

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

        # ------------------------ Calculate ------------------------
        # Calculate phenology metrics for each year and store in a new DataArray
        doo = racmo_sif_7d.groupby('sif_year').map(_doo_per_year)
        dor = racmo_sif_7d.groupby('sif_year').map(_dor_per_year)
        doa = racmo_sif_7d.groupby('sif_year').map(_doa_per_year)
        doc = racmo_sif_7d.groupby('sif_year').map(_doc_per_year)

        # ------------------------ Mask NaT values ------------------------
        # Mask doc if doo is not available
        mask = doo.isnull()
        doc = doc.where(~mask, np.datetime64("NaT"))

        # Mask doo if doc is not available
        mask = doc.isnull()
        doo = doo.where(~mask, np.datetime64("NaT"))

        # Mask dor, doa, oifp, iifp if doo is not available
        mask = doo.isnull()
        dor = dor.where(~mask, np.datetime64("NaT"))
        doa = doa.where(~mask, np.datetime64("NaT"))
        # oifp["oifp_doy"] = oifp["oifp_doy"].where(~mask, np.nan)
        # iifp["iifp_doy"] = iifp["iifp_doy"].where(~mask, np.nan)

        # Outer Ice-Free Period (OIFP) is the difference between DOC and DOO, which captures both open ocean and ice transition periods
        oifp = (doc - doo).dt.days # Difference in days (doc - doo)
        # Check for negative values in oifp.
        # These would indicate cases where DOC is before DOO, which is unexpected and may indicate an issue in the calculations.
        years, lats, lons = np.where(oifp < 0)
        print("Year indices:", years)
        print("Lat indices:", lats)
        print("Lon indices:", lons)

        # Inner Ice-Free Period (IIFP) is defined as the difference between DOA and DOR, which captures the period of open ocean when the grid cell ice fraction is less than 15%.
        iifp = (doa - dor).dt.days # Difference in days (doa - dor)
        # Check for negative values in iifp.
        # These would indicate cases where DOA is before DOR, which is unexpected and may indicate an issue in the calculations.
        years, lats, lons = np.where(iifp < 0)
        print("Year indices:", years)
        print("Lat indices:", lats)
        print("Lon indices:", lons)

        # ------------------------ Postprocessing ------------------------
        # Create Dataset with both datetime and DOY
        phenologies_ds = [doo, dor, doa, doc, oifp, iifp]
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenology_description_long_name_time = {'doo': 'Day of opening (timestamp)',
                                                'dor': 'Day of retreat (timestamp)',
                                                'doa': 'Day of advance (timestamp)',
                                                'doc': 'Day of complete (timestamp)'}
        phenology_description_long_name_doy = {'doo': 'Day of opening (day of year)',
                                                'dor': 'Day of retreat (day of year)',
                                                'doa': 'Day of advance (day of year)',
                                                'doc': 'Day of complete (day of year)',
                                                'oifp': 'Outer ice-free period (days)',
                                                'iifp': 'Inner ice-free period (days)'}
        phenology_description = {'doo': 'Last day SIC drops below 80% before the summer minimum',
                                    'dor': 'Last day SIC drops below 15% before the summer minimum',
                                    'doa': 'First day SIC increases above 15% following the final summer minimum',
                                    'doc': 'First day SIC increases above 80% following the final summer minimum',
                                    'oifp': 'Outer ice-free period (days)',
                                    'iifp': 'Inner ice-free period (days)'}
        phenology_description_title = {'doo': 'Day of sea-ice opening (Greenland Sea)',
                                        'dor': 'Day of sea-ice retreat (Greenland Sea)',
                                        'doa': 'Day of sea-ice advance (Greenland Sea)',
                                        'doc': 'Day of sea-ice complete (Greenland Sea)',
                                        'oifp': 'Outer ice-free period (Greenland Sea)',
                                        'iifp': 'Inner ice-free period (Greenland Sea)'}

        for phenology_ds, phenology in zip(phenologies_ds, phenologies):
            if phenology in ['oifp', 'iifp']:
                _save_phenology_variable(time_da=phenology_ds,
                                        var_name=phenology,
                                        long_name_doy=phenology_description_long_name_doy[phenology],
                                        description=phenology_description[phenology],
                                        title=phenology_description_title[phenology],
                                        output_path=Path(
                                            '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology'),
                                        filename=f'greenland_sea_sif_phenology_{phenology}.nc',)
            else:
                _save_phenology_variable(time_da=phenology_ds,
                                        var_name=phenology,
                                        long_name_time=phenology_description_long_name_time.pop(phenology),
                                        long_name_doy=phenology_description_long_name_doy.pop(phenology),
                                        description=phenology_description.pop(phenology),
                                        title=phenology_description_title.pop(phenology),
                                        output_path=Path(
                                            '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology'),
                                        filename=f'greenland_sea_sif_phenology_{phenology}.nc',)

        # %%
    def sanity_check_calculate_greenland_sea_sif_phenology():
        # %%
        # ------------------------ Sea ice ------------------------
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

        # ------------------------ Sea ice phenology ------------------------
        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        doo = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doo.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doa = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doa.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        dor = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_dor.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        oifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_oifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        iifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_iifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doc = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doc.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

        # %%
        # ------------------------ Sanity check ------------------------
        # Extract phenology DataArray for the same point and year
        sif_year = 2018 # TODO for the paper
        # rlon = racmo_sif_7d.rlon.values[75]; rlat = racmo_sif_7d.rlat.values[200] # TODO for the paper
        rlon = racmo_sif_7d.rlon.values[80]; rlat = racmo_sif_7d.rlat.values[200] # TODO for the paper
        # sif_year = 2018 # TODO for the paper
        # rlon = racmo_sif_7d.rlon.values[75]; rlat = racmo_sif_7d.rlat.values[200] # TODO for the paper

        # sif_year_point = 1990; rlat = 8.9; rlon = 7.05
        # rlon = racmo_sif_7d.rlon.values[130]; rlat = racmo_sif_7d.rlat.values[200]
        # rlon = racmo_sif_7d.rlon.values[50]; rlat = racmo_sif_7d.rlat.values[200]
        # # rlon = racmo_sif_7d.rlon.values[100]; rlat = racmo_sif_7d.rlat.values[200]

        _, ax = plt.subplots()
        racmo_sif_7d[-3, :, :].plot(ax=ax)
        ax.plot(rlon, rlat, 'x', color='r', ms=8, label='Selected Point')
        ax.legend()

        racmo_sif_7d_point = racmo_sif_7d.sel(rlon=[rlon], rlat=[rlat])
        racmo_sif_7d_point_df = racmo_sif_7d_point.to_dataframe().reset_index()
        racmo_sif_7d_point_df.index = racmo_sif_7d_point_df['time']
        racmo_sif_7d_point_df_yr = racmo_sif_7d_point_df.loc[racmo_sif_7d_point_df.sif_year.isin([sif_year])]

        doo_point = doo.sel(rlon=[rlon], rlat=[rlat])
        doo_point_df = doo_point.to_dataframe().reset_index()
        doo_point_df.index = doo_point['sif_year']
        # doo_point_df['doo_doy_convert'] = doo_point_df['doo_time'].dt.day_of_year
        doo_point_year_time = doo_point_df.loc[doo_point_df.sif_year == sif_year, 'doo_time'].values[0]
        doo_point_year_doy = doo_point_df.loc[doo_point_df.sif_year == sif_year, 'doo_doy'].values[0]

        dor_point = dor.sel(rlon=[rlon], rlat=[rlat])
        dor_point_df = dor_point.to_dataframe().reset_index()
        dor_point_df.index = dor_point['sif_year']
        dor_point_year_time = dor_point_df.loc[dor_point_df.sif_year == sif_year, 'dor_time'].values[0]
        dor_point_year_doy = dor_point_df.loc[dor_point_df.sif_year == sif_year, 'dor_doy'].values[0]

        doa_point = doa.sel(rlon=[rlon], rlat=[rlat])
        doa_point_df = doa_point.to_dataframe().reset_index()
        doa_point_df.index = doa_point['sif_year']
        doa_point_year_time = doa_point_df.loc[doa_point_df.sif_year == sif_year, 'doa_time'].values[0]
        doa_point_year_doy = doa_point_df.loc[doa_point_df.sif_year == sif_year, 'doa_doy'].values[0]

        doc_point = doc.sel(rlon=[rlon], rlat=[rlat])
        doc_point_df = doc_point.to_dataframe().reset_index()
        doc_point_df.index = doc_point['sif_year']
        doc_point_year_time = doc_point_df.loc[doc_point_df.sif_year == sif_year, 'doc_time'].values[0]
        doc_point_year_doy = doc_point_df.loc[doc_point_df.sif_year == sif_year, 'doc_doy'].values[0]

        oifp_point = oifp.sel(rlon=[rlon], rlat=[rlat])
        oifp_point_df = oifp_point.to_dataframe().reset_index()
        oifp_point_df.index = oifp_point['sif_year']
        oifp_point_year_doy = oifp_point_df.loc[oifp_point_df.sif_year == sif_year, 'oifp_doy'].values[0]

        iifp_point = iifp.sel(rlon=[rlon], rlat=[rlat])
        iifp_point_df = iifp_point.to_dataframe().reset_index()
        iifp_point_df.index = iifp_point['sif_year']
        iifp_point_year_doy = iifp_point_df.loc[iifp_point_df.sif_year == sif_year, 'iifp_doy'].values[0]

        fig, ax = plt.subplots(1)
        ax.plot(racmo_sif_7d_point_df_yr.index, racmo_sif_7d_point_df_yr.ci, lw=0, marker='x', label='SIC 7-day running mean')
        ax.axhline(0.15, color='orange', linestyle='--', label='15% SIC threshold')
        ax.axhline(0.8, color='orange', linestyle='--', label='80% SIC threshold')
        ax.axvline(doo_point_year_time, color='blue', linestyle='--', label='DOO')
        ax.axvline(dor_point_year_time, color='red', linestyle='--', label='DOR')
        ax.axvline(doa_point_year_time, color='green', linestyle='--', label='DOA')
        ax.axvline(doc_point_year_time, color='purple', linestyle='--', label='DOC')

        print(f"DOO: {doo_point_year_time} (DOY: {doo_point_year_doy})")
        print(f"DOR: {dor_point_year_time} (DOY: {dor_point_year_doy})")
        print(f"DOA: {doa_point_year_time} (DOY: {doa_point_year_doy})")
        print(f"DOC: {doc_point_year_time} (DOY: {doc_point_year_doy})")
        print(f"OIFP: {oifp_point_year_doy} days")
        print(f"IIFP: {iifp_point_year_doy} days")


        # %%
        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        doo = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doo.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doa = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doa.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        dor = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_dor.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doc = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doc.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        oifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_oifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        iifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_iifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

        # # Mask doc if doo is not available
        # mask = doo["doo_doy"].isnull() & doo["doo_time"].isnull()
        # doc["doc_doy"] = doc["doc_doy"].where(~mask, np.nan)
        # doc["doc_time"] = doc["doc_time"].where(~mask, np.datetime64("NaT"))

        # # Mask doo if doc is not available
        # mask = doc["doc_doy"].isnull() & doc["doc_time"].isnull()
        # doo["doo_doy"] = doo["doo_doy"].where(~mask, np.nan)
        # doo["doo_time"] = doo["doo_time"].where(~mask, np.datetime64("NaT"))

        # # Mask dor, doa, oifp, iifp if doo is not available
        # mask = doo["doo_doy"].isnull() & doo["doo_time"].isnull()
        # dor["dor_doy"] = dor["dor_doy"].where(~mask, np.nan)
        # dor["dor_time"] = dor["dor_time"].where(~mask, np.datetime64("NaT"))

        # doa["doa_doy"] = doa["doa_doy"].where(~mask, np.nan)
        # doa["doa_time"] = doa["doa_time"].where(~mask, np.datetime64("NaT"))

        # oifp["oifp_doy"] = oifp["oifp_doy"].where(~mask, np.nan)
        # iifp["iifp_doy"] = iifp["iifp_doy"].where(~mask, np.nan)

        # rlon = racmo_sif_7d.rlon.values[80]; rlat = racmo_sif_7d.rlat.values[200] # TODO for the paper
        rlon = racmo_sif_7d.rlon.values[95]; rlat = racmo_sif_7d.rlat.values[200] # TODO for the paper
        fig, ax = plt.subplots(1)
        time_id = 15
        doc['doc_doy'][time_id, :, :].plot(ax=ax)
        ax.plot(rlon, rlat, 'ro')

        doo_sel = doo.sel(rlon=[rlon], rlat=[rlat])
        doo_time = doo_sel['doo_time']
        doo_doy = doo_sel['doo_doy']
        print(f"DOO DOY: {doo_doy[time_id, :, :].values}")
        print(f"DOO Time: {doo_time[time_id, :, :].values}")

        doc_sel = doc.sel(rlon=[rlon], rlat=[rlat])
        doc_time = doc_sel['doc_time']
        doc_doy = doc_sel['doc_doy']
        print(f"DOC DOY: {doc_doy[time_id, :, :].values}")
        print(f"DOC Time: {doc_time[time_id, :, :].values}")

        oifp_sel = oifp.sel(rlon=[rlon], rlat=[rlat])
        oifp_doy = oifp_sel['oifp_doy']
        print(f"OIFP DOY: {oifp_doy[time_id, :, :].values}")

        dor_sel = dor.sel(rlon=[rlon], rlat=[rlat])
        dor_time = dor_sel['dor_time']
        dor_doy = dor_sel['dor_doy']
        print(f"DOR DOY: {dor_doy[time_id, :, :].values}")
        print(f"DOR Time: {dor_time[time_id, :, :].values}")

        doa_sel = doa.sel(rlon=[rlon], rlat=[rlat])
        doa_time = doa_sel['doa_time']
        doa_doy = doa_sel['doa_doy']
        print(f"DOA DOY: {doa_doy[time_id, :, :].values}")
        print(f"DOA Time: {doa_time[time_id, :, :].values}")

        iifp_sel = iifp.sel(rlon=[rlon], rlat=[rlat])
        iifp_doy = iifp_sel['iifp_doy']
        print(f"IIFP DOY: {iifp_doy[time_id, :, :].values}")

        # %%
    def sanity_check_calculate_greenland_sea_sif_phenology_one_point():
        # %%
        # Read sea ice raw data to check the results
        input_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
        racmo_sif = xr.open_dataset(f'{input_file_path}/ci_all.nc')['ci']
        racmo_sif = racmo_sif.squeeze('height', drop=True)
        # Drop coordinates from the DataArray
        coords_to_drop = ['height', 'lat', 'lon', 'pressure']
        racmo_sif = racmo_sif.drop_vars([c for c in coords_to_drop if c in racmo_sif.coords])

        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        greenland_sea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")

        # Apply mask
        racmo_sif = racmo_sif.where(greenland_sea_mask != 0)
        racmo_sif = racmo_sif.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
        # racmo_sif = racmo_sif.sel(rlon=slice(4, 5), rlat=slice(2, 4))

        # Point mask of closest Freya glacier mask racmo at 5 km
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        rlon, rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # %%
        # ------------------------- Select a grid point and a year -------------------------
        rlon_i = 65; rlat_i = 210
        # rlon_i = 107; rlat_i = 332; time_i = 30
        sif_year_point = 2020

        rlon = racmo_sif.rlon.values[rlon_i]
        rlat = racmo_sif.rlat.values[rlat_i]
        # time = racmo_sif.time.values[time_i]

        # sif_year_point = 1990; rlat = 8.9; rlon = 7.05
        rlon = racmo_sif.rlon.values[130];rlat = racmo_sif.rlat.values[200]
        _, ax = plt.subplots()
        racmo_sif[0, :, :].plot(ax=ax)
        ax.plot(rlon, rlat, 'x')

        racmo_sif_point = racmo_sif.sel(rlon=[rlon], rlat=[rlat])
        racmo_sif_point_df = racmo_sif_point.to_dataframe().reset_index()
        # racmo_sif_point = racmo_sif.sel(rlon=[racmo_sif.rlon.values[61]], rlat=[racmo_sif.rlat.values[200]])
        # racmo_sif_point = racmo_sif.sel(rlon=[racmo_sif.rlon.values[rlon_i]], rlat=[racmo_sif.rlat.values[rlat_i]])
        racmo_sif_point = racmo_sif.sel(rlon=[rlon], rlat=[rlat])
        racmo_sif_point_df['time'] = pd.to_datetime(racmo_sif_point_df['time'])
        # racmo_sif_point_df['doy'] = racmo_sif_point_df['time'].dt.dayofyear
        racmo_sif_point_df.index = racmo_sif_point_df['time']
        # Assign sea ice year
        # Month of the following year (January and February) are assigned to the current year, as they belong to the same sea ice season
        racmo_sif_point_df['sif_year'] = (racmo_sif_point_df.index.year - (racmo_sif_point_df.index.month <= 2))
        # 7-day running mean; This uses the current day and previous 6 days
        racmo_sif_point_df['ci_7d'] = racmo_sif_point_df['ci'].rolling(window=7).mean()
        # 1990 – 2020 sea ice year is 1991 – 2020, as January and February belong to the previous year
        racmo_sif_point_df = racmo_sif_point_df.loc[racmo_sif_point_df.sif_year.isin(range(1990, 2021))]

        # sif_year_point = 1990
        sif_yr = pd.DataFrame(racmo_sif_point_df.loc[racmo_sif_point_df['sif_year'] == sif_year_point])

        fig, ax = plt.subplots()
        ax.plot(sif_yr.index, sif_yr.ci_7d, color='blue', label='7-day running mean')
        ax.plot(sif_yr.index, sif_yr.ci, color='red', label='Daily SIC')
        ax.legend()
        plt.show()

        # # If sea ice concentration is less than 1% make it open water i.e. 0% sea ice concentration, to avoid noise in the following calculations
        # sea_ice_threshold = sif_yr.ci < 0.01
        # sif_yr.loc[sea_ice_threshold, 'ci'] = 0
        # # Minimum sea ice concentration (SIC) during the year
        # sif_yr_min = sif_yr.ci.min()
        sif_yr_min = sif_yr.ci_7d.min()

        # DOO Day of opening, last day sea ice concentration (SIC) drops below 80% before the first summer minimum
        # first_summer_min_date = sif_yr.loc[sif_yr.ci_7d == sif_yr_min, 'doy'].iloc[0]
        first_summer_min_date = sif_yr.loc[sif_yr.ci_7d == sif_yr_min].index[0]
        sif_yr_before_first_summer_min_date = pd.DataFrame(sif_yr.loc[sif_yr.index < first_summer_min_date])

        # Days with more than 80% SIC before the first summer minimum
        sif_yr_before_first_summer_min_date_80 = pd.DataFrame(sif_yr_before_first_summer_min_date[sif_yr_before_first_summer_min_date['ci_7d'] > 0.8])
        # The day of opening; last day with below 80% sea ice
        sif_yr_doo = sif_yr_before_first_summer_min_date_80.index[-1] + pd.Timedelta(days=1)
        sif_yr_doo_doy = sif_yr_doo.dayofyear

        # DOR Day of retreat, last day SIC drops below 15% before the first summer minimum
        sif_yr_before_first_summer_min_date_15 = pd.DataFrame(sif_yr_before_first_summer_min_date[sif_yr_before_first_summer_min_date['ci_7d'] < 0.15])
        if not sif_yr_before_first_summer_min_date_15.empty:
            # sif_yr_dor = sif_yr_before_first_summer_min_date_15.iloc[-1]['doy'] + 1
            sif_yr_dor = sif_yr_before_first_summer_min_date_15.index[-1]
        else:
            # If sea ice does not drop below 15% before the first summer minimum, set DOR to the day of the first summer minimum
            # sif_yr_dor = first_summer_min_date + 1
            sif_yr_dor = first_summer_min_date
        sif_yr_dor_doy = sif_yr_dor.dayofyear

        # DOA Day of advance, first day SIC increases above 15% following the final summer minimum
        final_summer_min_date = sif_yr.loc[sif_yr.ci_7d == sif_yr_min].index[-1]
        sif_yr_after_final_summer_min_date = pd.DataFrame(sif_yr.loc[sif_yr.index > final_summer_min_date])
        # Days with more than 15% SIC after the final summer minimum
        sif_yr_after_final_summer_min_date_15 = pd.DataFrame(sif_yr_after_final_summer_min_date[sif_yr_after_final_summer_min_date['ci_7d'] > 0.15])
        # The day of advance; first day with more than 15% sea ice
        # if first_summer_min_date == final_summer_min_date:
        #     sif_yr_doa = final_summer_min_date
        # else:
        #     sif_yr_doa = sif_yr_after_final_summer_min_date_15.index[0]
        # sif_yr_doa_doy = sif_yr_doa.dayofyear

        delta_days = (sif_yr_after_final_summer_min_date_15.index - final_summer_min_date).days
        if (delta_days[0] > 1) | ((delta_days[0] == 1) & (sif_yr_min < 0.15)):
            sif_yr_doa = sif_yr_after_final_summer_min_date_15.index[0]
        else:
            sif_yr_doa = final_summer_min_date
        sif_yr_doa_doy = sif_yr_doa.dayofyear

        # DOC Day of closing, first day SIC increases above 80% following the final summer minimum
        sif_yr_after_final_summer_min_date_80 = pd.DataFrame(sif_yr_after_final_summer_min_date[sif_yr_after_final_summer_min_date['ci_7d'] > 0.8])
        # The day of closing
        sif_yr_doc = sif_yr_after_final_summer_min_date_80.index[0]
        sif_yr_doc_doy = sif_yr_doc.dayofyear

        # IIFP
        sif_yr_iifp_doy = sif_yr_doa_doy - sif_yr_dor_doy

        # # Plot all the duration
        # fig, ax = plt.subplots(1)
        # ax.plot(sif_yr.index.dayofyear, sif_yr['ci_7d'], marker='x', ms=3, lw=0.1)
        # ax.axhline(y=0.8, color='r', lw=0.5, ls='--', zorder=1)
        # ax.axhline(y=0.15, color='r', lw=0.5, ls='--', zorder=1)
        # ax.axvline(x=sif_yr_doo_doy, color='r', lw=0.5, ls='--', zorder=1)
        # ax.axvline(x=sif_yr_dor_doy, color='r', lw=0.5, ls='--', zorder=1)
        # ax.axvline(x=sif_yr_doa_doy, color='r', lw=0.5, ls='--', zorder=1)
        # ax.axvline(x=sif_yr_doc_doy, color='r', lw=0.5, ls='--', zorder=1)

        # ------------ Comparison ---------------------
        # Compare with the dataset created by the calculate_greenland_sea_sif_phenology function ---------------------
        # Read the sea ice phenology dataset created by the calculate_greenland_sea_sif_phenology function
        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        doo = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doo.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doa = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doa.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        dor = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_dor.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        oifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_oifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        iifp = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_iifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doc = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_doc.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

        # Extract phenology DataArray for the same point and year
        # doo = phenology_ds['doo']
        doo_point = doo.sel(rlon=rlon, rlat=rlat)
        print(f"Day of opening (DOO) for year {sif_year_point}: {sif_yr_doo_doy} – from point mask: {doo_point.sel(sif_year=[sif_year_point]).values}")

        # dor = phenology_ds['dor']
        dor_point = dor.sel(rlon=rlon, rlat=rlat)
        print(f"Day of retreat (DOR) for year {sif_year_point}: {sif_yr_dor_doy} – from point mask: {dor_point.sel(sif_year=[sif_year_point]).values}")

        # doa = phenology_ds['doa']
        doa_point = doa.sel(rlon=rlon, rlat=rlat)
        print(f"Day of advance (DOA) for year {sif_year_point}: {sif_yr_doa_doy} – from point mask: {doa_point.sel(sif_year=[sif_year_point]).values}")

        # doc = phenology_ds['doc']
        doc_point = doc.sel(rlon=rlon, rlat=rlat)
        print(f"Day of closing (DOC) for year {sif_year_point}: {sif_yr_doc_doy} – from point mask: {doc_point.sel(sif_year=[sif_year_point]).values}")

        # iifp = phenology_ds['doc']
        iifp_point = iifp.sel(rlon=rlon, rlat=rlat)
        print(f"IIFP for year {sif_year_point}: {sif_yr_iifp_doy} – from point mask: {iifp_point.sel(sif_year=[sif_year_point]).values}")

         # Plot all the duration
        fig, ax = plt.subplots(1)
        ax.plot(sif_yr.index.dayofyear, sif_yr['ci_7d'], marker='x', ms=3, lw=0.1)
        ax.axhline(y=0.8, color='r', lw=0.5, ls='--', zorder=1)
        ax.axhline(y=0.15, color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=sif_yr_doo_doy, color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=doo_point.sel(sif_year=[sif_year_point]).values, color='b', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=sif_yr_dor_doy, color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=dor_point.sel(sif_year=[sif_year_point]).values, color='b', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=sif_yr_doa_doy, color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=doa_point.sel(sif_year=[sif_year_point]).values, color='b', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=sif_yr_doc_doy, color='r', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=doc_point.sel(sif_year=[sif_year_point]).values, color='b', lw=0.5, ls='--', zorder=1)

        # # years, lats, lons = np.where(iifp == 1)
        # # years, lats, lons = np.where(iifp <= 0)
        # years, lats, lons = np.where(iifp < 0)
        # print("Year indices:", years)
        # print("Lat indices:", lats)
        # print("Lon indices:", lons)

        # year_vals = iifp.sif_year[years].values
        # lat_vals  = iifp.rlat[lats].values
        # lon_vals  = iifp.rlon[lons].values

        # for y, la, lo in zip(year_vals, lat_vals, lon_vals):
        #     print(f"sif_year: {y}, rlat: {la}, rlon: {lo}")
        # %%
    def plot_greenland_sea_sif_phenology():
        # %%
        # Point mask of closest Freya glacier mask racmo at 5 km
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        rlon, rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # Get longitude and latitude plotting parameters
        lonlat = _get_latlon_meshgrid_east_greenland()
        bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat = _get_lonlat_params_east_greenland()
        xlims, xticks, xticklabels, ylims, yticks, yticklabels = _get_lonlat_ticks_east_greenland(bounds_lon=bounds_lon, bounds_lat=bounds_lat)
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude_east_greenland()

        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        gsea_mask = xr.where(gsea_mask > 0.5, 1, 0)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        colors = [(68, 1, 84), (71, 44, 122), (59, 81, 139), (44, 113, 142), (33, 144, 141), (39, 173, 129), (92, 200, 99), (170, 220, 50), (253, 231, 37)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]
        cbticklabel = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300]

        fig = plt.figure(figsize=(4.35, 4), dpi=600)
        # OUTER GridSpec: controls LARGE row gap (applies to all columns)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.12], hspace=0)   # large spacing between row groups
        gs_top = outer[0].subgridspec(2, 3, hspace=0.1, wspace=0.2)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        phenologies = [('doo', 0, 0, 'a'), ('doc', 0, 1, 'b'), ('oifp', 0, 2, 'c'), ('dor', 1, 0, 'd'), ('doa', 1, 1, 'e'), ('iifp', 1, 2, 'f')]
        for phenology, row, col, label in phenologies:
            # Read Sea ice phenology
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            ds = ds[f'{phenology}_doy']
            ds = ds.sel(sif_year=slice(1990, 2019))
            ds_mean = ds.mean(dim='sif_year')
            # ds_mean = ds_mean.where(ds_mean < 300, np.nan)

            ax = fig.add_subplot(gs_top[row, col])
            ax.contourf(lonlat[0], lonlat[1], ds_mean, cmap=cmap, norm=norm, levels=bounds, extend='max')
            ax.contourf(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, zorder=5)
            ax.contour(lonlat[0],lonlat[1], racmo_longitude, cmap=cmap_lon, norm=norm_lon, levels=bounds_lon, linewidths=0.2, linestyles='dashed', zorder=10)
            ax.contour(lonlat[0],lonlat[1], racmo_latitude, cmap=cmap_lat, norm=norm_lat, levels=bounds_lat, linewidths=0.2, linestyles='dashed', zorder=10)
            ax.plot(rlon, rlat, 'o', color='red', ms=2, label='Closest point to Freya glacier', zorder=12)
            cs = ax.contour(lonlat[0], lonlat[1], gsea_mask, colors='gray', linewidths=0.2, linestyles='dashed', zorder=1)
            cs.set_dashes([(0, (10, 10))])

            ax.text(0.02, 0.93, rf"$\mathbf{{({label}) {phenology.upper()}}}$", color='k', fontsize=9, zorder=12, transform=ax.transAxes, path_effects=[pe.withStroke(linewidth=0.75, foreground='w')])
            # [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'bottom', 'left']]
            # ax.axis('off')

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.set_xticklabels(xticklabels, color='k')
            ax.set_yticklabels(yticklabels, color='k')

            tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=9, width=0.5)
            ax.tick_params(**tick_params)
            [i.set_linewidth(0.3) for i in ax.spines.values()]

            ax.set_aspect('equal', adjustable='box')

        # Colorbar for phase plots
        ax = fig.add_subplot(gs_bot[0, 0])
        ax.axis('off')

        cbar_ax = ax.inset_axes([0.16, 0.01, 0.6, 0.225]) #[left, bottom, width, height]
        # cbar_ax_2 = fig.add_axes([0.95, 0.1, 0.015, 0.6]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm, extend='max')
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=9, pad=0.5, length=2, direction='in', width=0.5)
        cb.outline.set_linewidth(0.4)
        # cb.set_label(rf"# d$^{{\mathdefault{{-1}}}}$", size=9, labelpad=-2)
        cb.set_label('DOY, Days', size=9, labelpad=-1)
        # cb.ax.text(1.04, -0.001, '>300', transform=cb.ax.transAxes, ha='center', va='top', fontsize=9)

        # cb.ax.set_title(rf"# d$^{{\mathdefault{{-1}}}}$", fontsize=9, pad=2, loc='right')
        # cb.ax.yaxis.set_label_position('top')
        # cb.ax.yaxis.tick_left()

        targets = dict(left=157.5, right=25, top=15, bottom=135, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_greenland_sea_sif_phenology'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_sif_phenology.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
# %%
    def plot_greenland_sea_sif_phenology_one_point():
        # %%
        # Read Sea ice phenology
        file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        doo = xr.open_dataset(file_path / "greenland_sea_sif_phenology_doo.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doa = xr.open_dataset(file_path / "greenland_sea_sif_phenology_doa.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        dor = xr.open_dataset(file_path / "greenland_sea_sif_phenology_dor.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        oifp = xr.open_dataset(file_path / "greenland_sea_sif_phenology_oifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        iifp = xr.open_dataset(file_path / "greenland_sea_sif_phenology_iifp.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        doc = xr.open_dataset(file_path / "greenland_sea_sif_phenology_doc.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

        input_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        # Read original Sea ice data
        racmo_sif = xr.open_dataset(input_file_path / 'ci_all_masked_greenland_sea.nc')['ci'] # Try without chunking
        # racmo_sif = racmo_sif.squeeze('height', drop=True)
        # Read 7-day running mean Sea ice
        racmo_sif_7d = xr.open_dataset(input_file_path / 'ci_all_masked_greenland_sea_7d.nc')['ci']

        # Choose a grid point and a year
        sif_year = 2018 # TODO for the paper
        rlon_index = 80; rlat_index = 200 # TODO for the paper
        rlon = racmo_sif_7d.rlon.values[rlon_index]; rlat = racmo_sif_7d.rlat.values[rlat_index] # TODO for the paper

        racmo_sif_7d_point = racmo_sif_7d.sel(rlon=[rlon], rlat=[rlat])
        racmo_sif_7d_point_df = racmo_sif_7d_point.to_dataframe().reset_index()
        racmo_sif_7d_point_df.index = racmo_sif_7d_point_df['time']
        racmo_sif_7d_point_df_yr = racmo_sif_7d_point_df.loc[racmo_sif_7d_point_df.sif_year.isin([sif_year])]
        racmo_sif_7d_point_df_yr.loc[(racmo_sif_7d_point_df_yr.index.month == 3) & (racmo_sif_7d_point_df_yr['ci'] < 0.8), 'ci'] = np.nan

        doo_point = doo.sel(rlon=[rlon], rlat=[rlat])
        doo_point_df = doo_point.to_dataframe().reset_index()
        doo_point_df.index = doo_point_df['sif_year']
        # doo_point_df['doo_doy_convert'] = doo_point_df['doo_time'].dt.day_of_year
        doo_point_year_time = doo_point_df.loc[doo_point_df.sif_year == sif_year, 'doo_time'].values[0]
        doo_point_year_doy = doo_point_df.loc[doo_point_df.sif_year == sif_year, 'doo_doy'].values[0]

        dor_point = dor.sel(rlon=[rlon], rlat=[rlat])
        dor_point_df = dor_point.to_dataframe().reset_index()
        dor_point_df.index = dor_point_df['sif_year']
        dor_point_year_time = dor_point_df.loc[dor_point_df.sif_year == sif_year, 'dor_time'].values[0]
        dor_point_year_doy = dor_point_df.loc[dor_point_df.sif_year == sif_year, 'dor_doy'].values[0]

        doa_point = doa.sel(rlon=[rlon], rlat=[rlat])
        doa_point_df = doa_point.to_dataframe().reset_index()
        doa_point_df.index = doa_point_df['sif_year']
        doa_point_year_time = doa_point_df.loc[doa_point_df.sif_year == sif_year, 'doa_time'].values[0]
        doa_point_year_doy = doa_point_df.loc[doa_point_df.sif_year == sif_year, 'doa_doy'].values[0]

        doc_point = doc.sel(rlon=[rlon], rlat=[rlat])
        doc_point_df = doc_point.to_dataframe().reset_index()
        doc_point_df.index = doc_point_df['sif_year']
        doc_point_year_time = doc_point_df.loc[doc_point_df.sif_year == sif_year, 'doc_time'].values[0]
        doc_point_year_doy = doc_point_df.loc[doc_point_df.sif_year == sif_year, 'doc_doy'].values[0]

        oifp_point = oifp.sel(rlon=[rlon], rlat=[rlat])
        oifp_point_df = oifp_point.to_dataframe().reset_index()
        oifp_point_df.index = oifp_point_df['sif_year']
        oifp_point_year_doy = oifp_point_df.loc[oifp_point_df.sif_year == sif_year, 'oifp_doy'].values[0]

        iifp_point = iifp.sel(rlon=[rlon], rlat=[rlat])
        iifp_point_df = iifp_point.to_dataframe().reset_index()
        iifp_point_df.index = iifp_point_df['sif_year']
        iifp_point_year_doy = iifp_point_df.loc[iifp_point_df.sif_year == sif_year, 'iifp_doy'].values[0]

        fig, ax = plt.subplots(figsize=(5, 3), dpi=600)
        ax.plot(racmo_sif_7d_point_df_yr.index, racmo_sif_7d_point_df_yr.ci, lw=1, color='r', label='SIC 7-day running mean')
        ax.axhline(0.15, color='gray', linestyle=':', lw=0.6, label='15% SIC threshold')
        ax.axhline(0.8, color='gray', linestyle=':', lw=0.6, label='80% SIC threshold')
        # ax.axvline(doo_point_year_time, color='k', linestyle='--', lw=0.6, label='DOO')
        ax.vlines(doo_point_year_time, 0, 0.9, color='k', linestyle='--', lw=0.6, label='DOO')
        ax.vlines(dor_point_year_time, 0, 0.25, color='k', linestyle='--', lw=0.6, label='DOR')
        ax.vlines(doa_point_year_time, 0, 0.25, color='k', linestyle='--', lw=0.6, label='DOA')
        ax.vlines(doc_point_year_time, 0, 0.9, color='k', linestyle='--', lw=0.6, label='DOC')
        # ax.axvline(doc_point_year_time, color='k', linestyle='--', lw=0.6, label='DOC')
        ax.plot(doo_point_year_time, 0.8, marker='o', color='r', markersize=4)
        ax.plot(dor_point_year_time, 0.15, marker='o', color='r', markersize=4)
        ax.plot(doa_point_year_time, 0.15, marker='o', color='r', markersize=4)
        ax.plot(doc_point_year_time, 0.8, marker='o', color='r', markersize=4)

        # Add a double-headed arrow to indicate the duration of the open water season (IIFP)
        ax.annotate('', xy=(dor_point_year_time, 0.15), xytext=(dor_point_year_time + np.timedelta64(27, 'D'), 0.15), arrowprops=dict(arrowstyle='-|>', lw=0.5, ls='-', color='k', mutation_scale=15), clip_on=False)
        ax.annotate('', xy=(doa_point_year_time, 0.15), xytext=(doa_point_year_time - np.timedelta64(27, 'D'), 0.15), arrowprops=dict(arrowstyle='-|>', lw=0.5, ls='-', color='k', mutation_scale=15), clip_on=False)
        ax.text(dor_point_year_time + np.timedelta64(37, 'D'), 0.2, f'IIFP\n({int(iifp_point_year_doy)})', color='k', fontsize=9, ha='center', va='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

        # Add a double-headed arrow to indicate the duration of the open water season (OIFP)
        ax.annotate('', xy=(doo_point_year_time, 0.8), xytext=(doo_point_year_time + np.timedelta64(99, 'D'), 0.8), arrowprops=dict(arrowstyle='-|>', lw=0.5, ls='-', color='k', mutation_scale=15), clip_on=False)
        ax.annotate('', xy=(doc_point_year_time, 0.8), xytext=(doc_point_year_time - np.timedelta64(90, 'D'), 0.8), arrowprops=dict(arrowstyle='-|>', lw=0.5, ls='-', color='k', mutation_scale=15), clip_on=False)
        ax.text(dor_point_year_time + np.timedelta64(37, 'D'), 0.85, f'OIFP\n({int(oifp_point_year_doy)})', color='k', fontsize=9, ha='center', va='top', path_effects=[pe.withStroke(linewidth=2, foreground='w')])

        ax.set_xlim([pd.Timestamp(f"{sif_year}-03-01"), pd.Timestamp(f"{sif_year + 1}-02-28")])
        ax.set_xticks([pd.Timestamp(f"{sif_year}-03-01"), doo_point_year_time, dor_point_year_time, doa_point_year_time, doc_point_year_time, pd.Timestamp(f"{sif_year + 1}-02-28")])
        # ax.set_xticklabels([pd.Timestamp(f"{sif_year}-03-01").strftime('%m/%d') + '\n' + str(pd.Timestamp(f"{sif_year}-03-01").dayofyear), pd.Timestamp(doo_point_year_time).strftime('%m/%d') + '\n' + str(int(doo_point_year_doy)), pd.Timestamp(dor_point_year_time).strftime('%m/%d') + '\n' + str(int(dor_point_year_doy)), pd.Timestamp(doa_point_year_time).strftime('%m/%d') + '\n' + str(int(doa_point_year_doy)), pd.Timestamp(doc_point_year_time).strftime('%m/%d') + '\n' + str(int(doc_point_year_doy)), pd.Timestamp(f"{sif_year + 1}-02-28").strftime('%m/%d') + '\n' + str(pd.Timestamp(f"{sif_year + 1}-02-28").dayofyear)])
        ax.set_xticklabels([pd.Timestamp(f"{sif_year}-03-01").strftime('%m/%d'), pd.Timestamp(doo_point_year_time).strftime('%m/%d'), pd.Timestamp(dor_point_year_time).strftime('%m/%d'), pd.Timestamp(doa_point_year_time).strftime('%m/%d'), pd.Timestamp(doc_point_year_time).strftime('%m/%d'), pd.Timestamp(f"{sif_year + 1}-02-28").strftime('%m/%d')])

        ax.set_ylim([-0.01, 1])
        ax.set_yticks([0, 0.15, 0.5, 0.8, 1.0])
        ax.set_yticklabels([0, 15, 50, 80, 100])
        ax.set_ylabel('SIC (%)', color='k', fontsize=9)
        ax.yaxis.set_label_coords(-0.07, 0.5)

        # Customize spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(0, 1)
        ax.spines['left'].set_position(('outward', 3))
        ax.spines['bottom'].set_position(('outward', 20))
        [i.set_linewidth(0.4) for i in ax.spines.values()]

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=9)
        ax.tick_params(axis='both', which='major', **ytick_params)

        y_pos = -0.02
        ax.text(pd.Timestamp(doo_point_year_time), y_pos, f'DOO\n({int(doo_point_year_doy)})', color='k', fontsize=9, ha='center', va='top')
        ax.text(pd.Timestamp(dor_point_year_time), y_pos, f'DOR\n({int(dor_point_year_doy)})', color='k', fontsize=9, ha='center', va='top')
        ax.text(pd.Timestamp(doa_point_year_time), y_pos, f'DOA\n({int(doa_point_year_doy)})', color='k', fontsize=9, ha='center', va='top')
        ax.text(pd.Timestamp(doc_point_year_time), y_pos, f'DOC\n({int(doc_point_year_doy)})', color='k', fontsize=9, ha='center', va='top')

        # Resizeable plot with fixed aspect ratio
        # targets = dict(left=250, right=270, top=30, bottom=300, hspace=0, wspace=0)
        targets = dict(left=260, right=92, top=30, bottom=300, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Plot Greenland map
        ax_map = fig.add_axes([0.625, 0.35, 0.45, 0.33])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        [i.set_visible(False) for i in ax_map.spines.values()]
        ax_map.axis('off')
        variable = 'sif'
        lonlat = _get_latlon_meshgrid(variable=variable)
        colors_buffer = [(255, 255, 255), (169, 169, 169)]

        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)

        gr_masks_5km = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
        land_sea_mask = gr_masks_5km['LSM_GR']
        land_sea_mask = land_sea_mask.where(land_sea_mask == 1)  # Set land to 1 and sea to 0
        ax_map.contourf(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_buffer, norm=norm_buffer, zorder=20)

        # Add Iceland and Svalbard
        iceland_svalbard_mask = xr.open_dataarray('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/iceland_svalbard_mask.nc')
        ax_map.contourf(lonlat[0], lonlat[1], iceland_svalbard_mask, cmap=cmap_buffer, norm=norm_buffer, zorder=21)

        # Plot the selected point on the map
        # colors = [(255, 255, 255), (173, 216, 230), (135, 206, 235), (70, 130, 180), (65, 105, 225), (0, 0, 255), (0, 0, 139), (0, 0, 128), (25, 25, 112), (0, 0, 0)]
        colors = [(255, 255, 255), (173, 216, 230), (135, 206, 235), (70, 130, 180), (65, 105, 225), (0, 0, 255), (0, 0, 139), (0, 0, 128)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # cbticklabel = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        # cbticklabel = [0, '', 0.2, '', 0.4, '', 0.6, '', 0.8, '', 1]
        cbticklabel = [0, '', '20', '', '40', '', '60', '', '80', '', 100]
        # 1990-05-01
        racmo_sif_day = racmo_sif.sel(time='2018-5-1')
        ax_map.contourf(lonlat[0], lonlat[1], racmo_sif_day, cmap=cmap, norm=norm, extend='neither', levels=bounds)
        ax_map.plot(rlon, rlat, 'x', color='r', ms=4, label='Selected Point')

        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        binary_mask = xr.where(gsea_mask > 0.5, 1, 0)
        cs = ax_map.contour(lonlat[0], lonlat[1], binary_mask, colors='k', linewidths=0.2, linestyles='dashed', zorder=1)
        cs.set_dashes([(0, (10, 10))])

        [i.set_visible(False) for i in ax_map.spines.values()]  # Set visible spines to have a linewidth of 0.3
        ax_map.set_xlim([-7.5, 12.5])
        # ax.set_ylim([-13, 13])
        ax_map.set_ylim([-12, 11.6])
        ax_map.axis('off')
        ax_map.set_aspect('equal', adjustable='box')

        # cbar_ax = fig.add_axes([0.63, 0.05, 0.015, 0.25]) #[left, bottom, width, height]
        cbar_ax = fig.add_axes([0.935, 0.4, 0.015, 0.25]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='vertical', cmap=cmap, norm=norm)
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=8, pad=0.9, length=2, direction='in', width=0.5)
        cb.outline.set_linewidth(0.4)

        ax_map.text(1.14, 1.1, 'SIC (%)', color='k', fontsize=8, ha='center', va='top', transform=ax_map.transAxes, path_effects=[pe.withStroke(linewidth=2, foreground='w')])

        # cb.set_label('SIC (%)', size=9, labelpad=0.1)
        # cb.ax.yaxis.set_label_position('top')
        # cb.ax.yaxis.tick_left()

        logger.info(f"Saving plot of 'plot_greenland_sea_sif_phenology_one_point'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_sif_phenology_one_point.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
        lon_mask = gr_masks_5km['lon']
        lat_mask = gr_masks_5km['lat']
        lat_mask_point = lat_mask.sel(rlon=[rlon], rlat=[rlat]).values
        print(lat_mask_point)
        lon_mask_point = lon_mask.sel(rlon=[rlon], rlat=[rlat]).values
        print(lon_mask_point)

        # %%
    def calculate_greenland_sea_sif_phenology_trend():
        # %%
        file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            if phenology in ['doc']:
                # Adjust for leap years in the phenology day-of-year data
                # Add 365/366 if the day-of-year is before the threshold in Ice year
                # ‘ice year’ from March 1 to February 28 of the following year
                # 60/61 DOY is March to next year from Jan and Feb
                is_leap = ds[f"{phenology}_time"].dt.is_leap_year
                threshold = xr.where(is_leap, 61, 60)
                days_in_year = xr.where(is_leap, 366, 365)
                ds[f"{phenology}_doy"] = xr.where(ds[f"{phenology}_doy"] < threshold,
                                                ds[f"{phenology}_doy"] + days_in_year,
                                                ds[f"{phenology}_doy"])

            da = ds[f"{phenology}_doy"]
            da = da.sel(sif_year=slice(1990, 2019)) # Ensure the same years are used for trend analysis
            # Convert years to numeric (independent variable)
            years = da['sif_year'].values.astype(float)

            # Apply pixel-wise
            slope, pval = xr.apply_ufunc(_theil_sen_and_pvalue, da, years,
                                         input_core_dims=[['sif_year'], ['sif_year']],
                                         output_core_dims=[[], []],
                                         vectorize=True, dask='parallelized', output_dtypes=[float, float])

            # Name outputs
            slope = slope.rename(f"{phenology}_theilsen_slope")
            pval = pval.rename(f"{phenology}_trend_pvalue")

            # Add metadata
            slope.attrs.update({'long_name': f"Theil-Sen trend slope of {phenology.upper()}", 'units': 'days per year', 'method': 'Theil-Sen estimator'})
            pval.attrs.update({'long_name': f"Kendall tau trend p-value of {phenology.upper()}", 'method': 'Kendall rank correlation test'})

            # Combine into dataset
            trend_ds = xr.Dataset({f"{phenology}_theilsen_slope": slope, f"{phenology}_trend_pvalue": pval})

            # Save to NetCDF
            print(f"Saving trend results for {phenology} to NetCDF...")
            trend_ds.to_netcdf(f"{file_path}/greenland_sea_sif_phenology_{phenology}_trend_theilsen.nc")

        # --------- Uncomment for debugging purposes ---------------
        # phenology='doc'
        # ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
        # rlon = ds.rlon.values[90]; rlat = ds.rlat.values[200] # TODO for the paper
        # fig, ax = plt.subplots(1)
        # time_id = 25
        # ds[f"{phenology}_doy"][time_id, :, :].plot(ax=ax)
        # ax.plot(rlon, rlat, 'ro')
        # ds_sel = ds.sel(rlon=[rlon], rlat=[rlat])

        # rlon = ds.rlon.values[90]; rlat = ds.rlat.values[200] # TODO for the paper
        # ds_doy = ds[f"{phenology}_doy"].sel(rlon=[rlon], rlat=[rlat])
        # ds_time = ds[f"{phenology}_time"].sel(rlon=[rlon], rlat=[rlat])
        # print(f"DOC DOY: {ds_doy[time_id, :, :].values}")
        # print(f"DOC Time: {ds_time[time_id, :, :].values}")

        # --------- Uncomment for debugging purposes ---------------
        # %%

    def plot_greenland_sea_sif_phenology_trend():
        # %%
        # Point mask of closest Freya glacier mask racmo at 5 km
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        rlon, rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # Get longitude and latitude plotting parameters
        lonlat = _get_latlon_meshgrid_east_greenland()
        bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat = _get_lonlat_params_east_greenland()
        xlims, xticks, xticklabels, ylims, yticks, yticklabels = _get_lonlat_ticks_east_greenland(bounds_lon=bounds_lon, bounds_lat=bounds_lat)
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude_east_greenland()

        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        gsea_mask = xr.where(gsea_mask > 0.5, 1, 0)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 255, 255), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-6, -5, -4, -3, -2, -1, -0.0001, 0.0001, 1, 2, 3, 4, 5, 6])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        cbticklabel = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]

        fig = plt.figure(figsize=(4.35, 4), dpi=600)
        # OUTER GridSpec: controls LARGE row gap (applies to all columns)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.1], hspace=0)   # large spacing between row groups
        gs_top = outer[0].subgridspec(2, 3, hspace=0.1, wspace=0.2)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        phenologies = [('doo', 0, 0, 'a'), ('doc', 0, 1, 'b'), ('oifp', 0, 2, 'c'), ('dor', 1, 0, 'd'), ('doa', 1, 1, 'e'), ('iifp', 1, 2, 'f')]
        for phenology, row, col, label in phenologies:
            # Read Sea ice phenology trend
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}_trend_theilsen.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            ds[f'{phenology}_theilsen_slope'] = ds[f'{phenology}_theilsen_slope'].where(ds[f'{phenology}_trend_pvalue'] < 0.05, np.nan)

            ax = fig.add_subplot(gs_top[row, col])
            ax.contourf(lonlat[0], lonlat[1], ds[f'{phenology}_theilsen_slope'], cmap=cmap, norm=norm, extend='neither', levels=bounds)

            ax.contourf(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, zorder=5)
            ax.contour(lonlat[0],lonlat[1], racmo_longitude, cmap=cmap_lon, norm=norm_lon, levels=bounds_lon, linewidths=0.2, linestyles='dashed', zorder=10)
            ax.contour(lonlat[0],lonlat[1], racmo_latitude, cmap=cmap_lat, norm=norm_lat, levels=bounds_lat, linewidths=0.2, linestyles='dashed', zorder=10)
            ax.plot(rlon, rlat, 'o', color='red', ms=2, label='Closest point to Freya glacier', zorder=12)
            cs = ax.contour(lonlat[0], lonlat[1], gsea_mask, colors='gray', linewidths=0.2, linestyles='dashed', zorder=1)
            cs.set_dashes([(0, (10, 10))])

            ax.text(0.02, 0.93, rf"$\mathbf{{({label}) {phenology.upper()}}}$", color='k', fontsize=9, zorder=12, transform=ax.transAxes, path_effects=[pe.withStroke(linewidth=0.75, foreground='w')])
            # [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'bottom', 'left']]
            # ax.axis('off')

            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            ax.set_xticks(xticks)
            ax.set_yticks(yticks)

            ax.set_xticklabels(xticklabels, color='k')
            ax.set_yticklabels(yticklabels, color='k')

            tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=9, width=0.5)
            ax.tick_params(**tick_params)
            [i.set_linewidth(0.3) for i in ax.spines.values()]

            ax.set_aspect('equal', adjustable='box')

        # Colorbar for phase plots
        ax = fig.add_subplot(gs_bot[0, 0])
        ax.axis('off')

        cbar_ax = ax.inset_axes([0.16, 0.01, 0.6, 0.225]) #[left, bottom, width, height]
        # cbar_ax_2 = fig.add_axes([0.95, 0.1, 0.015, 0.6]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm, extend='neither')
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=9, pad=0.5, length=2, direction='in', width=0.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label(rf"Days decade$^{{\mathdefault{{-1}}}}$", size=9, labelpad=-1.5)
        # cb.set_label('DOY, Days', size=9, labelpad=-1)
        # cb.ax.text(1.04, -0.001, '>300', transform=cb.ax.transAxes, ha='center', va='top', fontsize=9)

        # cb.ax.set_title(rf"# d$^{{\mathdefault{{-1}}}}$", fontsize=9, pad=2, loc='right')
        # cb.ax.yaxis.set_label_position('top')
        # cb.ax.yaxis.tick_left()

        targets = dict(left=157.5, right=25, top=15, bottom=140, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_greenland_sea_sif_phenology'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_sif_phenology_trend.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def print_greenland_sea_sif_phenology_trend():
        # %%
        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            # Read Sea ice phenology trend
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}_trend_theilsen.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            ds[f'{phenology}_theilsen_slope'] = ds[f'{phenology}_theilsen_slope'].where(ds[f'{phenology}_trend_pvalue'] < 0.05, np.nan)
            ds_mean = ds[f'{phenology}_theilsen_slope'].mean(dim=['rlat', 'rlon'])
            ds_std = ds[f'{phenology}_theilsen_slope'].std(dim=['rlat', 'rlon'])
            print(f"{phenology}: mean={ds_mean.values * 10}, std={ds_std.values * 10}")
            # print(f"{phenology}: mean={ds_mean.values * 10}")
        # %%
    def plot_young_sound_ice_at_doo_dor_doa_doc():
        # %%
        # Read sea ice mask
        # mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/masks')
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        racmo_sif = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        racmo_sif = racmo_sif.sel(radius=100)
        racmo_ys = xr.open_dataarray(f"{mask_file_path}/young_sound_mask.nc")
        # Slice the racmo_sif to the same spatial extent as the phenology dataset to speed up the following calculations
        racmo_sif = racmo_sif.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
        racmo_ys = racmo_ys.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Create an empty DataFrame to store the phenology values for each year
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies_df_ys = pd.DataFrame(columns=phenologies, index=range(1990, 2021))
        phenologies_df = pd.DataFrame(columns=phenologies, index=range(1990, 2021))

        # Read young sound fjord sea ice combined dataset to compare the phenology with sea ice conditions
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)
        # SMB and Sea ice data only till 2020
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])
        young_sound_fjord_sea_ice_combined['sif_break_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_break_doy'] - young_sound_fjord_sea_ice_combined['sif_break_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_break_doy'].std()
        young_sound_fjord_sea_ice_combined['sif_formation_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_formation_doy'] - young_sound_fjord_sea_ice_combined['sif_formation_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_formation_doy'].std()
        young_sound_fjord_sea_ice_combined['open_water_duration_days_norm'] = (young_sound_fjord_sea_ice_combined['open_water_duration_days'] - young_sound_fjord_sea_ice_combined['open_water_duration_days'].mean()) / young_sound_fjord_sea_ice_combined['open_water_duration_days'].std()

        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        for phenology in phenologies:
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            da = ds[f"{phenology}_doy"]
            # Mask sea ice pixels to focus on the phenology of the open water areas
            field3d_mask = xr.broadcast(da, racmo_sif == 0)[1]
            doo_masked = da.where(~field3d_mask)
            doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            doo_masked_mean_df = doo_masked_mean.to_dataframe()
            phenologies_df[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

            # Mask young sound pixels to focus on the phenology of the open water areas
            field3d_mask = xr.broadcast(da, racmo_ys == 0)[1]
            doo_masked = da.where(~field3d_mask)
            doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            doo_masked_mean_df = doo_masked_mean.to_dataframe()
            phenologies_df_ys[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

        years = phenologies_df_ys.index
        ys_doy = pd.DataFrame(index=years, columns=np.arange(120, 360), data=0)
        ys_doy = ys_doy.astype(float)
        for year in years:
            doo = phenologies_df_ys.loc[year, 'doo']
            dor = phenologies_df_ys.loc[year, 'dor']
            doa = phenologies_df_ys.loc[year, 'doa']
            doc = phenologies_df_ys.loc[year, 'doc']

            # Case 1: dor and doa are NaN → fill doo to doc as 1
            if pd.isna(dor) and pd.isna(doa):
                ys_doy.loc[year, int(doo):int(doc)] = 0.5

            # Case 2: full phenology available
            else:
                ys_doy.loc[year, int(doo):int(dor)] = 0.5
                ys_doy.loc[year, int(dor):int(doa)] = 1.5
                ys_doy.loc[year, int(doa):int(doc)] = 0.5

        # Define colormap
        colors = [(255, 0, 0), (255, 255, 255), (135, 206, 235), (65, 105, 225)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.0001, 0.0001, 1, 2])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        x_min = ys_doy.columns.min()
        x_max = ys_doy.columns.max()
        y_min = ys_doy.index.min()
        y_max = ys_doy.index.max()

        # Plot
        fig = plt.figure(figsize=(2.2, 2), dpi=600)
        gs = plt.GridSpec(1, 1, hspace=0, wspace=0)   # small internal spacing

        ax = fig.add_subplot(gs[0, 0])
        im = ax.imshow(ys_doy, aspect='auto', cmap=cmap, norm=norm, origin='lower', extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], zorder=0)
        ax.axvline(x=phenologies_df_ys['doo'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)
        ax.axvline(x=phenologies_df_ys['dor'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        ax.axvline(x=phenologies_df_ys['doa'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        ax.axvline(x=phenologies_df_ys['doc'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)

        for year in young_sound_fjord_sea_ice_combined.index:
            ax.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_break_doy'], ymin=year-0.5, ymax=year+0.5, color='k', lw=0.5, ls='-', zorder=2)
            ax.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_formation_doy'], ymin=year-0.5, ymax=year+0.5, color='k', lw=0.5, ls='-', zorder=2)

        darker_color = '#00A651'
        lighter_color = '#7CFC00'
        for year in phenologies_df.index:
            ax.vlines(x=phenologies_df.loc[year, 'doo'], ymin=year-0.5, ymax=year+0.5, color=darker_color, lw=0.5, ls='-', zorder=2)
            ax.vlines(x=phenologies_df.loc[year, 'dor'], ymin=year-0.5, ymax=year+0.5, color=lighter_color , lw=0.5, ls='-', zorder=2)
            ax.vlines(x=phenologies_df.loc[year, 'doa'], ymin=year-0.5, ymax=year+0.5, color=lighter_color, lw=0.5, ls='-', zorder=2)
            ax.vlines(x=phenologies_df.loc[year, 'doc'], ymin=year-0.5, ymax=year+0.5, color=darker_color, lw=0.5, ls='-', zorder=2)

        ax.set_xlim([x_min - 0.5, x_max + 0.5])
        # ax.set_xticks([120, 160, 200, 240, 280, 320, 360])
        ax.set_xticks([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticklabels([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticks([135, 165, 195, 225, 255, 285, 315, 345], minor=True)
        ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        major_years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
        minor_years = [y for y in years if y not in major_years]
        ylim = (years[0] - 0.5, years[-1] + 0.5)
        ax.set_ylim(ylim)
        ax.set_yticks(major_years)
        ax.set_yticklabels(major_years)
        ax.set_yticks(minor_years, minor=True)

        ax.tick_params(axis='both', which='major', direction='out', pad=0.8, length=4, width=0.5, labelsize=6)
        # ax.tick_params(axis='both', which='minor', direction='out', length=1.5, width=0.5)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=4)
        ax.tick_params(axis='y', which='minor', direction='out', length=1.5, width=0.5)

        # ax.set_xlabel('DOY', horizontalalignment='center', verticalalignment='center', fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(major_years[0], major_years[-1])
        ax.spines['bottom'].set_bounds(x_min, x_max)
        ax.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.3) for i in ax.spines.values()]
        # ax.text(0.05, 1, 'a)', ha='center', va='center', fontsize=6, transform=ax.transAxes, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

        # Figure-relative coordinates (0–1)
        y = 0.09
        height = 0.02
        segments = [(0.2, 0.20, (135/255, 206/255, 235/255)),  # DOO–DOR
                    (0.4, 0.20, (65/255, 105/255, 225/255)),   # DOR–DOA
                    (0.6, 0.20, (135/255, 206/255, 235/255))]   # DOA–DOC

        for x, w, c in segments:
            fig.add_artist(Rectangle((x, y), w, height, transform=fig.transFigure, color=c, clip_on=False))

        # labels
        y = 0.03
        fig.text(0.2, y - 0.015, 'DOO\n(< 80%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.4, y - 0.015, 'DOR\n(< 15%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.6, y - 0.015, 'DOA\n(> 15%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.8, y - 0.015, 'DOC\n(> 80%)', ha='center', fontsize=6, linespacing=0.8)

        targets = dict(left=150, right=55, top=25, bottom=250, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_greenland_sea_ice_at_doo_dor_doa_doc'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_ice_at_doo_dor_doa_doc.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
    # %%

    def plot_greenland_sea_ice_at_doo_dor_doa_doc():
        # %%
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
        # Read Greenland Sea mask
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        # Slice the racmo_sif to the same spatial extent as the phenology dataset to speed up the following calculations
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        racmo_sif = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        gsea_mask = racmo_sif.sel(radius=800)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Greenland Sea mask for z500
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # racmo_ys = xr.open_dataarray(f"{mask_file_path}/young_sound_mask.nc")
        # Slice the racmo_sif to the same spatial extent as the phenology dataset to speed up the following calculations
        # racmo_ys = racmo_ys.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Create an empty DataFrame to store the phenology values for each year
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies_df = pd.DataFrame(columns=phenologies, index=range(1990, 2021))
        # phenologies_df_ys = pd.DataFrame(columns=phenologies, index=range(1990, 2021))

        # Read young sound fjord sea ice combined dataset to compare the phenology with sea ice conditions
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)
        # SMB and Sea ice data only till 2020
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])
        young_sound_fjord_sea_ice_combined['sif_break_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_break_doy'] - young_sound_fjord_sea_ice_combined['sif_break_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_break_doy'].std()
        young_sound_fjord_sea_ice_combined['sif_formation_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_formation_doy'] - young_sound_fjord_sea_ice_combined['sif_formation_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_formation_doy'].std()
        young_sound_fjord_sea_ice_combined['open_water_duration_days_norm'] = (young_sound_fjord_sea_ice_combined['open_water_duration_days'] - young_sound_fjord_sea_ice_combined['open_water_duration_days'].mean()) / young_sound_fjord_sea_ice_combined['open_water_duration_days'].std()

        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        for phenology in phenologies:
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

            if phenology in ['doc']:
                # Adjust for leap years in the phenology day-of-year data
                # Add 365/366 if the day-of-year is before the threshold in Ice year
                # ‘ice year’ from March 1 to February 28 of the following year
                # 60/61 DOY is March to next year from Jan and Feb
                is_leap = ds[f"{phenology}_time"].dt.is_leap_year
                threshold = xr.where(is_leap, 61, 60)
                days_in_year = xr.where(is_leap, 366, 365)
                ds[f"{phenology}_doy"] = xr.where(ds[f"{phenology}_doy"] < threshold,
                                                ds[f"{phenology}_doy"] + days_in_year,
                                                ds[f"{phenology}_doy"])

            da = ds[f"{phenology}_doy"]

            # Mask sea ice pixels to focus on the phenology of the open water areas
            field3d_mask = xr.broadcast(da, gsea_mask == 0)[1]
            doo_masked = da.where(~field3d_mask)
            # if (phenology == 'doc'):
                # doo_masked = xr.where(doo_masked < 60, doo_masked + 365, doo_masked)
                # doo_masked = xr.where(doo_masked < 120, doo_masked + 365, doo_masked)
            doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            doo_masked_mean_df = doo_masked_mean.to_dataframe()
            phenologies_df[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

            # # Mask young sound pixels to focus on the phenology of the open water areas
            # field3d_mask = xr.broadcast(da, racmo_ys == 0)[1]
            # doo_masked = da.where(~field3d_mask)
            # doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            # doo_masked_mean_df = doo_masked_mean.to_dataframe()
            # phenologies_df_ys[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

        # years = phenologies_df_ys.index
        phenologies_df = phenologies_df.iloc[:-1, :]
        years = phenologies_df.index
        ys_doy = pd.DataFrame(index=years, columns=np.arange(120, 360), data=0)
        # ys_doy = pd.DataFrame(index=years, columns=np.arange(60, 360), data=0)
        ys_doy = ys_doy.astype(float)
        for year in years:
            doo = phenologies_df.loc[year, 'doo']
            dor = phenologies_df.loc[year, 'dor']
            doa = phenologies_df.loc[year, 'doa']
            doc = phenologies_df.loc[year, 'doc']

            # Case 1: dor and doa are NaN → fill doo to doc as 1
            if pd.isna(dor) and pd.isna(doa):
                ys_doy.loc[year, int(doo):int(doc)] = 0.5

            # Case 2: full phenology available
            else:
                ys_doy.loc[year, int(doo):int(dor)] = 0.5
                ys_doy.loc[year, int(dor):int(doa)] = 1.5
                ys_doy.loc[year, int(doa):int(doc)] = 0.5

        # Define colormap
        colors = [(255, 0, 0), (255, 255, 255), (135, 206, 235), (65, 105, 225)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.0001, 0.0001, 1, 2])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        x_min = ys_doy.columns.min()
        x_max = ys_doy.columns.max()
        y_min = ys_doy.index.min()
        y_max = ys_doy.index.max()

        # Plot
        fig = plt.figure(figsize=(2.2, 2), dpi=600)
        gs = plt.GridSpec(1, 1, hspace=0, wspace=0)   # small internal spacing

        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(ys_doy, aspect='auto', cmap=cmap, norm=norm, origin='lower', extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], zorder=0)
        # ax.axvline(x=phenologies_df['doo'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)
        # ax.axvline(x=phenologies_df['dor'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        # ax.axvline(x=phenologies_df['doa'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        # ax.axvline(x=phenologies_df['doc'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)

        for year in young_sound_fjord_sea_ice_combined.index:
            ax.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_break_doy'], ymin=year-0.5, ymax=year+0.5, color='#F2C14E', lw=0.5, ls='-', zorder=2)
            ax.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_formation_doy'], ymin=year-0.5, ymax=year+0.5, color='#E76F51', lw=0.5, ls='-', zorder=2)

        # darker_color = '#00A651'
        # lighter_color = '#7CFC00'
        # for year in phenologies_df.index:
        #     ax.vlines(x=phenologies_df.loc[year, 'doo'], ymin=year-0.5, ymax=year+0.5, color=darker_color, lw=0.5, ls='-', zorder=2)
        #     ax.vlines(x=phenologies_df.loc[year, 'dor'], ymin=year-0.5, ymax=year+0.5, color=lighter_color , lw=0.5, ls='-', zorder=2)
        #     ax.vlines(x=phenologies_df.loc[year, 'doa'], ymin=year-0.5, ymax=year+0.5, color=lighter_color, lw=0.5, ls='-', zorder=2)
        #     ax.vlines(x=phenologies_df.loc[year, 'doc'], ymin=year-0.5, ymax=year+0.5, color=darker_color, lw=0.5, ls='-', zorder=2)

        ax.set_xlim([x_min - 0.5, x_max + 0.5])
        # ax.set_xticks([120, 160, 200, 240, 280, 320, 360])
        ax.set_xticks([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticklabels([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax.set_xticks([135, 165, 195, 225, 255, 285, 315, 345], minor=True)
        ax.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        # major_years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
        major_years = [1990, 1995, 2000, 2005, 2010, 2015, 2019]
        minor_years = [y for y in phenologies_df.index if y not in major_years]
        ylim = (phenologies_df.index[0] - 0.5, phenologies_df.index[-1] + 0.5)
        ax.set_ylim(ylim)
        ax.set_yticks(major_years)
        ax.set_yticklabels(major_years)
        ax.set_yticks(minor_years, minor=True)

        ax.tick_params(axis='both', which='major', direction='out', pad=0.8, length=4, width=0.3, labelsize=6)
        # ax.tick_params(axis='both', which='minor', direction='out', length=1.5, width=0.5)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=4, width=0.3)
        ax.tick_params(axis='y', which='minor', direction='out', length=2, width=0.3)

        # ax.set_xlabel('DOY', horizontalalignment='center', verticalalignment='center', fontsize=6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_bounds(major_years[0], major_years[-1])
        ax.spines['bottom'].set_bounds(x_min, x_max + 1)
        ax.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.3) for i in ax.spines.values()]
        # ax.text(0.05, 1, 'a)', ha='center', va='center', fontsize=6, transform=ax.transAxes, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
        # ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

        # Draw grid
        ax.grid(which='major', color='gray', linestyle='--', linewidth=0.2, alpha=0.3, zorder=0)
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

        # Figure-relative coordinates (0–1)
        y = 0.09
        height = 0.02
        segments = [(0.2, 0.20, (135/255, 206/255, 235/255)),  # DOO–DOR
                    (0.4, 0.20, (65/255, 105/255, 225/255)),   # DOR–DOA
                    (0.6, 0.20, (135/255, 206/255, 235/255))]   # DOA–DOC

        for x, w, c in segments:
            fig.add_artist(Rectangle((x, y), w, height, transform=fig.transFigure, color=c, clip_on=False))

        # labels
        y = 0.03
        fig.text(0.2, y - 0.015, 'DOO\n(< 80%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.4, y - 0.015, 'DOR\n(< 15%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.6, y - 0.015, 'DOA\n(> 15%)', ha='center', fontsize=6, linespacing=0.8)
        fig.text(0.8, y - 0.015, 'DOC\n(> 80%)', ha='center', fontsize=6, linespacing=0.8)

        # Legends
        line2d_param = dict(linestyle='-', lw=0.9, marker='None')
        break_up = Line2D([], [], label='Break up', color='#F2C14E', **line2d_param)
        formation = Line2D([], [], label='Formation', color='#E76F51', **line2d_param)
        legend_param = dict(loc='center', fontsize=6, frameon=False, handletextpad=0.5)
        ax.legend(handles=[break_up, formation], bbox_to_anchor=(0.95, 0.05), ncol=1, handlelength=0.8, labelspacing=0.5, markerscale=1, **legend_param)

        targets = dict(left=150, right=40, top=10, bottom=250, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_greenland_sea_ice_at_doo_dor_doa_doc'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_ice_at_doo_dor_doa_doc.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def print_plot_greenland_sea_ice_at_doo_dor_doa_doc():
        # %%
        mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')

        racmo_sif = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        gsea_mask = racmo_sif.sel(radius=800)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies_df = pd.DataFrame(columns=phenologies, index=range(1990, 2021))

        file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')
        for phenology in phenologies:
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            if phenology in ['doc']:
                # Adjust for leap years in the phenology day-of-year data
                # Add 365/366 if the day-of-year is before the threshold in Ice year
                # ‘ice year’ from March 1 to February 28 of the following year
                # 60/61 DOY is March to next year from Jan and Feb
                is_leap = ds[f"{phenology}_time"].dt.is_leap_year
                threshold = xr.where(is_leap, 61, 60)
                days_in_year = xr.where(is_leap, 366, 365)
                ds[f"{phenology}_doy"] = xr.where(ds[f"{phenology}_doy"] < threshold,
                                                ds[f"{phenology}_doy"] + days_in_year,
                                                ds[f"{phenology}_doy"])

            da = ds[f"{phenology}_doy"]
            # Mask sea ice pixels to focus on the phenology of the open water areas
            field3d_mask = xr.broadcast(da, gsea_mask == 0)[1]
            doo_masked = da.where(~field3d_mask)
            # if (phenology == 'doc'):
                # doo_masked = xr.where(doo_masked < 60, doo_masked + 365, doo_masked)
                # doo_masked = xr.where(doo_masked < 120, doo_masked + 365, doo_masked)
            doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            doo_masked_mean_df = doo_masked_mean.to_dataframe()
            phenologies_df[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

        mean_phenologies_df = phenologies_df.mean()
        std_phenologies_df = phenologies_df.std()
        print(mean_phenologies_df)
        print(std_phenologies_df)

    def plot_greenland_sea_ice_at_doo_dor_doa_doc_and_trend():
        # %%
        # fig = plt.figure(figsize=(4.35, 4.35), dpi=600)
        fig = plt.figure(figsize=(4, 5.25), dpi=600)
        # Outer GridSpec: controls the gap between the top and bottom groups
        # Increase gap between top colorbar and bottom plots
        outer = fig.add_gridspec(2, 1, height_ratios=[1.25, 2], hspace=0.03)
        # TOP group: plot + colorbar close together
        gs_top_group = outer[0].subgridspec(2, 1, height_ratios=[2, 0.1], hspace=0.3)
        gs_top = gs_top_group[0, 0].subgridspec(1, 1)
        gs_top_cbar = gs_top_group[1, 0].subgridspec(1, 1)
        # BOTTOM group: plots + colorbar close together
        gs_bot_group = outer[1].subgridspec(3, 1, height_ratios=[2, 2, 0.1], hspace=0.15)
        gs_bot = gs_bot_group[0:2, 0].subgridspec(2, 3, hspace=0.1, wspace=0)
        gs_bot_cbar = gs_bot_group[2, 0].subgridspec(1, 1)

        # Plot Greenland sea ice at DOO, DOR, DOA, DOC
        ax_top = fig.add_subplot(gs_top[0, 0])
        ax_top.axis('off')
        # ax_top = ax_top.inset_axes([0.16, 0.01, 0.6, 0.8]) #[left, bottom, width, height]
        ax_top = ax_top.inset_axes([0.25, 0.01, 0.5, 0.99]) #[left, bottom, width, height]

        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        # Read Greenland Sea mask
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        # Slice the racmo_sif to the same spatial extent as the phenology dataset to speed up the following calculations
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        racmo_sif = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        gsea_mask = racmo_sif.sel(radius=800)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Greenland Sea mask for z500
        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        # Create an empty DataFrame to store the phenology values for each year
        phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies_df = pd.DataFrame(columns=phenologies, index=range(1990, 2021))
        # phenologies_df_ys = pd.DataFrame(columns=phenologies, index=range(1990, 2021))

        # Read young sound fjord sea ice combined dataset to compare the phenology with sea ice conditions
        input_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)
        # SMB and Sea ice data only till 2020
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])
        young_sound_fjord_sea_ice_combined['sif_break_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_break_doy'] - young_sound_fjord_sea_ice_combined['sif_break_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_break_doy'].std()
        young_sound_fjord_sea_ice_combined['sif_formation_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_formation_doy'] - young_sound_fjord_sea_ice_combined['sif_formation_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_formation_doy'].std()
        young_sound_fjord_sea_ice_combined['open_water_duration_days_norm'] = (young_sound_fjord_sea_ice_combined['open_water_duration_days'] - young_sound_fjord_sea_ice_combined['open_water_duration_days'].mean()) / young_sound_fjord_sea_ice_combined['open_water_duration_days'].std()

        file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        for phenology in phenologies:
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))

            if phenology in ['doc']:
                # Adjust for leap years in the phenology day-of-year data
                # Add 365/366 if the day-of-year is before the threshold in Ice year
                # ‘ice year’ from March 1 to February 28 of the following year
                # 60/61 DOY is March to next year from Jan and Feb
                is_leap = ds[f"{phenology}_time"].dt.is_leap_year
                threshold = xr.where(is_leap, 61, 60)
                days_in_year = xr.where(is_leap, 366, 365)
                ds[f"{phenology}_doy"] = xr.where(ds[f"{phenology}_doy"] < threshold,
                                                ds[f"{phenology}_doy"] + days_in_year,
                                                ds[f"{phenology}_doy"])

            da = ds[f"{phenology}_doy"]

            # Mask sea ice pixels to focus on the phenology of the open water areas
            field3d_mask = xr.broadcast(da, gsea_mask == 0)[1]
            doo_masked = da.where(~field3d_mask)
            # if (phenology == 'doc'):
                # doo_masked = xr.where(doo_masked < 60, doo_masked + 365, doo_masked)
                # doo_masked = xr.where(doo_masked < 120, doo_masked + 365, doo_masked)
            doo_masked_mean = doo_masked.mean(dim=['rlat', 'rlon'])
            doo_masked_mean_df = doo_masked_mean.to_dataframe()
            phenologies_df[phenology] = doo_masked_mean_df[f"{phenology}_doy"].values

        # years = phenologies_df_ys.index
        phenologies_df = phenologies_df.iloc[:-1, :]
        years = phenologies_df.index
        ys_doy = pd.DataFrame(index=years, columns=np.arange(120, 360), data=0)
        # ys_doy = pd.DataFrame(index=years, columns=np.arange(60, 360), data=0)
        ys_doy = ys_doy.astype(float)
        for year in years:
            doo = phenologies_df.loc[year, 'doo']
            dor = phenologies_df.loc[year, 'dor']
            doa = phenologies_df.loc[year, 'doa']
            doc = phenologies_df.loc[year, 'doc']

            # Case 1: dor and doa are NaN → fill doo to doc as 1
            if pd.isna(dor) and pd.isna(doa):
                ys_doy.loc[year, int(doo):int(doc)] = 0.5

            # Case 2: full phenology available
            else:
                ys_doy.loc[year, int(doo):int(dor)] = 0.5
                ys_doy.loc[year, int(dor):int(doa)] = 1.5
                ys_doy.loc[year, int(doa):int(doc)] = 0.5

        # Define colormap
        colors = [(255, 0, 0), (255, 255, 255), (135, 206, 235), (65, 105, 225)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.0001, 0.0001, 1, 2])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        x_min = ys_doy.columns.min()
        x_max = ys_doy.columns.max()
        y_min = ys_doy.index.min()
        y_max = ys_doy.index.max()

        ax_top.imshow(ys_doy, aspect='auto', cmap=cmap, norm=norm, origin='lower', extent=[x_min - 0.5, x_max + 0.5, y_min - 0.5, y_max + 0.5], zorder=0)
        # ax_top.axvline(x=phenologies_df['doo'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)
        # ax_top.axvline(x=phenologies_df['dor'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        # ax_top.axvline(x=phenologies_df['doa'].median(), color='royalblue', lw=0.5, ls=':', zorder=1)
        # ax_top.axvline(x=phenologies_df['doc'].median(), color='skyblue', lw=0.5, ls='--', zorder=1)

        for year in young_sound_fjord_sea_ice_combined.index:
            ax_top.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_break_doy'], ymin=year-0.5, ymax=year+0.5, color='#F2C14E', lw=0.5, ls='-', zorder=2)
            ax_top.vlines(x=young_sound_fjord_sea_ice_combined.loc[year, 'sif_formation_doy'], ymin=year-0.5, ymax=year+0.5, color='#E76F51', lw=0.5, ls='-', zorder=2)

        ax_top.set_xlim([x_min - 0.5, x_max + 0.5])
        # ax_top.set_xticks([120, 160, 200, 240, 280, 320, 360])
        ax_top.set_xticks([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax_top.set_xticklabels([120, 150, 180, 210, 240, 270, 300, 330, 360])
        ax_top.set_xticks([135, 165, 195, 225, 255, 285, 315, 345], minor=True)
        ax_top.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        # major_years = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
        major_years = [1990, 1995, 2000, 2005, 2010, 2015, 2019]
        minor_years = [y for y in phenologies_df.index if y not in major_years]
        ylim = (phenologies_df.index[0] - 0.5, phenologies_df.index[-1] + 0.5)
        ax_top.set_ylim(ylim)
        ax_top.set_yticks(major_years)
        ax_top.set_yticklabels(major_years)
        ax_top.set_yticks(minor_years, minor=True)

        ax_top.tick_params(axis='both', which='major', direction='out', pad=0.8, length=4, width=0.3, labelsize=8)
        # ax_top.tick_params(axis='both', which='minor', direction='out', length=1.5, width=0.5)
        ax_top.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=5, width=0.3)
        ax_top.tick_params(axis='y', which='minor', direction='out', length=2, width=0.3)

        # ax_top.set_xlabel('DOY', horizontalalignment='center', verticalalignment='center', fontsize=6)
        ax_top.spines['top'].set_visible(False)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['left'].set_bounds(major_years[0], major_years[-1])
        ax_top.spines['bottom'].set_bounds(x_min, x_max + 1)
        ax_top.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.3) for i in ax_top.spines.values()]

        # Draw grid
        ax_top.grid(which='major', color='gray', linestyle='--', linewidth=0.2, alpha=0.3, zorder=0)
        # # Get spine bounds
        x0, x1 = ax_top.spines['bottom'].get_bounds()
        y0, y1 = ax_top.spines['left'].get_bounds()
        # x0, x1 = ax.get_xlim()
        # y0, y1 = ax.get_ylim()
        # Create clipping rectangle based on spine bounds
        grid_clip = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax_top.transData)
        # Clip gridlines
        for line in ax_top.get_xgridlines() + ax_top.get_ygridlines():
            line.set_clip_path(grid_clip)

        ax_top.text(-0.27, 0.955, rf"$\mathbf{{(a)}}$", color='k', fontsize=8, zorder=12, transform=ax_top.transAxes, path_effects=[pe.withStroke(linewidth=0.75, foreground='w')])

        # Colorbar of the DOO, DOR, DOA, DOC
        ax_top_cbar = fig.add_subplot(gs_top_cbar[0, 0])
        # Use axes coordinates (0–1)
        ax_top_cbar.set_axis_off()

        # Figure-relative coordinates (0–1)
        y = 0.65
        height = 1.2

        segments = [(0.2, 0.20, (135/255, 206/255, 235/255)),  # DOO–DOR
                    (0.4, 0.20, (65/255, 105/255, 225/255)),   # DOR–DOA
                    (0.6, 0.20, (135/255, 206/255, 235/255))]   # DOA–DOC

        for x, w, c in segments:
            ax_top_cbar.add_patch(Rectangle((x, y), w, height, transform=ax_top_cbar.transAxes, facecolor=c, edgecolor='none', clip_on=False))

        # Labels
        # labels = [(0.2, 'DOO\n(< 80%)'), (0.4, 'DOR\n(< 15%)'), (0.6, 'DOA\n(> 15%)'), (0.8, 'DOC\n(> 80%)')]
        labels = [(0.2, 'DOO (< 80%)'), (0.4, 'DOR (< 15%)'), (0.6, 'DOA (> 15%)'), (0.8, 'DOC (> 80%)')]

        for x, label in labels:
            # ax_top_cbar.text(x, 0.35, label, transform=ax_top_cbar.transAxes, ha='center', va='top', fontsize=8, linespacing=0.8)
            ax_top_cbar.text(x, 0.6, label, transform=ax_top_cbar.transAxes, ha='center', va='top', fontsize=8, linespacing=0.8)

        ax_top_cbar.set_xlim(0, 1)
        ax_top_cbar.set_ylim(0, 1)

        # Plot trends
        # Point mask of closest Freya glacier mask racmo at 5 km
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        rlon, rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # Get longitude and latitude plotting parameters
        lonlat = _get_latlon_meshgrid_east_greenland()
        bounds_lon, cmap_lon, norm_lon, bounds_lat, cmap_lat, norm_lat = _get_lonlat_params_east_greenland()
        xlims, xticks, xticklabels, ylims, yticks, yticklabels = _get_lonlat_ticks_east_greenland(bounds_lon=bounds_lon, bounds_lat=bounds_lat)
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude_east_greenland()

        gsea_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        gsea_mask = xr.where(gsea_mask > 0.5, 1, 0)
        gsea_mask = gsea_mask.sel(rlon=slice(2, 11.2), rlat=slice(-5, 12))

        colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 255, 255), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-6, -5, -4, -3, -2, -1, -0.0001, 0.0001, 1, 2, 3, 4, 5, 6])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        cbticklabel = [-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60]

        phenologies = [('doo', 0, 0, 'b'), ('doc', 0, 1, 'c'), ('oifp', 0, 2, 'd'), ('dor', 1, 0, 'e'), ('doa', 1, 1, 'f'), ('iifp', 1, 2, 'g')]
        for phenology, row, col, label in phenologies:
            ax_bot1 = fig.add_subplot(gs_bot[row, col])
            # Read Sea ice phenology trend
            ds = xr.open_dataset(f"{file_path}/greenland_sea_sif_phenology_{phenology}_trend_theilsen.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))
            ds[f'{phenology}_theilsen_slope'] = ds[f'{phenology}_theilsen_slope'].where(ds[f'{phenology}_trend_pvalue'] < 0.05, np.nan)

            ax_bot1.contourf(lonlat[0], lonlat[1], ds[f'{phenology}_theilsen_slope'], cmap=cmap, norm=norm, extend='neither', levels=bounds)
            ax_bot1.contourf(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, zorder=5)
            ax_bot1.contour(lonlat[0],lonlat[1], racmo_longitude, cmap=cmap_lon, norm=norm_lon, levels=bounds_lon, linewidths=0.2, linestyles='dashed', zorder=10)
            ax_bot1.contour(lonlat[0],lonlat[1], racmo_latitude, cmap=cmap_lat, norm=norm_lat, levels=bounds_lat, linewidths=0.2, linestyles='dashed', zorder=10)
            ax_bot1.plot(rlon, rlat, 'x', color='k', ms=4, label='Closest point to Freya glacier', zorder=12)
            ax_bot1.set_xticks(xticks)
            ax_bot1.set_xticklabels(xticklabels)
            ax_bot1.set_yticks(yticks)
            ax_bot1.set_yticklabels(yticklabels)
            cs = ax_bot1.contour(lonlat[0], lonlat[1], gsea_mask, colors='gray', linewidths=0.2, linestyles='dashed', zorder=1)
            cs.set_dashes([(0, (10, 10))])

            ax_bot1.text(0.02, 0.91, rf"$\mathbf{{({label}) {phenology.upper()}}}$", color='k', fontsize=8, zorder=12, transform=ax_bot1.transAxes, path_effects=[pe.withStroke(linewidth=0.75, foreground='w')])
            # [ax.spines[spine].set_visible(False) for spine in ['top', 'right', 'bottom', 'left']]
            # ax.axis('off')

            ax_bot1.set_xlim(xlims)
            ax_bot1.set_ylim(ylims)

            ax_bot1.set_xticks(xticks)
            ax_bot1.set_yticks(yticks)

            ax_bot1.set_xticklabels(xticklabels, color='k')
            ax_bot1.set_yticklabels(yticklabels, color='k')

            tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=8, width=0.5)
            ax_bot1.tick_params(**tick_params)
            [i.set_linewidth(0.3) for i in ax_bot1.spines.values()]

            ax_bot1.set_aspect('equal', adjustable='box')

        ax_bot_cbar = fig.add_subplot(gs_bot_cbar[0, 0])
        ax_bot_cbar.axis('off')

        cbar_ax = ax_bot_cbar.inset_axes([0.14, 0.01, 0.7, 1.2]) #[left, bottom, width, height]
        # cbar_ax = ax_bot_cbar.inset_axes([0.05, 0.01, 0.7, 1.2]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm, extend='neither')
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=8, pad=0.5, length=2, direction='in', width=0.5)
        cb.outline.set_linewidth(0.4)
        cb.set_label(rf"Days decade$^{{\mathdefault{{-1}}}}$", size=8, labelpad=-1.5)

        # Legends
        break_up = Line2D([], [], label='Fjord-ice \nbreakup', color='#F2C14E', ls='--', lw=0.75, marker='None')
        formation = Line2D([], [], label='Fjord-ice \nformation', color='#E76F51', ls='--', lw=0.75, marker='None')
        frg_location = Line2D([], [], label='FRG', linestyle='', marker='x', markersize=3, markerfacecolor='None', markeredgecolor='k')
        grsea_location = Line2D([], [], label='GrSea',  color='gray', ls='--', lw=0.75, marker='None')
        handles = [break_up, formation, frg_location, grsea_location]
        fig.legend(bbox_to_anchor=(0.97, 0.9), handles=handles, ncol=1, borderaxespad=0, fontsize=8, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')

        targets = dict(left=60, right=0, top=10, bottom=130, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_greenland_sea_ice_at_doo_dor_doa_doc_and_trend'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_greenland_sea_ice_at_doo_dor_doa_doc_and_trend.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

    def print_young_sound_fjord_sea_ice_breakup_formation():
        # %%
        # Read young sound fjord sea ice combined dataset to compare the phenology with sea ice conditions
        input_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)
        # SMB and Sea ice data only till 2020
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])
        # young_sound_fjord_sea_ice_combined['sif_break_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_break_doy'] - young_sound_fjord_sea_ice_combined['sif_break_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_break_doy'].std()
        # young_sound_fjord_sea_ice_combined['sif_formation_doy_norm'] = (young_sound_fjord_sea_ice_combined['sif_formation_doy'] - young_sound_fjord_sea_ice_combined['sif_formation_doy'].mean()) / young_sound_fjord_sea_ice_combined['sif_formation_doy'].std()
        # young_sound_fjord_sea_ice_combined['open_water_duration_days_norm'] = (young_sound_fjord_sea_ice_combined['open_water_duration_days'] - young_sound_fjord_sea_ice_combined['open_water_duration_days'].mean()) / young_sound_fjord_sea_ice_combined['open_water_duration_days'].std()





        # %%

class SMBAndSeaIceAnalysis:
    def plot_annual_cycle_sea_ice_buffers_versus_freya_smb():
        # %%
        # fig, ax = plt.subplots(figsize=(6, 4), dpi=600)
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=600)
        ax_twinx = ax.twinx()
        ax_twinx.set_frame_on(True)
        ax_twinx.patch.set_visible(False) # Prevents ax from hiding ax_twinx
        for sp in ax_twinx.spines.values():
            sp.set_visible(False)
        ax_twinx.spines['right'].set_visible(True)

        # Plot Sea ice concentration from RACMO2.3 5 km
        radii = ['young_sound', 100, 200, 300, 400, 500, 600, 700, 800]
        # radii = ['young_sound', 'YS_100', '100_200', '200_300', '300_400', '400_500', '500_600', '600_700', '700_800']
        # radii = ['young_sound', 'YS_100', '100_200', '200_300', '300_400', ]
        colors_sif = [(8, 48, 107), (8, 81, 156), (33, 113, 181), (49, 130, 189), (66, 146, 198), (107, 174, 214), (158, 202, 225), (198, 219, 239), (222, 235, 247)]
        colors = [tuple(c/255 for c in rgb) for rgb in colors_sif]
        sif_masked_5km_variables_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        for radius, color in zip(radii, colors):
            if radius == 'young_sound':
                masked_5km_variables = pd.read_csv(f"{sif_masked_5km_variables_path}/masked_5km_variables_within_young_sound_sea_ice.csv", parse_dates=['time'], index_col='time')
            else:
                # masked_5km_variables = pd.read_csv(f"{sif_masked_5km_variables_path}/masked_5km_variables_between_{radius}_sea_ice.csv", parse_dates=['time'], index_col='time')
                masked_5km_variables = pd.read_csv(f"{sif_masked_5km_variables_path}/masked_5km_variables_within_{radius}_sea_ice.csv", parse_dates=['time'], index_col='time')
            freq = masked_5km_variables.index.to_series().diff().mode()[0]
            masked_5km_variables = masked_5km_variables.resample(freq).mean()
            ci_in_100km_ci_daily = pd.DataFrame(masked_5km_variables['ci'])
            df_ci_mean, df_ci_std, df_ci_margin_of_error = _get_mean_std_margin_of_error(ci_in_100km_ci_daily, 'ci', critical_value=1.96)
            if radius == 'young_sound':
                lw=2
            else:
                lw=1
            ax.plot(df_ci_mean.index, df_ci_mean['ci'], color=color, lw=lw)

        # Plot Freya Glacier SMB from RACMO2.3 5 km
        smb_masked_5km_variables_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        smb_mod_all_daily = pd.read_csv(f"{smb_masked_5km_variables_path}/smb_masked_freya_1km.csv", header=0)
        smb_mod_all_daily['time'] = pd.to_datetime(smb_mod_all_daily['time'])
        smb_mod_all_daily.set_index('time', inplace=True)

        colors_smb = [(255, 210, 200), (255, 160, 122), (255, 69, 69), (178, 34, 34)]
        colors = [tuple(c/255 for c in rgb) for rgb in colors_smb]
        # Determine original time resolution
        time_delta = smb_mod_all_daily.index[1] - smb_mod_all_daily.index[0]
        smb_mod_all_daily = smb_mod_all_daily.resample(time_delta).mean()
        for smb_type, color in zip(smb_mod_all_daily.columns, colors):
            smb_mod_daily = pd.DataFrame(smb_mod_all_daily.loc[:, smb_type])
            df_smb_mean, df_smb_std, df_smb_margin_of_error = _get_mean_std_margin_of_error(smb_mod_daily, smb_type, critical_value=1.96)
            ax_twinx.plot(df_smb_mean.index, df_smb_mean[smb_type], color=color, lw=1)

        # Customize spines
        ax.spines['left'].set_color('b')
        ax_twinx.spines['right'].set_color('r')
        ax.spines['top'].set_visible(False)
        # ax.spines['left'].set_bounds(0, 1)  # Adjust the bounds as needed
        ax_twinx.spines['right'].set_bounds(-30, 20)
        ax.spines['bottom'].set_bounds(1, 365)
        ax.spines['bottom'].set_position(('outward', 3))
        [i.set_linewidth(0.4) for i in ax.spines.values()]
        [i.set_linewidth(0.4) for i in ax_twinx.spines.values()]

        # Customize ticks and labels
        ax.set_ylim([0, 1])
        sif_yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_yticks(sif_yticks)
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax.set_ylabel('Sea ice concentration (%)', color='b')
        ax.yaxis.set_label_coords(-0.05, 0.5)

        ax_twinx.set_ylim([-30, 20])
        smb_yticks = [-30, -20, -10, 0, 10, 20]
        ax_twinx.set_yticks(smb_yticks)
        ax_twinx.set_yticks([-25, -15, -5, 5, 15], minor=True)
        ax_twinx.set_ylabel('SMB (mm w.e.)', color='r')
        ax_twinx.yaxis.set_label_coords(1.06, 0.5)

        # Align both y-axis to have the same zero line
        _align_secondary_yaxis(ax, ax_twinx, sif_yticks, smb_yticks)

        # Set y-tick labels after alignment to ensure they are correct
        ax.set_yticklabels([int(v*100) for v in sif_yticks])
        ax_twinx.set_yticklabels(smb_yticks)

        ax.set_xlim([1, 365])
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticklabels([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticks([15, 46, 74, 105, 135, 166, 196, 225, 258, 288, 319, 349], minor=True)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=9)
        ax.tick_params(axis='y', which='major', labelcolor='b', color='b', **ytick_params)
        ax_twinx.tick_params(axis='y', which='major', labelcolor='r', color='r', **ytick_params)
        ax.tick_params(axis='x', which='major', direction='out', width=0.5, length=5, pad=2, labelsize=9)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=8)
        ytick_minor_params = dict(axis='y', which='minor', direction='out', width=0.5, length=2.5, pad=2)
        ax.tick_params(color='b', **ytick_minor_params)
        ax_twinx.tick_params(color='r', **ytick_minor_params)

        ax.grid(which='major', linewidth=0.3, linestyle=':', zorder=0, alpha=0.7)

        # Resizeable plot with fixed aspect ratio
        targets = dict(left=250, right=270, top=30, bottom=300, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Join lists with (255, 255, 255) in between
        combined_colors = colors_sif + [(255, 255, 255)] + colors_smb
        cmap = fp.make_cmap(combined_colors, bit=True)
        bounds = np.array([-0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0001, 0.0001, 0.1, 0.2, 0.3, 0.4])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-0.85, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0001, 0.05, 0.15, 0.25, 0.35,]
        # cbtick = [-0.85, -0.75, -0.65, -0.55, -0.45, -0.35, -0.25, -0.15, -0.05, 0.05, 0.15, 0.25, 0.35,]
        cbticklabel = ['YS', '100', '200', '300', '400', '500', '600', '700', '800', fr"FG$_{{\mathdefault{{AVG}}}}$", fr"FG$_{{\mathdefault{{AWS}}}}$", fr"FG$_{{\mathdefault{{AA}}}}$", fr"FG$_{{\mathdefault{{BA}}}}$"]

        # Add colorbar
        # cbar_ax = fig.add_axes([0.1, 0.07, 0.8, 0.03]) #[left, bottom, width, height]
        cbar_ax = fig.add_axes([0.1, 0.035, 0.8, 0.03]) #[left, bottom, width, height]
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm)
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbticklabel)
        cb.ax.tick_params(labelsize=7, pad=0.5, length=2, direction='inout', width=0.5)
        cb.outline.set_linewidth(0.1)
        # ax.text(0.35, -0.17, '(km)', ha='center', va='center', fontsize=8, transform=ax.transAxes, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

        logger.info(f"Saving plot of 'plot_annual_cycle_sea_ice_buffers_versus_freya_smb'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_annual_cycle_sea_ice_buffers_versus_freya_smb.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def plot_annual_cycle_young_sound_sea_ice_buffers_versus_freya_smb():
        # %%
        fig, ax = plt.subplots(figsize=(5, 2.5), dpi=600)
        ax_twinx = ax.twinx()
        ax_twinx.set_frame_on(True)
        ax_twinx.patch.set_visible(False) # Prevents ax from hiding ax_twinx
        for sp in ax_twinx.spines.values():
            sp.set_visible(False)
        ax_twinx.spines['right'].set_visible(True)

        # Plot Freya Glacier SMB from RACMO2.3 5 km
        smb_masked_5km_variables_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        smb_mod_all_daily = pd.read_csv(f"{smb_masked_5km_variables_path}/smb_masked_freya_1km.csv", header=0)
        smb_mod_all_daily['time'] = pd.to_datetime(smb_mod_all_daily['time'])
        smb_mod_all_daily.set_index('time', inplace=True)
        time_delta = smb_mod_all_daily.index[1] - smb_mod_all_daily.index[0]
        smb_mod_all_daily = smb_mod_all_daily.resample(time_delta).mean()

        smb_mod_daily = pd.DataFrame(smb_mod_all_daily.loc[:, 'smb_whole_racmo'])
        df_smb_mean, df_smb_std, df_smb_margin_of_error = _get_mean_std_margin_of_error(smb_mod_daily, 'smb_whole_racmo', critical_value=1.96)
        ax.plot(df_smb_mean.index, df_smb_mean['smb_whole_racmo'], color='r', lw=0.5, zorder=12)
        ax.fill_between(df_smb_mean.index, df_smb_mean['smb_whole_racmo'] - df_smb_margin_of_error['smb_whole_racmo'], df_smb_mean['smb_whole_racmo'] + df_smb_margin_of_error['smb_whole_racmo'], color='r', alpha=0.4, edgecolor='none', label="95% confidence interval")
        ax.fill_between(df_smb_mean.index, df_smb_mean['smb_whole_racmo'] - df_smb_std['smb_whole_racmo'], df_smb_mean['smb_whole_racmo'] + df_smb_std['smb_whole_racmo'], color='r', alpha=0.2, edgecolor='none', label="±1 standard deviation")

        # Plot Sea ice concentration from RACMO2.3 5 km
        sif_masked_5km_variables_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        masked_5km_variables = pd.read_csv(f"{sif_masked_5km_variables_path}/masked_5km_variables_between_YS_100_sea_ice.csv", parse_dates=['time'], index_col='time')
        freq = masked_5km_variables.index.to_series().diff().mode()[0]
        masked_5km_variables = masked_5km_variables.resample(freq).mean()
        ci_in_100km_ci_daily = pd.DataFrame(masked_5km_variables['ci'])
        df_ci_mean, df_ci_std, df_ci_margin_of_error = _get_mean_std_margin_of_error(ci_in_100km_ci_daily, 'ci', critical_value=1.96)
        ax_twinx.plot(df_ci_mean.index, df_ci_mean['ci'], color='b', lw=0.5, zorder=12)
        ax_twinx.fill_between(df_ci_mean.index, df_ci_mean['ci'] - df_ci_margin_of_error['ci'], df_ci_mean['ci'] + df_ci_margin_of_error['ci'], color='b', alpha=0.4, edgecolor='none', label="95% confidence interval")
        ax_twinx.fill_between(df_ci_mean.index, df_ci_mean['ci'] - df_ci_std['ci'], df_ci_mean['ci'] + df_ci_std['ci'], color='b', alpha=0.2, edgecolor='none', label="±1 standard deviation")

        # Customize ticks and labels
        ax.set_xlim([1, 365])
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticklabels([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticks([15, 46, 74, 105, 135, 166, 196, 225, 258, 288, 319, 349], minor=True)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        ax.set_ylim([-30, 20])
        smb_yticks = [-30, -20, -10, 0, 10, 20]
        ax.set_yticks(smb_yticks)
        ax.set_yticks([-25, -15, -5, 5, 15], minor=True)
        ax.set_ylabel('SMB (mm w.e.)', color='r', fontsize=7)
        ax.yaxis.set_label_coords(-0.078, 0.5)

        ax_twinx.set_ylim([0, 1])
        sif_yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax_twinx.set_yticks(sif_yticks)
        ax_twinx.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax_twinx.set_ylabel('Sea ice concentration (%)', color='b', fontsize=7)
        ax_twinx.yaxis.set_label_coords(1.071, 0.5)

        # Align both y-axis to have the same zero line
        _align_secondary_yaxis(ax, ax_twinx, smb_yticks, sif_yticks)

        # Set y-tick labels after alignment to ensure they are correct
        ax.set_yticklabels(smb_yticks)
        ax_twinx.set_yticklabels([int(v * 100) for v in sif_yticks])

        # Customize spines
        ax.spines['left'].set_color('r')
        ax_twinx.spines['right'].set_color('b')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_bounds(0, 1)  # Adjust the bounds as needed
        ax_twinx.spines['right'].set_bounds(0, 1)
        ax.spines['left'].set_bounds(-30, 20)
        ax.spines['bottom'].set_bounds(1, 365)
        ax.spines['bottom'].set_position(('outward', 3))
        ax.spines['left'].set_position(('outward', 3))
        ax_twinx.spines['right'].set_position(('outward', 3))
        [i.set_linewidth(0.4) for i in ax.spines.values()]
        [i.set_linewidth(0.4) for i in ax_twinx.spines.values()]

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=7)
        ax.tick_params(axis='y', which='major', labelcolor='r', color='r', **ytick_params)
        ax_twinx.tick_params(axis='y', which='major', labelcolor='b', color='b', **ytick_params)
        ax.tick_params(axis='x', which='major', direction='out', width=0.5, length=5, pad=2, labelsize=7)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=7)
        ytick_minor_params = dict(axis='y', which='minor', direction='out', width=0.5, length=2.5, pad=2)
        ax.tick_params(color='r', **ytick_minor_params)
        ax_twinx.tick_params(color='b', **ytick_minor_params)

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
        targets = dict(left=220, right=700, top=30, bottom=150, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Plot Greenland map
        # ax = fig.add_axes([0.65, 0.53, 0.6, 0.5])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax = fig.add_axes([0.65, 0.48, 0.6, 0.5])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_map = fig.add_axes([0.72, 0.58, 0.5, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        [i.set_visible(False) for i in ax_map.spines.values()]
        ax_map.axis('off')
        variable = 'sif'
        lonlat = _get_latlon_meshgrid(variable=variable)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax_map.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=5)

        # Land sea mask
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude(variable=variable)
        ax_map.contour(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, linewidths=0.05, zorder=9)

        # Plot station
        # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]
        ax_map.plot(station_rlon, station_rlat, marker='x', color='r', markeredgewidth=0.5, markersize=2, zorder=12, label='Closest point to Freya')

        # Read the sea ice buffer masks at different radii around the closest point to Freya in the glacier mask at 5 km resolution
        ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        ci_buffer_after = sea_ice_buffer_da.sel(radius=100)
        ci_buffer = ci_buffer_after - ci_buffer_before

        radius_int_x = 10
        radius_int_y = 120
        binary_mask = xr.where(ci_buffer > 0.5, 1, 0)
        cs = ax_map.contour(lonlat[0], lonlat[1], binary_mask, levels=[0.5], colors='grey', linewidths=0.5,  linestyles='dashed', zorder=1)

        # Draw rectangle for zoomed-in map
        zoom_xlim = (4, 5.5)
        zoom_ylim = (2, 4)
        rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),                     # lower-left corner
                         zoom_xlim[1] - zoom_xlim[0],                      # width
                         zoom_ylim[1] - zoom_ylim[0],                      # height
                         linewidth=0.5, edgecolor='k', facecolor='none', zorder=15)
        ax_map.add_patch(rect)

        [i.set_visible(False) for i in ax_map.spines.values()]  # Set visible spines to have a linewidth of 0.3
        ax_map.set_xlim([-7.5, 12.5])
        # ax.set_ylim([-13, 13])
        ax_map.set_ylim([-12, 11.6])
        ax_map.axis('off')
        ax_map.set_aspect('equal', adjustable='box')

        # Plot zoomed-in map
        # ax = fig.add_subplot(gs[0, 1])
        # ax = fig.add_axes([0.48, 0.53, 0.55, 0.45])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_zoom = fig.add_axes([0.79, 0.27, 0.3, 0.3])   # [left, bottom, width, height] in figure coordinates (0 to 1)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax_zoom.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=0)

        # Land sea mask
        # binary_mask = xr.where(land_sea_mask > 0.5, 1, 0)
        # ax_zoom.contour(lonlat[0], lonlat[1], binary_mask, levels=[0.5], colors='black', linewidths=0.5, zorder=9)

        # Plot station
        ax_zoom.plot(station_rlon, station_rlat, marker='x', color='red', markersize=3,  markeredgewidth=0.5, zorder=12, label='Closest point to Freya')

        # Plot sea ice buffer mask at 100-km radius
        colors_buffer = [(169, 169, 169), (65, 105, 225)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        binary_mask = xr.where(ci_buffer > 0.5, 1, np.nan)
        # ax.contour(lonlat[0], lonlat[1], sea_ice_buffer_da.sel(radius=100).data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)
        ax_zoom.contourf(lonlat[0], lonlat[1], binary_mask.data, cmap=cmap_buffer, norm=norm_buffer, zorder=0)# alpha=0.8)

        # circle_scale = 0.05 / 5.5
        circle_scale = 0.05 / 5.8
        _curved_circle_text(ax_zoom, station_rlon, station_rlat, 100, '100 km', scale=circle_scale, tune=55, angle_center=0)

        radius_int_x = 10
        radius_int_y = 120
        rlat_rlon_1km = 0.05 / 5.5
        # add_radius_x = rlat_rlon_1km * radius_int_x # rlon in 100 km, 200 km ...
        # add_radius_y = rlat_rlon_1km * radius_int_y
        # # ax.text(station_rlon + 0.7, station_rlat + 0.1, '100 km', size=4, color='k', path_effects=[pe.withStroke(linewidth=1, foreground='white')], zorder=12)

        # Add young sound fjord mask
        ys_mask = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        colors_buffer = [(169, 169, 169), (0, 0, 205)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        cs_young_sound = ax_zoom.contour(lonlat[0], lonlat[1], ys_mask, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)

        ax_zoom.set_xlim(zoom_xlim)
        ax_zoom.set_ylim(zoom_ylim)
        ax_zoom.set_aspect('equal', adjustable='box')

        xticks = [zoom_xlim[0]]
        xticklables_km = ['', 50, 100, 150, ]
        # xticklables_km = ['', 50, 100, 150, 200, 250]
        for distance in xticklables_km[1:]:
            xticks.append(zoom_xlim[0] + rlat_rlon_1km * distance)
        ax_zoom.set_xticks(xticks)
        ax_zoom.set_xticklabels(xticklables_km)

        yticks = [zoom_ylim[0]]
        yticklables_km = ['', 50, 100, 150, 200]
        for distance in yticklables_km[1:]:
            yticks.append(zoom_ylim[0] + rlat_rlon_1km * distance)
        ax_zoom.set_yticks(yticks)
        ax_zoom.set_yticklabels(yticklables_km)

        # Manually add zero at lower left
        ax_zoom.text(zoom_xlim[0] - 0.05, zoom_ylim[0] - 0.05, '0', ha='right', va='top', fontsize=5)
        ax_zoom.tick_params(axis='both', which='major', direction='inout', pad=0.8, length=4, width=0.5, labelsize=5)
        ax_zoom.text(0.5, -0.2, 'km', ha='center', va='center', fontsize=5, transform=ax_zoom.transAxes)
        ax_zoom.text(-0.35, 0.5, 'km', ha='center', va='center', fontsize=5, rotation=90, transform=ax_zoom.transAxes)

        [i.set_linewidth(0.3) for i in ax_zoom.spines.values()]

        # Connect the zoomed-in rectangle with the main map using a ConnectionPatch
        # Main rectangle top-right corner
        xyA = (zoom_xlim[1], zoom_ylim[0])
        # Zoom inset bottom-left corner
        xyB = (zoom_xlim[1], zoom_ylim[1])
        con = ConnectionPatch(xyA=xyA, coordsA=ax_map.transData, xyB=xyB, coordsB=ax_zoom.transData, color='k', lw=0.3, ls=':', zorder=30)
        fig.add_artist(con)

        # Main rectangle top-right corner
        xyA = (zoom_xlim[0], zoom_ylim[0])
        # Zoom inset bottom-left corner
        xyB = (zoom_xlim[0], zoom_ylim[1])
        con = ConnectionPatch(xyA=xyA, coordsA=ax_map.transData, xyB=xyB, coordsB=ax_zoom.transData, color='k', lw=0.3, ls=':', zorder=30)
        fig.add_artist(con)

        logger.info(f"Saving plot of 'plot_annual_cycle_young_sound_sea_ice_buffers_versus_freya_smb'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_annual_cycle_young_sound_sea_ice_buffers_versus_freya_smb.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def plot_annual_cycle_sea_ice_100km_vs_freya_smb_vs_z500():
        """ Plot the annual cycle of sea ice buffers versus Freya SMB and Z500."""
        # %%
        # Mask path for the extracted RACMO variables
        masked_variables_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var')

        # Plot Freya Glacier SMB from RACMO2.3 5 km
        smb_mod_all_daily = pd.read_csv(f"{masked_variables_path}/from_freya_glacier/smb_masked_freya_1km.csv", header=0)
        smb_mod_all_daily['time'] = pd.to_datetime(smb_mod_all_daily['time'])
        smb_mod_all_daily.set_index('time', inplace=True)
        time_delta = smb_mod_all_daily.index[1] - smb_mod_all_daily.index[0]
        smb_mod_all_daily = smb_mod_all_daily.resample(time_delta).mean()
        smb_mod_daily = pd.DataFrame(smb_mod_all_daily.loc['1990':'2019', 'smb_whole_racmo'])
        df_smb_mean, df_smb_std, df_smb_margin_of_error = _get_mean_std_margin_of_error(smb_mod_daily, 'smb_whole_racmo', critical_value=1.96)

        # # Calculate 7days running mean
        # df_smb_mean = _calculate_7day_smooth_365days(df_smb_mean, 'smb_whole_racmo')
        # df_smb_std = _calculate_7day_smooth_365days(df_smb_std, 'smb_whole_racmo')
        # df_smb_margin_of_error = _calculate_7day_smooth_365days(df_smb_margin_of_error, 'smb_whole_racmo')

        # Plot Sea ice concentration from RACMO2.3 5 km
        ci_masked_5km_variables = pd.read_csv(f"{masked_variables_path}/from_ci/masked_5km_variables_within_100_sea_ice.csv", parse_dates=['time'], index_col='time')
        # ci_masked_5km_variables = pd.read_csv(f"{masked_variables_path}/from_ci/masked_5km_variables_between_YS_100_sea_ice.csv", parse_dates=['time'], index_col='time')
        # ci_masked_5km_variables = pd.read_csv(f"{masked_variables_path}/from_ci/masked_5km_variables_within_greenland_sea_sea_ice.csv", parse_dates=['time'], index_col='time')
        freq = ci_masked_5km_variables.index.to_series().diff().mode()[0]
        ci_masked_5km_variables = ci_masked_5km_variables.resample(freq).mean()
        ci_in_100km_ci_daily = pd.DataFrame(ci_masked_5km_variables.loc['1990':'2019', 'ci'])
        df_ci_mean, df_ci_std, df_ci_margin_of_error = _get_mean_std_margin_of_error(ci_in_100km_ci_daily, 'ci', critical_value=1.96)

        # # Calculate 7days running mean
        # df_ci_mean = _calculate_7day_smooth_365days(df_ci_mean, 'ci')
        # df_ci_std = _calculate_7day_smooth_365days(df_ci_std, 'ci')
        # df_ci_margin_of_error = _calculate_7day_smooth_365days(df_ci_margin_of_error, 'ci')

        # Plot z500
        # z500_masked_5km_variables = pd.read_csv(f"{masked_variables_path}/from_z500/masked_z500hPa_east_greenland.csv", parse_dates=['time'], index_col='time')
        z500_masked_5km_variables = pd.read_csv(f"{masked_variables_path}/from_z500/masked_z500hPa_greenland_sea.csv", parse_dates=['time'], index_col='time')
        freq = z500_masked_5km_variables.index.to_series().diff().mode()[0]
        z500_masked_5km_variables = z500_masked_5km_variables.resample(freq).mean()
        z500_daily = pd.DataFrame(z500_masked_5km_variables.loc['1990':'2019', 'z500'])
        df_z500_mean, df_z500_std, df_z500_margin_of_error = _get_mean_std_margin_of_error(z500_daily, 'z500', critical_value=1.96)

        # # Calculate 7days running mean
        # df_z500_mean = _calculate_7day_smooth_365days(df_z500_mean, 'z500')
        # df_z500_std = _calculate_7day_smooth_365days(df_z500_std, 'z500')
        # df_z500_margin_of_error = _calculate_7day_smooth_365days(df_z500_margin_of_error, 'z500')

        vars = ['ci', 'smb', 'z500']
        colors = [(117, 112, 179), (217, 95, 2), (27, 158, 119)]
        rgb_hex = []
        for color in colors:
            rgb_hex.append(_rgb_to_hex(color))
        var_styles = {var: {'color': color, 'ls': '-', 'lw': 0.9} for var, color in zip(vars, rgb_hex)}
        var_styles_error = {var: {'alpha': 0.4, 'edgecolor':'none'} for var in vars}
        var_styles_std = {var: {'alpha': 0.2, 'edgecolor':'none'} for var in vars}

        # fig, ax = plt.subplots(figsize=(5, 2.5), dpi=600)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=600)
        ax_twinx = ax.twinx()
        ax_twinx.set_frame_on(True)
        ax_twinx.patch.set_visible(False) # Prevents ax from hiding ax_twinx
        for sp in ax_twinx.spines.values():
            sp.set_visible(False)
        ax_twinx.spines['right'].set_visible(True)

        ax_twinx2 = ax.twinx()
        ax_twinx2.set_frame_on(True)
        ax_twinx2.patch.set_visible(False) # Prevents ax from hiding ax_twinx2
        for sp in ax_twinx2.spines.values():
            sp.set_visible(False)
        ax_twinx2.spines['right'].set_visible(True)

        # Plot SMB
        ax.plot(df_smb_mean.index, df_smb_mean['smb_whole_racmo'], color=var_styles['smb']['color'], lw=var_styles['smb']['lw'], zorder=12)
        ax.fill_between(df_smb_mean.index, df_smb_mean['smb_whole_racmo'] - df_smb_margin_of_error['smb_whole_racmo'], df_smb_mean['smb_whole_racmo'] + df_smb_margin_of_error['smb_whole_racmo'], color=var_styles['smb']['color'], alpha=var_styles_error['smb']['alpha'], edgecolor=var_styles_error['smb']['edgecolor'], label="95% confidence interval")
        ax.fill_between(df_smb_mean.index, df_smb_mean['smb_whole_racmo'] - df_smb_std['smb_whole_racmo'], df_smb_mean['smb_whole_racmo'] + df_smb_std['smb_whole_racmo'], color=var_styles['smb']['color'], alpha=var_styles_std['smb']['alpha'], edgecolor=var_styles_std['smb']['edgecolor'], label="±1 standard deviation")

        # Plot Sea ice
        ax_twinx.plot(df_ci_mean.index, df_ci_mean['ci'], color=var_styles['ci']['color'], lw=var_styles['ci']['lw'], zorder=12)
        ax_twinx.fill_between(df_ci_mean.index, df_ci_mean['ci'] - df_ci_margin_of_error['ci'], df_ci_mean['ci'] + df_ci_margin_of_error['ci'], color=var_styles['ci']['color'], alpha=var_styles_error['ci']['alpha'], edgecolor=var_styles_error['ci']['edgecolor'], label="95% confidence interval")
        ax_twinx.fill_between(df_ci_mean.index, df_ci_mean['ci'] - df_ci_std['ci'], df_ci_mean['ci'] + df_ci_std['ci'], color=var_styles['ci']['color'], alpha=var_styles_std['ci']['alpha'], edgecolor=var_styles_std['ci']['edgecolor'], label="±1 standard deviation")

        # Plot Z500
        ax_twinx2.plot(df_z500_mean.index, df_z500_mean['z500'], color=var_styles['z500']['color'], lw=var_styles['z500']['lw'], zorder=12)
        ax_twinx2.fill_between(df_z500_mean.index, df_z500_mean['z500'] - df_z500_margin_of_error['z500'], df_z500_mean['z500'] + df_z500_margin_of_error['z500'], color=var_styles['z500']['color'], alpha=var_styles_error['z500']['alpha'], edgecolor=var_styles_error['z500']['edgecolor'], label="95% confidence interval")
        ax_twinx2.fill_between(df_z500_mean.index, df_z500_mean['z500'] - df_z500_std['z500'], df_z500_mean['z500'] + df_z500_std['z500'], color=var_styles['z500']['color'], alpha=var_styles_std['z500']['alpha'], edgecolor=var_styles_std['z500']['edgecolor'], label="±1 standard deviation")

        # Customize ticks and labels
        ax.set_xlim([1, 365])
        ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticklabels([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 365])
        ax.set_xticks([15, 46, 74, 105, 135, 166, 196, 225, 258, 288, 319, 349], minor=True)
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], minor=True)

        ax.set_ylim([-30, 20])
        smb_yticks = [-30, -20, -10, 0, 10, 20]
        ax.set_yticks(smb_yticks)
        ax.set_yticks([-25, -15, -5, 5, 15], minor=True)
        ax.set_ylabel(r'SMB$_\mathdefault{FRG}$ (mm w.e.)', color=var_styles['smb']['color'], fontsize=7)
        # ax.yaxis.set_label_coords(-0.078, 0.5)
        ax.yaxis.set_label_coords(-0.07, 0.5)

        ax_twinx.set_ylim([0, 1])
        sif_yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
        ax_twinx.set_yticks(sif_yticks)
        ax_twinx.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax_twinx.set_ylabel(r'SIC$_\mathdefault{100km}$ (%)', color=var_styles['ci']['color'], fontsize=7)
        ax_twinx.yaxis.set_label_coords(1.071, 0.5)

        ax_twinx2.set_ylim([5100, 5600])
        z500_yticks = [5100, 5200, 5300, 5400, 5500, 5600]
        ax_twinx2.set_yticks(z500_yticks)
        ax_twinx2.set_yticks([5150, 5250, 5350, 5450, 5550], minor=True)
        ax_twinx2.set_ylabel(r'Z500$_\mathdefault{GrSea}$ (m)', color=var_styles['z500']['color'], fontsize=7)
        ax_twinx2.yaxis.set_label_coords(1.195, 0.5)

        # Align both y-axis to have the same zero line
        _align_secondary_yaxis(ax, ax_twinx, smb_yticks, sif_yticks)

        # Set y-tick labels after alignment to ensure they are correct
        ax.set_yticklabels(smb_yticks)
        ax_twinx.set_yticklabels([int(v * 100) for v in sif_yticks])

        # Customize spines
        ax.spines['left'].set_color(var_styles['smb']['color'])
        ax_twinx.spines['right'].set_color(var_styles['ci']['color'])
        ax_twinx2.spines['right'].set_color(var_styles['z500']['color'])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.spines['left'].set_bounds(0, 1)  # Adjust the bounds as needed
        ax_twinx.spines['right'].set_bounds(0, 1)
        ax.spines['left'].set_bounds(-30, 20)
        ax.spines['bottom'].set_bounds(1, 365)
        ax.spines['bottom'].set_position(('outward', 3))
        ax.spines['left'].set_position(('outward', 3))
        ax_twinx.spines['right'].set_position(('outward', 3))
        ax_twinx2.spines['right'].set_position(('outward', 30))

        [i.set_linewidth(0.4) for i in ax.spines.values()]
        [i.set_linewidth(0.4) for i in ax_twinx.spines.values()]
        [i.set_linewidth(0.4) for i in ax_twinx2.spines.values()]

        # Customize ticks
        ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=7)
        ax.tick_params(axis='y', which='major', labelcolor=var_styles['smb']['color'], color=var_styles['smb']['color'], **ytick_params)
        ax_twinx.tick_params(axis='y', which='major', labelcolor=var_styles['ci']['color'], color=var_styles['ci']['color'], **ytick_params)
        ax_twinx2.tick_params(axis='y', which='major', labelcolor=var_styles['z500']['color'], color=var_styles['z500']['color'], **ytick_params)
        ax.tick_params(axis='x', which='major', direction='out', width=0.5, length=5, pad=2, labelsize=7)
        ax.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=7)
        ytick_minor_params = dict(axis='y', which='minor', direction='out', width=0.5, length=2.5, pad=2)
        ax.tick_params(color=var_styles['smb']['color'], **ytick_minor_params)
        ax_twinx.tick_params(color=var_styles['ci']['color'], **ytick_minor_params)
        ax_twinx2.tick_params(color=var_styles['z500']['color'], **ytick_minor_params)

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
        targets = dict(left=220, right=500, top=30, bottom=1100, hspace=0, wspace=0)
        # Hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Plot Greenland map
        # ax_map = fig.add_axes([0.72, 0.58, 0.5, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax_map = fig.add_axes([0.01, 0.01, 0.5, 0.4])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax_map = fig.add_axes([-0.05, 0.009, 0.5, 0.39])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax_map = fig.add_axes([0.01, 0.009, 0.5, 0.39])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_map = fig.add_axes([0.1, 0.009, 0.5, 0.39])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        [i.set_visible(False) for i in ax_map.spines.values()]
        ax_map.axis('off')
        variable = 'sif'
        lonlat = _get_latlon_meshgrid(variable=variable)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax_map.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=5)

        # Land sea mask
        land_sea_mask, racmo_latitude, racmo_longitude, colors_landseamask, cmap_landseamask, bounds_landseamask, norm_landseamask = _get_greenland_mask_latitude_longitude(variable=variable)
        ax_map.contour(lonlat[0], lonlat[1], land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, linewidths=0.05, zorder=9)

        # Z500 mask
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/')
        # z500_mask = xr.open_dataarray(f"{mask_file_path}/z500hPa_mask.nc")
        z500_mask = xr.open_dataarray(f"{mask_file_path}/greenland_sea_mask_5km_updated_coordinates.nc")
        # z500_mask = z500_mask.where(z500_mask == 1, np.nan)
        # colors_buffer = [(169, 169, 169), (27, 158, 119)]
        # cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        # bounds_buffer = np.array([-0.01, 0.01, 1.5])
        # norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # ax_map.contourf(lonlat[0], lonlat[1], z500_mask, cmap=cmap_buffer, norm=norm_buffer, alpha=0.25, zorder=6)

        colors_buffer = [(169, 169, 169), (169, 169, 169)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        ax_map.contour(lonlat[0], lonlat[1], z500_mask, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.2, linestyles='-.', zorder=12)# alpha=0.8)

        # Add Iceland and Svalbard
        iceland_svalbard_mask = xr.open_dataarray(f"{mask_file_path}/iceland_svalbard_mask.nc")
        colors_buffer = [(169, 169, 169), (169, 169, 169)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        ax_map.contourf(lonlat[0], lonlat[1], iceland_svalbard_mask, cmap=cmap_buffer, norm=norm_buffer, zorder=7)

        # Plot station
        # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        mask_file_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks')
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]
        ax_map.plot(station_rlon, station_rlat, marker='x', color='r', markeredgewidth=0.5, markersize=2, label='Closest point to Freya', zorder=12)

        # Read the sea ice buffer masks at different radii around the clo`sest point to Freya in the glacier mask at 5 km resolution
        # ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buffers_on_racmo5km.nc")['sea_ice_buffer']
        ci_buffer = sea_ice_buffer_da.sel(radius=100)
        # ci_buffer_before = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        # sea_ice_buffer_da = xr.open_dataset(f"{mask_file_path}/ci_all_buff`ers_on_racmo5km.nc")['sea_ice_buffer']
        # ci_buffer_after = sea_ice_buffer_da.sel(radius=100)
        # ci_buffer = ci_buffer_after - ci_buffer_before

        # radius_int_x = 10
        # radius_int_y = 120
        # binary_mask = xr.where(ci_buffer > 0.5, 1, 0)
        # cs = ax_map.contour(lonlat[0], lonlat[1], binary_mask, levels=[0.5], colors='grey', linewidths=0.5,  linestyles='dashed', zorder=14)
        # Plot sea ice buffer mask at 100-km radius
        colors_buffer = [(169, 169, 169), (0, 0, 0)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # binary_mask = xr.where(ci_buffer > 0.5, 1, np.nan)
        # ax.contour(lonlat[0], lonlat[1], sea_ice_buffer_da.sel(radius=100).data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)
        # ax_zoom.contourf(lonlat[0], lonlat[1], binary_mask.data, cmap=cmap_buffer, norm=norm_buffer, zorder=0)# alpha=0.8)
        ax_map.contour(lonlat[0], lonlat[1], ci_buffer.data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=12)# alpha=0.8)

        # Draw rectangle for zoomed-in map
        zoom_xlim = (4, 5.5)
        zoom_ylim = (2, 4)
        rect = Rectangle((zoom_xlim[0], zoom_ylim[0]),                     # lower-left corner
                         zoom_xlim[1] - zoom_xlim[0],                      # width
                         zoom_ylim[1] - zoom_ylim[0],                      # height
                         linewidth=0.5, edgecolor='k', facecolor='none', zorder=15)
        ax_map.add_patch(rect)

        [i.set_visible(False) for i in ax_map.spines.values()]  # Set visible spines to have a linewidth of 0.3
        ax_map.set_xlim([-7.5, 12.5])
        # ax.set_ylim([-13, 13])
        ax_map.set_ylim([-12, 11.6])
        ax_map.axis('off')
        ax_map.set_aspect('equal', adjustable='box')

        # Plot zoomed-in map
        # ax = fig.add_subplot(gs[0, 1])
        # ax_zoom = fig.add_axes([0.79, 0.27, 0.3, 0.3])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax_zoom = fig.add_axes([0.35, 0.09, 0.3, 0.3])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        # ax_zoom = fig.add_axes([0.35, 0.09, 0.3, 0.3])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_zoom = fig.add_axes([0.45, 0.09, 0.3, 0.3])   # [left, bottom, width, height] in figure coordinates (0 to 1)

        # Promice mask
        promice_mask_ramo_5km, cmap_promice_mask, bounds_promice_mask, norm_promice_mask = _get_promice_mask_params(variable=variable)
        ax_zoom.contourf(lonlat[0], lonlat[1], promice_mask_ramo_5km, cmap=cmap_promice_mask, norm=norm_promice_mask, zorder=0)

        # Plot station
        ax_zoom.plot(station_rlon, station_rlat, marker='x', color='red', markersize=3,  markeredgewidth=0.5, zorder=12, label='Closest point to Freya')

        # Plot sea ice buffer mask at 100-km radius
        colors_buffer = [(169, 169, 169), (0, 0, 0)]
        cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        bounds_buffer = np.array([-0.01, 0.01, 1.5])
        norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # binary_mask = xr.where(ci_buffer > 0.5, 1, np.nan)
        # ax.contour(lonlat[0], lonlat[1], sea_ice_buffer_da.sel(radius=100).data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)
        # ax_zoom.contourf(lonlat[0], lonlat[1], binary_mask.data, cmap=cmap_buffer, norm=norm_buffer, zorder=0)# alpha=0.8)
        ax_zoom.contour(lonlat[0], lonlat[1], ci_buffer.data, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=12)# alpha=0.8)

        # colors_buffer = [(169, 169, 169), (0, 0, 0)]
        # cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        # bounds_buffer = np.array([-0.01, 0.01, 1.5])
        # norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # cs_young_sound = ax_zoom.contour(lonlat[0], lonlat[1], ys_mask, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)

        # circle_scale = 0.05 / 5.5
        # circle_scale = 0.05 / 5.8
        circle_scale = 0.05 / 6
        _curved_circle_text(ax_zoom, station_rlon, station_rlat, 100, '100 km', scale=circle_scale, tune=55, angle_center=0)

        radius_int_x = 10
        radius_int_y = 120
        rlat_rlon_1km = 0.05 / 5.5
        # add_radius_x = rlat_rlon_1km * radius_int_x # rlon in 100 km, 200 km ...
        # add_radius_y = rlat_rlon_1km * radius_int_y
        # # ax.text(station_rlon + 0.7, station_rlat + 0.1, '100 km', size=4, color='k', path_effects=[pe.withStroke(linewidth=1, foreground='white')], zorder=12)

        # # Add young sound fjord mask
        # ys_mask = xr.open_dataset(f"{mask_file_path}/young_sound_mask.nc")['young_sound_mask']
        # colors_buffer = [(169, 169, 169), (0, 0, 205)]
        # cmap_buffer = fp.make_cmap(colors_buffer, bit=True)
        # bounds_buffer = np.array([-0.01, 0.01, 1.5])
        # norm_buffer = mpl.colors.BoundaryNorm(bounds_buffer, cmap_buffer.N)
        # cs_young_sound = ax_zoom.contour(lonlat[0], lonlat[1], ys_mask, cmap=cmap_buffer, norm=norm_buffer, levels=bounds_buffer, linewidths=0.5, linestyles='dashed', zorder=1)

        ax_zoom.set_xlim(zoom_xlim)
        ax_zoom.set_ylim(zoom_ylim)
        ax_zoom.set_aspect('equal', adjustable='box')

        xticks = [zoom_xlim[0]]
        xticklables_km = ['', 50, 100, 150, ]
        # xticklables_km = ['', 50, 100, 150, 200, 250]
        for distance in xticklables_km[1:]:
            xticks.append(zoom_xlim[0] + rlat_rlon_1km * distance)
        ax_zoom.set_xticks(xticks)
        ax_zoom.set_xticklabels(xticklables_km)

        yticks = [zoom_ylim[0]]
        yticklables_km = ['', 50, 100, 150, 200]
        for distance in yticklables_km[1:]:
            yticks.append(zoom_ylim[0] + rlat_rlon_1km * distance)
        ax_zoom.set_yticks(yticks)
        ax_zoom.set_yticklabels(yticklables_km)

        # Manually add zero at lower left
        ax_zoom.text(zoom_xlim[0] - 0.05, zoom_ylim[0] - 0.05, '0', ha='right', va='top', fontsize=6)
        ax_zoom.tick_params(axis='both', which='major', direction='inout', pad=0.8, length=4, width=0.5, labelsize=6)
        ax_zoom.text(0.5, -0.15, 'km', ha='center', va='center', fontsize=6, transform=ax_zoom.transAxes)
        ax_zoom.text(-0.25, 0.5, 'km', ha='center', va='center', fontsize=6, rotation=90, transform=ax_zoom.transAxes)

        [i.set_linewidth(0.3) for i in ax_zoom.spines.values()]

        # Connect the zoomed-in rectangle with the main map using a ConnectionPatch
        # Main rectangle top-right corner
        xyA = (zoom_xlim[1], zoom_ylim[1])
        # Zoom inset bottom-left corner
        xyB = (zoom_xlim[0], zoom_ylim[1])
        con = ConnectionPatch(xyA=xyA, coordsA=ax_map.transData, xyB=xyB, coordsB=ax_zoom.transData, color='k', lw=0.3, ls=':', zorder=30)
        fig.add_artist(con)

        # # Main rectangle top-right corner
        # xyA = (zoom_xlim[1], zoom_ylim[0])
        # # Zoom inset bottom-left corner
        # xyB = (zoom_xlim[1], zoom_ylim[1])
        # con = ConnectionPatch(xyA=xyA, coordsA=ax_map.transData, xyB=xyB, coordsB=ax_zoom.transData, color='k', lw=0.3, ls=':', zorder=30)
        # fig.add_artist(con)

        # Main rectangle top-right corner
        xyA = (zoom_xlim[1], zoom_ylim[0])
        # Zoom inset bottom-left corner
        xyB = (zoom_xlim[0], zoom_ylim[0])
        con = ConnectionPatch(xyA=xyA, coordsA=ax_map.transData, xyB=xyB, coordsB=ax_zoom.transData, color='k', lw=0.3, ls=':', zorder=30)
        fig.add_artist(con)

        # Legend items
        legend_param = dict(loc='center', fontsize=7, frameon=False, handletextpad=0.5)

        # colors_promice_mask = [(244, 164, 96), (95, 155, 230), (135, 206, 235), (27, 158, 119)]
        colors_promice_mask = [(178, 223, 138), (95, 155, 230), (135, 206, 235), (27, 158, 119)]
        # Convert RGB (0–255) to matplotlib RGB (0–1)
        colors_promice_mask = [tuple(c / 255 for c in color) for color in colors_promice_mask]

        grIS_label = Patch(facecolor=colors_promice_mask[2], edgecolor=colors_promice_mask[2], linewidth=0.5, label='Greenland Ice Sheet')
        peri_label = Patch(facecolor=colors_promice_mask[1], edgecolor=colors_promice_mask[1], linewidth=0.5, label='Peripheral Glacier')
        tundra_label = Patch(facecolor=colors_promice_mask[0], edgecolor=colors_promice_mask[0], linewidth=0.5, label='Tundra')

        # grSea_label = Patch(facecolor=(*colors_promice_mask[3], 0.25), edgecolor=(*colors_promice_mask[3], 0.25), linewidth=0.8, antialiased=False, label='Greenland Sea')
        # grSea_label = Patch(facecolor=(*colors_promice_mask[3], 0.25), edgecolor=colors_promice_mask[3], linewidth=0.5, label='Greenland Sea', alpha=0.25)
        grSea_label = Patch(facecolor=(*colors_promice_mask[3], 0.25), edgecolor='none', linewidth=0.8, label='Greenland Sea')

        # ax_legend = fig.add_axes([0.65, 0.2, 0.2, 0.2])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_legend = fig.add_axes([0.73, 0.23, 0.2, 0.2])   # [left, bottom, width, height] in figure coordinates (0 to 1)
        ax_legend.axis('off')  # Hide the axis for the legend subplot
        # ax_legend.legend(handles=[grIS_label, peri_label, tundra_label, grSea_label], bbox_to_anchor=(0.5, 0.1), ncol=1, handlelength=1, handleheight=1, labelspacing=1.3, markerscale=1.5, **legend_param)
        ax_legend.legend(handles=[grIS_label, peri_label, tundra_label], bbox_to_anchor=(0.5, 0.1), ncol=1, handlelength=1, handleheight=1, labelspacing=1.3, markerscale=1.5, **legend_param)

        logger.info(f"Saving plot of 'plot_annual_cycle_sea_ice_100km_vs_freya_smb_vs_z500'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_annual_cycle_sea_ice_100km_vs_freya_smb_vs_z500.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%

           # %%
    def calculate_spearman_correlation_among_z500_smb_sif():
        # %%
        # Read z500 masked data. Masked process is described in class ExtractZ500VariablesFromRacmo of 'FG0_preprocess_data.py' script.
        base_output_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_z500')
        # gph = pd.read_csv(f"{base_output_path}/masked_z500hPa_east_greenland.csv", parse_dates=['time'], index_col='time')
        # Use the masked z500 dataset with rlon slice to include more east Greenland grid points, which is more representative of z500 hPa.
        gph = pd.read_csv(f"{base_output_path}/masked_z500hPa_east_greenland_using_rlon_slice.csv", parse_dates=['time'], index_col='time')
        gph.index = pd.to_datetime(gph.index)
        # Resample to uniform frequency
        freq = gph.index.to_series().diff().mode()[0]
        gph_day = gph.resample(freq).mean()

        # # Read the GBI dataset from Hanna et al. (2021) to compare with the masked z500 dataset.
        # # The GBI dataset is used to compare with the masked z500 dataset to check if the masked z500 dataset can capture the GBI signal, which is a well-known atmospheric circulation pattern over Greenland.
        # base_output_path = Path('/nas/data/processed/people/sos/third_paper_data')
        # gph = pd.read_csv(f"{base_output_path}/GBI_1990_2020.csv")
        # # assuming your DataFrame is called gph
        # gph['time'] = pd.to_datetime(gph[['Years', 'Months', 'Days']])
        # gph_day = gph.set_index('time')
        # gph_day = gph_day.rename(columns={'GBI': 'z500'})
        # gph_day = pd.DataFrame(gph_day['z500'])

        # Read the masked 5km racmo variables within 100-km sea ice concentration
        masked_5km_variables_100km_sif_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        masked_5km_variables_100km_sif = pd.read_csv(f"{masked_5km_variables_100km_sif_path}/masked_5km_variables_within_100km_sea_ice.csv", parse_dates=['time'], index_col='time')
        masked_5km_variables_100km_sif.index = pd.to_datetime(masked_5km_variables_100km_sif.index)
        # Resample to uniform frequency
        freq = masked_5km_variables_100km_sif.index.to_series().diff().mode()[0]
        masked_5km_variables_100km_sif = masked_5km_variables_100km_sif.resample(freq).mean()
        sif_day = pd.DataFrame(masked_5km_variables_100km_sif['ci'])
        sif_day = sif_day.rename(columns={'ci': 'sif'})

        # Read the Freya SMB
        smb_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
        smb_day = pd.read_csv(f"{smb_path}/smb_masked_freya_1km.csv", parse_dates=['time'], index_col='time')
        smb_day.index = pd.to_datetime(smb_day.index)
        freq = smb_day.index.to_series().diff().mode()[0]
        smb_day = smb_day.resample(freq).mean()
        smb_day = pd.DataFrame(smb_day['smb_point_racmo'])
        smb_day = smb_day.rename(columns={'smb_point_racmo': 'smb'})

        # Combine all variables into one DataFrame and drop rows with NaN values to ensure we have complete data for correlation analysis.
        df_all = pd.concat([gph_day, sif_day, smb_day], axis=1).dropna()

        # months_list = (('May', [5]), ('June', [6]), ('July', [7]), ('August', [8]), ('September', [9]), ('MJJAS', [5, 6, 7, 8, 9]),('MJJAS_deseasonalize', [5, 6, 7, 8, 9]))
        months_list = (('May', [5]), ('June', [6]), ('July', [7]), ('Aug', [8]), ('Sep', [9]), ('MJJAS', [5, 6, 7, 8, 9]),)
        months_name = [month[0] for month in months_list]
        correlation_stats = ['Trend', 'Detrended']
        miindex = pd.MultiIndex.from_product([months_name, correlation_stats])
        correlation_pairs = ['smb_minus_z500_sif_minus_z500', 'smb_sif', 'smb_z500', 'z500_sif']
        # DataFrame to store Spearman correlation using trend and detrend data
        spearman_corr_df = pd.DataFrame(columns=miindex, index=correlation_pairs)
        spearman_corr_df_pval = pd.DataFrame(columns=miindex, index=correlation_pairs)
        spearman_corr_df_epval = pd.DataFrame(columns=miindex, index=correlation_pairs)
        # To draw a table with correlation values and confidence intervals, we create another DataFrame to store the formatted strings for each correlation pair and month.
        spearman_corr_df_table = pd.DataFrame(columns=miindex, index=correlation_pairs)
        spearman_corr_df_table_confint = pd.DataFrame(columns=miindex, index=correlation_pairs)
        for month_name, month in months_list:
            df_month = df_all[df_all.index.month.isin(month)]
            # gph_month = gph_day[gph_day.index.month.isin(month)]
            # sif_month = sif_day[sif_day.index.month.isin(month)]
            # smb_month = smb_day[smb_day.index.month.isin(month)]
            # df_month = pd.concat([gph_month, sif_month, smb_month], axis=1).dropna()

            # Detrend
            if month_name == 'MJJAS':
                # Deseasonalize and detrend
                # deseasonalized = df_month.groupby(df_month.index.month).transform(lambda x: x - x.mean())
                climatology = df_month.groupby(df_month.index.dayofyear).transform('mean')
                deseasonalized = df_month - climatology

                df_month_detrend = pd.DataFrame(index=deseasonalized.index)
                for var in deseasonalized.columns:
                    df_month_detrend[var] = signal.detrend(deseasonalized[var])
            else:
                df_month_detrend = pd.DataFrame(index=df_month.index)
                for var in df_month.columns:
                    df_month_detrend[var] = signal.detrend(df_month[var])

            for var1, var2 in [('smb_minus_z500', 'sif_minus_z500'), ('smb', 'sif'), ('smb', 'z500'), ('z500', 'sif')]:
                # ---------------------------------------
                # Trend Spearman correlation
                # ---------------------------------------
                if (var1 == 'smb_minus_z500') & (var2 == 'sif_minus_z500'): # Calculate partial correlation
                    spearman_corr, spearman_pvalue, lower, upper = _spearmanr_block_bootstrap_partial(df_month, x_column='sif', y_column='smb', covar='z500', block_size=10, n_boot=1000, alpha=5)
                else:
                    spearman_corr, spearman_pvalue, lower, upper = _spearmanr_block_bootstrap(df_month[var1].values, df_month[var2].values, block_size=10, n_boot=1000, alpha=5)
                # --- Apply EDOF Correction ---
                # Calculate N_eff based on the autocorrelation of the residuals ideally,
                # but using the raw series is a conservative approximation.
                if (var1 == 'smb_minus_z500') & (var2 == 'sif_minus_z500'): # Calculate partial correlation
                    k = 1 # number of covariates (Partial Correlation controls) is 1 i.e., z500
                    n_eff = _calculate_edof(df_month['smb'], df_month['sif'])
                else:
                    k = 0
                    n_eff = _calculate_edof(df_month[var1], df_month[var2])
                p_adj = _correct_pvalue(spearman_corr, n_eff, k=k)

                spearman_corr_df.loc[f'{var1}_{var2}', (month_name, 'Trend')] = spearman_corr
                spearman_corr_df_pval.loc[f'{var1}_{var2}', (month_name, 'Trend')] = spearman_pvalue
                spearman_corr_df_epval.loc[f'{var1}_{var2}', (month_name, 'Trend')] = p_adj

                if (np.absolute(np.round(lower, 1)) <= 0.09) & (np.absolute(np.round(upper, 1)) <= 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"(0, 0)"
                elif (np.absolute(np.round(lower, 1)) <= 0.09) & (np.absolute(np.round(upper, 1)) > 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"(0, {upper:.1f})"
                elif (np.absolute(np.round(lower, 1)) > 0.09) & (np.absolute(np.round(upper, 1)) <= 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"({lower:.1f}, 0)"
                elif (np.absolute(np.round(lower, 1)) > 0.09) & (np.absolute(np.round(upper, 1)) > 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"({lower:.1f}, {upper:.1f})"

                # To draw table with correlation values and confidence intervals, we format the string to include the correlation value, significance stars, and confidence intervals.
                if spearman_pvalue <= 0.01:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"{spearman_corr:.1f} $^{{**}}$\n({lower:.1f}, {upper:.1f})"
                elif (spearman_pvalue > 0.01) & (spearman_pvalue <= 0.05):
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Trend')] = f"{spearman_corr:.1f} $^{{*}}$\n({lower:.1f}, {upper:.1f})"
                else:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Trend')] = '–'
                if abs(spearman_corr) < 0.1:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Trend')] = '–'

                # ---------------------------------------
                # Detrended Spearman correlation
                # ---------------------------------------
                if (var1 == 'smb_minus_z500') & (var2 == 'sif_minus_z500'): # Calculate partial correlation
                    spearman_corr, spearman_pvalue, lower, upper = _spearmanr_block_bootstrap_partial(df_month_detrend, x_column='sif', y_column='smb', covar='z500', block_size=10, n_boot=1000, alpha=5)
                else:
                    spearman_corr, spearman_pvalue, lower, upper = _spearmanr_block_bootstrap(df_month_detrend[var1].values, df_month_detrend[var2].values, block_size=10, n_boot=1000, alpha=5)

                # --- Apply EDOF Correction ---
                # Calculate N_eff based on the autocorrelation of the residuals ideally,
                # but using the raw series is a conservative approximation.
                if (var1 == 'smb_minus_z500') & (var2 == 'sif_minus_z500'): # Calculate partial correlation
                    k = 1 # number of covariates (Partial Correlation controls) is 1 i.e., z500
                    n_eff = _calculate_edof(df_month_detrend['smb'], df_month_detrend['sif'])
                else:
                    k = 0
                    n_eff = _calculate_edof(df_month_detrend[var1], df_month_detrend[var2])
                p_adj = _correct_pvalue(spearman_corr, n_eff, k=k)

                spearman_corr_df.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = spearman_corr
                spearman_corr_df_pval.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = spearman_pvalue
                spearman_corr_df_epval.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = p_adj

                if (np.absolute(np.round(lower, 1)) <= 0.09) & (np.absolute(np.round(upper, 1)) <= 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"(0, 0)"
                elif (np.absolute(np.round(lower, 1)) <= 0.09) & (np.absolute(np.round(upper, 1)) > 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"(0, {upper:.1f})"
                elif (np.absolute(np.round(lower, 1)) > 0.09) & (np.absolute(np.round(upper, 1)) <= 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"({lower:.1f}, 0)"
                elif (np.absolute(np.round(lower, 1)) > 0.09) & (np.absolute(np.round(upper, 1)) > 0.09):
                    spearman_corr_df_table_confint.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"({lower:.1f}, {upper:.1f})"

                # To draw table with correlation values and confidence intervals, we format the string to include the correlation value, significance stars, and confidence intervals.
                # if spearman_pvalue <= 0.01:
                if p_adj <= 0.01:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"{spearman_corr:.1f} $^{{**}}$\n({lower:.1f}, {upper:.1f})"
                # elif (spearman_pvalue > 0.01) & (spearman_pvalue <= 0.05):
                elif (p_adj > 0.01) & (p_adj <= 0.05):
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = f"{spearman_corr:.1f} $^{{*}}$\n({lower:.1f}, {upper:.1f})"
                else:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = '–'
                if abs(spearman_corr) < 0.1:
                    spearman_corr_df_table.loc[f'{var1}_{var2}', (month_name, 'Detrended')] = '–'

        # Rename index for better presentation
        spearman_corr_df.rename(index={'smb_z500': r"r$_\mathdefault{s}$(SMB, Z500)", 'smb_sif': r"r$_\mathdefault{s}$(SMB, SIF)", 'z500_sif': r"r$_\mathdefault{s}$(Z500, SIF)", 'smb_minus_z500_sif_minus_z500': r"$r_{\mathrm{s}}(\mathrm{SMB}, \mathrm{SIF} \mid \mathrm{Z500})$"}, inplace=True)
        spearman_corr_df_pval.rename(index={'smb_z500': r"r$_\mathdefault{s}$(SMB, Z500)", 'smb_sif': r"r$_\mathdefault{s}$(SMB, SIF)", 'z500_sif': r"r$_\mathdefault{s}$(Z500, SIF)", 'smb_minus_z500_sif_minus_z500': r"$r_{\mathrm{s}}(\mathrm{SMB}, \mathrm{SIF} \mid \mathrm{Z500})$"}, inplace=True)
        spearman_corr_df_epval.rename(index={'smb_z500': r"r$_\mathdefault{s}$(SMB, Z500)", 'smb_sif': r"r$_\mathdefault{s}$(SMB, SIF)", 'z500_sif': r"r$_\mathdefault{s}$(Z500, SIF)", 'smb_minus_z500_sif_minus_z500': r"$r_{\mathrm{s}}(\mathrm{SMB}, \mathrm{SIF} \mid \mathrm{Z500})$"}, inplace=True)
        spearman_corr_df_table_confint.rename(index={'smb_z500': r"r$_\mathdefault{s}$(SMB, Z500)", 'smb_sif': r"r$_\mathdefault{s}$(SMB, SIF)", 'z500_sif': r"r$_\mathdefault{s}$(Z500, SIF)", 'smb_minus_z500_sif_minus_z500': r"$r_{\mathrm{s}}(\mathrm{SMB}, \mathrm{SIF} \mid \mathrm{Z500})$"}, inplace=True)

        spearman_corr_df_table.rename(index={'smb_z500': r"r$_\mathdefault{s}$(SMB, Z500)", 'smb_sif': r"r$_\mathdefault{s}$(SMB, SIF)", 'z500_sif': r"r$_\mathdefault{s}$(Z500, SIF)", 'smb_minus_z500_sif_minus_z500': r"$r_{\mathrm{s}}(\mathrm{SMB}, \mathrm{SIF} \mid \mathrm{Z500})$"}, inplace=True)
        # spearman_corr_df_table.rename(index={'smb_z500': 'SMB, Z500', 'smb_sif': 'SMB, SIF', 'z500_sif': 'Z500, SIF'}, inplace=True)
        spearman_corr_df_table.rename(columns={'Trend': '(x, y)', 'Detrended': r"(x', y')"}, inplace=True)

        # Save significance test results
        save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
        spearman_corr_df.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif.csv")
        spearman_corr_df_pval.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_pvalues.csv")
        spearman_corr_df_epval.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_epvalues.csv")
        spearman_corr_df_table_confint.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_table_confint.csv")
        spearman_corr_df_table.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_table.csv")

        # spearman_corr_df.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_gbi.csv")
        # spearman_corr_df_pval.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_gbi_pvalues.csv")
        # spearman_corr_df_table.to_csv(f"{save_path}/spearman_correlation_among_z500_smb_sif_gbi_table.csv")

        # # Uncomment to test and plot the original, trend, and detrended data for each variable
        # fig, ax = plt.subplots(1)
        # df_month['sif'].plot(ax=ax, label='Original')
        # df_trend['sif'].plot(ax=ax, label='Trend')
        # df_month_detrend['sif'].plot(ax=ax, label='Detrended')
        # ax.legend()

        # fig, ax = plt.subplots(1)
        # df_month['z500'].plot(ax=ax, label='Original')
        # df_trend['z500'].plot(ax=ax, label='Trend')
        # df_month_detrend['z500'].plot(ax=ax, label='Detrended')
        # ax.legend()

        # fig, ax = plt.subplots(1)
        # df_month['smb'].plot(ax=ax, label='Original')
        # df_month_detrend['smb'].plot(ax=ax, label='Trend')
        # df_month_detrend_des['smb'].plot(ax=ax, label='Deseasonalized + Detrended')
        # ax.legend()
        # %%

    def plot_spearman_correlation_among_z500_smb_sif():
        # %%
        input_file_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
        spearman_corr = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif.csv", header=[0, 1], index_col=0)
        spearman_corr_pval = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif_pvalues.csv", header=[0, 1], index_col=0)
        spearman_corr_epval = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif_epvalues.csv", header=[0, 1], index_col=0)
        spearman_corr_df_table_confint = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif_table_confint.csv", header=[0, 1], index_col=0)

        for trend_or_detrend in ['Trend', 'Detrended']:
            spearman_corr_trend = spearman_corr.loc[:, (slice(None), trend_or_detrend)]
            # spearman_corr_pval_trend = spearman_corr_pval.loc[:, (slice(None), trend_or_detrend)]
            spearman_corr_epval_trend = spearman_corr_epval.loc[:, (slice(None), trend_or_detrend)]
            spearman_corr_df_table_confint_trend = spearman_corr_df_table_confint.loc[:, (slice(None), trend_or_detrend)]
            # Mask insignificant correlations
            # spearman_corr_trend = spearman_corr_trend.where(spearman_corr_pval_trend <= 0.05, np.nan)
            spearman_corr_trend = spearman_corr_trend.where(spearman_corr_epval_trend <= 0.05, np.nan)
            spearman_corr_df_table_confint_trend = spearman_corr_df_table_confint_trend.where(spearman_corr_epval_trend <= 0.05, np.nan)

            # Define colormap
            colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 255, 255), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
            cmap = fp.make_cmap(colors, bit=True)
            bounds = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cbtick = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
            cbticklabel = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

            tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=4, width=0.5)

            fig = plt.figure(figsize=(2.5, 1), dpi=600)
            gs = plt.GridSpec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            im = ax.imshow(spearman_corr_trend, aspect='auto', cmap=cmap, norm=norm, origin='lower')

            month_name = spearman_corr_trend.columns.get_level_values(0)
            ax.set_xticks(np.arange(len(month_name.unique())))
            ax.set_xticklabels(month_name.unique())

            ax.set_yticks(np.arange(len(spearman_corr_trend.index)))
            if trend_or_detrend == 'Trend':
                yticklabels = [r"r$_{\mathdefault{s}}$" + "\n(SMB, SIF | Z500)",  r"r$_{\mathdefault{s}}$" + "\n(SMB, SIF)", r"r$_{\mathdefault{s}}$" + "\n(SMB, Z500)", r"r$_{\mathdefault{s}}$" + "\n(Z500, SIF)", ]
            else:
                yticklabels = [r"r$_{\mathdefault{s}}$" + "\n(SMB', SIF' | Z500')",  r"r$_{\mathdefault{s}}$" + "\n(SMB', SIF')", r"r$_{\mathdefault{s}}$" + "\n(SMB', Z500')", r"r$_{\mathdefault{s}}$" + "\n(Z500', SIF')", ]
            ax.set_yticklabels(yticklabels, fontsize=4, ha='center', va='center')

            # Style x, y ticks
            tick_params = dict(which='major', direction='inout', length=4, labelsize=4, width=0.5, color='k')
            ax.tick_params(axis='x', pad=1, **tick_params)
            ax.tick_params(axis='y', pad=12, **tick_params)

            # # Move first y-tick label to the left to avoid overlapping with the main figure
            yticklabels = ax.get_yticklabels()  # get fresh Text objects
            label = yticklabels[0]
            x, y = label.get_position()
            label.set_position((x - 0.035, y))  # move left (increase pad effect)

            [i.set_linewidth(0.4) for i in ax.spines.values()]
            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

            for i, col in enumerate(spearman_corr_epval_trend.columns):
                for j, row in enumerate(spearman_corr_epval_trend.index):
                    pvalue = spearman_corr_epval_trend.loc[row, col]
                    if pvalue <= 0.01:
                        stars = '**'
                    elif (pvalue <= 0.05) & (pvalue > 0.01):
                        stars = '*'
                    else:
                        stars = ''
                    ax.text(i, j, stars, ha='center', va='center', fontsize=5, color='k')

            # Add confidence interval text
            for i, col in enumerate(spearman_corr_df_table_confint_trend.columns):
                for j, row in enumerate(spearman_corr_df_table_confint_trend.index):
                    confint = spearman_corr_df_table_confint_trend.loc[row, col]
                    if pd.notna(confint):
                        ax.text(i, j - 0.2, confint, ha='center', va='bottom', fontsize=3, color='k')

            targets = dict(left=300, right=120, top=10, bottom=60, hspace=0, wspace=0)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            cbar_ax = fig.add_axes([0.925, 0.097, 0.03, 0.88]) #[left, bottom, width, height]
            cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='vertical', cmap=cmap, norm=norm)
            cb.set_ticks(cbtick)
            cb.minorticks_off()
            cb.update_ticks()
            cb.set_ticklabels(cbticklabel)
            cb.ax.tick_params(labelsize=4, pad=0.5, length=2, direction='in', width=0.5)
            cb.outline.set_linewidth(0.4)

            # Save the plot
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f"plot_spearman_correlation_among_z500_smb_sif_{trend_or_detrend}.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%
    def table_plot_spearman_correlation_among_z500_smb_sif():
        # %%
        input_file_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
        spearman_corr_df = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif_table.csv", header=[0, 1], index_col=0)
        # spearman_corr_df = pd.read_csv(f"{input_file_path}/spearman_correlation_among_z500_smb_sif_gbi.csv", header=[0, 1], index_col=0)

        # Plot significance test results as a table
        fig, ax = plt.subplots(figsize=(3, 0.95), dpi=600)  # Adjust size as needed
        ax.axis('off')  # Hide axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Flatten columns for multicolumn plotting
        flat_cols = [f"{c[1]}" for c in spearman_corr_df.columns]
        # Create the percentage of wind direction and wind speed data table
        spearman_corr_table = ax.table(cellText=spearman_corr_df.values,   # Round numbers to 2 decimals
                                colLabels=flat_cols,
                                rowLabels=spearman_corr_df.index,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 0.94]) # [x, y, width, height]
        spearman_corr_table.auto_set_font_size(False)
        spearman_corr_table.set_fontsize(4)

        # --- Set cell sizes and alignment ---
        for (row, col), cell in spearman_corr_table.get_celld().items():
            # Width for row labels
            if col == -1:
                cell.set_width(0.09)
                cell.set_fontsize(5)
            if row == 0:
                cell.set_fontsize(5)
            # cell.set_width(0.7 if col > -1 else 0.09)

            # # Heights
            cell.set_height(0.009 if row == 0 else 0.005)
            # Center text
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
            # Padding
            # cell.PAD = 0.01
            # Default borders
            cell.visible_edges = ''
            cell.set_edgecolor('#666666' if row > 0 else 'black')
            cell.set_linewidth(0.2 if row > 0 else 0.8)

        # --- Header styling ---
        for col in range(len(spearman_corr_df.columns)):
            cell = spearman_corr_table[(0, col)]
            cell.get_text().set_color('k')
            cell.visible_edges = 'B'  # top & bottom line
            cell.set_linewidth(0.4)

        # --- Add horizontal lines for the rest ---
        n_rows = len(spearman_corr_df.index) + 1
        n_cols = len(spearman_corr_df.columns)
        for row in range(1, n_rows):
            for col in range(n_cols):
                spearman_corr_table[(row, col)].visible_edges = 'B'

        # --- Add group labels above the table ---
        group_labels = list(dict.fromkeys([col[0] for col in spearman_corr_df.columns]))
        n_groups = len(group_labels)
        for i, label in enumerate(group_labels):
            x_center = (i + 0.5) / n_groups  # evenly distribute across axes
            ax.text(x_center, 0.93, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
        ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

        targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'table_plot_spearman_correlation_among_z500_smb_sif'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = f"table_plot_spearman_correlation_among_z500_smb_sif.png"
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

class LaggedRelationships:
    def plot_thf_t2m_q2m_for_young_sound_zac():
        # %%
        # Read turbulent fluxes on Young Sound
        # read_sif_data_from_racmo() # Print the name of variables available in RACMO SIF data
        ys_variables = read_sif_data_from_racmo(region='within_young_sound')
        ys_variables.rename(columns={'senf': 'senf_on_ys', 'latf': 'latf_on_ys'}, inplace=True)
        ys_variables = ys_variables * (-1)  # Flip the sign of turbulent fluxes on SIF to be consistent with the convention of turbulent fluxes on FG

        grsea_variables = read_sif_data_from_racmo(region='between_YS_100')
        grsea_variables.rename(columns={'senf': 'senf_on_grsea', 'latf': 'latf_on_grsea'}, inplace=True)
        grsea_variables = grsea_variables * (-1)  # Flip the sign of turbulent fluxes on SIF to be consistent with the convention of turbulent fluxes on FG

        # Read temperature and specific humidity on Zackenberg region
        zac_t2m = read_station_data_from_racmo(station='zac', variable='t2m')
        zac_t2m.rename(columns={'t2m': 't2m_on_zac'}, inplace=True)
        zac_q2m = read_station_data_from_racmo(station='zac', variable='q2m')
        zac_q2m.rename(columns={'q2m': 'q2m_on_zac'}, inplace=True)

        ys_zac_variables = pd.concat([ys_variables['senf_on_ys'], ys_variables['latf_on_ys'], zac_t2m, zac_q2m,
                                      grsea_variables['senf_on_grsea'], grsea_variables['latf_on_grsea']], axis=1)
        # Extract month
        ys_zac_variables['month'] = ys_zac_variables.index.month

        # Set up boxplot properties
        boxplot_prop = dict(showcaps=False, widths=0.4, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
        boxplot_prop_latf_q2m = dict(boxprops=dict(facecolor='royalblue', color='k', linewidth=0.5))
        boxplot_prop_senf_t2m = dict(boxprops=dict(facecolor='orangered', color='k', linewidth=0.5))

        # Define tick parameters and spines properties
        tick_params_major = dict(direction='out', length=4, pad=1, labelsize=8, width=0.4, color='black')
        tick_params_minor = dict(direction='out', length=2, width=0.4, color='black')
        tick_params_perc = dict(direction='out', length=2, pad=0.5, labelsize=5, width=0.4, color='black')
        ax_spine_shift = 4  # smaller than current default (e.g., 0.2)
        left_ylabel_pos = [-0.07, 0.55]
        right_ylabel_pos = [1.07, 0.55]

        fig = plt.figure(figsize=(5, 3), dpi=600)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.08], hspace=0.12)   # large spacing between row groups
        gs_top = outer[0].subgridspec(2, 1, hspace=0.4, wspace=0)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        # Select senf and latf columns
        data = ys_zac_variables[['month', 'senf_on_ys', 'latf_on_ys']].dropna()
        # Prepare data for boxplot: list of arrays per month
        senf_data = [data[data['month'] == m]['senf_on_ys'] for m in range(1, 13)]
        latf_data = [data[data['month'] == m]['latf_on_ys'] for m in range(1, 13)]
        # Positions for side-by-side plots
        positions_senf = [m - 0.2 for m in range(1, 13)]
        positions_latf = [m + 0.2 for m in range(1, 13)]

        ax1 = fig.add_subplot(gs_top[0, 0])
        # ax1.boxplot(senf_data, whis=[25, 75], positions=positions_senf, **boxplot_prop_senf_t2m, **boxplot_prop)
        # ax1.boxplot(latf_data, whis=[25, 75], positions=positions_latf, **boxplot_prop_latf_q2m, **boxplot_prop)
        ax1.boxplot(senf_data, whis=[5, 95], positions=positions_senf, **boxplot_prop_senf_t2m, **boxplot_prop)
        # ax1.boxplot(latf_data, whis=[5, 95], positions=positions_latf, **boxplot_prop_latf_q2m, **boxplot_prop)
        ax1.axhline(y=0, color='grey', lw=0.5, ls='--', zorder=2)

        ax2 = ax1.twinx()
        ax2.boxplot(latf_data, whis=[5, 95], positions=positions_latf, **boxplot_prop_latf_q2m, **boxplot_prop)
        # ax2.axhline(y=0, color='royalblue', lw=0.5, ls='--', zorder=2)

        # Labels and formatting
        ax1.set_xlim([0.3, 12.7])
        ax1.set_xticks(range(1, 13))
        ax1.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        # ax1.set_ylim(-10, 12)
        # ax1.set_yticks([-10, -5, 0, 10])
        # ax1.set_yticklabels(['-10', '-5', '0', '10'])
        # ax1.set_yticks([-7.5, -2.5, 2.5, 7.5], minor=True)

        ax1.set_ylim(-25, 25)
        ax1.set_yticks([-20, -10, 0, 10, 20])
        ax1.set_yticklabels(['-20', '-10', '0', '10', '20'])
        ax1.set_yticks([-15, -5, 5, 15], minor=True)
        ax1.set_ylabel(fr"SHF (W m$^\mathdefault{{-2}}$)", color='k', fontsize=8)
        ax1.yaxis.set_label_coords(left_ylabel_pos[0], left_ylabel_pos[1])

        ax1.tick_params(axis='both', which='major', **tick_params_major)
        ax1.tick_params(axis='y', which='minor', **tick_params_minor)
        [ax1.spines[i].set_visible(False) for i in ['top', 'right']]
        [i.set_linewidth(0.4) for i in ax1.spines.values()]
        ax1.spines['left'].set_bounds(-20, 20)
        # ax1.spines['left'].set_bounds(-10, 10)
        ax1.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax1.spines['bottom'].set_bounds(1, 12)
        ax1.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        ax1.text(0.06, 0.95, 'a) YS', transform=ax1.transAxes, ha='center', va='center', fontsize=10, fontweight='bold')
        # Add y-axis on the right just for the reference.

        # ax2.set_ylim([-5, 20])
        # ax2.set_yticks([-5, 0, 5, 10, 15, 20])
        # # ax2.set_yticks(ax1.get_yticks(minor=True), minor=True)
        # # ax2.set_yticklabels([tick.get_text() for tick in ax1.get_yticklabels()])
        # ax2.set_yticklabels(['-5', '0', '5', '10', '15', '20'])
        ax2.set_ylim(-25, 25)
        ax2.set_yticks([-20, -10, 0, 10, 20])
        ax2.set_yticklabels(['-20', '-10', '0', '10', '20'])
        ax2.set_yticks([-15, -5, 5, 15], minor=True)
        ax2.set_ylabel(fr"LHF (W m$^\mathdefault{{-2}}$)", color='k', fontsize=8)
        ax2.yaxis.set_label_coords(right_ylabel_pos[0], right_ylabel_pos[1])

        ax2.tick_params(axis='y', **tick_params_major)
        ax2.tick_params(axis='y', which='minor', **tick_params_minor)
        ax2.patch.set_visible(False) # Prevents ax from hiding ax2
        [ax2.spines[i].set_visible(False) for i in ['left', 'top', 'bottom']]
        [i.set_linewidth(0.4) for i in ax2.spines.values()]
        ax2.spines['right'].set_bounds(-20, 20)
        ax2.spines['right'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        # Draw grid
        ax1.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
        # # Get spine bounds
        x0, x1 = ax1.spines['bottom'].get_bounds()
        y0, y1 = ax1.spines['left'].get_bounds()
        # x0, x1 = ax1.get_xlim()
        # y0, y1 = ax1.get_ylim()
        # Create clipping rectangle based on spine bounds
        grid_clip = Rectangle((x0, y0), x1 - x0, y1 - y0, transform=ax1.transData)
        # Clip gridlines
        for line in ax1.get_xgridlines() + ax1.get_ygridlines():
            line.set_clip_path(grid_clip)

        # Select t2m and q2m columns
        data = ys_zac_variables[['month', 't2m_on_zac', 'q2m_on_zac']].dropna()
        # Prepare data for boxplot: list of arrays per month
        t2m_data = [data[data['month'] == m]['t2m_on_zac'] for m in range(1, 13)]
        t2m_data = [t - 273.15 for t in t2m_data]  # Convert temperature from K to °C for better interpretability
        q2m_data = [data[data['month'] == m]['q2m_on_zac'] for m in range(1, 13)]
        q2m_data = [q * 1000 for q in q2m_data]  # Convert specific humidity from kg/kg to g/kg for better interpretability
        # Positions for side-by-side plots
        positions_t2m = [m - 0.2 for m in range(1, 13)]
        positions_q2m = [m + 0.2 for m in range(1, 13)]

        ax3 = fig.add_subplot(gs_top[1, 0])
        ax3.boxplot(t2m_data, whis=[5, 95], positions=positions_t2m, **boxplot_prop_senf_t2m, **boxplot_prop)
        ax3.axhline(y=0, color='grey', lw=0.5, ls='--', zorder=2)

        ax4 = ax3.twinx()  # Create a secondary y-axis for q2m
        ax4.patch.set_visible(False) # Prevents ax from hiding ax4
        [ax4.spines[i].set_visible(False) for i in ['left', 'top', 'bottom']]

        [i.set_linewidth(0.4) for i in ax4.spines.values()]
        # ax4.axis(xy)  # Ensure ax4 shares the same x-axis as ax3
        ax4.boxplot(q2m_data, whis=[5, 95], positions=positions_q2m, **boxplot_prop_latf_q2m, **boxplot_prop)

        # Labels and formatting
        ax3.set_xlim([0.3, 12.7])
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

        ax3.set_ylim(-25, 10)
        ax3.set_yticks([-20, -10, 0, 10])
        ax3.set_yticklabels(['-20', '-10', '0', '10'])
        ax3.set_yticks([-15, -5, 5], minor=True)
        ax3.set_ylabel(fr"T$_\mathdefault{{2m}}$) ($^\circ$C)", color='k', fontsize=8)
        ax3.yaxis.set_label_coords(left_ylabel_pos[0], left_ylabel_pos[1])

        ax3.tick_params(axis='both', which='major', **tick_params_major)
        ax3.tick_params(axis='y', which='minor', **tick_params_minor)
        [ax3.spines[i].set_visible(False) for i in ['top', 'right']]
        [i.set_linewidth(0.4) for i in ax3.spines.values()]
        ax3.spines['left'].set_bounds(-20, 10)
        ax3.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax3.spines['bottom'].set_bounds(1, 12)
        ax3.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax3.text(0.06, 0.95, 'b) ZAC', transform=ax3.transAxes, ha='center', va='center', fontsize=10, fontweight='bold')

        ax4.set_ylim(0, 6)
        ax4.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax4.set_yticklabels(['0', '1', '2', '3', '4', '5', '6'])
        ax4.set_ylabel(fr"q$_\mathdefault{{2m}}$ (gkg$^\mathdefault{{-1}}$)", color='k', fontsize=8)
        ax4.yaxis.set_label_coords(right_ylabel_pos[0], right_ylabel_pos[1])
        ax4.spines['right'].set_bounds(0, 6)
        ax4.spines['right'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        ax4.tick_params(axis='y', **tick_params_major)
        ax4.tick_params(axis='y', which='minor', **tick_params_minor)

        # Legend axis
        ax_legend = fig.add_subplot(gs_bot[0, 0])
        [sp.set_visible(False) for sp in ax_legend.spines.values()]
        ax_legend.axis('off')

        # Dummy handle for legend
        latf_q2m = Rectangle((0, 0), 1, 1, facecolor='royalblue', edgecolor='k', linewidth=0.5, label='LATF' + '\n' + fr"q$_\mathdefault{{2m}}$")
        senf_t2m = Rectangle((0, 0), 1, 1, facecolor='orangered', edgecolor='k', linewidth=0.5, label='SENF' + '\n' + fr"T$_\mathdefault{{2m}}$")

        legend_boxplot = ax_legend.legend(bbox_to_anchor=(0.5, 1.5), loc='center', handles=[latf_q2m, senf_t2m], labels=["", ""],   # IMPORTANT: prevent default text rendering
                                          ncol=2, handler_map={Rectangle: HandlerVerticalBoxPlot2()}, handlelength=2, handleheight=3.5, frameon=False, fontsize=7)

        targets = dict(left=265, right=260, top=30, bottom=40, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_thf_t2m_q2m_for_young_sound_zac'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_thf_t2m_q2m_for_young_sound_zac.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    def calculate_trends():
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')

        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m']  * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        # variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2020']
        variables_on_zac = variables_on_zac.resample('ME').median()  # Resample to monthly mean to ensure consistent time series

        var = 'q2m'
        for month in [5, 6, 7, 8, 9]:
            variables_on_zac_month = variables_on_zac.loc[variables_on_zac.index.month == month]
            variables_on_zac_month['year'] = variables_on_zac_month.index.year
            fig, ax = plt.subplots()
            ax.plot(variables_on_zac_month['year'], variables_on_zac_month[var])

            years = variables_on_zac_month['year'].values
            y = variables_on_zac_month[var].values
            # Fit linear trend
            slope, intercept, r, p, stderr = linregress(years, y)
            # Estimated trend
            trend = intercept + slope * years
            ax.plot(variables_on_zac_month['year'], trend, '--', color='gray', label=f"Linear trend of {var} on zac")
            ax.text(0.1, 0.9, f"Month: {month} R² = {r**2:.2f} (p = {p:.3f})", transform=ax.transAxes, fontsize=12)
            ax.legend()

        # %%
        phenology = 'dor'
        mask_area = 'within_buffers'
        mask_area_var = mask_area.split('_')[0]
        radius = 800
        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")

        variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
        variables_on_ci.set_index('Unnamed: 0', inplace=True)
        variables_on_ci['doy'] = variables_on_ci.index.dayofyear
        variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
        variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed
        variables_on_ci = variables_on_ci.resample('ME').median()  # Resample to monthly mean to ensure consistent time series
        var = 'latf'
        for month in [5, 6, 7, 8, 9]:
            variables_on_ci_month = variables_on_ci.loc[    variables_on_ci.index.month == month]
            variables_on_ci_month['year'] = variables_on_ci_month.index.year
            fig, ax = plt.subplots()
            ax.plot(variables_on_ci_month['year'], variables_on_ci_month[var])

            years = variables_on_ci_month['year'].values
            y = variables_on_ci_month[var].values
            # Fit linear trend
            slope, intercept, r, p, stderr = linregress(years, y)
            # Estimated trend
            trend = intercept + slope * years
            ax.plot(variables_on_ci_month['year'], trend, '--', color='gray', label=f'Linear trend of {var}')
            ax.text(0.1, 0.9, f"Month: {month} R² = {r**2:.2f} (p = {p:.3f})", transform=ax.transAxes, fontsize=12)
            ax.legend()
        # %%
        # Correlation between q2m and latf, and t2m and senf.
        # var_pair = ['latf', 'q2m']
        var_pair = ['senf', 't2m']
        for month in [5, 6, 7, 8, 9]:
            variables_on_ci_month = variables_on_ci.loc[variables_on_ci.index.month == month]
            variables_on_zac_month = variables_on_zac.loc[variables_on_zac.index.month == month]

            x = variables_on_ci_month.loc[:, var_pair[0]]
            y = variables_on_zac_month.loc[:, var_pair[1]]

            # Align by index and remove NaNs
            df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

            # Calculate Spearman correlation
            rho, p_value = spearmanr(df[var_pair[0]], df[var_pair[1]])

            fig, ax = plt.subplots()
            ax.plot(x, y, 'o', label=f"Spearman's rho = {rho:.2f} (p = {p_value:.3f}), month = {month}")
            ax.set_xlabel(var_pair[0])
            ax.set_ylabel(var_pair[1])
            ax.legend()

        # %%
    def calculate_trends_early_late():
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')

        phenology = 'dor'
        mask_area = 'within_buffers'
        mask_area_var = mask_area.split('_')[0]
        radius = 800
        df_dor_doy = pd.read_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
        df_dor_doy.index = df_dor_doy['Unnamed: 0']
        df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
        variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
        variables_on_ci.set_index('Unnamed: 0', inplace=True)
        variables_on_ci['doy'] = variables_on_ci.index.dayofyear
        variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
        variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed

        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m']  * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        # variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2020']

        # Climatological mean of senf, latf, t2m and q2m during 1990–2020
        variables_on_ci_doy_mean = variables_on_ci.groupby(variables_on_ci.index.dayofyear).mean()
        variables_on_zac_doy_mean = variables_on_zac.groupby(variables_on_zac.index.dayofyear).mean()

        for period in ['early', ]:#early, 'late'
            years = df_dor_doy.loc[df_dor_doy['composite'] == period].index
            for year in years:
                variables_on_ci_year = variables_on_ci.loc[str(year)]
                variables_on_ci_year_doy = variables_on_ci_year.groupby(variables_on_ci_year.index.dayofyear).mean()
                variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy - variables_on_ci_doy_mean

                variables_on_zac_year = variables_on_zac.loc[str(year)]
                variables_on_zac_year_doy = variables_on_zac_year.groupby(variables_on_zac_year.index.dayofyear).mean()
                variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy - variables_on_zac_doy_mean

                for doy in [(121, 151, 'May'), (152, 181, 'June'), (182, 212, 'July'), (213, 243, 'August'), (244, 273, 'September')]:
                    start_doy, end_doy, month_name = doy
                    variables_on_ci_year_doy_anomalies_month = variables_on_ci_year_doy_anomalies.loc[start_doy:end_doy]
                    variables_on_zac_year_doy_anomalies_month = variables_on_zac_year_doy_anomalies.loc[start_doy:end_doy]


                    for var_pair in [['latf', 'q2m'],]: # ['senf', 't2m']
                        # Extract the two series
                        x = variables_on_ci_year_doy_anomalies_month.loc[:, var_pair[0]]
                        y = variables_on_zac_year_doy_anomalies_month.loc[:, var_pair[1]]

                        # Align by index and remove NaNs
                        df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

                        # Calculate Spearman correlation
                        rho, p_value = spearmanr(df[var_pair[0]], df[var_pair[1]])
                        print(f"SC bet {var_pair[0]} and {var_pair[1]} for year {year} and month {month_name} (DOY range {start_doy}-{end_doy}): rho = {rho:.3f}, p = {p_value:.3f}")


        # %%
    def calculate_early_late_sea_ice_dor_years():
        """Calculate early and late sea ice retreat years based on the detrended DOR phenology of sea ice in the Greenland Sea.

        """
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies = ['dor']
        # phenologies = ['iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            for mask_area in ['within_buffers']: # ['inbetween_buffers', 'within_buffers']
                sea_ice_mask = xr.open_dataset(f"{main_path}/input_dataset/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                for radius in [200, 300, 400, 500, 600, 700, 800]:
                    sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)['sea_ice_buffer']
                    ds_mask = ds.where(sea_ice_mask_radius == 1)
                    ds_mean = ds_mask.median(dim=['rlat', 'rlon'])
                    ds_mean_df = ds_mean.to_dataframe()
                    ds_mean_df = ds_mean_df.iloc[:-1, :]

                    # Detrend the phenology time series using linear regression
                    # Years
                    years = ds_mean_df.index.values
                    y = ds_mean_df[f"{phenology}_doy"].values
                    # Fit linear trend
                    slope, intercept, r, p, stderr = linregress(years, y)
                    # Estimated trend
                    trend = intercept + slope * years
                    # Detrended series (preserve mean)
                    ds_mean_df[f"{phenology}_doy_detrended"] = y - trend + y.mean()

                    # print(f"Slope = {slope:.3f} days/year")
                    # print(f"Trend over period = {slope*(years[-1]-years[0]):.2f} days")

                    # # Early years:
                    # early_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.25)
                    # early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] < early_doy].index
                    # # Late years:
                    # late_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.75)
                    # late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] > late_doy].index

                    # Early years using detrended phenology time series
                    early_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.25)
                    # early_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.05)
                    early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] <= early_doy].index
                    # Late years:
                    late_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.75)
                    # late_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.95)
                    late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] >= late_doy].index

                    # fig, ax = plt.subplots(figsize=(8, 4))
                    # ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy"], 'o-', label='Original')
                    # ax.plot(ds_mean_df.index, trend, 'r--', label='Linear trend')
                    # ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy_detrended"], 'o-', color='k', label='Detrended')
                    # for early_year in early_years:
                    #     ax.plot(early_year, ds_mean_df.loc[early_year, f"{phenology}_doy_detrended"], 'ro', markerfacecolor='none', markeredgewidth=2, ms=8, label='Early year' if early_year == early_years[0] else "")
                    # for late_year in late_years:
                    #     ax.plot(late_year, ds_mean_df.loc[late_year, f"{phenology}_doy_detrended"], 'bo', markerfacecolor='none', markeredgewidth=2, ms=8, label='Late year' if late_year == late_years[0] else "")
                    # ax.set_xlabel('Year')
                    # ax.set_ylabel(f'{phenology.upper()} (DOY)')
                    # ax.set_title(f'{phenology.upper()} Detrended Phenology Time Series ({mask_area}, {radius} km buffer)')
                    # ax.legend()
                    # plt.show()
                    df_dor_doy = pd.DataFrame(columns=[radius, 'composite', 'DOR'])
                    for early_year in early_years:
                        df_dor_doy.loc[early_year, radius] = ds_mean_df.loc[early_year, f"{phenology}_doy_detrended"]
                        df_dor_doy.loc[early_year, 'composite'] = 'early'
                        df_dor_doy.loc[early_year, 'DOR'] = early_doy
                    for late_year in late_years:
                        df_dor_doy.loc[late_year, radius] = ds_mean_df.loc[late_year, f"{phenology}_doy_detrended"]
                        df_dor_doy.loc[late_year, 'composite'] = 'late'
                        df_dor_doy.loc[late_year, 'DOR'] = late_doy

                    df_dor_doy.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
    # %%
    def plot_early_late_sea_ice_dor_years():
        """Calculate early and late sea ice retreat years based on the detrended DOR phenology of sea ice in the Greenland Sea.

        """
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies = ['dor']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            for mask_area in ['within_buffers']: # ['inbetween_buffers', 'within_buffers']
                sea_ice_mask = xr.open_dataset(f"{main_path}/input_dataset/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                # for radius in [200, 300, 400, 500, 600, 700, 800]:
                for radius in [800]:
                    # radius = 800
                    sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)['sea_ice_buffer']
                    ds_mask = ds.where(sea_ice_mask_radius == 1)
                    ds_mean = ds_mask.median(dim=['rlat', 'rlon'])
                    ds_mean_df = ds_mean.to_dataframe()
                    ds_mean_df = ds_mean_df.iloc[:-1, :]

                    # Detrend the phenology time series using linear regression
                    # Years
                    years = ds_mean_df.index.values
                    y = ds_mean_df[f"{phenology}_doy"].values
                    # Fit linear trend
                    slope, intercept, r, p, _ = linregress(years, y)
                    # Estimated trend
                    trend = intercept + slope * years
                    # Detrended series (preserve mean)
                    ds_mean_df[f"{phenology}_doy_detrended"] = y - trend + y.mean()

                    # print(f"Slope = {slope:.3f} days/year")
                    # print(f"Trend over period = {slope*(years[-1]-years[0]):.2f} days")

                    # # Early years:
                    # early_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.25)
                    # early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] < early_doy].index
                    # # Late years:
                    # late_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.75)
                    # late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] > late_doy].index

                    # Early years using detrended phenology time series
                    early_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.25)
                    early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] < early_doy].index
                    # Late years:
                    late_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.75)
                    late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] > late_doy].index

                    fig = plt.figure(figsize=(5, 3), dpi=600)
                    # OUTER GridSpec: controls LARGE row gap (applies to all columns)
                    outer = plt.GridSpec(2, 1, height_ratios=[2, 0.15], hspace=0.25)   # large spacing between row groups
                    gs_top = outer[0].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing
                    gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

                    ax = fig.add_subplot(gs_top[0, 0])
                    ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy"], 'o-', color='gray', label='Original')
                    ax.plot(ds_mean_df.index, trend, '--', color='gray', label='Linear trend')
                    ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy_detrended"], 'o-', color='k', label='Detrended')
                    for early_year in early_years:
                        ax.plot(early_year, ds_mean_df.loc[early_year, f"{phenology}_doy_detrended"], 'ro', markerfacecolor='none', markeredgewidth=2, ms=8, label='Early year' if early_year == early_years[0] else "")
                    for late_year in late_years:
                        ax.plot(late_year, ds_mean_df.loc[late_year, f"{phenology}_doy_detrended"], 'bo', markerfacecolor='none', markeredgewidth=2, ms=8, label='Late year' if late_year == late_years[0] else "")

                    ax.set_xlim([1989, 2021])
                    ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
                    ax.set_xticklabels([1990, 1995, 2000, 2005, 2010, 2015, 2020])
                    ax.set_xticks([1992.5, 1997.5, 2002.5, 2007.5, 2012.5, 2017.5], minor=True)
                    ax.set_xlabel('Year', size=9, color='k', labelpad=0.15)

                    ax.set_ylim([166.5, 245])
                    ax.set_yticks([170, 180, 190, 200, 210, 220, 230, 240])
                    ax.set_yticklabels([170, 180, 190, 200, 210, 220, 230, 240])
                    ax.set_yticks([175, 185, 195, 205, 215, 225, 235], minor=True)
                    ax.set_ylabel('DOR (DOY)', rotation='vertical', size=9, color='k', labelpad=0.15)

                    [ax.spines[i].set_visible(False) for i in ['top', 'right']]
                    [i.set_linewidth(0.6) for i in ax.spines.values()]
                    ax.spines['left'].set_bounds(170, 240)
                    ax_spine_shift = 2 # Adjust this value to control the outward shift of the spines
                    ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
                    ax.spines['bottom'].set_bounds(1990, 2020)
                    ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

                    tick_params_major = dict(direction='out', length=5, pad=1, labelsize=9, width=0.6, color='k')
                    tick_params_minor = dict(direction='out', length=2.5, width=0.6, color='k')
                    ax.tick_params(axis='both', which='major', **tick_params_major)
                    ax.tick_params(axis='both', which='minor', **tick_params_minor)

                    # Draw grid
                    ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
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

                    # Define legend
                    marker_style_legend = dict(ms=5, lw=0, mew=0.5)
                    handleba2 = Line2D([], [], label='N: ' + str(len(ds_mean_df)), **marker_style_legend)
                    handleba3 = Line2D([], [], label='b0: ' + str(np.round(slope, 2)) + ' ' + r'days year$^\mathdefault{-1}$', **marker_style_legend)
                    handleba4 = Line2D([], [], label='b1: ' + str(np.round(intercept, 2)), **marker_style_legend)
                    # Assign stars depending on p-value
                    if p <= 0.01:
                        stars = r"$^{**}$"
                    elif p <= 0.05:
                        stars = r"$^{*}$"
                    else:
                        stars = ""
                    # # build LaTeX-friendly label
                    # r2_value = np.round(r**2, 2)
                    r2_value = np.round(r, 1)
                    # r2_plabel = fr"R$^\mathdefault{{2}}$: {r2_value}{stars}"
                    r2_plabel = fr"r: {r2_value}{stars}"
                    handleba5= Line2D([], [], label=r2_plabel, **marker_style_legend)
                    handleba = [handleba2, handleba3, handleba5]
                    # first_legend = ax.legend(bbox_to_anchor=(0.73, 0.005), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=11, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')
                    first_legend = ax.legend(bbox_to_anchor=(0.3, 0.03), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=9, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')

                    # Set legend
                    ax = fig.add_subplot(gs_bot[0, 0])
                    ax.axis('off')

                    legend_trend_detrend = dict(ms=6, marker='o')
                    original = Line2D([0], [0], color='gray', label=rf'DOR', mfc='gray', mec='gray', lw=1, **legend_trend_detrend)
                    trend_line = Line2D([0], [0], color='gray', label=r'Linear trend', marker='', ls='--', lw=1)
                    detrended = Line2D([0], [0], color='k', label=rf'Detrended DOR', mfc='k', mec='k', lw=1, **legend_trend_detrend)
                    early = Line2D([0], [0], color='r', label='Early retreat year', mfc='none', markeredgewidth=2, ls='None', **legend_trend_detrend)
                    late = Line2D([0], [0], color='b', label='Late retreat year', mfc='none', markeredgewidth=2, ls='None', **legend_trend_detrend)
                    leg = ax.legend(bbox_to_anchor=(0.5, 0.5), loc='center', handles=[original, trend_line, detrended, early, late],
                                ncol=5, columnspacing=1, borderaxespad=0, fontsize=8, handletextpad=0.1, labelspacing=0.3, frameon=False, handlelength=1.2, handleheight=1.2,
                            edgecolor='none') # facecolor='w', framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None)}

                    targets = dict(left=265, right=10, top=10, bottom=0, hspace=0, wspace=0)
                    fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
                    fp.adjust_borders(fig, targets)

                    logger.info(f"Saving plot of 'plot_early_late_sea_ice_dor_years'...")
                    base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
                    fig_name = 'plot_early_late_sea_ice_dor_years.png'
                    fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    def plot_early_late_sea_ice_dor_years_grsea_ys():
        """Calculate early and late sea ice retreat years based on the detrended DOR phenology of sea ice in the Greenland Sea.

        """
        # %%
        fig = plt.figure(figsize=(5, 3.1), dpi=600)
        # OUTER GridSpec: controls LARGE row gap (applies to all columns)
        outer = plt.GridSpec(2, 1, height_ratios=[2, 0.15], hspace=0.33)   # large spacing between row groups
        gs_top = outer[0].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing
        gs_bot = outer[1].subgridspec(1, 1, hspace=0, wspace=0)   # small internal spacing

        # Plot Young Sound fjord
        input_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        young_sound_fjord_sea_ice_combined = pd.read_csv(f"{input_path}/young_sound_fjord_sea_ice_combined.csv", header=0, index_col=0)
        young_sound_fjord_sea_ice_combined = pd.DataFrame(young_sound_fjord_sea_ice_combined.loc[:2020, :])

        # Plot Greenland Sea
        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        phenologies = ['dor']
        ds = xr.open_dataset(f"{main_path}/greenland_sea_sif_phenology_dor.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))["dor_doy"]
        sea_ice_mask = xr.open_dataset('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/ci_all_within_buffers_dor_trend_racmo5km.nc')
        sea_ice_mask_radius = sea_ice_mask.sel(radius=800)['sea_ice_buffer']
        ds_mask = ds.where(sea_ice_mask_radius == 1)
        ds_mean = ds_mask.median(dim=['rlat', 'rlon'])
        ds_mean_df = ds_mean.to_dataframe()
        ds_mean_df = ds_mean_df.iloc[:-1, :]

        # Detrend the phenology time series using linear regression
        # Years
        years = ds_mean_df.index.values
        y = ds_mean_df['dor_doy'].values
        # Fit linear trend
        slope, intercept, r, p, _ = linregress(years, y)
        # Estimated trend
        trend = intercept + slope * years
        # Detrended series (preserve mean)
        ds_mean_df['dor_doy_detrended'] = y - trend + y.mean()

        # Early years using detrended phenology time series
        early_doy = ds_mean_df['dor_doy_detrended'].quantile(0.25)
        early_years = ds_mean_df.loc[ds_mean_df['dor_doy_detrended'] < early_doy].index
        # Late years:
        late_doy = ds_mean_df['dor_doy_detrended'].quantile(0.75)
        late_years = ds_mean_df.loc[ds_mean_df['dor_doy_detrended'] > late_doy].index

        ax = fig.add_subplot(gs_top[0, 0])
        ax.plot(ds_mean_df.index, ds_mean_df['dor_doy'], 'o-', color='gray', label='Original')
        ax.plot(ds_mean_df.index, trend, '--', color='gray', label='Linear trend')
        ax.plot(ds_mean_df.index, ds_mean_df['dor_doy_detrended'], 'o-', color='k', label='Detrended')
        ax.plot(young_sound_fjord_sea_ice_combined.index, young_sound_fjord_sea_ice_combined['sif_break_doy'], 'o-', color='g', label='Fjord break-up')

        for early_year in early_years:
            ax.plot(early_year, ds_mean_df.loc[early_year, 'dor_doy_detrended'], 'ro', markerfacecolor='none', markeredgewidth=2, ms=8, label='Early year' if early_year == early_years[0] else "")
        for late_year in late_years:
            ax.plot(late_year, ds_mean_df.loc[late_year, 'dor_doy_detrended'], 'bo', markerfacecolor='none', markeredgewidth=2, ms=8, label='Late year' if late_year == late_years[0] else "")

        ax.set_xlim([1989, 2021])
        ax.set_xticks([1990, 1995, 2000, 2005, 2010, 2015, 2020])
        ax.set_xticklabels([1990, 1995, 2000, 2005, 2010, 2015, 2020])
        ax.set_xticks([1992.5, 1997.5, 2002.5, 2007.5, 2012.5, 2017.5], minor=True)
        ax.set_xlabel('Year', size=9, color='k', labelpad=0.15)

        ax.set_ylim([166.5, 245])
        ax.set_yticks([170, 180, 190, 200, 210, 220, 230, 240])
        ax.set_yticklabels([170, 180, 190, 200, 210, 220, 230, 240])
        ax.set_yticks([175, 185, 195, 205, 215, 225, 235], minor=True)
        ax.set_ylabel('DOR (DOY)', rotation='vertical', size=9, color='k', labelpad=0.15)

        [ax.spines[i].set_visible(False) for i in ['top', 'right']]
        [i.set_linewidth(0.6) for i in ax.spines.values()]
        ax.spines['left'].set_bounds(170, 240)
        ax_spine_shift = 2 # Adjust this value to control the outward shift of the spines
        ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        ax.spines['bottom'].set_bounds(1990, 2020)
        ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        tick_params_major = dict(direction='out', length=5, pad=1, labelsize=9, width=0.6, color='k')
        tick_params_minor = dict(direction='out', length=2.5, width=0.6, color='k')
        ax.tick_params(axis='both', which='major', **tick_params_major)
        ax.tick_params(axis='both', which='minor', **tick_params_minor)

        # Draw grid
        ax.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, alpha=0.7)
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

        # Define legend
        marker_style_legend = dict(ms=5, lw=0, mew=0.5)
        handleba2 = Line2D([], [], label='N: ' + str(len(ds_mean_df)), **marker_style_legend)
        handleba3 = Line2D([], [], label='b0: ' + str(np.round(slope, 2)) + ' ' + r'days year$^\mathdefault{-1}$', **marker_style_legend)
        handleba4 = Line2D([], [], label='b1: ' + str(np.round(intercept, 2)), **marker_style_legend)
        # Assign stars depending on p-value
        if p <= 0.01:
            stars = r"$^{**}$"
        elif p <= 0.05:
            stars = r"$^{*}$"
        else:
            stars = ""
        # # build LaTeX-friendly label
        # r2_value = np.round(r**2, 2)
        r2_value = np.round(r, 1)
        # r2_plabel = fr"R$^\mathdefault{{2}}$: {r2_value}{stars}"
        r2_plabel = fr"r: {r2_value}{stars}"
        handleba5= Line2D([], [], label=r2_plabel, **marker_style_legend)
        handleba = [handleba2, handleba3, handleba5]
        # first_legend = ax.legend(bbox_to_anchor=(0.73, 0.005), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=11, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')
        first_legend = ax.legend(bbox_to_anchor=(0.3, 0.03), loc='lower right', handles=handleba, ncol=1, borderaxespad=0, fontsize=9, frameon=False, handletextpad=0.005, labelspacing=0.2, markerscale=1.5, labelcolor='k')

        # Set legend
        ax = fig.add_subplot(gs_bot[0, 0])
        ax.axis('off')

        legend_trend_detrend = dict(ms=6, marker='o')
        original = Line2D([0], [0], color='gray', label=rf'DOR', mfc='gray', mec='gray', lw=1, **legend_trend_detrend)
        trend_line = Line2D([0], [0], color='gray', label=r'Linear trend', marker='', ls='--', lw=1)
        detrended = Line2D([0], [0], color='k', label=rf'Detrended DOR', mfc='k', mec='k', lw=1, **legend_trend_detrend)
        early = Line2D([0], [0], color='r', label='Early retreat year', mfc='none', markeredgewidth=2, ls='None', **legend_trend_detrend)
        late = Line2D([0], [0], color='b', label='Late retreat year', mfc='none', markeredgewidth=2, ls='None', **legend_trend_detrend)
        fjord_ice_breakup = Line2D([0], [0], color='g', label='Fjord-ice breakup', mfc='g', mec='g', lw=1, **legend_trend_detrend)
        leg = ax.legend(bbox_to_anchor=(0.5, 0.5), loc='center', handles=[original, detrended, early, late, trend_line, fjord_ice_breakup],
                    ncol=3, columnspacing=1, borderaxespad=0, fontsize=8, handletextpad=0.1, labelspacing=0.3, frameon=False, handlelength=1.2, handleheight=1.2,
                edgecolor='none') # facecolor='w', framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None)}

        targets = dict(left=265, right=10, top=10, bottom=30, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_early_late_sea_ice_dor_years'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_early_late_sea_ice_dor_years_grsea_ys.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

    def print_instantaneous_correlation_between_sea_ice_and_zac_variables_for_all_dor():
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        # phenologies = ['iifp']
        phenology = 'dor'
        mask_area = 'within_buffers'
        mask_area_var = mask_area.split('_')[0]
        radius = 800

        df_dor_doy = pd.read_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
        df_dor_doy.index = df_dor_doy['Unnamed: 0']
        df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
        variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
        variables_on_ci.set_index('Unnamed: 0', inplace=True)
        variables_on_ci['doy'] = variables_on_ci.index.dayofyear
        variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
        variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed
        variables_on_ci = variables_on_ci.loc['1990':'2019']

        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m']  * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        # variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2019']

        # Select month
        # variables_on_zac = variables_on_zac[variables_on_zac.index.month.isin([5, 6, 7, 8, 9])]  # Select summer months (June, July, August)
        # variables_on_ci = variables_on_ci[variables_on_ci.index.month.isin([5, 6, 7, 8, 9])]  # Select summer months (June, July, August)
        # variables_on_zac = variables_on_zac[variables_on_zac.index.month.isin([9])]  # Select summer months (June, July, August)
        # variables_on_ci = variables_on_ci[variables_on_ci.index.month.isin([9])]  # Select summer months (June, July, August)

        # Detrend correlation
        print(f"Spearman correlation between detrended")
        # -----------------------------------------------------------
        # Between Latent heat flux on CI and specific humidity on ZAC
        # -----------------------------------------------------------
        time = np.array(range(1, len(variables_on_ci.index) + 1))
        slope, intercept, r, p, _ = linregress(time, variables_on_ci['latf'])
        # Estimated trend
        trend = intercept + slope * time
        # Detrended series (preserve mean)
        variables_on_ci[f"latf_detrended"] = variables_on_ci['latf'] - trend + variables_on_ci['latf'].mean()

        time = np.array(range(1, len(variables_on_zac.index) + 1))
        slope, intercept, r, p, _ = linregress(time, variables_on_zac['q2m'])
        # Estimated trend
        trend = intercept + slope * time
        # Detrended series (preserve mean)
        variables_on_zac[f"q2m_detrended"] = variables_on_zac['q2m'] - trend + variables_on_zac['q2m'].mean()

        x = variables_on_ci['latf_detrended']
        y = variables_on_zac['q2m_detrended']
        rho, p_value = spearmanr(x, y)
        print(f"latf on CI and q2m on ZAC: rho = {rho}, p-value = {p_value}")

        # -----------------------------------------------------------
        # Between Sensible heat flux on CI and T2m on ZAC
        # -----------------------------------------------------------
        time = np.array(range(1, len(variables_on_ci.index) + 1))
        slope, intercept, r, p, _ = linregress(time, variables_on_ci['senf'])
        # Estimated trend
        trend = intercept + slope * time
        # Detrended series (preserve mean)
        variables_on_ci[f"senf_detrended"] = variables_on_ci['senf'] - trend + variables_on_ci['senf'].mean()

        time = np.array(range(1, len(variables_on_zac.index) + 1))
        slope, intercept, r, p, _ = linregress(time, variables_on_zac['t2m'])
        # Estimated trend
        trend = intercept + slope * time
        # Detrended series (preserve mean)
        variables_on_zac[f"t2m_detrended"] = variables_on_zac['t2m'] - trend + variables_on_zac['t2m'].mean()

        x = variables_on_ci['senf_detrended']
        y = variables_on_zac['t2m_detrended']
        rho, p_value = spearmanr(x, y)
        print(f"senf on CI and t2m on ZAC: rho = {rho}, p-value = {p_value}")

        # %%
    def calculate_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor_month():
        # %%
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        # phenologies = ['iifp']
        phenology = 'dor'
        mask_area = 'within_buffers'
        mask_area_var = mask_area.split('_')[0]
        radius = 800

        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        df_dor_doy = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
        df_dor_doy.index = df_dor_doy['Unnamed: 0']
        df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var/from_ci')
        variables_on_ci = pd.read_csv(f"{main_path}/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
        variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
        variables_on_ci.set_index('Unnamed: 0', inplace=True)
        variables_on_ci['doy'] = variables_on_ci.index.dayofyear
        variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
        variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
        variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed

        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var/from_zac/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv')
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/extracted_racmo_var/from_zac/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv')
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m']  * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        # variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2019']

        # Climatological mean of senf, latf, t2m and q2m during 1990–2019
        variables_on_ci_doy_mean = variables_on_ci.groupby(variables_on_ci.index.dayofyear).mean()
        variables_on_zac_doy_mean = variables_on_zac.groupby(variables_on_zac.index.dayofyear).mean()

        months = [5, 6, 7, 8, 9]
        df_corr_early_latf_q2m_r = pd.DataFrame(index= months)
        df_corr_late_latf_q2m_r = pd.DataFrame(index= months)
        df_corr_early_senf_t2m_r = pd.DataFrame(index= months)
        df_corr_late_senf_t2m_r = pd.DataFrame(index= months)
        df_corr_early_latf_q2m_p = pd.DataFrame(index= months)
        df_corr_late_latf_q2m_p = pd.DataFrame(index= months)
        df_corr_early_senf_t2m_p = pd.DataFrame(index= months)
        df_corr_late_senf_t2m_p = pd.DataFrame(index= months)
        for period in ['early', 'late']:#early, 'late'
            years = df_dor_doy.loc[df_dor_doy['composite'] == period].index
            for year in years:
                variables_on_ci_year = variables_on_ci.loc[str(year)]
                variables_on_ci_year_doy = variables_on_ci_year.groupby(variables_on_ci_year.index.dayofyear).mean()
                # variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy - variables_on_ci_doy_mean
                variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy.copy()

                variables_on_zac_year = variables_on_zac.loc[str(year)]
                variables_on_zac_year_doy = variables_on_zac_year.groupby(variables_on_zac_year.index.dayofyear).mean()
                # variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy - variables_on_zac_doy_mean
                variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy.copy()

                for doy in [(121, 151, 5), (152, 181, 6), (182, 212, 7), (213, 243, 8), (244, 273, 9)]:
                    start_doy, end_doy, month = doy
                    variables_on_ci_year_doy_anomalies_month = variables_on_ci_year_doy_anomalies.loc[start_doy:end_doy]
                    variables_on_zac_year_doy_anomalies_month = variables_on_zac_year_doy_anomalies.loc[start_doy:end_doy]

                    for var_pair in [['latf', 'q2m'], ['senf', 't2m']]:
                        # Extract the two series
                        x = variables_on_ci_year_doy_anomalies_month.loc[:, var_pair[0]]
                        y = variables_on_zac_year_doy_anomalies_month.loc[:, var_pair[1]]

                        # Align by index and remove NaNs
                        df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

                        # Calculate Spearman correlation
                        rho, p_value = spearmanr(df[var_pair[0]], df[var_pair[1]])
                        if var_pair == ['latf', 'q2m']:
                            if period == 'early':
                                df_corr_early_latf_q2m_r.loc[month, year] = rho
                                df_corr_early_latf_q2m_p.loc[month, year] = p_value
                            elif period == 'late':
                                df_corr_late_latf_q2m_r.loc[month, year] = rho
                                df_corr_late_latf_q2m_p.loc[month, year] = p_value
                        elif var_pair == ['senf', 't2m']:
                            if period == 'early':
                                df_corr_early_senf_t2m_r.loc[month, year] = rho
                                df_corr_early_senf_t2m_p.loc[month, year] = p_value
                            elif period == 'late':
                                df_corr_late_senf_t2m_r.loc[month, year] = rho
                                df_corr_late_senf_t2m_p.loc[month, year] = p_value

        # Save the outputs
        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        df_corr_early_latf_q2m_r.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_latf_q2m_r_month.csv")
        df_corr_early_latf_q2m_p.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_latf_q2m_p_month.csv")
        df_corr_late_latf_q2m_r.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_latf_q2m_r_month.csv")
        df_corr_late_latf_q2m_p.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_latf_q2m_p_month.csv")
        df_corr_early_senf_t2m_r.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_senf_t2m_r_month.csv")
        df_corr_early_senf_t2m_p.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_senf_t2m_p_month.csv")
        df_corr_late_senf_t2m_r.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_senf_t2m_r_month.csv")
        df_corr_late_senf_t2m_p.to_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_senf_t2m_p_month.csv")
        # %%
    def plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor_month():
        # %%
        """Plot instantaneous correlation between sea ice phenology and ZAC variables for early and late DOR years.

        """
        # %%
        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')

        # Read latent heat flux and specific humidity correlation results
        # Early
        df_corr_early_latf_q2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_latf_q2m_r_month.csv")
        df_corr_early_latf_q2m_r.index = df_corr_early_latf_q2m_r['Unnamed: 0']
        df_corr_early_latf_q2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_latf_q2m_p_month.csv")
        df_corr_early_latf_q2m_p.index = df_corr_early_latf_q2m_p['Unnamed: 0']
        df_corr_early_latf_q2m_r[df_corr_early_latf_q2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_early_latf_q2m_r = df_corr_early_latf_q2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_early_latf_q2m_r = df_corr_early_latf_q2m_r[sorted(df_corr_early_latf_q2m_r.columns, key=int)]
        # Late
        df_corr_late_latf_q2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_latf_q2m_r_month.csv")
        df_corr_late_latf_q2m_r.index = df_corr_late_latf_q2m_r['Unnamed: 0']
        df_corr_late_latf_q2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_latf_q2m_p_month.csv")
        df_corr_late_latf_q2m_p.index = df_corr_late_latf_q2m_p['Unnamed: 0']
        df_corr_late_latf_q2m_r[df_corr_late_latf_q2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_late_latf_q2m_r = df_corr_late_latf_q2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_late_latf_q2m_r = df_corr_late_latf_q2m_r[sorted(df_corr_late_latf_q2m_r.columns, key=int)]

        # Read Sensible heat flux and air temperature correlation results
        # Early
        df_corr_early_senf_t2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_senf_t2m_r_month.csv")
        df_corr_early_senf_t2m_r.index = df_corr_early_senf_t2m_r['Unnamed: 0']
        df_corr_early_senf_t2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_senf_t2m_p_month.csv")
        df_corr_early_senf_t2m_p.index = df_corr_early_senf_t2m_p['Unnamed: 0']
        df_corr_early_senf_t2m_r[df_corr_early_senf_t2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_early_senf_t2m_r = df_corr_early_senf_t2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_early_senf_t2m_r = df_corr_early_senf_t2m_r[sorted(df_corr_early_senf_t2m_r.columns, key=int)]

        df_corr_late_senf_t2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_senf_t2m_r_month.csv")
        df_corr_late_senf_t2m_r.index = df_corr_late_senf_t2m_r['Unnamed: 0']
        df_corr_late_senf_t2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_senf_t2m_p_month.csv")
        df_corr_late_senf_t2m_p.index = df_corr_late_senf_t2m_p['Unnamed: 0']
        df_corr_late_senf_t2m_r[df_corr_late_senf_t2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_late_senf_t2m_r = df_corr_late_senf_t2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_late_senf_t2m_r = df_corr_late_senf_t2m_r[sorted(df_corr_late_senf_t2m_r.columns, key=int)]

        # Plot all the correlations
        colors = [(65, 105, 225), (70, 130, 180), (135, 206, 235), (173, 216, 230), (255, 255, 255), (250, 235, 215), (255, 160, 122), (255, 0 ,0), (178, 34, 34)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0001, 0.0001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]

        fig = plt.figure(figsize=(5.1, 4), dpi=600)
        tick_params_major = dict(direction='out', length=5, pad=1, labelsize=9, width=0.6, color='k')

        for df_var, ax, sub_fig in zip([df_corr_early_latf_q2m_r, df_corr_late_latf_q2m_r,
                               df_corr_early_senf_t2m_r, df_corr_late_senf_t2m_r],
                              [fig.add_axes([0.03, 0.63, 0.53, 0.33]),
                               fig.add_axes([0.03, 0.17, 0.53, 0.33]),
                               fig.add_axes([0.52, 0.63, 0.53, 0.33]),
                               fig.add_axes([0.52, 0.17, 0.53, 0.33])],
                              ['(a)', '(c)', '(b)', '(d)']):
            sns.heatmap(df_var, ax=ax, cmap=cmap, norm=norm, linewidths=0.5, linecolor='w', cbar=False)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep'], rotation=0)
            ax.set_ylabel('')

            # if df_var is df_corr_early_senf_t2m_r or df_var is df_corr_late_senf_t2m_r:
            #     ax.set_ylabel('')
            # else:
            #     ax.set_ylabel('Buffer radius (km)')

            ax.tick_params(axis='both', which='major', **tick_params_major)
            ax.text(-0.01, 1.05, f"{sub_fig}", ha='center', va='center', fontsize=11, fontweight='bold', transform=ax.transAxes)

        fig.text(0.014, 0.8, 'Early retreat years', ha='center', va='center', fontsize=10, rotation=90)
        fig.text(0.014, 0.35, 'Late retreat years', ha='center', va='center', fontsize=10, rotation=90)
        fig.text(0.3, 0.98, r'r(LH$_\mathdefault{GrSea}$, q$_\mathdefault{2m, ZAC}$)', ha='center', va='center', fontsize=10)
        fig.text(0.77, 0.98, r'r(SH$_\mathdefault{GrSea}$, T$_\mathdefault{2m, ZAC}$)', ha='center', va='center', fontsize=10)

        cbar_ax = fig.add_axes([0.23, 0.035, 0.6, 0.03])
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm)
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbtick)
        cb.ax.tick_params(labelsize=8, pad=1, length=2, direction='in', width=0.7)
        cb.outline.set_linewidth(0.4)

        logger.info(f"Saving plot of 'plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor_month'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor_month.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def calculate_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor():
        # %%
        """Calculate instantaneous correlation between sea ice phenology and ZAC variables for early and late DOR years.

        """
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        # phenologies = ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        # phenologies = ['iifp']
        phenologies = ['dor']
        for phenology in phenologies:
            for mask_area in ['within_buffers']:
                df_corr_early_latf_q2m_r = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_late_latf_q2m_r = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_early_senf_t2m_r = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_late_senf_t2m_r = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_early_latf_q2m_p = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_late_latf_q2m_p = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_early_senf_t2m_p = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                df_corr_late_senf_t2m_p = pd.DataFrame(index= [200, 300, 400, 500, 600, 700, 800])
                for radius in [200, 300, 400, 500, 600, 700, 800]:
                    df_dor_doy = pd.read_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
                    df_dor_doy.index = df_dor_doy['Unnamed: 0']
                    df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

                    # Extract variables for early and late phenology years
                    # Composite of the masked sea ice variables during early and late years
                    if mask_area == 'inbetween_buffers':
                        mask_area_var = 'between'
                        if radius == 100:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_YS_100_{phenology}_sea_ice.csv")
                        else:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius-100}_{radius}_{phenology}_sea_ice.csv")
                    elif mask_area == 'within_buffers':
                        mask_area_var = mask_area.split('_')[0]
                        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
                    variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
                    variables_on_ci.set_index('Unnamed: 0', inplace=True)
                    variables_on_ci['doy'] = variables_on_ci.index.dayofyear
                    variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
                    variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed

                    # Atmoshperic variables on Zackenberg during early and late years
                    zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
                    zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
                    zac_t2m.set_index('time', inplace=True)
                    zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

                    zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
                    zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
                    zac_q2m.set_index('time', inplace=True)
                    zac_q2m['q2m'] = zac_q2m['q2m']  * 1000  # Convert specific humidity from kg/kg to g/kg

                    variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
                    variables_on_zac['doy'] = variables_on_zac.index.dayofyear
                    variables_on_zac = variables_on_zac.loc['1990':'2020']

                    # Climatological mean of senf, latf, t2m and q2m during 1990–2020
                    variables_on_ci_doy_mean = variables_on_ci.groupby(variables_on_ci.index.dayofyear).mean()
                    variables_on_zac_doy_mean = variables_on_zac.groupby(variables_on_zac.index.dayofyear).mean()

                    # Calculate Anomalies
                    for period in ['early', 'late']:
                        years = df_dor_doy.loc[df_dor_doy['composite'] == period].index
                        for year in years:
                            variables_on_ci_year = variables_on_ci.loc[str(year)]
                            variables_on_ci_year_doy = variables_on_ci_year.groupby(variables_on_ci_year.index.dayofyear).mean()
                            # variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy - variables_on_ci_doy_mean
                            variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy.copy()

                            variables_on_zac_year = variables_on_zac.loc[str(year)]
                            variables_on_zac_year_doy = variables_on_zac_year.groupby(variables_on_zac_year.index.dayofyear).mean()
                            # variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy - variables_on_zac_doy_mean
                            variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy.copy()

                            for var_pair in [['latf', 'q2m'], ['senf', 't2m']]:
                                # Extract the two series
                                x = variables_on_ci_year_doy_anomalies.loc[121:273, var_pair[0]]
                                y = variables_on_zac_year_doy_anomalies.loc[121:273, var_pair[1]]

                                # Align by index and remove NaNs
                                df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

                                # Calculate Spearman correlation
                                rho, p_value = spearmanr(df[var_pair[0]], df[var_pair[1]])

                                if var_pair == ['latf', 'q2m']:
                                    if period == 'early':
                                        df_corr_early_latf_q2m_r.loc[radius, year] = rho
                                        df_corr_early_latf_q2m_p.loc[radius, year] = p_value
                                    elif period == 'late':
                                        df_corr_late_latf_q2m_r.loc[radius, year] = rho
                                        df_corr_late_latf_q2m_p.loc[radius, year] = p_value
                                elif var_pair == ['senf', 't2m']:
                                    if period == 'early':
                                        df_corr_early_senf_t2m_r.loc[radius, year] = rho
                                        df_corr_early_senf_t2m_p.loc[radius, year] = p_value
                                    elif period == 'late':
                                        df_corr_late_senf_t2m_r.loc[radius, year] = rho
                                        df_corr_late_senf_t2m_p.loc[radius, year] = p_value

                # Save the outputs
                df_corr_early_latf_q2m_r.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_latf_q2m_r.csv")
                df_corr_early_latf_q2m_p.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_latf_q2m_p.csv")
                df_corr_late_latf_q2m_r.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_latf_q2m_r.csv")
                df_corr_late_latf_q2m_p.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_latf_q2m_p.csv")
                df_corr_early_senf_t2m_r.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_senf_t2m_r.csv")
                df_corr_early_senf_t2m_p.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_early_senf_t2m_p.csv")
                df_corr_late_senf_t2m_r.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_senf_t2m_r.csv")
                df_corr_late_senf_t2m_p.to_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_detrended_doy_late_senf_t2m_p.csv")

# %%
    def plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor():
        # %%
        """Plot instantaneous correlation between sea ice phenology and ZAC variables for early and late DOR years.

        """
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/sea_ice_smb_melt_indicator')

        # Read latent heat flux and specific humidity correlation results
        # Early
        df_corr_early_latf_q2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_latf_q2m_r.csv")
        df_corr_early_latf_q2m_r.index = df_corr_early_latf_q2m_r['Unnamed: 0']
        df_corr_early_latf_q2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_latf_q2m_p.csv")
        df_corr_early_latf_q2m_p.index = df_corr_early_latf_q2m_p['Unnamed: 0']
        df_corr_early_latf_q2m_r[df_corr_early_latf_q2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_early_latf_q2m_r = df_corr_early_latf_q2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_early_latf_q2m_r = df_corr_early_latf_q2m_r[sorted(df_corr_early_latf_q2m_r.columns, key=int)]
        # Late
        df_corr_late_latf_q2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_latf_q2m_r.csv")
        df_corr_late_latf_q2m_r.index = df_corr_late_latf_q2m_r['Unnamed: 0']
        df_corr_late_latf_q2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_latf_q2m_p.csv")
        df_corr_late_latf_q2m_p.index = df_corr_late_latf_q2m_p['Unnamed: 0']
        df_corr_late_latf_q2m_r[df_corr_late_latf_q2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_late_latf_q2m_r = df_corr_late_latf_q2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_late_latf_q2m_r = df_corr_late_latf_q2m_r[sorted(df_corr_late_latf_q2m_r.columns, key=int)]

        # Read Sensible heat flux and air temperature correlation results
        # Early
        df_corr_early_senf_t2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_senf_t2m_r.csv")
        df_corr_early_senf_t2m_r.index = df_corr_early_senf_t2m_r['Unnamed: 0']
        df_corr_early_senf_t2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_early_senf_t2m_p.csv")
        df_corr_early_senf_t2m_p.index = df_corr_early_senf_t2m_p['Unnamed: 0']
        df_corr_early_senf_t2m_r[df_corr_early_senf_t2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_early_senf_t2m_r = df_corr_early_senf_t2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_early_senf_t2m_r = df_corr_early_senf_t2m_r[sorted(df_corr_early_senf_t2m_r.columns, key=int)]

        df_corr_late_senf_t2m_r = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_senf_t2m_r.csv")
        df_corr_late_senf_t2m_r.index = df_corr_late_senf_t2m_r['Unnamed: 0']
        df_corr_late_senf_t2m_p = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_dor_within_buffers_detrended_doy_late_senf_t2m_p.csv")
        df_corr_late_senf_t2m_p.index = df_corr_late_senf_t2m_p['Unnamed: 0']
        df_corr_late_senf_t2m_r[df_corr_late_senf_t2m_p > 0.05] = np.nan  # Set non-significant correlations to NaN
        df_corr_late_senf_t2m_r = df_corr_late_senf_t2m_r.drop(columns=['Unnamed: 0'])
        # Sort the columns of the DataFrame based on the integer values of the column names
        df_corr_late_senf_t2m_r = df_corr_late_senf_t2m_r[sorted(df_corr_late_senf_t2m_r.columns, key=int)]

        # Plot all the correlations
        colors_main_rgb = [(8, 64, 129), (8, 104, 172), (43, 140, 190), (78, 179, 211), (123, 204, 196), (168, 221, 181), (204, 235, 197), (224, 243, 219), (247, 252, 240), (255, 255, 255)]
        cmap = fp.make_cmap(colors_main_rgb, bit=True)
        bounds = np.array([-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.0001, 0.0001])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [-1, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0]

        fig = plt.figure(figsize=(6, 4), dpi=600)
        tick_params_major = dict(direction='out', length=5, pad=1, labelsize=9, width=0.6, color='k')

        for df_var, ax in zip([df_corr_early_latf_q2m_r, df_corr_late_latf_q2m_r,
                               df_corr_early_senf_t2m_r, df_corr_late_senf_t2m_r],
                              [fig.add_axes([0.05, 0.63, 0.53, 0.33]),
                               fig.add_axes([0.04, 0.17, 0.53, 0.33]),
                               fig.add_axes([0.53, 0.63, 0.53, 0.33]),
                               fig.add_axes([0.515, 0.17, 0.53, 0.33])]):
            sns.heatmap(df_var, ax=ax, cmap=cmap, norm=norm, linewidths=0.5, linecolor='w', cbar=False)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            if df_var is df_corr_early_senf_t2m_r or df_var is df_corr_late_senf_t2m_r:
                ax.set_ylabel('')
            else:
                ax.set_ylabel('Buffer radius (km)')

            ax.tick_params(axis='both', which='major', **tick_params_major)

        fig.text(0.012, 0.8, 'Early Years', ha='center', va='center', fontsize=10, rotation=90)
        fig.text(0.012, 0.4, 'Late Years', ha='center', va='center', fontsize=10, rotation=90)
        fig.text(0.25, 0.98, r'r(LHF, q$_\mathdefault{2m}$)', ha='center', va='center', fontsize=10)
        fig.text(0.75, 0.98, r'r(SHF, T$_\mathdefault{2m}$)', ha='center', va='center', fontsize=10)

        cbar_ax = fig.add_axes([0.23, 0.035, 0.6, 0.03])
        cb = mpl.colorbar.ColorbarBase(cbar_ax, orientation='horizontal', cmap=cmap, norm=norm)
        cb.set_ticks(cbtick)
        cb.minorticks_off()
        cb.update_ticks()
        cb.set_ticklabels(cbtick)
        cb.ax.tick_params(labelsize=8, pad=1, length=2, direction='in', width=0.7)
        cb.outline.set_linewidth(0.4)

        logger.info(f"Saving plot of 'plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_instantaneous_correlation_between_sea_ice_and_zac_variables_for_early_late_dor.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def calculate_lagged_correlation_between_sea_ice_and_zac_variables_for_early_late_dor():
        # %%
        # %%
        """Calculate lagged correlation between sea ice phenology and ZAC variables for early and late DOR."""

        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data')
        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m'] * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2020']

        phenologies = ['dor'] # ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sic_phenology/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            for mask_area in ['within_buffers']: #  ['inbetween_buffers']:
                sea_ice_mask = xr.open_dataset(f"/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                # for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                for radius in [800]:
                    df_dor_doy = pd.read_csv(f"{main_path}/sic_phenology/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
                    df_dor_doy.index = df_dor_doy['Unnamed: 0']
                    df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

                    # Extract variables for early and late phenology years
                    # Composite of the masked sea ice variables during early and late years
                    if mask_area == 'inbetween_buffers':
                        mask_area_var = 'between'
                        if radius == 100:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_YS_100_{phenology}_sea_ice.csv")
                        else:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius-100}_{radius}_{phenology}_sea_ice.csv")
                    elif mask_area == 'within_buffers':
                        mask_area_var = mask_area.split('_')[0]
                        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
                    variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
                    variables_on_ci.set_index('Unnamed: 0', inplace=True)
                    variables_on_ci['doy'] = variables_on_ci.index.dayofyear
                    variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
                    variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed

                    # Climatological mean of senf, latf, t2m and q2m during 1990–2020
                    variables_on_ci_doy_mean = variables_on_ci.groupby(variables_on_ci.index.dayofyear).mean()
                    variables_on_zac_doy_mean = variables_on_zac.groupby(variables_on_zac.index.dayofyear).mean()

                    # Calculate Anomalies
                    for period in ['early', 'late']:
                        years = df_dor_doy.loc[df_dor_doy['composite'] == period].index
                        for year in years:
                            variables_on_ci_year = variables_on_ci.loc[str(year)]
                            variables_on_ci_year_doy = variables_on_ci_year.groupby(variables_on_ci_year.index.dayofyear).mean()
                            variables_on_ci_year_doy_anomalies = variables_on_ci_year_doy - variables_on_ci_doy_mean

                            variables_on_zac_year = variables_on_zac.loc[str(year)]
                            variables_on_zac_year_doy = variables_on_zac_year.groupby(variables_on_zac_year.index.dayofyear).mean()
                            variables_on_zac_year_doy_anomalies = variables_on_zac_year_doy - variables_on_zac_doy_mean

                            for var_pair in [['senf', 't2m'], ['latf', 'q2m']]:
                                # Extract the two series
                                x = variables_on_ci_year_doy_anomalies.loc[:, var_pair[0]]
                                y = variables_on_zac_year_doy_anomalies.loc[:, var_pair[1]]

                                # Align by index and remove NaNs
                                df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

                                # Calculate lagged correlation
                                rolling_corr = pd.DataFrame(index=df.index)
                                # for water_residence_time in [7, 14]:
                                for water_residence_time in [14]:
                                    # fig, ax = plt.subplots(figsize=(10, 4))
                                    for lag in range(0, 10):
                                        rolling_corr[f'lag_{lag}_{water_residence_time}'] = (df[var_pair[0]].shift(lag).rolling(water_residence_time).corr(df[var_pair[1]]))
                                        rolling_corr.loc[rolling_corr[f'lag_{lag}_{water_residence_time}'].abs() < 0.5, f'lag_{lag}_{water_residence_time}'] = np.nan
                                    rolling_corr.to_csv(f"{main_path}/sic_phenology/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy_{period}_{var_pair[0]}_{var_pair[1]}_{year}_lagged_correlation.csv")
    # %%
    def plot_lagged_correlation_between_sea_ice_and_zac_variables_for_early_late_dor():
        # %%
        """Plot lagged correlation between sea ice phenology and ZAC variables for early and late DOR years.
        """
        # %%
        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/sic_phenology')
        # Read all the lagged correlation results
        phenology = 'dor'
        mask_area = 'within_buffers'
        radius = 800
        var_pair = ['latf', 'q2m']
        # var_pair = ['senf', 't2m']

        # for period in ['early', 'late']:
        df_dor_doy = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
        df_dor_doy.index = df_dor_doy['Unnamed: 0']
        df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

        # Plot all the correlations
        colors = [(3, 37, 76), (5, 48, 97), (8, 81, 156), (33, 113, 181), (66, 146, 198), (107, 174, 214), (141, 211, 199), (173, 221, 190),  (199, 233, 180), (116, 196, 118), (35, 139, 69), (0, 90, 50)]
        colors = colors[::-1]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for period in ['early', 'late']:
            fig = plt.figure(figsize=(7, 6), dpi=600)
            # left years | right years | colorbar
            outer = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 0.04], hspace=0.08, wspace=0.15)
            # left four years
            gs_left = outer[0, 0].subgridspec(4, 1, hspace=0.275)
            # right four years
            gs_right = outer[0, 1].subgridspec(4, 1, hspace=0.275)

            years = df_dor_doy.loc[df_dor_doy['composite'] == period].index
            for years, gs, fig_labels in zip([years[:4], years[4:]], [gs_left, gs_right], [['(a)', '(c)', '(e)', '(g)'], ['(b)', '(d)', '(f)', '(h)']]):
                zorder = 16
                for i, (year, fig_label) in enumerate(zip(years, fig_labels)):
                    gs_pair = gs[i].subgridspec(2, 1, hspace=0.1)    # SMALL gap between + and -
                    # positive correlation
                    ax_pos = fig.add_subplot(gs_pair[0], zorder=zorder)
                    # negative correlation
                    zorder = zorder - 1
                    ax_neg = fig.add_subplot(gs_pair[1], zorder=zorder)
                    # for var_pair in [['latf', 'q2m']]:
                    rolling_corr = pd.read_csv(f"{main_path}/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy_{period}_{var_pair[0]}_{var_pair[1]}_{year}_lagged_correlation.csv")
                    rolling_corr.set_index('Unnamed: 0', inplace=True)
                    # Create dictionary
                    lag_colors = dict(zip(rolling_corr.columns, colors))

                    for lag in rolling_corr.columns:
                        if not lag.startswith("lag_"):
                            continue

                        pos = rolling_corr[lag].where(rolling_corr[lag] >= 0.5)
                        neg = rolling_corr[lag].where(rolling_corr[lag] <= -0.5)

                        ax_pos.plot(rolling_corr.index, pos, lw=1, marker='o', ms=2, color=lag_colors[lag])
                        ax_neg.plot(rolling_corr.index, neg, lw=1, marker='o', ms=2, color=lag_colors[lag])
                    ax_pos.axvline(x=df_dor_doy.loc[year, f"{radius}"], color='gray', lw=0.6, ls='--', zorder=1,)
                    ax_neg.axvline(x=df_dor_doy.loc[year, f"{radius}"], color='gray', lw=0.6, ls='--', zorder=1,)

                    ax_pos.text(0.001, 1.05, rf"$\mathbf{{{fig_label}}}$ {year}", transform=ax_pos.transAxes, fontsize=8, fontweight='bold', color='k', ha='left', va='top')

                    ax_pos.text(0.001, 0.01, '+', fontsize=6, ha='center', va='center', bbox=dict(boxstyle='circle', fc='white', ec='black'), transform=ax_pos.transAxes)
                    ax_neg.text(0.001, 0.01, '−', fontsize=6, ha='center', va='center', bbox=dict(boxstyle='circle', fc='white', ec='black'), transform=ax_neg.transAxes)

                    ax_pos.set_xlim(121, 273)
                    ax_pos.set_xticks([121, 152, 182, 213, 244, 273])
                    ax_pos.set_xticklabels(['', '', '', '', '', ''])
                    ax_pos.set_xticks([135, 166, 196, 225, 258], minor=True)
                    ax_pos.set_xticklabels(['', '', '', '', ''], minor=True)

                    ax_neg.set_xlim(121, 273)
                    ax_neg.set_xticks([121, 152, 182, 213, 244, 273])
                    ax_neg.set_xticklabels([121, 152, 182, 213, 244, 273])
                    ax_neg.set_xticks([135, 166, 196, 228, 258], minor=True)
                    ax_neg.set_xticklabels(['May', 'Jun', 'Jul', 'Aug', 'Sep'], minor=True)

                    ax_pos.set_ylim(0.45, 1)
                    ax_pos.set_yticks([0.5, 0.75, 1])
                    ax_pos.set_yticklabels([0.5, 0.75, 1])
                    ax_pos.set_yticks([0.625, 0.875], minor=True)

                    ax_neg.set_ylim(-1, -0.45)
                    ax_neg.set_yticks([-1, -0.75, -0.5])
                    ax_neg.set_yticklabels([-1, -0.75, -0.5])
                    ax_neg.set_yticks([-0.875, -0.625], minor=True)

                    [ax_pos.spines[i].set_visible(False) for i in ['top', 'right']]
                    [i.set_linewidth(0.6) for i in ax_pos.spines.values()]
                    ax_pos.spines['left'].set_bounds(0.5, 1)
                    ax_spine_shift = 3 # Adjust this value to control the outward shift of the spines
                    ax_pos.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
                    ax_pos.spines['bottom'].set_bounds(121, 273)
                    # ax_pos.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

                    [ax_neg.spines[i].set_visible(False) for i in ['top', 'right']]
                    [i.set_linewidth(0.6) for i in ax_neg.spines.values()]
                    ax_neg.spines['left'].set_bounds(-1, -0.5)
                    ax_spine_shift = 3 # Adjust this value to control the outward shift of the spines
                    ax_neg.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
                    ax_neg.spines['bottom'].set_bounds(121, 273)
                    ax_neg.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

                    # Customize ticks
                    ytick_params = dict(direction='out', width=0.5, length=5, pad=2, labelsize=8, color='k')
                    ax_pos.tick_params(axis='both', which='major', **ytick_params)
                    ax_neg.tick_params(axis='both', which='major', **ytick_params)

                    ax_pos.tick_params(axis='x', which='minor', length=2.5, pad=0.5, labelsize=7)
                    ax_neg.tick_params(axis='x', which='minor', length=0, pad=0.5, labelsize=7)
                    ytick_minor_params = dict(axis='y', which='minor', direction='out', width=0.5, length=2.5)
                    ax_pos.tick_params(**ytick_minor_params)
                    ax_neg.tick_params(**ytick_minor_params)

            # Colorbar axis spanning both columns
            ax_cbar = fig.add_subplot(outer[1, :])
            ax_cbar.axis('off')  # Hide the axes for the legend

            ax = ax_cbar.inset_axes([0.2, 0.25, 0.6, 0.5]) #[left, bottom, width, height]
            cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap, norm=norm)
            cb.set_ticks(cbtick)
            cb.minorticks_off()
            cb.update_ticks()
            cb.set_ticklabels(cbtick)
            cb.ax.tick_params(labelsize=8, pad=1, length=2, direction='in', width=0.7)
            cb.outline.set_linewidth(0.6)
            # Add title to the left of colorbar
            ax_cbar.text(0.18, 0.4, 'Lag \ndays', transform=ax_cbar.transAxes, fontsize=9, va='center', ha='center')

            targets = dict(left=250, right=55, top=30, bottom=30, hspace=0, wspace=0)
            # Hook up a function to adjust the borders when the window is resized
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'plot_lagged_correlation_between_sea_ice_and_zac_variables_for_early_late_dor'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f"plot_lagged_correlation_between_sea_ice_and_zac_variables_for_{period}_{var_pair[0]}_{var_pair[1]}_dor.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

    def plot_wind_directions_during_early_late_dor():
        # %%
        phenology = 'dor'
        mask_area = 'within_buffers'
        radius = 800
        rlon_slice = slice(2, 11.2)
        rlat_slice = slice(-5, 12)

        main_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos')

        # Read Greenland mask and merge it with Svalbard and Iceland
        gr_masks_5km = xr.open_dataset(f"{main_path}/data/raw_data/masks/FGRN055_Masks_5.5km_updated_coordinates.nc")
        gr_masks_5km = gr_masks_5km.sel(rlon=rlon_slice, rlat=rlat_slice)
        land_sea_mask = gr_masks_5km['LSM_GR']
        topo = gr_masks_5km['Topography']
        topo_c = topo.copy()
        # Mask Svalbard and Iceland
        topo_c.values[(topo_c.values > 0) & (land_sea_mask.values == 0)] = 0
        # Mask Greenland to seperate Iceland and Svalbard
        topo.values[(topo.values == topo_c.values)] = 0
        # Add Iceland and Svalbard in land sea mask
        land_sea_mask.values[(topo.values > 100)] = 1

        land_sea_mask = xr.where((land_sea_mask == 0), np.nan, land_sea_mask) # to avoid hiding the colors below
        colors_landseamask = [(255, 255, 255), (169, 169, 169)]

        cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
        bounds_landseamask = np.array([-0.01, 0.01, 1.5])
        norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

        # Read wind components from RACMO5km
        input_file_path = Path(f"{main_path}/data/raw_data/complex_mca_input")

        da_u10m = xr.open_dataset(f"{input_file_path}/u10m_all.nc")['u10m']
        da_u10m = da_u10m.sel(height=0)
        da_u10m_latlon = da_u10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        da_v10m = xr.open_dataset(f"{input_file_path}/v10m_all.nc")['v10m']
        da_v10m = da_v10m.sel(height=0)
        da_v10m_latlon = da_v10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        da_ff10m = xr.open_dataset(f"{input_file_path}/ff10m_all.nc")['ff10m']
        da_ff10m = da_ff10m.sel(height=0)
        da_ff10m_latlon = da_ff10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        # Read the mask for sea ice within buffers for DOR significant trend at RACMO 5 km resolution
        sea_ice_mask = xr.open_dataset(f"{main_path}/data/raw_data/masks/ci_all_within_buffers_dor_trend_racmo5km.nc")['sea_ice_buffer']
        sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)

        # Read Freya geographic coordinates from station metadata
        freya_aws_meta_data = read_station_metadata(station='FRE2 (promice type, tripod)')
        station_lon, station_lat = freya_aws_meta_data.loc[['longitude', 'latitude'], 'FRE2 (promice type, tripod)']

        # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{main_path}/data/raw_data/masks/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # Read station data from RACMO 5.5 km for Zackenberg (ZAC) to get wind direction
        zac_wd_daily_mod = read_station_data_from_racmo(station='zac', variable='wd')

        # Read the detrended DOY for early and late DOR years
        df_dor_doy = pd.read_csv(f"{main_path}/data/processed_data/sic_phenology/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
        df_dor_doy.index = df_dor_doy['Unnamed: 0']

        # Round the average DOY to the nearest integer
        df_dor_doy['date'] = pd.to_datetime(df_dor_doy['Unnamed: 0'].astype(int).astype(str) +
                                            df_dor_doy[str(radius)].astype(int).astype(str).str.zfill(3), format='%Y%j')

        # Latitude and longitude
        racmo_latitude = gr_masks_5km['lat']
        racmo_longitude = gr_masks_5km['lon']

        # Wind speed color
        colors = [(3, 37, 76), (5, 48, 97), (8, 81, 156), (33, 113, 181), (66, 146, 198),
                  (107, 174, 214), (141, 211, 199), (173, 221, 190), (199, 233, 180), (116, 196, 118),
                  (35, 139, 69), (0, 90, 50), (0, 70, 35), (0, 50, 20), (0, 30, 10)]      # darkest
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 2, 4, 6, 8, 10, 12]

        for period in ['early', 'late']:
            fig = plt.figure(figsize=(25, 15), dpi=600)
            outer = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 0.04], hspace=0)
            gs_top = outer[0, :].subgridspec(2, 4, hspace=0., wspace=0.)

            dates = df_dor_doy.loc[df_dor_doy['composite'] == period, 'date']
            for i, date in enumerate(dates):
                ax = fig.add_subplot(gs_top[i])
                ax.axis('off')  # Hide the axes for the legend

                da_u10m_date = da_u10m_latlon.sel(time=date)
                da_v10m_date = da_v10m_latlon.sel(time=date)
                da_ff10m_date = da_ff10m_latlon.sel(time=date)

                # Plot wind speed with land-sea mask and sea ice mask
                ax.contourf(racmo_longitude, racmo_latitude, da_ff10m_date, cmap=cmap, norm=norm, levels=bounds, extend='max', zorder=1)
                ax.contourf(racmo_longitude, racmo_latitude, land_sea_mask, cmap=cmap_landseamask, norm=norm_landseamask, zorder=2)

                # Plot sea ice mask contour where DOR trend is significant
                ax.contour(racmo_longitude, racmo_latitude, sea_ice_mask_radius, levels=[0.5], colors='k', linewidths=2, linestyles='-', zorder=3)

                # Plot wind direction
                step = 20
                ax.quiver(racmo_longitude.values[::step,::step], racmo_latitude.values[::step,::step], da_u10m_date.values[::step,::step], da_v10m_date.values[::step,::step],
                          color='w', scale=150, zorder=4) # scale=250

                # Plot the closest point to Freya on the map
                ax.plot(station_lon, station_lat, marker='x', color='k', markeredgewidth=5, markersize=20, label='Closest point to Freya', zorder=5)
                ax.text(0.51, 0.95, date.strftime('%Y-%m-%d'), transform=ax.transAxes, fontsize=26, path_effects=[pe.withStroke(linewidth=2, foreground='w')], zorder=12, fontweight='bold')

                # Plot wind rose in new axis showing wind direction in ZAC
                lag_day = pd.Timedelta(days=0)  # 7 days before the date
                zac_wd_daily_mod_date = pd.DataFrame(zac_wd_daily_mod.loc[date + lag_day]).T
                zac_wd_daily_mod_date_cd = vs.classify_wind_direction_in_degrees_into_cardinal_direction(zac_wd_daily_mod_date, 'wd')
                zac_wd_daily_mod_date_rad = np.deg2rad(zac_wd_daily_mod_date_cd['wd'])

                lag_day = pd.Timedelta(days=8)  # 7 days before the date
                zac_wd_daily_mod_date_lag = pd.DataFrame(zac_wd_daily_mod.loc[date + lag_day]).T
                zac_wd_daily_mod_date_cd_lag = vs.classify_wind_direction_in_degrees_into_cardinal_direction(zac_wd_daily_mod_date_lag, 'wd')
                zac_wd_daily_mod_date_rad_lag = np.deg2rad(zac_wd_daily_mod_date_cd_lag['wd'])

                ax_inset = ax.inset_axes([0.05, 0.55, 0.25, 0.25],  polar=True) #[left, bottom, width, height]
                # # If you want the first axis to be on top:
                ax_inset.set_theta_offset(np.pi / 2)
                ax_inset.set_theta_direction(-1)

                ax_inset.plot(0, 0, marker='x', color='k', markeredgewidth=3, markersize=14, linestyle='None', zorder=1, clip_on=False)

                theta = zac_wd_daily_mod_date_rad.iloc[0]
                ax_inset.annotate('',
                    xy=(theta, 0.1),      # arrow head at center
                    xytext=(theta, 15),  # arrow tail at outer edge
                    arrowprops=dict(arrowstyle='-|>', color='b', lw=2, zorder=10, mutation_scale=30))

                # theta_lag = zac_wd_daily_mod_date_rad_lag.iloc[0]
                # ax_inset.annotate('',
                #     xy=(theta_lag, 0.15),      # arrow head at center
                #     xytext=(theta_lag, 15),  # arrow tail at outer edge
                #     arrowprops=dict(arrowstyle='-|>', color='b', lw=2, zorder=10, mutation_scale=30))

                # angles_deg = [0, 90, 135, 157.5, 180, 270, 292.5, 315]
                # ax_inset.set_xticks(np.deg2rad(angles_deg))
                ax_inset.set_xticks([n / float(8) * 2 * np.pi for n in range(8)])
                direction_8 = np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                ax_inset.set_xticklabels(direction_8, color='grey', size=14, fontweight='bold', zorder=12)
                # Move labels closer to the circle
                for label in ax_inset.get_xticklabels():
                    label.set_clip_on(False)      # <-- critical
                    label.set_y(0.45)     # default ~1.1, higher = closer
                    label.set_horizontalalignment('center')
                    label.set_verticalalignment('center')
                    label.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
                ax_inset.xaxis.grid(True, which='major', linewidth=0.8, color='grey', linestyle='--', zorder=1)

                # Set yticks and labels
                ax_inset.set_ylim(0, 16)
                ax_inset.set_yticklabels([])
                ax_inset.yaxis.grid(False)

                # Polar spine (outer circle)
                ax_inset.spines['polar'].set_linewidth(0.6)
                ax_inset.spines['polar'].set_zorder(1)
                ax_inset.spines['polar'].set_color('k')

            # Colorbar axis spanning both columns
            gs_cbar = outer[1, :].subgridspec(1, 1)
            ax_cbar = fig.add_subplot(gs_cbar[0, 0])
            ax_cbar.axis('off')  # Hide the axes for the legend

            ax = ax_cbar.inset_axes([0.2, 0.25, 0.6, 0.6]) #[left, bottom, width, height]
            cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal', cmap=cmap, norm=norm, extend='max')
            cb.set_ticks(cbtick)
            cb.minorticks_off()
            cb.update_ticks()
            cb.set_ticklabels(cbtick)
            cb.ax.tick_params(labelsize=24, pad=1.5, length=2, direction='in', width=0.7)
            cb.outline.set_linewidth(0.6)
            # Add title to the left of colorbar
            ax_cbar.text(0.16, 0.6, r'WS (ms$^{\mathdefault{-1}}$)', transform=ax_cbar.transAxes, fontsize=24, va='center', ha='center')
            cb.ax.text(1.04, -0.001, '>13', transform=cb.ax.transAxes, ha='center', va='top', fontsize=24)

            targets = dict(left=100, right=55, top=30, bottom=90, hspace=0, wspace=0)
            # Hook up a function to adjust the borders when the window is resized
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'plot_wind_directions_during_early_late_dor'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f"plot_wind_directions_during_{period}_dor.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def plot_wind_directions_during_fog_event():
        # %%
        # Freya Glacier wind components from AWS and RACMO
        # AWS
        freya_wind_hr = []
        # read_freya_aws_data() # Print the name of variables and their long name available in Freya AWS data
        for v in ['wspd_u', 'wdir_u']:
            df = read_freya_aws_data(variable=v)
            freya_wind_hr.append(df)
        freya_wind_hr = pd.concat(freya_wind_hr, axis=1).sort_index()
        # Calculate daily mean wind direction
        freya_wd_daily_obs = vs.calculate_wind_direction_daily_mean(freya_wind_hr['wspd_u'], freya_wind_hr['wdir_u'], max_values_per_day=24)
        freya_wd_daily_obs.rename(columns={'wd': 'wd_freya_obs'}, inplace=True)

        # RACMO
        # read_station_data_from_racmo(station='freya') # Print the name of variables available in Freya RACMO data
        freya_wd_daily_mod = read_station_data_from_racmo(station='freya', variable='wd')
        freya_wd_daily_mod.rename(columns={'wd': 'wd_freya_mod'}, inplace=True)

        # Zackenberg wind components from AWS and RACMO
        # AWS
        # Print the name of variables and their long name available in Zackenberg AWS data
        # read_zackenberg_aws_data()
        zac_wind_hr = []
        for v in ['ws', 'wd']:
            df = read_zackenberg_aws_data(variable=v)
            zac_wind_hr.append(df)
        zac_wind_hr = pd.concat(zac_wind_hr, axis=1).sort_index()
        # Calculate daily mean wind direction
        zac_wd_daily_obs = vs.calculate_wind_direction_daily_mean(zac_wind_hr['ws'], zac_wind_hr['wd'], max_values_per_day=24)
        zac_wd_daily_obs.rename(columns={'wd': 'wd_zac_obs'}, inplace=True)

        # RACMO
        # read_station_data_from_racmo(station='zac') # Print the name of variables available in Zackenberg RACMO data
        zac_wd_daily_mod = read_station_data_from_racmo(station='zac', variable='wd')
        zac_wd_daily_mod.rename(columns={'wd': 'wd_zac_mod'}, inplace=True)

        # Concatenate all wind direction data
        wd_combined = pd.concat([freya_wd_daily_obs, freya_wd_daily_mod, zac_wd_daily_obs, zac_wd_daily_mod], axis=1)

        # Read the fog classification data
        results_fog_all = read_fog_classification()

        # Read wind components from RACMO5km
        main_path = Path('/nas/data/processed/people/sos/third_paper_data')

        # Read Greenland mask and merge it with Svalbard and Iceland
        gr_masks_5km = xr.open_dataset(f"{main_path}/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc")
        gr_masks_5km = gr_masks_5km.sel(rlon=rlon_slice, rlat=rlat_slice)
        land_sea_mask = gr_masks_5km['LSM_GR']
        topo = gr_masks_5km['Topography']
        topo_c = topo.copy()
        # Mask Svalbard and Iceland
        topo_c.values[(topo_c.values > 0) & (land_sea_mask.values == 0)] = 0
        # Mask Greenland to seperate Iceland and Svalbard
        topo.values[(topo.values == topo_c.values)] = 0
        # Add Iceland and Svalbard in land sea mask
        land_sea_mask.values[(topo.values > 100)] = 1

        land_sea_mask = xr.where((land_sea_mask == 0), np.nan, land_sea_mask) # to avoid hiding the colors below
        colors_landseamask = [(255, 255, 255), (169, 169, 169)]

        cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
        bounds_landseamask = np.array([-0.01, 0.01, 1.5])
        norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

        input_file_path = Path(f"{main_path}/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile")
        rlon_slice = slice(2, 11.2)
        rlat_slice = slice(-5, 12)

        da_u10m = xr.open_dataset(f"{input_file_path}/u10m_all.nc")['u10m']
        da_u10m = da_u10m.sel(height=0)
        da_u10m_latlon = da_u10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        da_v10m = xr.open_dataset(f"{input_file_path}/v10m_all.nc")['v10m']
        da_v10m = da_v10m.sel(height=0)
        da_v10m_latlon = da_v10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        da_ff10m = xr.open_dataset(f"{input_file_path}/ff10m_all.nc")['ff10m']
        da_ff10m = da_ff10m.sel(height=0)
        da_ff10m_latlon = da_ff10m.sel(rlon=rlon_slice, rlat=rlat_slice)

        # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
        freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{main_path}/freyagl_svd_analysis/input_dataset/masks/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
        freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
        station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

        # Latitude and longitude
        racmo_latitude = gr_masks_5km['lat']
        racmo_longitude = gr_masks_5km['lon']

        # Wind speed color
        colors = [(3, 37, 76), (5, 48, 97), (8, 81, 156), (33, 113, 181), (66, 146, 198),
                  (107, 174, 214), (141, 211, 199), (173, 221, 190), (199, 233, 180), (116, 196, 118),
                  (35, 139, 69), (0, 90, 50), (0, 70, 35), (0, 50, 20), (0, 30, 10)]      # darkest
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        cbtick = [0, 2, 4, 6, 8, 10, 12]
        # %%

        results_fog_event = results_fog_all.loc[results_fog_all['events_fixed'] == 11]
        results_fog_event_start = results_fog_event.loc[:, 'start_time']
        results_fog_event_start = pd.to_datetime(results_fog_event_start.iloc[0])
        lag_day = pd.Timedelta(days=0)
        date = results_fog_event_start.normalize() - lag_day
        print(date)

        da_u10m_date = da_u10m_latlon.sel(time=date)
        da_v10m_date = da_v10m_latlon.sel(time=date)
        da_ff10m_date = da_ff10m_latlon.sel(time=date)

        fig, ax = plt.subplots(figsize=(25, 15), dpi=600)
        # Plot wind direction
        step = 20
        ax.quiver(racmo_longitude.values[::step,::step], racmo_latitude.values[::step,::step],
                  da_u10m_date.values[::step,::step], da_v10m_date.values[::step,::step],
                    color='k', scale=150, zorder=4) # scale=250

        # Plot wind rose in new axis showing wind direction in ZAC
        # lag_day = pd.Timedelta(days=0)  # 7 days before the date
        wd_date = pd.DataFrame(wd_combined.loc[date + lag_day]).T
        # ZAC
        wd_zac_obs_date = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_date, 'wd_zac_obs')
        wd_zac_mod_date = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_date, 'wd_zac_mod')
        print(f"ZAC observed wind direction: {wd_zac_obs_date['wd_zac_obs_class'].values}")
        print(f"ZAC modeled wind direction: {wd_zac_mod_date['wd_zac_mod_class'].values}")
        # FRE
        wd_fre_obs_date = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_date, 'wd_freya_obs')
        wd_fre_mod_date = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_date, 'wd_freya_mod')
        print(f"FRE observed wind direction: {wd_fre_obs_date['wd_freya_obs_class'].values}")
        print(f"FRE modeled wind direction: {wd_fre_mod_date['wd_freya_mod_class'].values}")






        # %%
    def calculate_lagged_correlation_between_sea_ice_and_zac_variables_for_early_late_dor_anomalies():
        # %%
        """Calculate lagged correlation between sea ice phenology and ZAC variables for early and late DOR."""

        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        # Atmoshperic variables on Zackenberg during early and late years
        zac_t2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/t2m_point_zac_5km.csv")
        zac_t2m['time'] = pd.to_datetime(zac_t2m['time'])
        zac_t2m.set_index('time', inplace=True)
        zac_t2m['t2m'] = zac_t2m['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius

        zac_q2m = pd.read_csv(f"{main_path}/extracted_racmo_var/from_zac/q2m_point_zac_5km.csv")
        zac_q2m['time'] = pd.to_datetime(zac_q2m['time'])
        zac_q2m.set_index('time', inplace=True)
        zac_q2m['q2m'] = zac_q2m['q2m'] * 1000  # Convert specific humidity from kg/kg to g/kg

        variables_on_zac = pd.merge(zac_t2m, zac_q2m, left_index=True, right_index=True)
        variables_on_zac['doy'] = variables_on_zac.index.dayofyear
        variables_on_zac = variables_on_zac.loc['1990':'2020']

        phenologies = ['dor'] # ['doo', 'dor', 'doa', 'doc', 'oifp', 'iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            for mask_area in ['within_buffers']: #  ['inbetween_buffers']:
                sea_ice_mask = xr.open_dataset(f"{main_path}/input_dataset/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                # for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                for radius in [800]:
                    df_dor_doy = pd.read_csv(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}_{mask_area}_{radius}km_detrended_doy.csv")
                    df_dor_doy.index = df_dor_doy['Unnamed: 0']
                    df_dor_doy = df_dor_doy.drop(columns=['Unnamed: 0'])

                    # Extract variables for early and late phenology years
                    # Composite of the masked sea ice variables during early and late years
                    if mask_area == 'inbetween_buffers':
                        mask_area_var = 'between'
                        if radius == 100:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_YS_100_{phenology}_sea_ice.csv")
                        else:
                            variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius-100}_{radius}_{phenology}_sea_ice.csv")
                    elif mask_area == 'within_buffers':
                        mask_area_var = mask_area.split('_')[0]
                        variables_on_ci = pd.read_csv(f"{main_path}/extracted_racmo_var/from_ci/masked_5km_variables_{mask_area_var}_{radius}_{phenology}_sea_ice.csv")
                    variables_on_ci['Unnamed: 0'] = pd.to_datetime(variables_on_ci['Unnamed: 0'])
                    variables_on_ci.set_index('Unnamed: 0', inplace=True)
                    variables_on_ci['doy'] = variables_on_ci.index.dayofyear
                    variables_on_ci['tskin'] = variables_on_ci['tskin'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['t2m'] = variables_on_ci['t2m'] - 273.15  # Convert temperature from Kelvin to Celsius
                    variables_on_ci['latf'] = variables_on_ci['latf'] * (-1) # Latitude, no conversion needed
                    variables_on_ci['senf'] = variables_on_ci['senf'] * (-1) # Latitude, no conversion needed

                    # Climatological mean of senf, latf, t2m and q2m during 1990–2020
                    variables_on_ci_doy_mean = variables_on_ci.groupby(variables_on_ci.index.dayofyear).mean()
                    variables_on_zac_doy_mean = variables_on_zac.groupby(variables_on_zac.index.dayofyear).mean()

                    for period in ['early',]:
                        years = df_dor_doy.loc[df_dor_doy['composite'] == period].index

                        # Sea ice variables on Clavering Island during early and late years
                        variables_on_ci_years = variables_on_ci.loc[variables_on_ci.index.year.isin(years)]
                        variables_on_ci_years_doy = variables_on_ci_years.groupby(variables_on_ci_years.index.dayofyear).mean()
                        variables_on_ci_years_doy_anomalies = variables_on_ci_years_doy - variables_on_ci_doy_mean
                        #  Zackenberg variables during early and late years
                        variables_on_zac_years = variables_on_zac.loc[variables_on_zac.index.year.isin(years)]
                        variables_on_zac_years_doy = variables_on_zac_years.groupby(variables_on_zac_years.index.dayofyear).mean()
                        variables_on_zac_years_doy_anomalies = variables_on_zac_years_doy - variables_on_zac_doy_mean
                        for var_pair in [['latf', 'q2m'],]:
                            # Extract the two series
                                x = variables_on_ci_years_doy_anomalies.loc[:, var_pair[0]]
                                y = variables_on_zac_years_doy_anomalies.loc[:, var_pair[1]]

                                # Align by index and remove NaNs
                                df = pd.concat([x, y], axis=1, keys=var_pair).dropna()

                                # Calculate lagged correlation
                                rolling_corr = pd.DataFrame(index=df.index)
                                # for water_residence_time in [7, 14]:
                                for water_residence_time in [14]:
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    for lag in range(0, 11):
                                        rolling_corr[f'lag_{lag}_{water_residence_time}'] = (df[var_pair[0]].shift(lag).rolling(water_residence_time).corr(df[var_pair[1]]))
                                        rolling_corr.loc[rolling_corr[f'lag_{lag}_{water_residence_time}'].abs() < 0.5, f'lag_{lag}_{water_residence_time}'] = np.nan
                                        ax.plot(rolling_corr.index, rolling_corr[f'lag_{lag}_{water_residence_time}'], marker='o', ms=3, label=f"Lag {lag} days ({water_residence_time})", lw=1.5)

                                    ax.axhline(y=0, color='k', lw=1.5, ls='--', zorder=1)
                                    ax.axhline(y=-0.5, color='k', lw=1.5, ls='--', zorder=1)
                                    ax.axhline(y=0.5, color='k', lw=1.5, ls='--', zorder=1)
                                    ax.set_xlim(121, 273)

                                    for year in years:
                                        ax.axvline(x=df_dor_doy.loc[year, f"{radius}"], color='red', lw=0.4, ls='--', zorder=1,)

                                    # ax.set_ylim(0.5, 1)
                                    ax.legend(loc='upper right')
                                    # ax.set_title(f"Year {year}")
                                    ax.set_xlim(121, 273)

# %%
    def wind_direction_analysis():
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')

        phenologies = ['dor']
        # phenologies = ['iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            # for mask_area in ['inbetween_buffers']:
            # for mask_area in ['within_buffers']:
            for mask_area in ['within_buffers']:
                sea_ice_mask = xr.open_dataset(f"{main_path}/input_dataset/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                # for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                for radius in [800]:
                    sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)['sea_ice_buffer']
                    ds_mask = ds.where(sea_ice_mask_radius == 1)
                    ds_mean = ds_mask.median(dim=['rlat', 'rlon'])
                    ds_mean_df = ds_mean.to_dataframe()

                    fig, ax = plt.subplots()
                    ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy"], color='k', lw=1.5, ls='-', zorder=1, label=f"{phenology} {mask_area} {radius} km buffer")
                    ax.legend()

                    # Early years:
                    early_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.25)
                    # early_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.05)
                    early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] < early_doy].index
                    # Late years:
                    late_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.75)
                    # late_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.95)
                    late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy"] > late_doy].index

                    early_years_phenology = ds_mean_df.loc[ds_mean_df.index.isin(early_years)]
                    late_years_phenology = ds_mean_df.loc[ds_mean_df.index.isin(late_years)]

                    early_years_phenology = early_years_phenology.reset_index()
                    late_years_phenology = late_years_phenology.reset_index()

                    # Convert year + DOY to datetime
                    early_years_phenology['date'] = pd.to_datetime(
                        early_years_phenology['sif_year'].astype(int).astype(str) +
                        early_years_phenology[f"{phenology}_doy"].astype(int).astype(str).str.zfill(3),
                        format='%Y%j')
                    late_years_phenology['date'] = pd.to_datetime(
                        late_years_phenology['sif_year'].astype(int).astype(str) +
                        late_years_phenology[f"{phenology}_doy"].astype(int).astype(str).str.zfill(3),
                        format='%Y%j')


            # Read wind components from RACMO5km
            input_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')
            da_u10m = xr.open_dataset(f"{input_file_path}/u10m_all.nc")['u10m']
            da_u10m = da_u10m.sel(height=0)

            da_v10m = xr.open_dataset(f"{input_file_path}/v10m_all.nc")['v10m']
            da_v10m = da_v10m.sel(height=0)

            da_ff10m = xr.open_dataset(f"{input_file_path}/ff10m_all.nc")['ff10m']
            da_ff10m = da_ff10m.sel(height=0)

            gr_masks_5km = xr.open_dataset('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/FGRN055_Masks_5.5km_updated_coordinates.nc')
            land_sea_mask = gr_masks_5km['LSM_GR']
            colors_landseamask = [(0, 0, 0), (169, 169, 169)]
            cmap_landseamask = fp.make_cmap(colors_landseamask, bit=True)
            bounds_landseamask = np.array([0, 0.5, 1.5])
            norm_landseamask = mpl.colors.BoundaryNorm(bounds_landseamask, cmap_landseamask.N)

            # Read the point mask for the closest point to Freya in the glacier mask at 5 km resolution
            mask_file_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks')
            freya_glacier_point_mask_5km_clavering_island = pd.read_csv(f"{mask_file_path}/freya_glacier_point_mask_5km_clavering_island.csv") # Check encoding if needed on bash file -i northeast_stations_metadata.csv
            freya_glacier_point_mask_5km_clavering_island.index = freya_glacier_point_mask_5km_clavering_island['Unnamed: 0']
            freya_glacier_point_mask_5km_clavering_island.drop('Unnamed: 0', axis=1, inplace=True)
            station_rlon, station_rlat = freya_glacier_point_mask_5km_clavering_island.loc['closest_point_to_freya_point', ['rlon', 'rlat']]

            # Select early year
            da_u10m_early = da_u10m.sel(time=pd.DatetimeIndex(early_years_phenology['date']))
            da_v10m_early = da_v10m.sel(time=pd.DatetimeIndex(early_years_phenology['date']))
            da_ff10m_early = da_ff10m.sel(time=pd.DatetimeIndex(early_years_phenology['date']))

            step = 10
            for t in da_u10m_early.time:
                # t = da_u10m_early.time[]
                u = da_u10m_early.sel(time=t)
                v = da_v10m_early.sel(time=t)
                ff = da_ff10m_early.sel(time=t)

                fig, ax = plt.subplots(figsize=(8,10))
                pcm = ax.pcolormesh(
                    u.lon,
                    u.lat,
                    ff,
                    shading='auto',
                    cmap='viridis'
                )
                # ax.contour(
                #     u.lon,
                #     u.lat,
                #     land_sea_mask,
                #     levels=10,
                #     colors='k',
                #     lw=0.5
                # )


                ax.quiver(
                    u.lon.values[::step,::step],
                    u.lat.values[::step,::step],
                    u.values[::step,::step],
                    v.values[::step,::step],
                    scale=250
                )
                freya_aws_meta_data = read_station_metadata(station='FRE2 (promice type, tripod)')
                station_lon, station_lat = freya_aws_meta_data.loc[['longitude', 'latitude'], 'FRE2 (promice type, tripod)']
                ax.plot(station_lon, station_lat, marker='x', color='r', markeredgewidth=0.5, markersize=10, label='Closest point to Freya', zorder=12)

                ax.set_title(f"Early {str(t.dt.strftime('%Y-%m-%d').values)}")
                plt.colorbar(pcm, ax=ax, label='Wind speed (m/s)')

                plt.show()

            # Select late year
            da_u10m_late = da_u10m.sel(time=pd.DatetimeIndex(late_years_phenology['date']))
            da_v10m_late = da_v10m.sel(time=pd.DatetimeIndex(late_years_phenology['date']))
            da_ff10m_late = da_ff10m.sel(time=pd.DatetimeIndex(late_years_phenology['date']))

            step = 7
            for t in da_u10m_late.time:
                t = t + pd.Timedelta(days=10)  # Adjust the time if needed
                # t = da_u10m_late.time[]
                u = da_u10m_late.sel(time=t)
                v = da_v10m_late.sel(time=t)
                ff = da_ff10m_late.sel(time=t)

                fig, ax = plt.subplots(figsize=(8,10))
                pcm = ax.pcolormesh(
                    u.lon,
                    u.lat,
                    ff,
                    shading='auto',
                    cmap='viridis'
                )
                # ax.contour(
                #     u.lon,
                #     u.lat,
                #     land_sea_mask,
                #     levels=10,
                #     colors='k',
                #     lw=0.5
                # )

                ax.quiver(
                    u.lon.values[::step,::step],
                    u.lat.values[::step,::step],
                    u.values[::step,::step],
                    v.values[::step,::step],
                    scale=250
                )
                freya_aws_meta_data = read_station_metadata(station='FRE2 (promice type, tripod)')
                station_lon, station_lat = freya_aws_meta_data.loc[['longitude', 'latitude'], 'FRE2 (promice type, tripod)']
                ax.plot(station_lon, station_lat, marker='x', color='r', markeredgewidth=0.5, markersize=10, label='Closest point to Freya', zorder=12)

                ax.set_title(f"Late {str(t.dt.strftime('%Y-%m-%d').values)}")
                plt.colorbar(pcm, ax=ax, label='Wind speed (m/s)')

                plt.show()

            # %%
            # RACMO
            # read_station_data_from_racmo(station='freya') # Print the name of variables available in Freya RACMO data
            zac_wd_daily_mod = read_station_data_from_racmo(station='zac', variable='wd')

            start = early_years_phenology.loc[3, 'date']
            end = start + pd.Timedelta(days=10)

            zac_wd_daily_mod_early = zac_wd_daily_mod.loc[start:end]
            vs.classify_wind_direction_in_degrees_into_cardinal_direction(zac_wd_daily_mod_early, 'wd')
            # %%

            # %%

    def sanity_check_dor_early_late_distribution_wiht_radius_plot():
        # %%
        main_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis')
        input_file_path = Path('/nas/data/processed/people/sos/third_paper_data/preprocessed_racmo_dataset/surface_levels/variable_5km/racmo_whole_domain/rearranged/onefile')

        df = pd.DataFrame(columns=np.arange(1990, 2021, 1), index=np.arange(100, 801, 100))
        df_doy = pd.DataFrame(columns=np.arange(1990, 2021, 1), index=np.arange(100, 801, 100))
        phenologies = ['dor']
        # phenologies = ['iifp']
        for phenology in phenologies:
            ds = xr.open_dataset(f"{main_path}/sea_ice_smb_melt_indicator/greenland_sea_sif_phenology_{phenology}.nc", decode_timedelta=CFTimedeltaCoder(decode_via_units=True))[f"{phenology}_doy"]
            for mask_area in ['inbetween_buffers']:
            # for mask_area in ['within_buffers']:
            # for mask_area in ['within_buffers']:
                sea_ice_mask = xr.open_dataset(f"{main_path}/input_dataset/masks/ci_all_{mask_area}_{phenology}_trend_racmo5km.nc")
                for radius in [100, 200, 300, 400, 500, 600, 700, 800]:
                # for radius in [800]:
                    sea_ice_mask_radius = sea_ice_mask.sel(radius=radius)['sea_ice_buffer']
                    ds_mask = ds.where(sea_ice_mask_radius == 1)
                    ds_mean = ds_mask.median(dim=['rlat', 'rlon'])
                    ds_mean_df = ds_mean.to_dataframe()

                    fig, ax = plt.subplots()
                    ax.plot(ds_mean_df.index, ds_mean_df[f"{phenology}_doy"], color='k', lw=1.5, ls='-', zorder=1, label=f"{phenology} {mask_area} {radius} km buffer")
                    ax.legend()

                    # Detrend the phenology time series using linear regression
                    # Years
                    years = ds_mean_df.index.values
                    y = ds_mean_df['dor_doy'].values

                    # Fit linear trend
                    slope, intercept, r, p, stderr = linregress(years, y)

                    # Estimated trend
                    trend = intercept + slope * years

                    # Detrended series (preserve mean)
                    ds_mean_df['dor_doy_detrended'] = y - trend + y.mean()

                    # Early years:
                    early_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.25)
                    # early_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.05)
                    early_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] < early_doy].index
                    # Late years:
                    late_doy = ds_mean_df[f"{phenology}_doy_detrended"].quantile(0.75)
                    # late_doy = ds_mean_df[f"{phenology}_doy"].quantile(0.95)
                    late_years = ds_mean_df.loc[ds_mean_df[f"{phenology}_doy_detrended"] > late_doy].index

                    early_years_phenology = ds_mean_df.loc[ds_mean_df.index.isin(early_years)]
                    late_years_phenology = ds_mean_df.loc[ds_mean_df.index.isin(late_years)]

                    early_years_phenology = early_years_phenology.reset_index()
                    late_years_phenology = late_years_phenology.reset_index()

                    # Convert year + DOY to datetime
                    early_years_phenology['date'] = pd.to_datetime(
                        early_years_phenology['sif_year'].astype(int).astype(str) +
                        early_years_phenology[f"{phenology}_doy_detrended"].astype(int).astype(str).str.zfill(3),
                        format='%Y%j')
                    late_years_phenology['date'] = pd.to_datetime(
                        late_years_phenology['sif_year'].astype(int).astype(str) +
                        late_years_phenology[f"{phenology}_doy_detrended"].astype(int).astype(str).str.zfill(3),
                        format='%Y%j')

                    for early_year in early_years:
                        df.loc[radius, early_year] = 1
                        df_doy.loc[radius, early_year] = early_years_phenology.loc[early_years_phenology['sif_year'] == early_year, f"{phenology}_doy_detrended"].values[0]
                    for late_year in late_years:
                        df.loc[radius, late_year] = 2
                        df_doy.loc[radius, late_year] = late_years_phenology.loc[late_years_phenology['sif_year'] == late_year, f"{phenology}_doy_detrended"].values[0]

        df_numeric = df.apply(pd.to_numeric)
        df_doy_numeric = df_doy.apply(pd.to_numeric)
        # df = df.fillna(0)


        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        masked = np.ma.masked_invalid(df_numeric.values)

        cmap = mcolors.ListedColormap(["royalblue", "firebrick"])
        norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5], cmap.N)

        fig, ax = plt.subplots(figsize=(14,4))

        im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='auto')

        cbar = plt.colorbar(im, ax=ax, ticks=[1,2])
        cbar.set_ticklabels(["Early", "Late"])

        ax.set_xticks(np.arange(len(df_numeric.columns)))
        ax.set_xticklabels(df_numeric.columns, rotation=90)

        ax.set_yticks(np.arange(len(df_numeric.index)))
        ax.set_yticklabels(df_numeric.index)

        ax.set_xlabel("Year")
        ax.set_ylabel("Buffer radius (km)")

        plt.tight_layout()
        plt.show()


        # %%
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(16, 4))

        sns.heatmap(
            df_doy_numeric,
            cmap="viridis",      # or "YlGnBu", "plasma", "magma"
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "Day of Year (DOY)"},
        )

        plt.xlabel("Year")
        plt.ylabel("Buffer radius (km)")
        plt.title("Median DOY of Phenology")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.show()

class WindDirectionCompositeAnalysis:
    def NEED_CORRECTION_plot_wind_speed_and_direction_at_zac_and_freya():
        # %%
        processed_aws_data_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')

        # Read processed Freya wind direction and speed
        freya_wd_hour = pd.read_csv(f"{processed_aws_data_path}/freya_wd_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_wd_hour.index = pd.to_datetime(freya_wd_hour.index)
        freq = freya_wd_hour.index.to_series().diff().mode()[0]
        freya_wd_hour = freya_wd_hour.resample(freq).mean()
        freya_wd_hour = pd.DataFrame(freya_wd_hour['wd'])

        freya_ws_hour = pd.read_csv(f"{processed_aws_data_path}/freya_ws_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_ws_hour.index = pd.to_datetime(freya_ws_hour.index)
        freq = freya_ws_hour.index.to_series().diff().mode()[0]
        freya_ws_hour = freya_ws_hour.resample(freq).mean()
        freya_ws_hour = pd.DataFrame(freya_ws_hour['ws'])

        # Read processed ZAC wind direction and speed
        zac_wd_hour = pd.read_csv(f"{processed_aws_data_path}/zac_wd_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_wd_hour.index = pd.to_datetime(zac_wd_hour.index)
        freq = zac_wd_hour.index.to_series().diff().mode()[0]
        zac_wd_hour = zac_wd_hour.resample(freq).mean()
        zac_wd_hour = pd.DataFrame(zac_wd_hour['wd'])

        zac_ws_hour = pd.read_csv(f"{processed_aws_data_path}/zac_ws_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_ws_hour.index = pd.to_datetime(zac_ws_hour.index)
        freq = zac_ws_hour.index.to_series().diff().mode()[0]
        zac_ws_hour = zac_ws_hour.resample(freq).mean()
        zac_ws_hour = pd.DataFrame(zac_ws_hour['WS (m/s)'])
        zac_ws_hour.rename(columns={'WS (m/s)': 'ws'}, inplace=True)

        # Merge Wind Speed and Direction
        wd_merged = pd.merge(freya_wd_hour.rename(columns={'wd': 'FRE'}), zac_wd_hour.rename(columns={'wd': 'ZAC'}), left_index=True, right_index=True, how='outer')
        # wd_merged = wd_merged.dropna()

        # Select stations and their colors
        stations_colors = dict({'ZAC': '#7FC97F', 'FRE':'royalblue'})
        stations = list(stations_colors.keys())

        # Classify wind direction into cardinal directions for each station daily average
        date_range = pd.date_range(start=wd_merged.index[0], end=wd_merged.index[-1], freq='h')
        wind_direction_classification_stations_hourly = pd.DataFrame(columns=stations, index=date_range)

        for station in stations:
            wind_direction_classification_stations_hourly.loc[:, station] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_merged, station)[f'{station}_wd_class']
        wind_direction_classification_stations_hourly = wind_direction_classification_stations_hourly.dropna()

        # Calculate percentage of classified daily average wind direction per month
        # Select months May to September
        months = [5, 6, 7, 8, 9]
        miindex = pd.MultiIndex.from_product([stations, months])
        cardinal_directions = list(np.array('N NE E SE S SW W NW'.split()))
        wind_direction_classification_percentage  = pd.DataFrame(index=miindex, columns=cardinal_directions)
        wind_direction_classification_total  = pd.DataFrame(index=miindex, columns=['sample_size'])

        for station in stations:
            wd_percentage, wd_total  = _calculate_percentage_of_wind_blowing_from_certain_direction_per_month(wind_direction_classification_stations_hourly, station)
            for month in months:
                wind_direction_classification_percentage.loc[(station, month), :] = wd_percentage.loc[month, :]
                wind_direction_classification_total.loc[(station, month), 'sample_size'] = wd_total.loc[month, :].values

        fig = plt.figure(figsize=(4, 3), dpi=600)
        # OUTER GridSpec: controls LARGE row gap (applies to all columns)
        outer = plt.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.35)   # large spacing between row groups
        # TOP block: rows 0–1, all columns
        gs_top = outer[0].subgridspec(2, 3, hspace=0.15, wspace=0.25)   # small internal spacing
        # BOTTOM block: rows 2–3, all columns
        gs_bot = outer[1].subgridspec(2, 3, hspace=0.15, wspace=0.25)   # small internal spacing

        subplot_loops = [('May', 5, 0), ('Jun', 6, 1), ('July', 7, 2), ('Aug', 8, 0), ('Sep', 9, 1)] # (month, column)
        for subplot_loop in subplot_loops:
            month_name = subplot_loop[0]
            month = subplot_loop[1]
            col = subplot_loop[2]
            wind_direction_row = 0
            wind_speed_row = 1
            if month in [5, 6, 7]:
                gs_wd = gs_top[wind_direction_row, col]
                gs_ws = gs_top[wind_speed_row, col]
            elif month in [8, 9]:
                gs_wd = gs_bot[wind_direction_row, col]
                gs_ws = gs_bot[wind_speed_row, col]

            # Plot wind rose
            ax = fig.add_subplot(gs_wd, polar=True)
            for station_key in stations_colors.keys():
                cardinal_directions_number = len(cardinal_directions)
                # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
                angles = [n / float(cardinal_directions_number) * 2 * np.pi for n in range(cardinal_directions_number)]
                angles += angles[:1]

                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)

                obs = (wind_direction_classification_percentage.loc[(station_key, month), :]).tolist()
                obs += obs[:1]
                ax.plot(angles, obs, color=stations_colors[station_key], linewidth=0.5, linestyle='-')

            # Polar spine (outer circle)
            ax.spines['polar'].set_linewidth(0.4)
            ax.spines['polar'].set_zorder(1)
            ax.spines['polar'].set_color('k')

            # Grid styling
            ax.yaxis.grid(True, linewidth=0.3)
            ax.xaxis.grid(True, linewidth=0.3)

            # Draw one axe per variable + add labels
            ax.set_xticks([n / float(8) * 2 * np.pi for n in range(8)])
            ax.set_xticklabels(cardinal_directions, color='grey', size=6)

            # Move labels closer to the circle
            for label in ax.get_xticklabels():
                label.set_clip_on(False)      # <-- critical
                label.set_y(0.51)     # default ~1.1, lower = closer
                # label.set_zorder(12)
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')

            # Draw ylabels
            ax.set_rlabel_position(15)
            ax.set_ylim(0, 60)
            ax.set_yticks([20, 40, 60])
            ax.set_yticklabels(['20%', '40%', '60%'], color='grey', size=4, zorder=12)
            ax.tick_params(axis='y', which='major', width=0, length=0.1, pad=0.0001)
            for label in ax.get_yticklabels():
                label.set_zorder(12)
                label.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

            # Add subplot letter
            if month == 5:
                subplot_number = 'a)'
            elif month == 6:
                subplot_number = 'b)'
            elif month == 7:
                subplot_number = 'c)'
            elif month == 8:
                subplot_number = 'd)'
            elif month == 9:
                subplot_number = 'e)'
            ax.text(-0.8, 1, f"{subplot_number} {month_name}", horizontalalignment='left', transform=ax.transAxes, size=6, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # Create legend
            nan_patch = Patch(color='none', label=f"n = {int(np.int64(wind_direction_classification_total.loc[('ZAC', month), :].values[0]))}")
            station_legend = ax.legend(bbox_to_anchor= (1.5, 0.05), handles=[nan_patch], ncol=1, borderaxespad=0, fontsize=6, frameon=False, handletextpad=0.005, labelspacing=0.3)
            for station_legend_text in station_legend.get_texts():
                station_legend_text.set_color('k')
                station_legend_text.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
                station_legend_text.set_ha('center')

            # Plot wind speed at different wind directions
            ax = fig.add_subplot(gs_ws)
            first_station = 'ZAC'
            first_station_ws_wd = pd.DataFrame({'wd': wind_direction_classification_stations_hourly.loc[wind_direction_classification_stations_hourly.index.month == month, first_station], variable: ws_merged_variable_map[variable].loc[ws_merged_variable_map[variable].index.month == month, first_station]}).dropna()

            first_station_ws_wd['station'] = first_station

            second_station = 'FRE'
            second_station_ws_wd = pd.DataFrame({'wd': wind_direction_classification_stations_hourly.loc[wind_direction_classification_stations_hourly.index.month == month, second_station], variable: ws_merged_variable_map[variable].loc[ws_merged_variable_map[variable].index.month == month, second_station]}).dropna()
            second_station_ws_wd['station'] = second_station

            station_ws_wd = pd.concat([first_station_ws_wd, second_station_ws_wd], ignore_index=True)
            station_ws_wd = station_ws_wd.dropna()

            ax_violin = sns.violinplot(x='wd', y='ws', hue='station', split=True, inner='quartiles',
                                palette={ 'ZAC':stations_colors[first_station], 'FRE':stations_colors[second_station]}, data=station_ws_wd, density_norm='area',
                                ax=ax, order=cardinal_directions, linewidth=0.4)

            for l in ax_violin.lines:
                l.set_linestyle('--')
                l.set_linewidth(0.3)
                l.set_color('k')
                #l.set_alpha(0.8)
            for l in ax_violin.lines[1::3]:
                l.set_linestyle('-')
                l.set_linewidth(1)
                l.set_color('k')
                #l.set_alpha(0.8)

            plt.setp(ax_violin.collections, zorder=2, label='')
            ax.get_legend().set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('')

            ax.set_ylim(0, 8)
            ax.set_yticks([0, 2, 4, 6, 8])
            ax.set_yticklabels([0, 2, 4, 6, 8])

            # ax.set_ylabel(rf"Wind speed (m s$^\mathdefault{{-1}}$)", rotation='vertical', horizontalalignment='center', fontsize=6, labelpad=2)
            ax.set_ylabel(rf"m s$^\mathdefault{{-1}}$", rotation='vertical', horizontalalignment='center', fontsize=6)
            ax.yaxis.label.set_rotation(90)
            ax.yaxis.label.set_horizontalalignment('center')
            ax.yaxis.label.set_verticalalignment('center')
            ax.yaxis.set_label_coords(-0.15, 0.5)

            ax.grid(which='major', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)
            ax.tick_params(axis='both', direction='inout', which='major', width=0.5, length=3, labelsize=6, color='k')

            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_linewidth(0.4)
            ax.spines['bottom'].set_linewidth(0.4)

        # Add stations legend in new axis
        col = 2
        gs_ideal = gs_bot[0, col]
        ax = fig.add_subplot(gs_ideal)
        [sp.set_visible(False) for sp in ax.spines.values()]
        ax.tick_params(axis='both', left=False, labelleft=False, labelbottom=False)
        ax.set_yticks([])
        ax.set_xticks([])

        marker_style_legend_wd = dict(lw=0.5, ec='k', fill=True)
        zac_handle = Patch(fc=stations_colors['ZAC'], **marker_style_legend_wd)
        fre_handle = Patch(fc=stations_colors['FRE'], **marker_style_legend_wd)
        ax.legend(bbox_to_anchor=(0.5, 0.8), handles=[zac_handle, fre_handle], labels=['ZAC', 'FRE'], ncol=1,
                borderaxespad=0, fontsize=9, handletextpad=0.1, labelspacing=0.2, frameon=False, handlelength=1, handleheight=1,
                edgecolor='none') # facecolor='w', framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None)}

        # Add idealized violin plot in new axis
        gs_ideal = gs_bot[1, col]
        ax = fig.add_subplot(gs_ideal)
        station_ws_wd_SE_SW = station_ws_wd[station_ws_wd['wd'].isin(['SE', 'SW'])]
        # Plot only ZAC and SE side of the violin plot
        station_ws_wd_SE_SW.loc[station_ws_wd_SE_SW.station == 'FRE', 'ws'] = np.nan
        station_ws_wd_SE_SW.loc[station_ws_wd_SE_SW.wd == 'SW', 'ws'] = np.nan

        ax_violin = sns.violinplot(x='wd', y='ws', hue='station', split=True, inner='quartiles',
                                palette={ 'ZAC':'white', 'FRE':'royalblue'}, data=station_ws_wd_SE_SW, density_norm='area',
                                ax=ax, order=['SE', 'SW'], linewidth=0.4)

        [sp.set_visible(False) for sp in ax.spines.values()]
        plt.setp(ax_violin.collections, zorder=2, label='')
        ax.get_legend().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set_xlim([-0.9, 1.5])
        ax.set_ylim([-0.4, 5.5])

        ax.tick_params(axis='both', left=False, labelleft=False, labelbottom=False)
        ax.set_yticks([])
        ax.set_xticks([])

        for l in ax_violin.lines:
            l.set_linestyle('--')
            l.set_linewidth(0.6)
            l.set_color('k')
            #l.set_alpha(0.8)
        for l in ax_violin.lines[1::3]:
            l.set_linestyle('-')
            l.set_linewidth(1.2)
            l.set_color('k')
            #l.set_alpha(0.8)

        pos = 0.01
        quantile_fontsize = 6
        # ax.text(pos, station_ws_wd_SE_SW.ws.quantile(0.95) + 0.005, '95%', fontsize=quantile_fontsize)
        ax.text(pos, station_ws_wd_SE_SW.ws.quantile(0.75), '75%', fontsize=quantile_fontsize)
        # ax.text(pos, station_ws_wd_SE_SW.ws.quantile(0.50), '50%', color='#d95f02', fontsize=quantile_fontsize)
        ax.text(pos, station_ws_wd_SE_SW.ws.mean() - 0.3, 'Mean', color='k', fontsize=quantile_fontsize)
        ax.text(pos, station_ws_wd_SE_SW.ws.quantile(0.25) - 0.33, '25%', fontsize=quantile_fontsize)
        # ax.text(pos, station_ws_wd_SE_SW.ws.quantile(0.05) - 0.007, '5%', fontsize=quantile_fontsize)

        # Plot arrows and labels
        ax.annotate('', xy=(-0.49, 5), xytext=(0.071, 5), arrowprops=dict(arrowstyle='<|-|>', lw=0.3, ls='--', color='gray'))
        ax.plot([0.2, 0.2], [0.3, 0.9], transform=ax.transAxes, color='gray', ls='--', lw=0.3)
        ax.text(0.3, 1.05, 'Frequency', transform=ax.transAxes, va='center', ha='center', fontsize=6)

        targets = dict(left=129, right=10, top=60, bottom=90, hspace=300, wspace=600)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_wind_speed_and_direction_at_zac_and_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_wind_speed_and_direction_at_zac_and_freya.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

    def plot_wind_direction_at_zac_and_freya():
        # %%
        processed_aws_data_path = Path('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed')
        # Select stations and their colors
        stations_colors = dict({'zac': '#7FC97F', 'freya':'royalblue'})
        stations = list(stations_colors.keys())

        # Read processed
        # Freya wind direction
        freya_wd_hour = pd.read_csv(f"{processed_aws_data_path}/freya_wd_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        freya_wd_hour.index = pd.to_datetime(freya_wd_hour.index)
        freq = freya_wd_hour.index.to_series().diff().mode()[0]
        freya_wd_hour = freya_wd_hour.resample(freq).mean()
        freya_wd_hour = pd.DataFrame(freya_wd_hour['wd'])
        # Zac wind direction
        zac_wd_hour = pd.read_csv(f"{processed_aws_data_path}/zac_wd_hour.csv", parse_dates=['Unnamed: 0'], index_col='Unnamed: 0')
        zac_wd_hour.index = pd.to_datetime(zac_wd_hour.index)
        freq = zac_wd_hour.index.to_series().diff().mode()[0]
        zac_wd_hour = zac_wd_hour.resample(freq).mean()
        zac_wd_hour = pd.DataFrame(zac_wd_hour['wd'])

        # Merge wind direction
        wd_merged = pd.merge(freya_wd_hour.rename(columns={'wd': 'freya'}), zac_wd_hour.rename(columns={'wd': 'zac'}), left_index=True, right_index=True, how='outer')

        # Classify wind direction into cardinal directions for each station daily average
        date_range = pd.date_range(start=wd_merged.index[0], end=wd_merged.index[-1], freq='h')
        cardinal_directions_hourly = pd.DataFrame(columns=stations, index=date_range)
        for station in stations:
            cardinal_directions_hourly.loc[:, station] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(wd_merged, station)[f'{station}_class']
        cardinal_directions_hourly = cardinal_directions_hourly.dropna()

        # Calculate percentage of classified daily average wind direction per month
        # Select months May to September
        months = [5, 6, 7, 8, 9]
        miindex = pd.MultiIndex.from_product([stations, months])
        directions_16 = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        cardinal_directions_percentage  = pd.DataFrame(index=miindex, columns=directions_16)
        cardinal_directions_total  = pd.DataFrame(index=miindex, columns=['sample_size'])

        for station in stations:
            wd_percentage, wd_total  = _calculate_percentage_of_wind_blowing_from_certain_direction_per_month(cardinal_directions_hourly, station)
            for month in months:
                cardinal_directions_percentage.loc[(station, month), :] = wd_percentage.loc[month, :]
                cardinal_directions_total.loc[(station, month), 'sample_size'] = wd_total.loc[month, :].values

        fig = plt.figure(figsize=(4.5, 3), dpi=600)
        gs = plt.GridSpec(2, 3, hspace=0.27, wspace=0.18)
        subplot_loops = [('May', 5, 0, 0), ('Jun', 6, 0, 1), ('July', 7, 0, 2), ('Aug', 8, 1, 0), ('Sep', 9, 1, 1)] # (month, column)

        cardinal_directions_number = len(directions_16)
        direction_8 = np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        for subplot_loop in subplot_loops:
            month_name = subplot_loop[0]
            month = subplot_loop[1]
            row = subplot_loop[2]
            col = subplot_loop[3]
            ax = fig.add_subplot(gs[row, col], polar=True)
            for station in stations:
                # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
                angles = [n / float(cardinal_directions_number) * 2 * np.pi for n in range(cardinal_directions_number)]
                angles += angles[:1]

                # If you want the first axis to be on top:
                ax.set_theta_offset(np.pi / 2)
                ax.set_theta_direction(-1)

                obs = (cardinal_directions_percentage.loc[(station, month), :]).tolist()
                obs += obs[:1]
                ax.plot(angles, obs, color=stations_colors[station], linewidth=0.9, linestyle='-')

            # Set xticks and labels
            ax.set_xticks([n / float(8) * 2 * np.pi for n in range(8)])
            ax.set_xticklabels(direction_8, color='grey', size=6)

            # Set minor ticks
            directions_8_minor = np.array(['NNE', 'ENE', 'ESE', 'SSE', 'SSW', 'WSW', 'WNW', 'NNW'])
            ax.set_xticks([(n + 0.5) / 8 * 2 * np.pi for n in range(8)], minor=True)
            ax.set_xticklabels(directions_8_minor, color='grey', size=5, minor=True)

            # Grid styling
            ax.xaxis.grid(True, which='major', linewidth=0.3, linestyle='-')
            ax.xaxis.grid(True, which='minor', linewidth=0.3, linestyle='--', color='lightgrey')

            # Move labels closer to the circle
            for label in ax.get_xticklabels():
                label.set_clip_on(False)      # <-- critical
                # label.set_y(0.51)     # default ~1.1, lower = closer
                label.set_y(0.22)     # default ~1.1, lower = closer
                # label.set_zorder(12)
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')

            # Move labels closer to the circle
            for label in ax.get_xminorticklabels():
                label.set_clip_on(False)      # <-- critical
                # label.set_y(0.51)     # default ~1.1, lower = closer
                label.set_y(0.15)     # default ~1.1, lower = closer
                # label.set_zorder(12)
                label.set_horizontalalignment('center')
                label.set_verticalalignment('center')

            # Set yticks and labels
            ax.yaxis.grid(True, linewidth=0.3)
            ax.set_rlabel_position(15)
            ax.set_ylim(0, 40)
            ax.set_yticks([10, 20, 30])
            ax.set_yticklabels(['10%', '20%', '30%'], color='grey', size=4, zorder=12)
            ax.tick_params(axis='y', which='major', width=0, length=0.1, pad=0.0001)
            for label in ax.get_yticklabels():
                label.set_zorder(12)
                label.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])

            # Polar spine (outer circle)
            ax.spines['polar'].set_linewidth(0.4)
            ax.spines['polar'].set_zorder(1)
            ax.spines['polar'].set_color('k')

            # Add subplot letter
            if month == 5:
                subplot_number = 'a)'
            elif month == 6:
                subplot_number = 'b)'
            elif month == 7:
                subplot_number = 'c)'
            elif month == 8:
                subplot_number = 'd)'
            elif month == 9:
                subplot_number = 'e)'
            ax.text(-0.1, 1, f"{subplot_number} {month_name}", horizontalalignment='left', transform=ax.transAxes, size=6, color='k', path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
            # Create legend
            nan_patch = Patch(color='none', label=f"n = {int(np.int64(cardinal_directions_total.loc[('zac', month), :].values[0]))}")
            station_legend = ax.legend(bbox_to_anchor= (0.2, 0.05), handles=[nan_patch], ncol=1, borderaxespad=0, fontsize=6, frameon=False, handletextpad=0.005, labelspacing=0.3)
            for station_legend_text in station_legend.get_texts():
                station_legend_text.set_color('k')
                station_legend_text.set_path_effects([pe.withStroke(linewidth=1.5, foreground='white')])
                station_legend_text.set_ha('center')

        # Plot legend in new axis
        ax = fig.add_subplot(gs[1, 2])
        [sp.set_visible(False) for sp in ax.spines.values()]
        ax.tick_params(axis='both', left=False, labelleft=False, labelbottom=False)
        ax.set_yticks([])
        ax.set_xticks([])
        marker_style_legend_wd = dict(lw=0, marker='d', markersize=6, markerfacecolor='none', markeredgewidth=0.8)

        zac_handle = Line2D([0], [0], color=stations_colors['zac'],  markeredgecolor=stations_colors['zac'], **marker_style_legend_wd)
        fre_handle = Line2D([0], [0], color=stations_colors['freya'],  markeredgecolor=stations_colors['freya'], **marker_style_legend_wd)

        ax.legend(bbox_to_anchor=(0.5, 0.8), handles=[zac_handle, fre_handle], labels=['ZAC', 'FRE'], ncol=1,
                borderaxespad=0, fontsize=9, handletextpad=0.1, labelspacing=0.2, frameon=False, handlelength=1, handleheight=1,
                edgecolor='none') # facecolor='w', framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None)}

        targets = dict(left=45, right=30, top=70, bottom=90, hspace=300, wspace=500)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_wind_direction_at_zac_and_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_wind_direction_at_zac_and_freya.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
        # %%

    def plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya():
        # %%
        variables = [dict(variable='wd', column_name='wd'),
                    dict(variable='ws', column_name='ws'),
                    dict(variable='t2m', column_name='t2m'),
                    dict(variable='q2m', column_name='fre_rh_corr_adjusted_qm_q', rename_column='q2m'),
                    dict(variable='atm_pressure', column_name='atm_pressure'),
                    dict(variable='t2m_tsurf_diff', column_name='t2m_tsurf_diff')]

        freya_variables = []
        for v in variables:
            df = read_preprocessed_station_data(station='freya', variable=v['variable'], column_name=v['column_name'], time_freq='hour', rename_column=v.get('rename_column'))
            freya_variables.append(df)
        freya_variables = pd.concat(freya_variables, axis=1).sort_index()
        freya_variables['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_variables, 'wd')['wd_class']

        # Box plot properties
        boxplot_prop = dict(showcaps=False, widths=0.16, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1.5, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
        boxplot_prop_katabatic = dict(boxprops=dict(facecolor='royalblue', color='k', linewidth=0.5))
        boxplot_prop_non_katabatic = dict(boxprops=dict(facecolor='#7FC97F', color='k', linewidth=0.5))

        # Plotting
        fig = plt.figure(figsize=(6, 4), dpi=600)
        gs = plt.GridSpec(5, 7, hspace=0.2, wspace=0.01)

        # variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4)]
        variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4)]
        months = [('May', 5, 0), ('Jun', 6, 1), ('July', 7, 2), ('Aug', 8, 3), ('Sep', 9, 4), ('MJJAS', 56789, 5)]
        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        non_katabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions

        for variable, row in variables:
            for month_name, month, col in months:
                if month_name == 'MJJAS':
                    freya_variables_month = freya_variables.loc[freya_variables.index.month.isin([5, 6, 7, 8, 9]), [variable, 'wd_cardinal']].dropna()
                else:
                    freya_variables_month = freya_variables.loc[freya_variables.index.month == month, [variable, 'wd_cardinal']].dropna()

                # Select only katabatic and non-katabatic wind directions
                freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][variable])
                freya_variables_month_katabatic_wind['wind_type'] = 'katabatic'

                freya_variables_month_non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][variable])
                freya_variables_month_non_katabatic_wind['wind_type'] = 'non-katabatic'

                freya_variables_month_interested = pd.concat([freya_variables_month_katabatic_wind, freya_variables_month_non_katabatic_wind], ignore_index=True)
                freya_variables_month_interested['time'] = month_name

                # Wind Speed and other variables
                ax = fig.add_subplot(gs[row, col])
                pos = 0.08
                # Create box plot properties based on wind type
                ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'katabatic', variable], whis=[5, 95], positions=np.array([-pos]), **boxplot_prop_katabatic, **boxplot_prop)
                ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non-katabatic', variable], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_non_katabatic, **boxplot_prop)
                # Create violin plot
                ax_violin = sns.violinplot(x='time', y=variable, hue='wind_type', hue_order=['katabatic', 'non-katabatic'], split=True, inner=None,
                                palette={'non-katabatic':'#7FC97F', 'katabatic':'royalblue'}, data=freya_variables_month_interested, density_norm='area',
                                ax=ax, linewidth=0.4) #
                # Remove default labels and legend
                plt.setp(ax_violin.collections, zorder=2, label='')
                ax.get_legend().set_visible(False)
                ax.set_ylabel('')
                ax.set_xlabel('')

                text_params = dict(verticalalignment='center', transform=ax.transAxes, clip_on=False, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                if variable == 'ws':
                    ax.text(0.5, 1.1, month_name, size=12, color='k', horizontalalignment='center', **text_params)

                ax.text(0.49, -0.01, f"{len(freya_variables_month_katabatic_wind)}", horizontalalignment='right', size=6, color='royalblue', **text_params)
                ax.text(0.52, -0.01, f"{len(freya_variables_month_non_katabatic_wind)}", horizontalalignment='left', size=6, color='#7FC97F', **text_params)

                if variable == 'ws':
                    ylim = [0, 10]
                    yticks = [0, 2, 4, 6, 8, 10]
                    yticklabels = [0, 2, 4, 6, 8, 10]
                    # var_label = r'WS' + '\n' + r'(m s$^\mathdefault{-1}$)'
                    var_label = r'WS (m s$^\mathdefault{-1}$)'
                elif variable == 'atm_pressure':
                    ylim = [900, 950]
                    yticks = [900, 910, 920, 930, 940, 950]
                    yticklabels = [900, 910, 920, 930, 940, 950]
                    # var_label = r'P$_\mathdefault{surf}$' + '\n' + r'(hPa)'
                    var_label = r'P$_\mathdefault{surf}$ (hPa)'
                elif variable == 't2m':
                    ylim = [-15, 15]
                    yticks = [-15, -10, -5, 0, 5, 10, 15]
                    yticklabels = [-15, -10, -5, 0, 5, 10, 15]
                    # var_label = r'T$_\mathdefault{2m}$' + '\n' + r'($^\circ{\mathdefault{C}}$)'
                    var_label = r'T$_\mathdefault{2m}$ ($^\circ{\mathdefault{C}}$)'
                elif variable == 'q2m':
                    ylim = [0, 6]
                    yticks = [0, 1, 2, 3, 4, 5, 6]
                    yticklabels = [0, 1, 2, 3, 4, 5, 6]
                    # var_label = r'q$_\mathdefault{2m}$' +'\n'+ r'(g kg$^\mathdefault{-1}$)'
                    var_label = r'q$_\mathdefault{2m}$ (g kg$^\mathdefault{-1}$)'
                elif variable == 't2m_tsurf_diff':
                    ylim = [-5, 15]
                    yticks = [-5, 0, 5, 10, 15]
                    yticklabels = [-5, 0, 5, 10, 15]
                    # var_label = r'T$_\mathdefault{2m}$ - T$_\mathdefault{surf}$' + '\n' + r'($^\circ{\mathdefault{C}}$)'
                    var_label = r'T$_\mathdefault{2m}$ - T$_\mathdefault{surf}$ ($^\circ{\mathdefault{C}}$)'

                ax.set_ylim(ylim)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)

                ax.set_xlim([-0.5, 0.5])
                ax.set_xticks([0])
                ax.set_xticklabels([''])

                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                remove_tick_params = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.tick_params(axis='x', **remove_tick_params)
                if col != 0:
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='y', **remove_tick_params)

                if col == 0:
                    ax.tick_params(axis='y', direction='out', length=6, pad=1, labelsize=7, width=0.7)
                    ax.set_ylabel(var_label, rotation='vertical', size=7, color='k', labelpad=0.1)

                ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

        # ------------------------------------
        # Plot idealized wind rose in new axis
        # ------------------------------------
        cardinal_directions = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        ax = fig.add_subplot(gs[0, 6], polar=True)
        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Each sector is 22.5 degrees wide
        # Katabatic wind sector (degrees)
        katabatic_start = np.deg2rad(123.75) # 135 (SE) - 11.25
        katabatic_end   = np.deg2rad(168.75) # 157.5 (SSE) + 11.25

        # Non-katabatic wind sector (degrees)
        non_katabatic_start = np.deg2rad(281.25) # 292.5 (WNW) - 11.25
        non_katabatic_end   = np.deg2rad(326.25) # 315 (NW) + 11.25

        for wind_type in [(katabatic_start, katabatic_end, 'royalblue'), (non_katabatic_start, non_katabatic_end, '#7FC97F')]:
            # Sector properties
            theta_center = (wind_type[0] + wind_type[1]) / 2
            theta_width  = wind_type[1] - wind_type[0]
            radius = 100  # magnitude of katabatic wind
            # Plot filled wind-rose sector
            ax.bar(theta_center, radius, width=theta_width, bottom=0, color=wind_type[2], linewidth=0, alpha=0.9, zorder=12)

        angles_deg = [0, 90, 135, 157.5, 180, 270, 292.5, 315]
        ax.set_xticks(np.deg2rad(angles_deg))
        ax.set_xticklabels(['N', 'E', 'SE', 'SSE', 'S', 'W', 'WNW', 'NW'], color='grey', size=6)
        # Move labels closer to the circle
        for label in ax.get_xticklabels():
            label.set_clip_on(False)      # <-- critical
            label.set_y(0.35)     # default ~1.1, higher = closer
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            label.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
        ax.xaxis.grid(True, which='major', linewidth=0.2, color='grey', linestyle='--', zorder=12)

        # Set yticks and labels
        ax.set_yticklabels([])
        ax.yaxis.grid(False)

        # Polar spine (outer circle)
        ax.spines['polar'].set_linewidth(0.4)
        ax.spines['polar'].set_zorder(1)
        ax.spines['polar'].set_color('k')

        # ------------------------------------
        # Add legend in new axis
        # ------------------------------------
        ax = fig.add_subplot(gs[1, 6])
        # Define a cone (triangle) pointing downward
        cone_vertices = np.array([[0.0,  -1.8], [-0.9, 1.0], [0.9, 1.0], [0.0,  -1.8]])
        # ['long tip', 'narrow left base', 'narrow right base', 'close']
        cone_codes = [mplpath.MOVETO, mplpath.LINETO, mplpath.LINETO, mplpath.CLOSEPOLY]
        cone_marker = mplpath(cone_vertices, cone_codes)

        legend_handle_params = dict(linestyle='None', marker=cone_marker, markersize=10,  markeredgewidth=0.5)
        non_katabatic_handle = Line2D([], [], markerfacecolor='#7FC97F', markeredgecolor='#7FC97F', **legend_handle_params)
        katabatic_handle = Line2D([], [], markerfacecolor='royalblue', markeredgecolor='royalblue', **legend_handle_params)

        wind_legend = ax.legend(handles=[non_katabatic_handle, katabatic_handle],
                                labels=[f"Non-katabatic wind\n({int(np.round(np.rad2deg(non_katabatic_start), 0))}° – {int(np.round(np.rad2deg(non_katabatic_end), 0))}°)",
                                        f"Katabatic wind\n({int(np.round(np.rad2deg(katabatic_start), 0))}° – {int(np.round(np.rad2deg(katabatic_end), 0))}°)"],
                                bbox_to_anchor=(0.5, 0.5), loc='center', fontsize=6, frameon=False, handletextpad=0.05, labelspacing=0.8)

        for wind_legend_text in wind_legend.get_texts():
            wind_legend_text.set_color('k')
            wind_legend_text.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
            wind_legend_text.set_ha('center')
        ax.axis('off')

        # ------------------------------------
        # Add idealized violin plot in new axis
        # ------------------------------------
        ideal_variable = 'ws'
        freya_variables_month = freya_variables.loc[freya_variables.index.month == 5, [ideal_variable, 'wd_cardinal']].dropna()
        # Select only katabatic and upvalley wind directions
        freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][ideal_variable])
        freya_variables_month_katabatic_wind['wind_type'] = 'katabatic'
        freya_variables_month_non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][ideal_variable])
        freya_variables_month_non_katabatic_wind['wind_type'] = 'non_katabatic'
        freya_variables_month_interested = pd.concat([freya_variables_month_katabatic_wind, freya_variables_month_non_katabatic_wind], ignore_index=True)
        freya_variables_month_interested['time'] = month_name
        # Plot only one side of the violin plot as idealized
        freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'katabatic', ideal_variable] = np.nan

        ax = fig.add_subplot(gs[2, 6])
        pos = 0.08
        # Create box plot properties based on wind type
        boxplot_prop_non_katabatic_ideal = dict(boxprops=dict(facecolor='w', color='k', linewidth=0.5))
        ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non_katabatic', ideal_variable], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_non_katabatic_ideal, **boxplot_prop)
        # Create violin plot
        ax_violin = sns.violinplot(x='time', y=ideal_variable, hue='wind_type', hue_order=['non_katabatic', 'katabatic'], split=True, inner=None,
                        palette={'non_katabatic':'w', 'katabatic':'royalblue'}, data=freya_variables_month_interested, density_norm='area',
                        ax=ax, linewidth=0.4)

        [sp.set_visible(False) for sp in ax.spines.values()]
        plt.setp(ax_violin.collections, zorder=2, label='')
        ax.get_legend().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set_xlim([-0.4, 0.5])
        ax.set_ylim([-2, 10])
        ax.axis('off')

        # Plot arrows and labels
        pos = 0.2
        quantile_fontsize = 6
        ideal_data = freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non_katabatic', ideal_variable]
        ax.text(pos, ideal_data.quantile(0.75), '75%', fontsize=quantile_fontsize)
        ax.text(pos, ideal_data.mean() - 0.3, 'Mean', color='k', fontsize=quantile_fontsize, fontweight='bold')
        ax.text(pos, ideal_data.quantile(0.50) - 0.2, '50%', color='grey', fontsize=quantile_fontsize, fontweight='bold')
        ax.text(pos, ideal_data.quantile(0.25) - 0.8, '25%', fontsize=quantile_fontsize)

        ax.annotate('', xy=(-0.4, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle='<|-|>', lw=0.3, ls='--', color='gray'), clip_on=False)
        ax.plot([0.005, 0.005], [0.05, 0.22], transform=ax.transAxes, color='gray', ls='--', lw=0.3)
        ax.text(0.3, -0.09, 'Frequency', transform=ax.transAxes, va='center', ha='center', fontsize=6, clip_on=False)

        targets = dict(left=220, right=40, top=100, bottom=30, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya_():
        # %%
        variables = [dict(variable='wd', column_name='wd'),
                    dict(variable='ws', column_name='ws'),
                    dict(variable='t2m', column_name='t2m'),
                    dict(variable='q2m', column_name='fre_rh_corr_adjusted_qm_q', rename_column='q2m'),
                    dict(variable='atm_pressure', column_name='atm_pressure'),
                    dict(variable='t2m_tsurf_diff', column_name='t2m_tsurf_diff')]

        freya_variables = []
        for v in variables:
            df = read_preprocessed_station_data(station='freya', variable=v['variable'], column_name=v['column_name'], time_freq='hour', rename_column=v.get('rename_column'))
            freya_variables.append(df)
        freya_variables = pd.concat(freya_variables, axis=1).sort_index()
        freya_variables['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_variables, 'wd')['wd_class']

        # Box plot properties
        boxplot_prop = dict(showcaps=False, widths=0.16, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1.5, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
        boxplot_prop_katabatic = dict(boxprops=dict(facecolor='royalblue', color='k', linewidth=0.5))
        boxplot_prop_non_katabatic = dict(boxprops=dict(facecolor='#7FC97F', color='k', linewidth=0.5))

        # Plotting
        fig = plt.figure(figsize=(6, 4), dpi=600)
        gs = plt.GridSpec(5, 7, hspace=0.2, wspace=0.01)

        # variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4)]
        variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4), ('fog_event', 5)]
        months = [('May', 5, 0), ('Jun', 6, 1), ('July', 7, 2), ('Aug', 8, 3), ('Sep', 9, 4), ('MJJAS', 56789, 5)]
        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        non_katabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions

        for variable, row in variables:
            for month_name, month, col in months:
                if variable != 'fog_event':
                    if month_name == 'MJJAS':

                        freya_variables_month = freya_variables.loc[freya_variables.index.month.isin([5, 6, 7, 8, 9]), [variable, 'wd_cardinal']].dropna()
                    else:
                        freya_variables_month = freya_variables.loc[freya_variables.index.month == month, [variable, 'wd_cardinal']].dropna()
                elif variable == 'fog_event':
                    ifile_path = '/nas/data/processed/people/sos/fog_detection_using_cnn/fog_image_annotation/fog_timeseries/'
                    fog_class_all = []
                    for year in [2016, 2019]:
                        if month_name == 'MJJAS':
                            fog_class = pd.concat([pd.read_csv(f"{ifile_path}fog_on_glacier_{year}_{m}.csv", parse_dates=['date']).set_index('date')
                                                for m in [5, 6, 7, 8, 9]]).sort_index()
                        else:
                            fog_class = pd.read_csv(f"{ifile_path}fog_on_glacier_{year}_{month}.csv", parse_dates=['date']).set_index('date')
                        fog_class_all.append(fog_class)
                    fog_class_all = pd.concat(fog_class_all).sort_index()
                    # Join fog for that month and freya variables
                    freya_variables_month = fog_class_all.join(freya_variables, how='left')

                if variable != 'fog_event':
                    # Select only katabatic and non-katabatic wind directions
                    freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][variable])
                    freya_variables_month_katabatic_wind['wind_type'] = 'katabatic'

                    freya_variables_month_non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][variable])
                    freya_variables_month_non_katabatic_wind['wind_type'] = 'non-katabatic'

                    freya_variables_month_interested = pd.concat([freya_variables_month_katabatic_wind, freya_variables_month_non_katabatic_wind], ignore_index=True)
                    freya_variables_month_interested['time'] = month_name

                    # Wind Speed and other variables
                    ax = fig.add_subplot(gs[row, col])
                    pos = 0.08
                    # Create box plot properties based on wind type
                    ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'katabatic', variable], whis=[5, 95], positions=np.array([-pos]), **boxplot_prop_katabatic, **boxplot_prop)
                    ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non-katabatic', variable], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_non_katabatic, **boxplot_prop)
                    # Create violin plot
                    ax_violin = sns.violinplot(x='time', y=variable, hue='wind_type', hue_order=['katabatic', 'non-katabatic'], split=True, inner=None,
                                    palette={'non-katabatic':'#7FC97F', 'katabatic':'royalblue'}, data=freya_variables_month_interested, density_norm='area',
                                    ax=ax, linewidth=0.4) #
                    # Remove default labels and legend
                    plt.setp(ax_violin.collections, zorder=2, label='')
                    ax.get_legend().set_visible(False)
                    ax.set_ylabel('')
                    ax.set_xlabel('')

                    text_params = dict(verticalalignment='center', transform=ax.transAxes, clip_on=False, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                    if variable == 'ws':
                        ax.text(0.5, 1.1, month_name, size=12, color='k', horizontalalignment='center', **text_params)

                    ax.text(0.49, -0.01, f"{len(freya_variables_month_katabatic_wind)}", horizontalalignment='right', size=6, color='royalblue', **text_params)
                    ax.text(0.52, -0.01, f"{len(freya_variables_month_non_katabatic_wind)}", horizontalalignment='left', size=6, color='#7FC97F', **text_params)

                    if variable == 'ws':
                        ylim = [0, 10]
                        yticks = [0, 2, 4, 6, 8, 10]
                        yticklabels = [0, 2, 4, 6, 8, 10]
                        # var_label = r'WS' + '\n' + r'(m s$^\mathdefault{-1}$)'
                        var_label = r'WS (m s$^\mathdefault{-1}$)'
                    elif variable == 'atm_pressure':
                        ylim = [900, 950]
                        yticks = [900, 910, 920, 930, 940, 950]
                        yticklabels = [900, 910, 920, 930, 940, 950]
                        # var_label = r'P$_\mathdefault{surf}$' + '\n' + r'(hPa)'
                        var_label = r'P$_\mathdefault{surf}$ (hPa)'
                    elif variable == 't2m':
                        ylim = [-15, 15]
                        yticks = [-15, -10, -5, 0, 5, 10, 15]
                        yticklabels = [-15, -10, -5, 0, 5, 10, 15]
                        # var_label = r'T$_\mathdefault{2m}$' + '\n' + r'($^\circ{\mathdefault{C}}$)'
                        var_label = r'T$_\mathdefault{2m}$ ($^\circ{\mathdefault{C}}$)'
                    elif variable == 'q2m':
                        ylim = [0, 6]
                        yticks = [0, 1, 2, 3, 4, 5, 6]
                        yticklabels = [0, 1, 2, 3, 4, 5, 6]
                        # var_label = r'q$_\mathdefault{2m}$' +'\n'+ r'(g kg$^\mathdefault{-1}$)'
                        var_label = r'q$_\mathdefault{2m}$ (g kg$^\mathdefault{-1}$)'
                    elif variable == 't2m_tsurf_diff':
                        ylim = [-5, 15]
                        yticks = [-5, 0, 5, 10, 15]
                        yticklabels = [-5, 0, 5, 10, 15]
                        # var_label = r'T$_\mathdefault{2m}$ - T$_\mathdefault{surf}$' + '\n' + r'($^\circ{\mathdefault{C}}$)'
                        var_label = r'T$_\mathdefault{2m}$ - T$_\mathdefault{surf}$ ($^\circ{\mathdefault{C}}$)'

                    ax.set_ylim(ylim)
                    ax.set_yticks(yticks)
                    ax.set_yticklabels(yticklabels)

                    ax.set_xlim([-0.5, 0.5])
                    ax.set_xticks([0])
                    ax.set_xticklabels([''])

                    ax.spines['bottom'].set_visible(False)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)

                    remove_tick_params = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
                    ax.tick_params(axis='x', **remove_tick_params)
                    if col != 0:
                        ax.spines['left'].set_visible(False)
                        ax.tick_params(axis='y', **remove_tick_params)

                    if col == 0:
                        ax.tick_params(axis='y', direction='out', length=6, pad=1, labelsize=7, width=0.7)
                        ax.set_ylabel(var_label, rotation='vertical', size=7, color='k', labelpad=0.1)

                    ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

                elif variable == 'fog_event':
                    freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions), ['fog_on_glacier', 'fog_cover_aws']])
                    freya_variables_month_katabatic_wind['no_fog_on_glacier'] = 1 - freya_variables_month_katabatic_wind['fog_on_glacier']

                    freya_variables_month_katabatic_wind_sum = pd.DataFrame(freya_variables_month_katabatic_wind.sum())

                    freya_variables_month_katabatic_wind_sum['fog_on_glacier_perc'] = (freya_variables_month_katabatic_wind_sum['fog_on_glacier'] / freya_variables_month_katabatic_wind_sum.sum()) * 100

                    freya_variables_month_katabatic_wind_fog_no_fog = freya_variables_month_katabatic_wind.loc[:, ['fog_on_glacier', 'no_fog_on_glacier']].value_counts().reset_index()
                    freya_variables_month_katabatic_wind_fog_no_fog['perc'] = (freya_variables_month_katabatic_wind_fog_no_fog['count'] / freya_variables_month_katabatic_wind_fog_no_fog['count'].sum()) * 100


                    # df_fog_event_count = df_fog_event[variable].value_counts().reset_index()





                    # # Select only katabatic and non-katabatic wind directions
                    # freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][variable])
                    # freya_variables_month_katabatic_wind['wind_type'] = 'katabatic'

                    # freya_variables_month_non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][variable])
                    # freya_variables_month_non_katabatic_wind['wind_type'] = 'non-katabatic'

                    # freya_variables_month_interested = pd.concat([freya_variables_month_katabatic_wind, freya_variables_month_non_katabatic_wind], ignore_index=True)
                    # freya_variables_month_interested['time'] = month_name

                    # # Wind Speed and other variables
                    # ax = fig.add_subplot(gs[row, col])
                    # pos = 0.08






        # ------------------------------------
        # Plot idealized wind rose in new axis
        # ------------------------------------
        cardinal_directions = np.array(['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'])
        ax = fig.add_subplot(gs[0, 6], polar=True)
        # If you want the first axis to be on top:
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        # Each sector is 22.5 degrees wide
        # Katabatic wind sector (degrees)
        katabatic_start = np.deg2rad(123.75) # 135 (SE) - 11.25
        katabatic_end   = np.deg2rad(168.75) # 157.5 (SSE) + 11.25

        # Non-katabatic wind sector (degrees)
        non_katabatic_start = np.deg2rad(281.25) # 292.5 (WNW) - 11.25
        non_katabatic_end   = np.deg2rad(326.25) # 315 (NW) + 11.25

        for wind_type in [(katabatic_start, katabatic_end, 'royalblue'), (non_katabatic_start, non_katabatic_end, '#7FC97F')]:
            # Sector properties
            theta_center = (wind_type[0] + wind_type[1]) / 2
            theta_width  = wind_type[1] - wind_type[0]
            radius = 100  # magnitude of katabatic wind
            # Plot filled wind-rose sector
            ax.bar(theta_center, radius, width=theta_width, bottom=0, color=wind_type[2], linewidth=0, alpha=0.9, zorder=12)

        angles_deg = [0, 90, 135, 157.5, 180, 270, 292.5, 315]
        ax.set_xticks(np.deg2rad(angles_deg))
        ax.set_xticklabels(['N', 'E', 'SE', 'SSE', 'S', 'W', 'WNW', 'NW'], color='grey', size=6)
        # Move labels closer to the circle
        for label in ax.get_xticklabels():
            label.set_clip_on(False)      # <-- critical
            label.set_y(0.35)     # default ~1.1, higher = closer
            label.set_horizontalalignment('center')
            label.set_verticalalignment('center')
            label.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
        ax.xaxis.grid(True, which='major', linewidth=0.2, color='grey', linestyle='--', zorder=12)

        # Set yticks and labels
        ax.set_yticklabels([])
        ax.yaxis.grid(False)

        # Polar spine (outer circle)
        ax.spines['polar'].set_linewidth(0.4)
        ax.spines['polar'].set_zorder(1)
        ax.spines['polar'].set_color('k')

        # ------------------------------------
        # Add legend in new axis
        # ------------------------------------
        ax = fig.add_subplot(gs[1, 6])
        # Define a cone (triangle) pointing downward
        cone_vertices = np.array([[0.0,  -1.8], [-0.9, 1.0], [0.9, 1.0], [0.0,  -1.8]])
        # ['long tip', 'narrow left base', 'narrow right base', 'close']
        cone_codes = [mplpath.MOVETO, mplpath.LINETO, mplpath.LINETO, mplpath.CLOSEPOLY]
        cone_marker = mplpath(cone_vertices, cone_codes)

        legend_handle_params = dict(linestyle='None', marker=cone_marker, markersize=10,  markeredgewidth=0.5)
        non_katabatic_handle = Line2D([], [], markerfacecolor='#7FC97F', markeredgecolor='#7FC97F', **legend_handle_params)
        katabatic_handle = Line2D([], [], markerfacecolor='royalblue', markeredgecolor='royalblue', **legend_handle_params)

        wind_legend = ax.legend(handles=[non_katabatic_handle, katabatic_handle],
                                labels=[f"Non-katabatic wind\n({int(np.round(np.rad2deg(non_katabatic_start), 0))}° – {int(np.round(np.rad2deg(non_katabatic_end), 0))}°)",
                                        f"Katabatic wind\n({int(np.round(np.rad2deg(katabatic_start), 0))}° – {int(np.round(np.rad2deg(katabatic_end), 0))}°)"],
                                bbox_to_anchor=(0.5, 0.5), loc='center', fontsize=6, frameon=False, handletextpad=0.05, labelspacing=0.8)

        for wind_legend_text in wind_legend.get_texts():
            wind_legend_text.set_color('k')
            wind_legend_text.set_path_effects([pe.withStroke(linewidth=1, foreground='white')])
            wind_legend_text.set_ha('center')
        ax.axis('off')

        # ------------------------------------
        # Add idealized violin plot in new axis
        # ------------------------------------
        ideal_variable = 'ws'
        freya_variables_month = freya_variables.loc[freya_variables.index.month == 5, [ideal_variable, 'wd_cardinal']].dropna()
        # Select only katabatic and upvalley wind directions
        freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][ideal_variable])
        freya_variables_month_katabatic_wind['wind_type'] = 'katabatic'
        freya_variables_month_non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][ideal_variable])
        freya_variables_month_non_katabatic_wind['wind_type'] = 'non_katabatic'
        freya_variables_month_interested = pd.concat([freya_variables_month_katabatic_wind, freya_variables_month_non_katabatic_wind], ignore_index=True)
        freya_variables_month_interested['time'] = month_name
        # Plot only one side of the violin plot as idealized
        freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'katabatic', ideal_variable] = np.nan

        ax = fig.add_subplot(gs[2, 6])
        pos = 0.08
        # Create box plot properties based on wind type
        boxplot_prop_non_katabatic_ideal = dict(boxprops=dict(facecolor='w', color='k', linewidth=0.5))
        ax.boxplot(freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non_katabatic', ideal_variable], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_non_katabatic_ideal, **boxplot_prop)
        # Create violin plot
        ax_violin = sns.violinplot(x='time', y=ideal_variable, hue='wind_type', hue_order=['non_katabatic', 'katabatic'], split=True, inner=None,
                        palette={'non_katabatic':'w', 'katabatic':'royalblue'}, data=freya_variables_month_interested, density_norm='area',
                        ax=ax, linewidth=0.4)

        [sp.set_visible(False) for sp in ax.spines.values()]
        plt.setp(ax_violin.collections, zorder=2, label='')
        ax.get_legend().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.set_xlim([-0.4, 0.5])
        ax.set_ylim([-2, 10])
        ax.axis('off')

        # Plot arrows and labels
        pos = 0.2
        quantile_fontsize = 6
        ideal_data = freya_variables_month_interested.loc[freya_variables_month_interested['wind_type'] == 'non_katabatic', ideal_variable]
        ax.text(pos, ideal_data.quantile(0.75), '75%', fontsize=quantile_fontsize)
        ax.text(pos, ideal_data.mean() - 0.3, 'Mean', color='k', fontsize=quantile_fontsize, fontweight='bold')
        ax.text(pos, ideal_data.quantile(0.50) - 0.2, '50%', color='grey', fontsize=quantile_fontsize, fontweight='bold')
        ax.text(pos, ideal_data.quantile(0.25) - 0.8, '25%', fontsize=quantile_fontsize)

        ax.annotate('', xy=(-0.4, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle='<|-|>', lw=0.3, ls='--', color='gray'), clip_on=False)
        ax.plot([0.005, 0.005], [0.05, 0.22], transform=ax.transAxes, color='gray', ls='--', lw=0.3)
        ax.text(0.3, -0.09, 'Frequency', transform=ax.transAxes, va='center', ha='center', fontsize=6, clip_on=False)

        targets = dict(left=220, right=40, top=100, bottom=30, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_wind_speed_temperature_humidity_pressure_inversion_based_on_wind_direction_at_freya.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)









# %%
    def test_normality_of_variables_based_on_wind_direction_at_freya():
        # %%
        variables = [dict(variable='wd', column_name='wd'),
                        dict(variable='ws', column_name='ws'),
                        dict(variable='t2m', column_name='t2m'),
                        dict(variable='q2m', column_name='fre_rh_corr_adjusted_qm_q', rename_column='q2m'),
                        dict(variable='atm_pressure', column_name='atm_pressure'),
                        dict(variable='t2m_tsurf_diff', column_name='t2m_tsurf_diff')]

        freya_variables = []
        for v in variables:
            df = read_preprocessed_station_data(station='freya', variable=v['variable'], column_name=v['column_name'], time_freq='hour', rename_column=v.get('rename_column'))
            freya_variables.append(df)
        freya_variables = pd.concat(freya_variables, axis=1).sort_index()
        freya_variables['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_variables, 'wd')['wd_class']

        # Normality test results storage
        var_list = [v['variable'] for v in variables]
        normality_tests = ['Shapiro-Wilk', 'DAgostino', 'Anderson-Darling']
        wind_direction_types = ['KAT', 'NKAT']
        miindex = pd.MultiIndex.from_product([normality_tests, wind_direction_types])

        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        non_katabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions
        months = [[5], [6], [7], [8], [9], [5, 6, 7, 8, 9]]  # May to September
        for month in months:
            normality_results = pd.DataFrame(index=var_list, columns=miindex)
            for variable in var_list:
                freya_variables_month = freya_variables.loc[freya_variables.index.month.isin(month), [variable, 'wd_cardinal']].dropna()

                # Select only katabatic and non-katabatic wind directions
                katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][variable])
                non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][variable])

                for data, label in [(katabatic_wind[variable], 'KAT'), (non_katabatic_wind[variable], 'NKAT')]:
                    # print(f"\nNormality tests for {label} - variable: {variable}")
                    # print('\nPerforming Shapiro-Wilk test')
                    stat, p = shapiro(data) # Perform Shapiro-Wilk test
                    print('Statistics=%.3f, p=%.3f' % (stat, p))
                    # interpret
                    alpha = 0.05
                    if p > alpha:
                        # print(f"{label} looks Gaussian (fail to reject H0)")
                        # normality_results.loc['Shapiro-Wilk', variable] = 'Gaussian'
                        normality_results.loc[variable, ('Shapiro-Wilk', label)] = 'G'
                    else:
                        # print(f"{label} does not look Gaussian (reject H0)")
                        normality_results.loc[variable, ('Shapiro-Wilk', label)] = 'NG'

                    # print('\nPerforming d\'agostino and pearson\'s test')
                    stat, p = normaltest(data)
                    # print('Statistics=%.3f, p=%.3f' % (stat, p))
                    # interpret
                    alpha = 0.05
                    if p > alpha:
                        # print(f"{label} looks Gaussian (fail to reject H0)")
                        normality_results.loc[variable, ('DAgostino', label)] = 'G'
                    else:
                        # print(f"{label} does not look Gaussian (reject H0)")
                        normality_results.loc[variable, ('DAgostino', label)] = 'NG'

                    # print("\n Performing the anderson-darling test")
                    result = anderson(data)
                    # print('Statistic: %.3f' % result.statistic)
                    # for i in range(len(result.critical_values)):
                    i = 4  # 1% significance level
                    sl, cv = result.significance_level[i], result.critical_values[i]
                    if result.statistic < result.critical_values[i]:
                        # print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                        normality_results.loc[variable, ('Anderson-Darling', label)] = 'G'
                    else:
                        # print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
                        normality_results.loc[variable, ('Anderson-Darling', label)] = 'NG'

            # Save normality test results
            normality_results.rename(columns={'DAgostino': fr"D'Agostino's $K^\mathdefault{{2}}$"}, inplace=True)
            normality_results.rename(index={'wd':fr'WD ($^\circ$)',
                                        'ws': fr"WS (m s$^\mathdefault{{-1}}$)",
                                        't2m': fr"T$_\mathdefault{{2m}}$ ($^\circ$C)",
                                        'q2m': fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)" ,
                                        'atm_pressure': fr"P$_\mathdefault{{atm}}$ (hPa)",
                                        't2m_tsurf_diff': fr"$\Delta$T ($^\circ$C)"}, inplace=True)
            flat_cols = [f"{c[1]}" for c in normality_results.columns]

            # Plot normality test results as a table
            fig, ax = plt.subplots(figsize=(2.5, 0.8), dpi=600)  # Adjust size as needed
            ax.axis('off')  # Hide axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Create the percentage of wind direction and wind speed data table
            normality_table = ax.table(cellText=normality_results.values,   # Round numbers to 2 decimals
                                    colLabels=flat_cols,
                                    rowLabels=normality_results.index,
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0, 0, 1, 0.85]) # [x, y, width, height]
            normality_table.auto_set_font_size(False)
            normality_table.set_fontsize(4)

            # --- Set cell sizes and alignment ---
            for (row, col), cell in normality_table.get_celld().items():
                # Width for row labels
                if col == -1:
                    cell.set_width(0.09)
                    cell.set_fontsize(6)
                if row == 0:
                    cell.set_fontsize(6)

                # # Heights
                cell.set_height(0.005 if row == 0 else 0.005)
                # Center text
                cell.get_text().set_horizontalalignment('center')
                cell.get_text().set_verticalalignment('center')
                # Padding
                cell.PAD = 0.01
                # Default borders
                cell.visible_edges = ''
                cell.set_edgecolor('#666666' if row > 0 else 'black')
                cell.set_linewidth(0.2 if row > 0 else 0.8)

            # --- Header styling ---
            for col in range(len(normality_results.columns)):
                cell = normality_table[(0, col)]
                cell.get_text().set_color('k')
                cell.visible_edges = 'B'  # top & bottom line
                cell.set_linewidth(0.4)

            # --- Add horizontal lines for the rest ---
            n_rows = len(normality_results.index) + 1
            n_cols = len(normality_results.columns)
            for row in range(1, n_rows):
                for col in range(n_cols):
                    normality_table[(row, col)].visible_edges = 'B'

            # --- Add group labels above the table ---
            group_labels = list(dict.fromkeys([col[0] for col in normality_results.columns]))
            n_groups = len(group_labels)
            for i, label in enumerate(group_labels):
                x_center = (i + 0.5) / n_groups  # evenly distribute across axes
                ax.text(x_center, 0.89, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
            ax.plot([0.007, 1], [0.965, 0.965], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

            targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'test_normality_of_variables_based_on_wind_direction_at_freya'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            if month == [5]:
                month_str = 'May'
            elif month == [6]:
                month_str = 'Jun'
            elif month == [7]:
                month_str = 'Jul'
            elif month == [8]:
                month_str = 'Aug'
            elif month == [9]:
                month_str = 'Sep'
            elif month == [5, 6, 7, 8, 9]:
                month_str = 'MJJAS'
            fig_name = f"test_normality_of_variables_based_on_wind_direction_at_freya_{month_str}.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def report_ratio_katabatic_versus_non_katabatic_wind_direction_at_freya():
        # %%
        freya_wd = read_preprocessed_station_data(station='freya', variable='wd', column_name='wd', time_freq='hour')
        freya_wd['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_wd, 'wd')['wd_class']

        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        non_katabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions

        # Select only katabatic and non-katabatic wind directions
        katabatic_wind = pd.DataFrame(freya_wd.loc[freya_wd.wd_cardinal.isin(katabatic_wind_directions)]['wd'])
        non_katabatic_wind = pd.DataFrame(freya_wd.loc[freya_wd.wd_cardinal.isin(non_katabatic_wind_directions)]['wd'])

        ratio_katabatic_to_non_katabatic = len(katabatic_wind) / len(non_katabatic_wind)
        print(f"Ratio of katabatic to non-katabatic wind direction occurrences at Freya station (MJJAS): {ratio_katabatic_to_non_katabatic:.3f}")
        print("Interpretation:")
        print(f"Katabatic winds were observed {ratio_katabatic_to_non_katabatic:.3f} times more often than non-katabatic winds in the measurement period (MJJAS = May, June, July, August, September).")
        print(f"In other words, for roughly every 1 non-katabatic wind occurrence, there were about {ratio_katabatic_to_non_katabatic:.3f} katabatic wind occurrences.")

        # How to Interpret in Context
        katabatic_fraction = ratio_katabatic_to_non_katabatic / (ratio_katabatic_to_non_katabatic + 1)
        non_katabatic_fraction = 1 / (ratio_katabatic_to_non_katabatic + 1)
        print("\nContextual Interpretation:")
        print(f"if you imagine 100 wind direction events:\nAbout {katabatic_fraction * 100:.0f} events would be katabatic,\nAbout {non_katabatic_fraction * 100:.0f} events would be non-katabatic.")

        # total = len(katabatic_wind) + len(non_katabatic_wind)
        # katabatic_percentage = (len(katabatic_wind) / total) * 100
        # non_katabatic_percentage = (len(non_katabatic_wind) / total) * 100
        # %%
    # %%
    def test_significant_difference_in_variable_based_on_wind_direction_at_freya():
        # %%
        # Is a randomly chosen katabatic wind speed likely to be larger than a randomly chosen non-katabatic wind speed?
        # %%
        variables = [dict(variable='wd', column_name='wd'),
                    dict(variable='ws', column_name='ws'),
                    dict(variable='t2m', column_name='t2m'),
                    dict(variable='q2m', column_name='fre_rh_corr_adjusted_qm_q', rename_column='q2m'),
                    dict(variable='atm_pressure', column_name='atm_pressure'),
                    dict(variable='t2m_tsurf_diff', column_name='t2m_tsurf_diff')]
        var_list = [v['variable'] for v in variables if v['variable'] != 'wd']

        freya_variables = []
        for v in variables:
            df = read_preprocessed_station_data(station='freya', variable=v['variable'], column_name=v['column_name'], time_freq='hour', rename_column=v.get('rename_column'))
            freya_variables.append(df)
        freya_variables = pd.concat(freya_variables, axis=1).sort_index()
        freya_variables['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_variables, 'wd')['wd_class']

        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        non_katabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions

        months = [[5], [6], [7], [8], [9], [5, 6, 7, 8, 9]]  # May to September
        for month in months:
            # Statistically significant difference test results storage
            # significance_diff_results = pd.DataFrame(index=var_list, columns=['N', 'KAT:NKAT', 'Median\n(KAT-NKAT)', fr"Cliff's $\delta$", 'P (KAT > NKAT) %', 'Mann–Whitney\np', 'Perm\np'])
            significance_diff_results = pd.DataFrame(index=var_list, columns=['Median\n(KAT-NKAT)', "Cliff's\nδ", 'P\n(KAT>NKAT) %', 'U–test\np', 'Perm\np'])
            # Store observed difference and p-value from permutation test
            obs_diff_perm = pd.DataFrame(index=var_list, columns=['Observed_Median_Difference', 'P_value_Perm_test)'])

            for variable in var_list:
                # variable = 't2m_tsurf_diff'  # Wind speed Change variable here for testing
                freya_variables_month = freya_variables.loc[freya_variables.index.month.isin(month), [variable, 'wd_cardinal']].dropna()

                # Select only katabatic and non-katabatic wind directions
                katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)][variable])
                non_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(non_katabatic_wind_directions)][variable])

                # Store the number of samples
                # significance_diff_results.loc[variable, 'N'] = len(katabatic_wind) + len(non_katabatic_wind)
                # ratio_katabatic_to_non_katabatic = len(katabatic_wind) / len(non_katabatic_wind)
                # significance_diff_results.loc[variable, 'KAT:NKAT'] = f"{ratio_katabatic_to_non_katabatic:.2f}:1"

                # Store the cliff delta
                delta, p_x_greater_y, p_y_greater_x = _cliffs_delta_fast(katabatic_wind[variable], non_katabatic_wind[variable])
                if np.absolute(np.round(delta, 1)) < 0.09:
                    significance_diff_results.loc[variable, "Cliff's\nδ"] = f"0\n({p_x_greater_y})"
                else:
                    significance_diff_results.loc[variable, "Cliff's\nδ"] = f"{delta:.1f}\n({p_x_greater_y})"
                significance_diff_results.loc[variable, 'P\n(KAT>NKAT) %'] = f"{p_x_greater_y}"

                # Store the mannwhitneyu test p-value
                u_stat, p_value = mannwhitneyu(katabatic_wind[variable], non_katabatic_wind[variable], alternative='two-sided')
                if p_value <= 0.01:
                    significance_diff_results.loc[variable, 'U–test\np'] = '<0.01'
                elif (p_value > 0.01) & (p_value <= 0.05):
                    significance_diff_results.loc[variable, 'U–test\np'] = '<0.05'
                elif (p_value > 0.05) & (p_value <= 0.1):
                    significance_diff_results.loc[variable, 'U–test\np'] = '<0.1'
                else:
                    significance_diff_results.loc[variable, 'U–test\np'] = '–'

                # Store permutation test for median difference
                save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
                obs_diff, p_perm, perm_dist = _permutation_test(katabatic_wind[variable].values, non_katabatic_wind[variable].values, n_perm=10000)
                if month == [5]:
                    month_str = 'May'
                elif month == [6]:
                    month_str = 'Jun'
                elif month == [7]:
                    month_str = 'Jul'
                elif month == [8]:
                    month_str = 'Aug'
                elif month == [9]:
                    month_str = 'Sep'
                elif month == [5, 6, 7, 8, 9]:
                    month_str = 'MJJAS'
                # Save permutation distribution
                np.save(f"{save_path}/perm_dist_{variable}_{month_str}.npy", perm_dist)

                # Store results
                obs_diff_perm.loc[variable, 'Observed_Median_Difference'] = obs_diff
                obs_diff_perm.loc[variable, 'P_value_Perm_test)'] = p_perm

                if p_perm <= 0.01:
                    significance_diff_results.loc[variable, 'Perm\np'] = '<0.01'
                    significance_diff_results.loc[variable, 'Median\n(KAT-NKAT)'] = f"{np.round(obs_diff, 1)}$^{{**}}$"
                elif (p_perm > 0.01) & (p_perm <= 0.05):
                    significance_diff_results.loc[variable, 'Perm\np'] = '<0.05'
                    significance_diff_results.loc[variable, 'Median\n(KAT-NKAT)'] = f"{np.round(obs_diff, 1)}$^{{*}}$"
                # elif (p_perm > 0.05) & (p_perm <= 0.1):
                #     significance_diff_results.loc[variable, 'Perm\np'] = '<0.1'
                #     significance_diff_results.loc[variable, 'Median\n(KAT-NKAT)'] = f"{np.round(obs_diff, 1)}$^{{*}}$"
                else:
                    significance_diff_results.loc[variable, 'Perm\np'] = '–'
                    significance_diff_results.loc[variable, 'Median\n(KAT-NKAT)'] = '-'

            significance_diff_results.rename(index={'ws': fr"WS (m s$^\mathdefault{{-1}}$)",
                                        't2m': fr"T$_\mathdefault{{2m}}$ ($^\circ$C)",
                                        'q2m': fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)" ,
                                        'atm_pressure': fr"P$_\mathdefault{{atm}}$ (hPa)",
                                        't2m_tsurf_diff': fr"$\Delta$T ($^\circ$C)"}, inplace=True)

            # Save significance test results
            significance_diff_results.to_csv(f"{save_path}/significance_diff_results_{month_str}.csv")
            obs_diff_perm.to_csv(f"{save_path}/obs_diff_perm_{month_str}.csv")
        # %%
    def plot_significant_difference_in_variable_based_on_wind_direction_at_freya():
        # %%
        months_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'MJJAS']  # May to September
        significance_stats = [fr'$\Delta$Med', 'δ\n(P%)']
        miindex = pd.MultiIndex.from_product([months_names, significance_stats])
        var_list = [fr"WS (m s$^\mathdefault{{-1}}$)", fr"T$_\mathdefault{{2m}}$ ($^\circ$C)", fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)" , fr"P$_\mathdefault{{atm}}$ (hPa)", fr"$\Delta$T ($^\circ$C)"]
        significance_diff_results_all  = pd.DataFrame(index=var_list, columns=miindex)

        for month_str in months_names:
            # Load significance test results
            save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
            significance_diff_results = pd.read_csv(f'{save_path}/significance_diff_results_{month_str}.csv', index_col=0)

            # Combine results into a single dataframe
            significance_diff_results_all.loc[:, (month_str, fr'$\Delta$Med')] = significance_diff_results.loc[:, "Median\n(KAT-NKAT)"].values
            significance_diff_results_all.loc[:, (month_str, 'δ\n(P%)')] = significance_diff_results.loc[:, "Cliff's\nδ"].values

        # Plot significance test results as a table
        fig, ax = plt.subplots(figsize=(3, 0.95), dpi=600)  # Adjust size as needed
        ax.axis('off')  # Hide axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Flatten columns for multicolumn plotting
        flat_cols = [f"{c[1]}" for c in significance_diff_results_all.columns]
        # Create the percentage of wind direction and wind speed data table
        significance_diff_table = ax.table(cellText=significance_diff_results_all.values,   # Round numbers to 2 decimals
                                colLabels=flat_cols,
                                rowLabels=significance_diff_results_all.index,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 0.94]) # [x, y, width, height]
        significance_diff_table.auto_set_font_size(False)
        significance_diff_table.set_fontsize(4)

        # --- Set cell sizes and alignment ---
        for (row, col), cell in significance_diff_table.get_celld().items():
            # Width for row labels
            if col == -1:
                cell.set_width(0.09)
                cell.set_fontsize(5)
            if row == 0:
                cell.set_fontsize(5)

            # # Heights
            cell.set_height(0.009 if row == 0 else 0.005)
            # Center text
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
            # Padding
            # cell.PAD = 0.01
            # Default borders
            cell.visible_edges = ''
            cell.set_edgecolor('#666666' if row > 0 else 'black')
            cell.set_linewidth(0.2 if row > 0 else 0.8)

        # --- Header styling ---
        for col in range(len(significance_diff_results_all.columns)):
            cell = significance_diff_table[(0, col)]
            cell.get_text().set_color('k')
            cell.visible_edges = 'B'  # top & bottom line
            cell.set_linewidth(0.4)

        # --- Add horizontal lines for the rest ---
        n_rows = len(significance_diff_results_all.index) + 1
        n_cols = len(significance_diff_results_all.columns)
        for row in range(1, n_rows):
            for col in range(n_cols):
                significance_diff_table[(row, col)].visible_edges = 'B'

        # --- Add group labels above the table ---
        group_labels = list(dict.fromkeys([col[0] for col in significance_diff_results_all.columns]))
        n_groups = len(group_labels)
        for i, label in enumerate(group_labels):
            x_center = (i + 0.5) / n_groups  # evenly distribute across axes
            ax.text(x_center, 0.93, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
        ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

        targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_significant_difference_in_variables_based_on_wind_direction_at_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_significant_difference_in_variables_based_on_wind_direction_at_freya.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def plot_permutation_test_for_variable_based_on_wind_direction_at_freya():
        """
        Permutation tests were performed using a fixed random seed, and permutation distributions were generated using non-overlapping random permutations to ensure full reproducibility.
        """
        # %%
        var_list = [('a)', 'ws'), ('b)', 't2m'), ('c)', 'q2m'), ('d)', 'atm_pressure'), ('e)', 't2m_tsurf_diff')]
        tick_params = dict(axis='both', direction='inout', length=4, pad=1, labelsize=8, width=0.5, color='black')

        # for month in ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'MJJAS']:
        for month in ['MJJAS']:
            # Load the permutation results
            save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
            obs_diff_perm = pd.read_csv(f'{save_path}/obs_diff_perm_{month}.csv', index_col=0)

            # Plotting
            fig, ax = plt.subplots(2, 3, figsize=(6, 3), dpi=600)
            ax = ax.flatten()
            for i, (label, variable) in enumerate(var_list):
                # Load permutation test for median difference
                perm_dist = np.load(f"{save_path}/perm_dist_{variable}_{month}.npy")
                obs_diff = obs_diff_perm.loc[variable, 'Observed_Median_Difference']
                p_perm = obs_diff_perm.loc[variable, 'P_value_Perm_test)']

                ax[i].hist(perm_dist, bins=50, density=True, color='lightgray')
                ax[i].axvline(obs_diff, linestyle='--', linewidth=1, color='k', label='Observed median difference')

                if variable == 'ws':
                    if month == 'MJJAS':
                        xlim = [-1.2, 0.1]
                        xticks = [-1, -0.5, 0]
                        ylim = [0, 20]
                        yticks = [0, 5, 10, 15, 20]
                    elif month == 'May':
                        xlim = [-0.25, 1]
                        xticks = [0, 0.5, 1]
                        ylim = [0, 10]
                        yticks = [0, 5, 10]
                    else:
                        xlim = [-0.5, 0.5]
                        xticks = [-0.5, 0, 0.5]
                        ylim = [0, 15]
                        yticks = [0, 5, 10, 15]
                    var_label = rf"WS (m s$^\mathdefault{{-1}}$)"
                elif variable == 't2m':
                    if month == 'MJJAS':
                        xlim = [-1, 4.1]
                        xticks = [-1, 0, 1, 2, 3, 4]
                        ylim = [0, 2]
                        yticks = [0, 0.5, 1, 1.5, 2]
                    elif month == 'May':
                        xlim = [-1, 1]
                        xticks = [-1, -0.5, 0, 0.5, 1]
                        ylim = [0, 2]
                        yticks = [0, 0.5, 1, 1.5, 2]
                    else:
                        xlim = [-1, 1]
                        xticks = [-1, -0.5, 0, 0.5, 1]
                        ylim = [0, 2]
                        yticks = [0, 0.5, 1, 1.5, 2]
                    var_label = rf"T$_\mathdefault{{2m}}$ ($^\circ{{\mathdefault{{C}}}}$)"
                elif variable == 'q2m':
                    if month == 'MJJAS':
                        xlim = [-0.1, 0.5]
                        xticks = [0, 0.15, 0.3, 0.45]
                        ylim = [0, 13]
                        yticks = [0, 5, 10]
                    elif month == 'May':
                        xlim = [-0.2, 0.2]
                        xticks = [-0.2, 0.1, 0.2]
                        ylim = [0, 8]
                        yticks = [0, 4, 8]
                    else:
                        xlim = [-0.1, 0.3]
                        xticks = [0, 0.1, 0.2, 0.3]
                        ylim = [0, 10]
                        yticks = [0, 5, 10]
                    var_label = rf"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)"
                elif variable == 'atm_pressure':
                    if month == 'MJJAS':
                        xlim = [-0.75, 0.9]
                        xticks = [-0.5, 0, 0.5]
                        ylim = [0, 2.6]
                        yticks = [0, 1, 2]
                    elif month == 'May':
                        xlim = [-1, 1.5]
                        xticks = [-0.5, 0, 0.5, 1]
                        ylim = [0, 1.6]
                        yticks = [0, 0.5, 1, 1.5]
                    else:
                        xlim = [-0.75, 0.75]
                        xticks = [-0.5, 0, 0.5]
                        ylim = [0, 2.6]
                        yticks = [0, 1, 2]
                    var_label = rf"P$_\mathdefault{{surf}}$ (hPa)"
                elif variable == 't2m_tsurf_diff':
                    xlim = [-0.5, 3.5]
                    xticks = [0, 1, 2, 3]
                    ylim = [0, 5]
                    yticks = [0, 1, 2, 3, 4, 5]
                    var_label = rf"T$_\mathdefault{{2m}}$ - T$_\mathdefault{{surf}}$ ($^\circ{{\mathdefault{{C}}}}$)"

                ax[i].set_xlim(xlim)
                ax[i].set_xticks(xticks)
                ax[i].set_xticklabels(xticks)

                ax[i].set_ylim(ylim)
                ax[i].set_yticks(yticks)
                ax[i].set_yticklabels(yticks)

                # Style x, y ticks
                ax[i].tick_params(**tick_params)

                # ax[i].set_xlabel(f"{var_label}", labelpad=0.05, fontsize=9)
                # ax[i].set_ylabel('Density', labelpad=0.05, fontsize=9)
                ax[i].text(0.5, -0.15, f"{var_label}", color='k', horizontalalignment='center', verticalalignment='center', size=9, transform=ax[i].transAxes, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                ax[i].text(-0.11, 0.5, 'Density', color='k', horizontalalignment='center', verticalalignment='center', size=9, transform=ax[i].transAxes, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')], rotation=90)
                ax[i].text(0.05, 0.93, f"{label}", color='k', horizontalalignment='left', verticalalignment='center', size=9, transform=ax[i].transAxes, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])

            # Add legend in new axis
            # Remove unused subplot's axes
            [spine.set_visible(False) for spine in ax[5].spines.values()]
            ax[5].axis('off')
            # Legend handles
            month_handle = Line2D([0], [0], color='none', label=f'{month}', marker='')
            hist_handle = Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='k', linewidth=0.4, label='Permutation distribution')
            zac_handle = Line2D([0], [0], color='k',  linestyle='--', linewidth=1)
            leg = ax[5].legend(bbox_to_anchor=(0.5, 0.5), loc='center', handles=[month_handle, hist_handle, zac_handle], handler_map={hist_handle: HandlerHistogramBars()}, labels=[f'{month}', 'Permutation distribution\n (n = 10000)', 'Observed median\n(KAT–NKAT)'], ncol=1,
                    borderaxespad=0, fontsize=9, handletextpad=0.1, labelspacing=0.3, frameon=False, handlelength=1.2, handleheight=1.2,
                    edgecolor='none') # facecolor='w', framealpha=0.9, handler_map={tuple: HandlerTuple(ndivide=None)}

            # for text in leg.get_texts():
            #     x, y = text.get_position()
            #     text.set_position((x, y + 0.02))
            #     # text.set_va('center_baseline')

            targets = dict(left=160, right=40, top=25, bottom=160, hspace=390, wspace=600)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'plot_permutation_test_for_variable_based_on_wind_direction_at_freya'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f'plot_permutation_test_for_variable_based_on_wind_direction_at_freya_{month}.png'
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

class FogWindDirectionAnalysis:
    def report_fog_occurrence_based_on_wind_direction_at_freya():
        # %%
        variables = [dict(variable='wd', column_name='wd'),
                    dict(variable='ws', column_name='ws'),
                    dict(variable='t2m', column_name='t2m'),
                    dict(variable='q2m', column_name='fre_rh_corr_adjusted_qm_q', rename_column='q2m'),
                    dict(variable='atm_pressure', column_name='atm_pressure'),
                    dict(variable='t2m_tsurf_diff', column_name='t2m_tsurf_diff')]

        freya_variables = []
        for v in variables:
            df = read_preprocessed_station_data(station='freya', variable=v['variable'], column_name=v['column_name'], time_freq='hour', rename_column=v.get('rename_column'))
            freya_variables.append(df)
        freya_variables = pd.concat(freya_variables, axis=1).sort_index()
        freya_variables['wd_cardinal'] = vs.classify_wind_direction_in_degrees_into_cardinal_direction(freya_variables, 'wd')['wd_class']

        # variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4)]
        variables = [('ws', 0), ('t2m', 1), ('q2m', 2), ('atm_pressure', 3), ('t2m_tsurf_diff', 4), ('fog_event', 5)]
        months = [('May', 5, 0), ('Jun', 6, 1), ('July', 7, 2), ('Aug', 8, 3), ('Sep', 9, 4), ('MJJAS', 56789, 5)]
        katabatic_wind_directions = ['SE', 'SSE'] # Katabatic wind directions
        anabatic_wind_directions = ['NW', 'WNW'] # Non-katabatic wind directions

        ifile_path = '/nas/data/processed/people/sos/fog_detection_using_cnn/fog_image_annotation/fog_timeseries/'
        # variable = 'ws'
        variable = 'ws'
        print(variable)
        for month_name, month, col in months:
            print(f"\n------------------------- {month_name} --------------------------")
            fog_class_all = []
            for year in [2016, 2019]:
                if month_name == 'MJJAS':
                    fog_class = pd.concat([pd.read_csv(f"{ifile_path}fog_on_glacier_{year}_{m}.csv", parse_dates=['date']).set_index('date')
                                        for m in [5, 6, 7, 8, 9]]).sort_index()
                else:
                    fog_class = pd.read_csv(f"{ifile_path}fog_on_glacier_{year}_{month}.csv", parse_dates=['date']).set_index('date')
                fog_class_all.append(fog_class)
            fog_class_all = pd.concat(fog_class_all).sort_index()
            # Join fog for that month and freya variables
            freya_variables_month = fog_class_all.join(freya_variables, how='left')
            freya_variables_month['no_fog_on_glacier'] = 1 - freya_variables_month['fog_on_glacier']

            # Anabatic wind speed during Fog and no Fog event
            print("\n---- Anabatic Wind ----")
            freya_variables_month_anabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(anabatic_wind_directions)])
            freya_variables_month_anabatic_wind_fog_cover_aws_variable = pd.DataFrame(freya_variables_month_anabatic_wind.loc[(freya_variables_month_anabatic_wind.fog_on_glacier == 1), variable])
            print(f"FAWS: {freya_variables_month_anabatic_wind_fog_cover_aws_variable.mean()}")
            freya_variables_month_anabatic_wind_no_fog_on_glacier_variable = pd.DataFrame(freya_variables_month_anabatic_wind.loc[(freya_variables_month_anabatic_wind.fog_on_glacier == 0), variable])
            print(f"FN: {freya_variables_month_anabatic_wind_no_fog_on_glacier_variable.mean()}")
            # Store the cliff delta
            delta, p_x_greater_y, p_y_greater_x = _cliffs_delta_fast(freya_variables_month_anabatic_wind_fog_cover_aws_variable[variable], freya_variables_month_anabatic_wind_no_fog_on_glacier_variable[variable])
            print(f"Cliff's δ: {delta:.3f}, P(X>Y): {p_x_greater_y:.3f}, P(Y>X): {p_y_greater_x:.3f}")

            # Katabatic wind speed during Fog and no Fog event
            print("\n---- Katabatic Wind ----")
            freya_variables_month_katabatic_wind = pd.DataFrame(freya_variables_month.loc[freya_variables_month.wd_cardinal.isin(katabatic_wind_directions)])
            freya_variables_month_katabatic_wind_fog_cover_aws_variable = pd.DataFrame(freya_variables_month_katabatic_wind.loc[(freya_variables_month_katabatic_wind.fog_on_glacier == 1), variable])
            print(f"FAWS: {freya_variables_month_katabatic_wind_fog_cover_aws_variable.mean()}")
            freya_variables_month_katabatic_wind_no_fog_on_glacier_variable = pd.DataFrame(freya_variables_month_katabatic_wind.loc[(freya_variables_month_katabatic_wind.fog_on_glacier == 0), variable])
            print(f"FN: {freya_variables_month_katabatic_wind_no_fog_on_glacier_variable.mean()}")
            # Store the cliff delta
            delta, p_x_greater_y, p_y_greater_x = _cliffs_delta_fast(freya_variables_month_katabatic_wind_fog_cover_aws_variable[variable], freya_variables_month_katabatic_wind_no_fog_on_glacier_variable[variable])
            print(f"Cliff's δ: {delta:.3f}, P(X>Y): {p_x_greater_y:.3f}, P(Y>X): {p_y_greater_x:.3f}")

            # # Fog covers AWS
            # # freya_variables_month_fog_covers_aws = pd.DataFrame(freya_variables_month.loc[freya_variables_month.fog_cover_aws == 1])
            # freya_variables_month_fog_covers_aws = pd.DataFrame(freya_variables_month.loc[freya_variables_month.fog_on_glacier == 0])
            # variable = 'ws'
            # print(month_name)
            # print(variable)
            # # Fog covers AWS and katabatic wind direction occurrences
            # freya_variables_month_fog_covers_aws_katabatic_wind = pd.DataFrame(freya_variables_month_fog_covers_aws.loc[freya_variables_month_fog_covers_aws.wd_cardinal.isin(katabatic_wind_directions)])
            # freya_variables_month_fog_covers_aws_katabatic_wind_variable = pd.DataFrame(freya_variables_month_fog_covers_aws_katabatic_wind[variable])
            # print(f"KAT {freya_variables_month_fog_covers_aws_katabatic_wind_variable[variable].mean()}")


            # # Fog covers AWS and Anabatic wind direction occurrences
            # freya_variables_month_fog_covers_aws_anabatic_wind = pd.DataFrame(freya_variables_month_fog_covers_aws.loc[freya_variables_month_fog_covers_aws.wd_cardinal.isin(anabatic_wind_directions)])
            # freya_variables_month_fog_covers_aws_anabatic_wind_variable = pd.DataFrame(freya_variables_month_fog_covers_aws_anabatic_wind[variable])
            # print(f"ANA {freya_variables_month_fog_covers_aws_anabatic_wind_variable[variable].mean()}")







        # %%



class SeaIceCompositeAnalysis:
    def plot_variable_composites_based_on_high_low_sif():
        """Plot a violin plot of variable composites based on high and low sea ice years for the period 1990 to 2020.
        Variables: SMB, 2m air temperature, 2m specific humidity, incoming shortwave radiation, incoming longwave radiation

        TODO: Check L1_data_analysis.py function plot_violinplot_daily_var_on_freya_composites_based_on_high_low_ci_with_bootstrap
        for more details.
        """
        # %%
        # Get high and low sea ice days based on 100km sea ice concentration
        ci_high_low_days = _get_high_low_sea_ice_days()

        # Box plot properties
        boxplot_prop = dict(showcaps=False, widths=0.16, patch_artist=True, whiskerprops=dict(color='k', linewidth=0.9),
                            meanprops=dict(linewidth=1.5, ls='-', color='k'),
                            medianprops=dict(linewidth=1, ls='-', color='grey'),
                            showmeans=True, meanline=True, showfliers=None, sym='', zorder=3)
        boxplot_prop_high = dict(boxprops=dict(facecolor='#08519C', color='k', linewidth=0.5))
        boxplot_prop_low = dict(boxprops=dict(facecolor='#EF6548', color='k', linewidth=0.5))

        racmo_vars = [('smb', 0), ('t2m', 1), ('q2m', 2), ('swsd', 3), ('lwsd', 4)]
        months = [('May', [5], 0), ('Jun', [6], 1), ('July', [7], 2), ('Aug', [8], 3), ('Sep', [9], 4), ('MJJAS', [5, 6, 7, 8, 9], 5)]

        # Plotting
        fig = plt.figure(figsize=(6, 4), dpi=600)
        gs = plt.GridSpec(5, 7, hspace=0.2, wspace=0.01)

        # Select variable to analyze
        for racmo_var, row in racmo_vars:
            # row = 0
            # racmo_var = 'smb'  # 't2m', 'q2m', 'swsd', 'lwsd'
            # # File path and name for modeled Freya SMB from RACMO2.3 1 km
            racmo_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
            if racmo_var == 'smb':
                racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_1km.csv", parse_dates=['time'], index_col='time')
            else:
                racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
            racmo_day.index = pd.to_datetime(racmo_day.index)
            freq = racmo_day.index.to_series().diff().mode()[0]
            racmo_day = racmo_day.resample(freq).mean()

            if racmo_var == 'smb':
                racmo_var_name = f"{racmo_var}_point_racmo"
            else:
                racmo_var_name = f"{racmo_var}_freya_point_racmo"
            racmo_day = pd.DataFrame(racmo_day[racmo_var_name])

            if racmo_var == 't2m':
                racmo_day = racmo_day - 273.15 # K into deg C
            if racmo_var == 'q2m':
                racmo_day = racmo_day * 1000 # kg/kg into gm/kg

            # Variable composites for high and low sea ice days
            racmo_day_high_ci = pd.DataFrame(racmo_day.loc[ci_high_low_days['high_or_low_sif'] == 'high_sif'])
            racmo_day_high_ci['ci_composite'] = 'High'

            racmo_day_low_ci = pd.DataFrame(racmo_day.loc[ci_high_low_days['high_or_low_sif'] == 'low_sif'])
            racmo_day_low_ci['ci_composite'] = 'Low'

            # Select specific month
            for month_name, month, col in months:
                # month = 5
                # month_name = 'May'
                # col = 0
                racmo_day_high_ci_month = pd.DataFrame(racmo_day_high_ci.loc[racmo_day_high_ci.index.month.isin(month)])
                racmo_day_low_ci_month = pd.DataFrame(racmo_day_low_ci.loc[racmo_day_low_ci.index.month.isin(month)])

                racmo_day_high_low_ci_month = pd.concat([racmo_day_high_ci_month, racmo_day_low_ci_month], ignore_index=True)
                racmo_day_high_low_ci_month['month'] = month_name

                # Wind Speed and other variables
                ax = fig.add_subplot(gs[row, col])
                pos = 0.08
                # Create box plot properties based on wind type
                ax.boxplot(racmo_day_high_ci_month[racmo_var_name], whis=[5, 95], positions=np.array([-pos]), **boxplot_prop_high, **boxplot_prop)
                ax.boxplot(racmo_day_low_ci_month[racmo_var_name], whis=[5, 95], positions=np.array([pos]), **boxplot_prop_low, **boxplot_prop)
                # Create violin plot
                ax_violin = sns.violinplot(x='month', y=racmo_var_name, hue='ci_composite', hue_order=['High', 'Low'], split=True, inner=None,
                                palette={'High':'#08519C', 'Low':'#EF6548'}, data=racmo_day_high_low_ci_month, density_norm='area',
                                ax=ax, linewidth=0.4) #
                # Remove default labels and legend
                plt.setp(ax_violin.collections, zorder=2, label='')
                ax.get_legend().set_visible(False)
                ax.set_ylabel('')
                ax.set_xlabel('')

                text_params = dict(verticalalignment='center', transform=ax.transAxes, clip_on=False, path_effects=[pe.withStroke(linewidth=1.5, foreground='white')])
                if racmo_var == 'smb':
                    ax.text(0.5, 1.22, month_name, size=9, color='k', horizontalalignment='center', fontweight='bold', **text_params)
                    ax.text(0.49, 1.05, f"{len(racmo_day_high_ci_month)}", horizontalalignment='right', size=6, color='#08519C', **text_params)
                    ax.text(0.52, 1.05, f"{len(racmo_day_low_ci_month)}", horizontalalignment='left', size=6, color='#EF6548', **text_params)

                if racmo_var == 'smb':
                    ylim = [-60, 20]
                    yticks = [-60, -40, -20, 0, 20]
                    yticklabels = [-60, -40, -20, 0, 20]
                    var_label = r'SMB' + '\n' + r'(mm w.e. d$^{\mathdefault{-1}}$)'
                    var_name_lag = 'SMB'
                elif racmo_var == 't2m':
                    ylim = [-15, 15]
                    yticks = [-15, -5, 5, 15]
                    yticklabels = [-15, -5, 5, 15]
                    var_label = r'T$_{\mathdefault{2m}}$' + '\n' + r'($^\circ{\mathdefault{C}}$)'
                elif racmo_var == 'q2m':
                    ylim = [0, 5]
                    yticks = [0, 1.25, 2.5, 3.75, 5]
                    yticklabels = [0, 1.25, 2.5, 3.75, 5]
                    var_label = r'q$_{\mathdefault{2m}}$' + '\n' + r'(g kg$^\mathdefault{-1}$)'
                elif racmo_var == 'swsd':
                    ylim = [50, 450]
                    yticks = [50, 150, 250, 350, 450]
                    yticklabels = [50, 150, 250, 350, 450]
                    var_label = r'SWR$_{\mathdefault{in}}$' + '\n' + r'(W m$^{\mathdefault{-2}}$)'
                elif racmo_var == 'lwsd':
                    ylim = [150, 350]
                    yticks = [150, 200, 250, 300, 350]
                    yticklabels = [150, 200, 250, 300, 350]
                    var_label = r'LWR$_{\mathdefault{in}}$' + '\n' + r'(W m$^{\mathdefault{-2}}$)'

                ax.set_ylim(ylim)
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)

                ax.set_xlim([-0.5, 0.5])
                ax.set_xticks([0])
                ax.set_xticklabels([''])

                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                remove_tick_params = dict(left=False, bottom=False, labelleft=False, labelbottom=False)
                ax.tick_params(axis='x', **remove_tick_params)
                if col != 0:
                    ax.spines['left'].set_visible(False)
                    ax.tick_params(axis='y', **remove_tick_params)

                if col == 0:
                    ax.tick_params(axis='y', direction='out', length=6, pad=1, labelsize=7, width=0.7)
                    ax.set_ylabel(var_label, rotation='vertical', size=7, color='k', labelpad=0.1)

                ax.grid(which='major', axis='y', color='gray', linestyle='--', linewidth=0.3, alpha=0.3, zorder=1)

        # ------------------------------------
        # Add idealized violin plot in new axis
        # ------------------------------------
        # Generate skewed data (right-skewed)
        np.random.seed(42)
        # core = np.random.gamma(shape=1.0, scale=1.0, size=950)
        # tail = np.random.gamma(shape=2.0, scale=2.0, size=50)  # long tail
        # example_high = pd.DataFrame(np.concatenate([core, tail]), columns=['example_variable'])
        example_high = pd.DataFrame(np.random.gamma(shape=1.2, scale=2.0, size=1000), columns=['example_variable'])
        example_high['sif'] = 'High'
        example_low = pd.DataFrame(np.random.gamma(shape=2.0, scale=2.0, size=1000), columns=['example_variable'])
        example_low['sif'] = 'Low'
        example_data = pd.concat([example_high, example_low], ignore_index=True)
        example_data['time'] = 'idealized'
        # Plot only one side of the violin plot
        example_data.loc[example_data['sif'] == 'Low', 'example_variable'] = np.nan

        pos = 0.08
        # Create box plot properties based on wind type
        boxplot_prop_ideal = dict(boxprops=dict(facecolor='w', color='k', linewidth=0.5))

        ax_violin_idealized = fig.add_subplot(gs[0, 6])
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

        # Plot arrows and labels
        pos = 0.2
        quantile_fontsize = 5
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.75) + 0.2, '75%', fontsize=quantile_fontsize)
        ax_violin_idealized.text(pos, example_high['example_variable'].mean() - 0.3, 'Mean', color='k', fontsize=quantile_fontsize, fontweight='bold')
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.50) - 0.9, '50%', color='grey', fontsize=quantile_fontsize, fontweight='bold')
        ax_violin_idealized.text(pos, example_high['example_variable'].quantile(0.25) - 1.2, '25%', fontsize=quantile_fontsize)

        ax_violin_idealized.annotate('', xy=(-0.4, -1.5), xytext=(0, -1.5), arrowprops=dict(arrowstyle='<|-|>', lw=0.3, ls='--', color='gray'), clip_on=False)
        ax_violin_idealized.plot([0.005, 0.005], [0.05, 0.22], transform=ax_violin_idealized.transAxes, color='gray', ls='--', lw=0.3)
        ax_violin_idealized.text(0.3, -0.09, 'Frequency', transform=ax_violin_idealized.transAxes, va='center', ha='center', fontsize=6, clip_on=False)

        # ------------------------------------
        # Add idealized legend in new axis
        # ------------------------------------
        ax_legend = fig.add_subplot(gs[2, 6])
        [sp.set_visible(False) for sp in ax_legend.spines.values()]
        ax_legend.axis('off')

        # Dummy handle for legend'High':'#08519C', 'Low':'#EF6548'

        # Dummy handle for legend
        high_sif_handle = Rectangle((0, 0), 1, 1, facecolor='#08519C')
        low_sif_handle = Rectangle((0, 0), 1, 1, facecolor='#EF6548')
        ax_legend.legend(bbox_to_anchor=(0.5, 0.5), loc='center', handles=[high_sif_handle, low_sif_handle], labels=['High SIF', 'Low SIF'], handler_map={Rectangle: HandlerVerticalBoxPlot()},
                                                            handlelength=1.5, handleheight=2.5,
                                                            frameon=False, handletextpad=-2.5, labelspacing=2)

        targets = dict(left=300, right=20, top=130, bottom=30, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_variable_composites_based_on_high_low_sif'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = 'plot_variable_composites_based_on_high_low_sif.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def test_normality_of_variables_based_on_high_low_sif_at_freya():
        # %%
        # Get high and low sea ice days based on 100km sea ice concentration
        ci_high_low_days = _get_high_low_sea_ice_days()

        # Normality test results storage
        racmo_vars = ['smb', 't2m', 'q2m', 'swsd', 'lwsd']
        normality_tests = ['Shapiro-Wilk', 'DAgostino', 'Anderson-Darling']
        sea_ice_types = ['HSIF', 'LSIF']
        miindex = pd.MultiIndex.from_product([normality_tests, sea_ice_types])

        # months = [[5], [6], [7], [8], [9], [5, 6, 7, 8, 9]]  # May to September
        months = [[5, 6, 7, 8, 9]]  # May to September
        for month in months:
            normality_results = pd.DataFrame(index=racmo_vars, columns=miindex)
            for racmo_var in racmo_vars:
                # racmo_var = 'smb'  # 't2m', 'q2m', 'swsd', 'lwsd'
                # File path and name for modeled Freya SMB from RACMO2.3 1 km
                racmo_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
                if racmo_var == 'smb':
                    racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_1km.csv", parse_dates=['time'], index_col='time')
                else:
                    racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
                racmo_day.index = pd.to_datetime(racmo_day.index)
                freq = racmo_day.index.to_series().diff().mode()[0]
                racmo_day = racmo_day.resample(freq).mean()

                if racmo_var == 't2m':
                    racmo_day = racmo_day - 273.15 # K into deg C
                if racmo_var == 'q2m':
                    racmo_day = racmo_day * 1000 # kg/kg into gm/kg

                if racmo_var == 'smb':
                    racmo_var_name = f"{racmo_var}_point_racmo"
                else:
                    racmo_var_name = f"{racmo_var}_freya_point_racmo"
                racmo_day = pd.DataFrame(racmo_day[racmo_var_name])

                # Variable composites for high and low sea ice days
                racmo_day_high_ci = pd.DataFrame(racmo_day.loc[ci_high_low_days['high_or_low_sif'] == 'high_sif'])
                racmo_day_high_ci_month = pd.DataFrame(racmo_day_high_ci.loc[racmo_day_high_ci.index.month.isin(month)])

                racmo_day_low_ci = pd.DataFrame(racmo_day.loc[ci_high_low_days['high_or_low_sif'] == 'low_sif'])
                racmo_day_low_ci_month = pd.DataFrame(racmo_day_low_ci.loc[racmo_day_low_ci.index.month.isin(month)])

                for data, label in [(racmo_day_high_ci_month[racmo_var_name], 'HSIF'), (racmo_day_low_ci_month[racmo_var_name], 'LSIF')]:
                    # Perform Shapiro-Wilk test
                    stat, p = shapiro(data)
                    alpha = 0.05
                    if p > alpha:
                        normality_results.loc[racmo_var, ('Shapiro-Wilk', label)] = 'G'
                    else:
                        normality_results.loc[racmo_var, ('Shapiro-Wilk', label)] = 'NG'

                    # Performing d\'agostino and pearson\'s test')
                    stat, p = normaltest(data)
                    alpha = 0.05
                    if p > alpha:
                        # print(f"{label} looks Gaussian (fail to reject H0)")
                        normality_results.loc[racmo_var, ('DAgostino', label)] = 'G'
                    else:
                        # print(f"{label} does not look Gaussian (reject H0)")
                        normality_results.loc[racmo_var, ('DAgostino', label)] = 'NG'

                    # Performing the anderson-darling test
                    result = anderson(data)
                    # for i in range(len(result.critical_values)):
                    i = 4  # 1% significance level
                    sl, cv = result.significance_level[i], result.critical_values[i]
                    if result.statistic < result.critical_values[i]:
                        # print('%.3f: %.3f, data looks normal (fail to reject H0)' % (sl, cv))
                        normality_results.loc[racmo_var, ('Anderson-Darling', label)] = 'G'
                    else:
                        # print('%.3f: %.3f, data does not look normal (reject H0)' % (sl, cv))
                        normality_results.loc[racmo_var, ('Anderson-Darling', label)] = 'NG'

            # Save normality test results
            normality_results.rename(columns={'DAgostino': fr"D'Agostino's $K^\mathdefault{{2}}$"}, inplace=True)
            normality_results.rename(index={'smb': fr"SMB (mm w.e. d$^\mathdefault{{-1}}$)",
                                        't2m': fr"T$_\mathdefault{{2m}}$ ($^\circ$C)",
                                        'q2m': fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)",
                                        'swsd': fr"SWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)",
                                        'lwsd': fr"LWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)",}, inplace=True)
            flat_cols = [f"{c[1]}" for c in normality_results.columns]

            # Plot normality test results as a table
            fig, ax = plt.subplots(figsize=(2.5, 0.8), dpi=600)  # Adjust size as needed
            ax.axis('off')  # Hide axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Create the percentage of wind direction and wind speed data table
            normality_table = ax.table(cellText=normality_results.values,   # Round numbers to 2 decimals
                                    colLabels=flat_cols,
                                    rowLabels=normality_results.index,
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0, 0, 1, 0.85]) # [x, y, width, height]
            normality_table.auto_set_font_size(False)
            normality_table.set_fontsize(4)

            # --- Set cell sizes and alignment ---
            for (row, col), cell in normality_table.get_celld().items():
                # Width for row labels
                if col == -1:
                    cell.set_width(0.09)
                    cell.set_fontsize(6)
                if row == 0:
                    cell.set_fontsize(6)

                # # Heights
                cell.set_height(0.005 if row == 0 else 0.005)
                # Center text
                cell.get_text().set_horizontalalignment('center')
                cell.get_text().set_verticalalignment('center')
                # Padding
                cell.PAD = 0.01
                # Default borders
                cell.visible_edges = ''
                cell.set_edgecolor('#666666' if row > 0 else 'black')
                cell.set_linewidth(0.2 if row > 0 else 0.8)

            # --- Header styling ---
            for col in range(len(normality_results.columns)):
                cell = normality_table[(0, col)]
                cell.get_text().set_color('k')
                cell.visible_edges = 'B'  # top & bottom line
                cell.set_linewidth(0.4)

            # --- Add horizontal lines for the rest ---
            n_rows = len(normality_results.index) + 1
            n_cols = len(normality_results.columns)
            for row in range(1, n_rows):
                for col in range(n_cols):
                    normality_table[(row, col)].visible_edges = 'B'

            # --- Add group labels above the table ---
            group_labels = list(dict.fromkeys([col[0] for col in normality_results.columns]))
            n_groups = len(group_labels)
            for i, label in enumerate(group_labels):
                x_center = (i + 0.5) / n_groups  # evenly distribute across axes
                ax.text(x_center, 0.89, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
            ax.plot([0.007, 1], [0.965, 0.965], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

            targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'test_normality_of_variables_based_on_high_low_sif_at_freya'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            if month == [5]:
                month_str = 'May'
            elif month == [6]:
                month_str = 'Jun'
            elif month == [7]:
                month_str = 'Jul'
            elif month == [8]:
                month_str = 'Aug'
            elif month == [9]:
                month_str = 'Sep'
            elif month == [5, 6, 7, 8, 9]:
                month_str = 'MJJAS'
            fig_name = f"test_normality_of_variables_based_on_high_low_sif_at_freya_{month_str}.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

    # %%
    def test_significant_difference_in_variable_based_on_high_low_sif_at_freya_using_change_points():
        # %%
        # Read the change points detected with kernel change detection
        change_points = pd.read_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/change_points_with_kernel_change_detection_in_sif.csv', index_col=[0, 1, 2])
        change_points.index.names = ['penalty', 'min_size', 'bkp_id']
        # Select penalty=2.5 and min_size=7 change points for stationarity test, as they are more consistent across different months compared to the change points detected with penalty=3 and min_size=7s.
        penalty = 2.5
        min_size = 7
        idx = pd.IndexSlice
        change_points = change_points.loc[idx[penalty, min_size, :], :]

        racmo_vars = ['smb', 't2m', 'q2m', 'swsd', 'lwsd']
        for month_name, month in (('May', [5]), ('June', [6]), ('July', [7]), ('August', [8]), ('September', [9]), ('MJJAS', [5, 6, 7, 8, 9]),):
            # Use calculated change points to split the time series into regimes and test stationarity
            change_points_month = pd.DataFrame(change_points.loc[(penalty, min_size), month_name])
            change_points_month.dropna(inplace=True)
            change_points_month['year'] = (pd.to_datetime(change_points_month[month_name], errors='coerce').dt.year)
            num_regimes = len(change_points_month) + 1

            # Regime 1 is the whole period before splitting, which is used to compare with the calculated regimes.
            # If no change point is detected, the whole period will be considered as one regime (regime 1).
            sif_regimes1 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=1990, period='whole', change_point_2=2020)
            if num_regimes == 2:
                # Split the time series into two regimes based on the change point
                sif_regime2 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='before')
                sif_regime3 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='after')
                # Create a list of calculated regimes for stationarity test
                sif_regimes = [sif_regimes1, sif_regime2, sif_regime3]
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3']
            elif num_regimes == 3:
                # Split the time series into three regimes based on the change points
                sif_regime2 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='before')
                sif_regime3 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], change_point_2=change_points_month.loc[2, 'year'], period='between')
                sif_regime4 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[2, 'year'], period='after')
                # Create a list of calculated regimes for stationarity test
                sif_regimes = [sif_regimes1, sif_regime2, sif_regime3, sif_regime4]
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3', 'sif_regime_4']

            # Test stationarity for each regime using Augmented Dickey-Fuller test
            for sif_regime, sif_regime_name in zip(sif_regimes, sif_regimes_names):
                # ind = 1
                # sif_regime = sif_regimes[ind]
                # sif_regime_name = sif_regimes_names[ind]
                # Statistically significant difference test results storage
                significance_diff_results = pd.DataFrame(index=racmo_vars, columns=['Median\n(HSI-LSI)', "Cliff's\nδ", 'P\n(HSI>LSI) %', 'U–test\np', 'Perm\np'])
                # Store observed difference and p-value from permutation test
                obs_diff_perm = pd.DataFrame(index=racmo_vars, columns=['Observed_Median_Difference', 'P_value_Perm_test)'])
                for racmo_var in racmo_vars:
                    # racmo_var = 'smb'  # 't2m', 'q2m', 'swsd', 'lwsd'
                    # File path and name for modeled Freya SMB from RACMO2.3 1 km
                    racmo_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_freya_glacier')
                    if racmo_var == 'smb':
                        racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_1km.csv", parse_dates=['time'], index_col='time')
                    else:
                        racmo_day = pd.read_csv(f"{racmo_path}/{racmo_var}_masked_freya_5km.csv", parse_dates=['time'], index_col='time')
                    racmo_day.index = pd.to_datetime(racmo_day.index)
                    freq = racmo_day.index.to_series().diff().mode()[0]
                    racmo_day = racmo_day.resample(freq).mean()

                    if racmo_var == 't2m':
                        racmo_day = racmo_day - 273.15 # K into deg C
                    if racmo_var == 'q2m':
                        racmo_day = racmo_day * 1000 # kg/kg into gm/kg

                    if racmo_var == 'smb':
                        racmo_var_name = f"{racmo_var}_point_racmo"
                    else:
                        racmo_var_name = f"{racmo_var}_freya_point_racmo"
                    racmo_day = pd.DataFrame(racmo_day[racmo_var_name])

                    # Variable composites for high and low sea ice days
                    racmo_day_ci = pd.DataFrame(racmo_day.loc[racmo_day.index.isin(sif_regime.index)])
                    # Variables during High Sea Ice and Low Sea Ice days in the regime
                    racmo_day_ci_high = pd.DataFrame(racmo_day_ci.loc[sif_regime['high_or_low_sif'] == 'high_sif'])
                    racmo_day_ci_low = pd.DataFrame(racmo_day_ci.loc[sif_regime['high_or_low_sif'] == 'low_sif'])

                    # Store the mannwhitneyu test p-value
                    u_stat, p_mw = mannwhitneyu(racmo_day_ci_high[racmo_var_name], racmo_day_ci_low[racmo_var_name], alternative='two-sided')
                    if p_mw <= 0.01:
                        significance_diff_results.loc[racmo_var, 'U–test\np'] = '<0.01'
                    elif (p_mw > 0.01) & (p_mw <= 0.05):
                        significance_diff_results.loc[racmo_var, 'U–test\np'] = '<0.05'
                    elif (p_mw > 0.05) & (p_mw <= 0.1):
                        significance_diff_results.loc[racmo_var, 'U–test\np'] = '<0.1'
                    else:
                        significance_diff_results.loc[racmo_var, 'U–test\np'] = '–'

                    # Store the cliff delta
                    delta, p_x_greater_y, p_y_greater_x = _cliffs_delta_fast(racmo_day_ci_high[racmo_var_name], racmo_day_ci_low[racmo_var_name])
                    if np.absolute(np.round(delta, 1)) < 0.09:
                        if p_mw <= 0.01:
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"0\n({p_x_greater_y}) $^{{**}}$"
                        elif (p_mw > 0.01) & (p_mw <= 0.05):
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"0\n({p_x_greater_y}) $^{{*}}$"
                        else:
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"0\n({p_x_greater_y})"

                    else:
                        if p_mw <= 0.01:
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"{delta:.1f}\n({p_x_greater_y}) $^{{**}}$"
                        elif (p_mw > 0.01) & (p_mw <= 0.05):
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"{delta:.1f}\n({p_x_greater_y}) $^{{*}}$"
                        else:
                            significance_diff_results.loc[racmo_var, "Cliff's\nδ"] = f"{delta:.1f}\n({p_x_greater_y})"


                    significance_diff_results.loc[racmo_var, 'P\n(HSI>LSI) %'] = f"{p_x_greater_y}"

                    # Store permutation test for median difference
                    save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
                    obs_diff, p_perm, perm_dist = _permutation_test(racmo_day_ci_high[racmo_var_name].values, racmo_day_ci_low[racmo_var_name].values, n_perm=10000)
                    if month == [5]:
                        month_str = 'May'
                    elif month == [6]:
                        month_str = 'Jun'
                    elif month == [7]:
                        month_str = 'Jul'
                    elif month == [8]:
                        month_str = 'Aug'
                    elif month == [9]:
                        month_str = 'Sep'
                    elif month == [5, 6, 7, 8, 9]:
                        month_str = 'MJJAS'
                    # Save permutation distribution
                    np.save(f"{save_path}/perm_dist_hsfi_vs_lsfi_{racmo_var_name}_{month_name}_{sif_regime_name}.npy", perm_dist)

                    # Store results
                    obs_diff_perm.loc[racmo_var, 'Observed_Median_Difference'] = obs_diff
                    obs_diff_perm.loc[racmo_var, 'P_value_Perm_test)'] = p_perm

                    if p_perm <= 0.01:
                        significance_diff_results.loc[racmo_var, 'Perm\np'] = '<0.01'
                        significance_diff_results.loc[racmo_var, 'Median\n(HSI-LSI)'] = f"{np.round(obs_diff, 1)}$^{{**}}$"
                    elif (p_perm > 0.01) & (p_perm <= 0.05):
                        significance_diff_results.loc[racmo_var, 'Perm\np'] = '<0.05'
                        significance_diff_results.loc[racmo_var, 'Median\n(HSI-LSI)'] = f"{np.round(obs_diff, 1)}$^{{*}}$"
                    # elif (p_perm > 0.05) & (p_perm <= 0.1):
                    #     significance_diff_results.loc[racmo_var, 'Perm\np'] = '<0.1'
                    #     significance_diff_results.loc[racmo_var, 'Median\n(HSI-LSI)'] = f"{np.round(obs_diff, 1)}$^{{*}}$"
                    else:
                        significance_diff_results.loc[racmo_var, 'Perm\np'] = '–'
                        significance_diff_results.loc[racmo_var, 'Median\n(HSI-LSI)'] = '-'

                significance_diff_results.rename(index={'smb': fr"SMB (mm w.e. d$^\mathdefault{{-1}}$)",
                                            't2m': fr"T$_\mathdefault{{2m}}$ ($^\circ$C)",
                                            'q2m': fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)",
                                            'swsd': fr"SWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)",
                                            'lwsd': fr"LWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)",}, inplace=True)

                # Save significance test results
                significance_diff_results.to_csv(f"{save_path}/significance_diff_results_hsfi_vs_lsfi_{month_name}_{sif_regime_name}.csv")
                obs_diff_perm.to_csv(f"{save_path}/obs_diff_perm_hsfi_vs_lsfi_{month_name}_{sif_regime_name}.csv")
    # %%

    def plot_significant_difference_in_variable_based_on_high_low_sif_at_freya_using_change_points():
        # %%
        significance_stats = [fr'$\Delta$Med', 'δ\n(P%)']

        months_names = ['May', 'June', 'July', 'August', 'September', 'MJJAS']  # May to September
        for month_name in months_names:
            # month_name = 'MJJAS'
            if month_name in ['September', 'MJJAS']: # September and MJJAS has 4 regimes
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3', 'sif_regime_4']
            else:
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3']
            miindex = pd.MultiIndex.from_product([sif_regimes_names, significance_stats])
            var_list =[fr"SMB (mm w.e. d$^\mathdefault{{-1}}$)", fr"T$_\mathdefault{{2m}}$ ($^\circ$C)", fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)" , fr"SWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)", fr"LWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)"]
            significance_diff_results_all  = pd.DataFrame(index=var_list, columns=miindex)

            for sif_regime_name in sif_regimes_names:
                # Load significance test results
                save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
                significance_diff_results = pd.read_csv(f'{save_path}/significance_diff_results_hsfi_vs_lsfi_{month_name}_{sif_regime_name}.csv', index_col=0)

                # Combine results into a single dataframe
                significance_diff_results_all.loc[:, (sif_regime_name, fr'$\Delta$Med')] = significance_diff_results.loc[:, "Median\n(HSI-LSI)"].values
                significance_diff_results_all.loc[:, (sif_regime_name, 'δ\n(P%)')] = significance_diff_results.loc[:, "Cliff's\nδ"].values

            # Plot significance test results as a table
            fig, ax = plt.subplots(figsize=(3, 0.95), dpi=600)  # Adjust size as needed
            ax.axis('off')  # Hide axes
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Flatten columns for multicolumn plotting
            flat_cols = [f"{c[1]}" for c in significance_diff_results_all.columns]
            # Create the percentage of wind direction and wind speed data table
            significance_diff_table = ax.table(cellText=significance_diff_results_all.values,   # Round numbers to 2 decimals
                                    colLabels=flat_cols,
                                    rowLabels=significance_diff_results_all.index,
                                    cellLoc='center',
                                    loc='center',
                                    bbox=[0, 0, 1, 0.94]) # [x, y, width, height]
            significance_diff_table.auto_set_font_size(False)
            significance_diff_table.set_fontsize(4)

            # --- Set cell sizes and alignment ---
            for (row, col), cell in significance_diff_table.get_celld().items():
                # Width for row labels
                if col == -1:
                    cell.set_width(0.09)
                    cell.set_fontsize(5)
                if row == 0:
                    cell.set_fontsize(5)
                # Heights
                cell.set_height(0.009 if row == 0 else 0.005)
                # Center text
                cell.get_text().set_horizontalalignment('center')
                cell.get_text().set_verticalalignment('center')
                # Padding
                # cell.PAD = 0.01
                # Default borders
                cell.visible_edges = ''
                cell.set_edgecolor('#666666' if row > 0 else 'black')
                cell.set_linewidth(0.2 if row > 0 else 0.8)

            # --- Header styling ---
            for col in range(len(significance_diff_results_all.columns)):
                cell = significance_diff_table[(0, col)]
                cell.get_text().set_color('k')
                cell.visible_edges = 'B'  # top & bottom line
                cell.set_linewidth(0.4)

            # --- Add horizontal lines for the rest ---
            n_rows = len(significance_diff_results_all.index) + 1
            n_cols = len(significance_diff_results_all.columns)
            for row in range(1, n_rows):
                for col in range(n_cols):
                    significance_diff_table[(row, col)].visible_edges = 'B'

            # --- Add group labels above the table ---
            group_labels = list(dict.fromkeys([col[0] for col in significance_diff_results_all.columns]))
            n_groups = len(group_labels)
            for i, label in enumerate(group_labels):
                x_center = (i + 0.5) / n_groups  # evenly distribute across axes
                ax.text(x_center, 0.93, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
            ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line
            ax.text(-0.1, 0.93, month_name, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',

            targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
            fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
            fp.adjust_borders(fig, targets)

            logger.info(f"Saving plot of 'plot_significant_difference_in_variable_based_on_high_low_sif_at_freya_using_change_points'...")
            base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
            fig_name = f"plot_significant_difference_in_variable_based_on_high_low_sif_at_freya_using_change_points_{month_name}.png"
            fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%
    def plot_significant_difference_in_variable_based_on_high_low_sif_at_freya():
        # %%
        months_names = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'MJJAS']  # May to September
        significance_stats = [fr'$\Delta$Med', 'δ\n(P%)']
        miindex = pd.MultiIndex.from_product([months_names, significance_stats])

        var_list =[fr"SMB (mm w.e. d$^\mathdefault{{-1}}$)", fr"T$_\mathdefault{{2m}}$ ($^\circ$C)", fr"q$_\mathdefault{{2m}}$ (g kg$^\mathdefault{{-1}}$)" , fr"SWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)", fr"LWR$_\mathdefault{{in}}$ (W m$^\mathdefault{{-2}}$)"]
        significance_diff_results_all  = pd.DataFrame(index=var_list, columns=miindex)

        for month_name in months_names:
            # Load significance test results
            save_path = '/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result'
            significance_diff_results = pd.read_csv(f'{save_path}/significance_diff_results_hsfi_vs_lsfi_{month_name}.csv', index_col=0)

            # Combine results into a single dataframe
            significance_diff_results_all.loc[:, (month_name, fr'$\Delta$Med')] = significance_diff_results.loc[:, "Median\n(KAT-NKAT)"].values
            significance_diff_results_all.loc[:, (month_name, 'δ\n(P%)')] = significance_diff_results.loc[:, "Cliff's\nδ"].values

        # Plot significance test results as a table
        fig, ax = plt.subplots(figsize=(3, 0.95), dpi=600)  # Adjust size as needed
        ax.axis('off')  # Hide axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Flatten columns for multicolumn plotting
        flat_cols = [f"{c[1]}" for c in significance_diff_results_all.columns]
        # Create the percentage of wind direction and wind speed data table
        significance_diff_table = ax.table(cellText=significance_diff_results_all.values,   # Round numbers to 2 decimals
                                colLabels=flat_cols,
                                rowLabels=significance_diff_results_all.index,
                                cellLoc='center',
                                loc='center',
                                bbox=[0, 0, 1, 0.94]) # [x, y, width, height]
        significance_diff_table.auto_set_font_size(False)
        significance_diff_table.set_fontsize(4)

        # --- Set cell sizes and alignment ---
        for (row, col), cell in significance_diff_table.get_celld().items():
            # Width for row labels
            if col == -1:
                cell.set_width(0.09)
                cell.set_fontsize(5)
            if row == 0:
                cell.set_fontsize(5)
            # Heights
            cell.set_height(0.009 if row == 0 else 0.005)
            # Center text
            cell.get_text().set_horizontalalignment('center')
            cell.get_text().set_verticalalignment('center')
            # Padding
            # cell.PAD = 0.01
            # Default borders
            cell.visible_edges = ''
            cell.set_edgecolor('#666666' if row > 0 else 'black')
            cell.set_linewidth(0.2 if row > 0 else 0.8)

        # --- Header styling ---
        for col in range(len(significance_diff_results_all.columns)):
            cell = significance_diff_table[(0, col)]
            cell.get_text().set_color('k')
            cell.visible_edges = 'B'  # top & bottom line
            cell.set_linewidth(0.4)

        # --- Add horizontal lines for the rest ---
        n_rows = len(significance_diff_results_all.index) + 1
        n_cols = len(significance_diff_results_all.columns)
        for row in range(1, n_rows):
            for col in range(n_cols):
                significance_diff_table[(row, col)].visible_edges = 'B'

        # --- Add group labels above the table ---
        group_labels = list(dict.fromkeys([col[0] for col in significance_diff_results_all.columns]))
        n_groups = len(group_labels)
        for i, label in enumerate(group_labels):
            x_center = (i + 0.5) / n_groups  # evenly distribute across axes
            ax.text(x_center, 0.93, label, ha='center', va='center', fontsize=6, transform=ax.transAxes) #fontweight='bold',
        ax.plot([0.007, 1], [0.99, 0.99], color='k', transform=ax.transAxes, linewidth=0.4, clip_on=False)  # top line

        targets = dict(left=263, right=10, top=0, bottom=10, hspace=0, wspace=0)
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        logger.info(f"Saving plot of 'plot_significant_difference_in_variable_based_on_high_low_sif_at_freya'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/FG1_data_analysis-')
        fig_name = f"plot_significant_difference_in_variable_based_on_high_low_sif_at_freya.png"
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)
    # %%
    def change_points_with_kernel_change_detection_in_sif():
        """Detecting Change Points With Kernel Change Detection.
        Reference: https://forecastegy.com/posts/change-point-detection-time-series-python /

        """
        # %%
        # Read the masked 5km racmo variables within 100km sea ice concentration
        masked_5km_variables_100km_sif_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        masked_5km_variables_100km_sif = pd.read_csv(f"{masked_5km_variables_100km_sif_path}/masked_5km_variables_within_100km_sea_ice.csv", parse_dates=['time'], index_col='time')
        masked_5km_variables_100km_sif.index = pd.to_datetime(masked_5km_variables_100km_sif.index)
        # Resample to uniform frequency
        freq = masked_5km_variables_100km_sif.index.to_series().diff().mode()[0]
        masked_5km_variables_100km_sif = masked_5km_variables_100km_sif.resample(freq).mean()
        sif_in_100km_sif = pd.DataFrame(masked_5km_variables_100km_sif['ci'])
        sif_in_100km_sif_month_median = sif_in_100km_sif.resample('ME').median()

        # change_points = pd.DataFrame(index=['May', 'June', 'July', 'August', 'September', 'MJJAS'], columns=['daily_average'])
        # Define MultiIndex for change points DataFrame
        penalties = [2.5, 3]
        break_points = [1, 2, 3,]  # We are interested in the first 3 change points
        years = [5, 6, 7, 8, 9] # A regime must last at least 5 years to be considered physically meaningful.
        # index = pd.MultiIndex.from_product([penalties, break_points], names=['penalty', 'bkp_id'])
        index = pd.MultiIndex.from_product([penalties, years, break_points], names=['penalty', 'year', 'bkp_id'])

        months = ['May', 'June', 'July', 'August', 'September', 'MJJAS']
        kernels = ['linear']
        # columns = pd.MultiIndex.from_product([months, kernels], names=['month', 'kernel'])
        # columns = pd.MultiIndex.from_product([months, kernels], names=['month', 'kernel'])

        change_points = pd.DataFrame(columns=months, index=index)
        for month_name, month in (('May', [5]), ('June', [6]), ('July', [7]), ('August', [8]), ('September', [9]), ('MJJAS', [5, 6, 7, 8, 9]),):
        # for month_name, month in (('MJJAS', [5, 6, 7, 8, 9]),):
            # for period in ['daily_average', 'monthly_average']:
            sif_in_100km_sif_month_median_single_year = sif_in_100km_sif[sif_in_100km_sif.index.month.isin(month)]
            signal = StandardScaler().fit_transform(sif_in_100km_sif_month_median_single_year.values.reshape(-1, 1)).flatten()

            for i, kernel in enumerate(kernels):
                # fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=300)
                for pen in penalties:
                    for year in years:
                        # year = 6 # A regime must last at least 6 years to be considered physically meaningful.
                        if month_name in ['May', 'June', 'July', 'August', 'September']:
                            if month_name in ['May', 'July', 'August']:
                                min_size = 31 * year # A regime must last at least 3 months (31 * 5) to be considered physically meaningful.
                            elif month_name in ['June', 'September']:
                                min_size = 30 * year # A regime must last at least 5 years (30 * 5) to be considered physically meaningful.
                        else:
                            # min_size = 153 * 5  # 5 years (31+30+31+31+30 * 5) (MJJAS)
                            min_size = 153 * year  # 5 years (31+30+31+31+30 * 5) (MJJAS)
                            # algo = rpt.KernelCPD(kernel=kernel, min_size=int(0.1 * len(signal)))
                        algo = rpt.KernelCPD(kernel=kernel, min_size=min_size)
                        algo.fit(signal)

                        result = algo.predict(pen=pen * np.log(len(signal)))
                        bkp_dates = [sif_in_100km_sif_month_median_single_year.index[bkp - 1].strftime("%Y-%m") for bkp in result[:-1]]

                        for i, date in enumerate(bkp_dates):
                            # change_points.loc[(pen, break_points[i]), (month_name, kernel)] = date
                            change_points.loc[(pen, year, break_points[i]), month_name] = date
        change_points.to_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/change_points_with_kernel_change_detection_in_sif.csv')

        # %%
        #             ax.plot(signal)
        #             ax.set_xticks(ticks=np.arange(0, len(sif_in_100km_sif_month_median_single_year.index), 1),
        #                             labels=[d.strftime('%Y') for d in sif_in_100km_sif_month_median_single_year.index], rotation=45)
        #             ax.text(0.05, 0.9, f"{month_name}", fontsize=12, transform=ax.transAxes)
        #             print(f"{month_name}\nDetected change point at index:")

        #             for bkp in result[:-1]:
        #                 print(f"{bkp} using {kernel} kernel")
        #                 ax.axvline(x=bkp, color='k', linestyle='--')
        #                 change_points.loc[month_name, 'daily_average'] = sif_in_100km_sif_month_median_single_year.index[bkp-1].strftime('%Y-%m')
        #                 print(sif_in_100km_sif_month_median_single_year.index[bkp-1].strftime('%Y-%m'))
        #                 if len(result[:-1]) == 1:
        #                     ax.text(0.05, 0.8, f"{sif_in_100km_sif_month_median_single_year.index[bkp-1].strftime('%Y-%m')}", fontsize=12, transform=ax.transAxes)
        #                 elif len(result[:-1]) > 1:
        #                     y = 0.7 + 0.05 * (len(result[:-1]) - 1 - result[:-1].index(bkp))
        #                     ax.text(0.05, y, f"{sif_in_100km_sif_month_median_single_year.index[bkp-1].strftime('%Y-%m')}", fontsize=12, transform=ax.transAxes)

        #             print('----------------------------')
        #         fig.tight_layout()
        # change_points.to_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/change_points_with_kernel_change_detection_in_sif.csv')
            # %%
    def stationarity_test_of_sif():
        # %%
        # Read the masked 5km racmo variables within 100km sea ice concentration
        masked_5km_variables_100km_sif_path = Path('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/extracted_racmo_var/from_ci')
        masked_5km_variables_100km_sif = pd.read_csv(f"{masked_5km_variables_100km_sif_path}/masked_5km_variables_within_100km_sea_ice.csv", parse_dates=['time'], index_col='time')
        masked_5km_variables_100km_sif.index = pd.to_datetime(masked_5km_variables_100km_sif.index)
        # Resample to uniform frequency
        freq = masked_5km_variables_100km_sif.index.to_series().diff().mode()[0]
        masked_5km_variables_100km_sif = masked_5km_variables_100km_sif.resample(freq).mean()
        sif_in_100km_sif = pd.DataFrame(masked_5km_variables_100km_sif['ci'])

        # Read the change points detected with kernel change detection
        change_points = pd.read_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/change_points_with_kernel_change_detection_in_sif.csv', index_col=[0, 1, 2])
        change_points.index.names = ['penalty', 'min_size', 'bkp_id']
        # Select penalty=2.5 and min_size=7 change points for stationarity test, as they are more consistent across different months compared to the change points detected with penalty=3 and min_size=7s.
        penalty = 2.5
        min_size = 7

        months = ['May', 'June', 'July', 'August', 'September', 'MJJAS']
        # Change points detected with kernel change detection
        calculated_regimes = ['calculated_regime_1', 'calculated_regime_2', 'calculated_regime_3', 'calculated_regime_4']
        calculated_regimes_stationary_results = pd.DataFrame(columns=pd.MultiIndex.from_product([months, calculated_regimes]), index=['ADF', 'KPSS', 'PP'])

        # # Predefined change points
        # fixed_change_points_month = pd.to_datetime(['2001', '2007']) # For June, the change point detected with kernel change detection is not clear, so I set it to June 2007 based on visual inspection of the time series and considering the change points detected in other months.
        # fixed_regimes = ['all_period', 'fixed_regime_2', 'fixed_regime_3', 'fixed_regime_4', 'fixed_regime_5', 'fixed_regime_6']
        # fixed_regimes_stationary_results = pd.DataFrame(columns=pd.MultiIndex.from_product([months, fixed_regimes]), index=['ADF', 'KPSS', 'PP'])

        for month_name, month in (('May', [5]), ('June', [6]), ('July', [7]), ('August', [8]), ('September', [9]), ('MJJAS', [5, 6, 7, 8, 9]),):
            sif_in_100km_sif_month = sif_in_100km_sif[sif_in_100km_sif.index.month.isin(month)]

            # Use calculated change points to split the time series into regimes and test stationarity
            # Use penalty=2.5 and min_size=7 change points for stationarity test, as they are more consistent across different months
            # compared to the change points detected with penalty=3 and min_size=7s
            change_points_month = pd.DataFrame(change_points.loc[(penalty, min_size), month_name])
            change_points_month.dropna(inplace=True)
            change_points_month['year'] = (pd.to_datetime(change_points_month[month_name], errors='coerce').dt.year)
            num_regimes = len(change_points_month) + 1

            # Regime 1 is the whole period before splitting, which is used to compare with the calculated regimes.
            # If no change point is detected, the whole period will be considered as one regime (regime 1).
            calculated_regimes1 = sif_in_100km_sif_month.copy()
            if num_regimes == 2:
                # Split the time series into two regimes based on the change point
                calculated_regime2 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index.year < change_points_month.loc[1, 'year'])]
                calculated_regime3 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index.year >= change_points_month.loc[1, 'year'])]
                # Create a list of calculated regimes for stationarity test
                calculated_regimes = [calculated_regimes1, calculated_regime2, calculated_regime3]
                calculated_regimes_names = ['calculated_regime_1', 'calculated_regime_2', 'calculated_regime_3']
            elif num_regimes == 3:
                # Split the time series into three regimes based on the change points
                calculated_regime2 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index.year < change_points_month.loc[1, 'year'])]
                calculated_regime3 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index.year >= change_points_month.loc[1, 'year']) & (sif_in_100km_sif_month.index.year < change_points_month.loc[2, 'year'])]
                calculated_regime4 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index.year >= change_points_month.loc[2, 'year'])]
                # Create a list of calculated regimes for stationarity test
                calculated_regimes = [calculated_regimes1, calculated_regime2, calculated_regime3, calculated_regime4]
                calculated_regimes_names = ['calculated_regime_1', 'calculated_regime_2', 'calculated_regime_3', 'calculated_regime_4']

            # Test stationarity for each regime using Augmented Dickey-Fuller test
            for regime, regime_name in zip(calculated_regimes, calculated_regimes_names):
                # Augmented Dickey — Fuller
                result = adfuller(regime['ci'], regression='ct', autolag='AIC')
                p_value = result[1]
                calculated_regimes_stationary_results.loc['ADF', (month_name, regime_name)] = 'Stationary' if p_value < 0.05 else 'Non-stationary'

                # Kwiatkowski–Phillips–Schmidt–Shin
                result = kpss(regime['ci'], regression='ct', nlags='auto')
                p_value = result[1]
                calculated_regimes_stationary_results.loc['KPSS', (month_name, regime_name)] = 'Stationary' if p_value > 0.05 else 'Non-stationary'

                # Phillips–Perron
                result = PhillipsPerron(regime['ci'], trend='ct')
                p_value = result.pvalue
                calculated_regimes_stationary_results.loc['PP', (month_name, regime_name)] = 'Stationary' if p_value < 0.05 else 'Non-stationary'

        calculated_regimes_stationary_results.to_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/stationarity_test_of_sif.csv')
            # # Split the time series into fixed regimes based on the change point
            # all_period_regime = sif_in_100km_sif_month.copy()
            # fixed_regime2 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index < fixed_change_points_month[0])]
            # fixed_regime3 = sif_in_100km_sif_month[(sif_in_100km_sif_month.index >= fixed_change_points_month[0]) & (sif_in_100km_sif_month.index < fixed_change_points_month[1])]
            # fixed_regime4 = sif_in_100km_sif_month[sif_in_100km_sif_month.index >= fixed_change_points_month[1]]
            # fixed_regime5 = sif_in_100km_sif_month[sif_in_100km_sif_month.index < fixed_change_points_month[1]]
            # fixed_regime6 = sif_in_100km_sif_month[sif_in_100km_sif_month.index >= fixed_change_points_month[1]]

            # # Test stationarity for each regime using Augmented Dickey-Fuller test
            # for regime, regime_name in zip([all_period_regime, fixed_regime2, fixed_regime3, fixed_regime4, fixed_regime5, fixed_regime6], fixed_regimes):
            #     # Augmented Dickey — Fuller
            #     result = adfuller(regime['ci'], regression='ct', autolag='AIC')
            #     p_value = result[1]
            #     fixed_regimes_stationary_results.loc['ADF', (month_name, regime_name)] = 'Stationary' if p_value < 0.05 else 'Non-stationary'

            #     # Kwiatkowski–Phillips–Schmidt–Shin
            #     result = kpss(regime['ci'], regression='ct', nlags='auto')
            #     p_value = result[1]
            #     fixed_regimes_stationary_results.loc['KPSS', (month_name, regime_name)] = 'Stationary' if p_value > 0.05 else 'Non-stationary'

            #     # Phillips–Perron
            #     result = PhillipsPerron(regime['ci'], trend='ct')
            #     p_value = result.pvalue
            #     fixed_regimes_stationary_results.loc['PP', (month_name, regime_name)] = 'Stationary' if p_value < 0.05 else 'Non-stationary'
        # %%

    def high_low_sif_distribution():
        # %%
        # Read the change points detected with kernel change detection
        change_points = pd.read_csv('/nas/data/processed/people/sos/third_paper_data/freya_aws_processed/statistic_result/change_points_with_kernel_change_detection_in_sif.csv', index_col=[0, 1, 2])
        change_points.index.names = ['penalty', 'min_size', 'bkp_id']
        # Select penalty=2.5 and min_size=7 change points for stationarity test, as they are more consistent across different months compared to the change points detected with penalty=3 and min_size=7s.
        penalty = 2.5
        min_size = 7
        idx = pd.IndexSlice
        change_points = change_points.loc[idx[penalty, min_size, :], :]

        # Define colormap
        colors = [(255, 0, 0), (255, 255, 255), (65, 105, 225)]
        cmap = fp.make_cmap(colors, bit=True)
        bounds = np.array([-1, -0.0001, 0.0001, 1])
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        cbtick = [-1, 0,  1]
        cbticklabel = [-1, 0, 1]
        tick_params = dict(axis='both', direction='inout', length=2, pad=1, labelsize=6, width=0.5)
        months = (('May', [5]), ('June', [6]), ('July', [7]), ('August', [8]), ('September', [9]), ('MJJAS', [5, 6, 7, 8, 9]),)
            # %%
        for month_name, month in months:
            # %%
            month_name, month = months[5]
            # Use calculated change points to split the time series into regimes and test stationarity
            change_points_month = pd.DataFrame(change_points.loc[(penalty, min_size), month_name])
            change_points_month.dropna(inplace=True)
            change_points_month['year'] = (pd.to_datetime(change_points_month[month_name], errors='coerce').dt.year)
            num_regimes = len(change_points_month) + 1

            # Regime 1 is the whole period before splitting, which is used to compare with the calculated regimes.
            # If no change point is detected, the whole period will be considered as one regime (regime 1).
            sif_regimes1 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=1990, period='whole', change_point_2=2020)
            if num_regimes == 2:
                # Split the time series into two regimes based on the change point
                sif_regime2 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='before')
                sif_regime3 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='after')
                # Create a list of calculated regimes for stationarity test
                sif_regimes = [sif_regimes1, sif_regime2, sif_regime3]
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3']
            elif num_regimes == 3:
                # Split the time series into three regimes based on the change points
                sif_regime2 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], period='before')
                sif_regime3 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[1, 'year'], change_point_2=change_points_month.loc[2, 'year'], period='between')
                sif_regime4 = _get_high_low_sea_ice_days_based_on_doy(month_name=month_name, change_point_1=change_points_month.loc[2, 'year'], period='after')
                # Create a list of calculated regimes for stationarity test
                sif_regimes = [sif_regimes1, sif_regime2, sif_regime3, sif_regime4]
                sif_regimes_names = ['sif_regime_1', 'sif_regime_2', 'sif_regime_3', 'sif_regime_4']

            # %%
            # Test stationarity for each regime using Augmented Dickey-Fuller test
            for sif_regime, sif_regime_name in zip(sif_regimes, sif_regimes_names):
            # %%
                sif_regime = sif_regimes[0]
                sif_regime['year'] = sif_regime.index.year
                sif_regime['doy'] = sif_regime.index.day_of_year

                label_map = {'high_sif': 1, 'low_sif': -1}
                sif_regime['value'] = sif_regime['high_or_low_sif'].map(label_map).fillna(0).astype(int)

                # _, days_in_month = calendar.monthrange(2020, month[0])
                doy = np.sort(sif_regime['doy'].unique())
                years = np.sort(sif_regime['year'].unique())

                # Create an empty dataframe with days as index and years as columns
                # sif_regime_binary = pd.DataFrame(0, index=days, columns=years)
                sif_regime_binary = pd.DataFrame(0, index=doy, columns=years)
                sif_regime_binary.index.name = 'doy'
                sif_regime_binary.columns.name = 'year'

                for _, row in sif_regime.iterrows():
                    sif_regime_binary.loc[row['doy'], row['year']] = row['value']

                fig, ax = plt.subplots(1, figsize=(2, 2), dpi=600)
                im = ax.imshow(sif_regime_binary, aspect='auto', cmap=cmap, norm=norm, origin='lower')
                # %%

if __name__ == '__main__':
    # analysis = SeaIceAndSMBAnalysis()
    # SeaIceAndSMBAnalysis.calculate_greenland_sea_sif_phenology()
    SeaIceAndSMBAnalysis.calculate_greenland_sea_sif_phenology_trend()
