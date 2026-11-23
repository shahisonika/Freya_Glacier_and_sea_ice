# !/usr/bin/env python3
"""
Freya Glacier Analysis Level 0 (FG0) - Plot Freya Glacier preprocessed dataset.

Author: Sonika Shahi
Updated: 24 Aug 2025

Environment: xeofs_env (conda activate xeofs_env on walrus server)
"""
# %%
import os
import sys
import xarray as xr
from pathlib import Path
# from pyproj import Transformer, CRS
from matplotlib.collections import LineCollection
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Rectangle, Patch, Polygon
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
plt.rc('font',family='Helvetica')
from matplotlib.legend_handler import HandlerPatch, HandlerTuple, HandlerBase
from shapely.geometry import box
from shapely.ops import unary_union

import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utility helpers
# -------------------------
os.chdir('/nas/home/sos/third_paper_work/codes/freyagl_analysis')
# from FG_modules import read_preprocessed_station_data, classify_wind_direction_in_degrees_into_cardinal_direction

# Import modules from different directory
# Reference: https://www.geeksforgeeks.org/python-import-module-from-different-directory/
module_path = "/nas/home/sos/third_paper_work/codes/add_modules/"
if os.path.exists(module_path):
    sys.path.append(module_path)
    from figure_properties import FigureProperties as fp
    from kdimensional_tree import kdtree_fast, kdtree_fast_batch
else:
    raise ImportError(f"Module path '{module_path}' does not exist.")

def _get_xy_limits_plot_freya_glacier_mask(resolution, xlims, ylims):
    """
    Generate ticks and tick labels for RACMO grid plots.

    Parameters
    ----------
    resolution : str
        '1km' or '5km'
    xlims, ylims : tuple or list
        (min, max) coordinate limits for x and y.

    Returns
    -------
    xticks, yticks : list of float
        Tick positions for x and y.
    xticklabels, yticklabels : list of str
        Labels in km for x and y axes.
    """
    if resolution == '1km':
        spacing = 1000 # Grid spacing in meters (default: RACMO 1000 = 1 km)
        # X-axis ticks and labels
        xticks = np.arange(xlims[0], xlims[1], spacing).tolist()
        xticklabels = [f"{i} km" for i in range(len(xticks))]

        # Y-axis ticks and labels
        yticks = np.arange(ylims[0], ylims[1] + spacing, spacing).tolist()
        yticklabels = [f"{i} km" if i > 0 else "" for i in range(len(yticks))]
    elif resolution == '5km':
        spacing = 0.045 # Grid spacing in meters (default: RACMO 0.05 rlat rlon grid space is 5.5 km; 5 km is 0.045 degree)
        # X-axis ticks and labels
        xticks = np.arange(xlims[0], xlims[1], spacing).tolist()
        xticklabels = [f"{i * 5} km" for i in range(len(xticks))]

        # Y-axis ticks and labels
        yticks = np.arange(ylims[0], ylims[1] + spacing, spacing).tolist()
        yticklabels = [f"{i * 5} km" if i > 0 else "" for i in range(len(yticks))]

    return xticks, yticks, xticklabels, yticklabels

def _get_edges(resolution, cell_size, xindexs_on, yindexs_on, x_values, y_values, freya_glacier_on_racmo_da, dataset_type):
        edges = []
        for xindex_on, yindex_on in zip(xindexs_on, yindexs_on):
            if resolution == '1km':
                condition = freya_glacier_on_racmo_da.sel(y=y_values[yindex_on], x=x_values[xindex_on]) == 0
            elif resolution == '5km':
                condition = freya_glacier_on_racmo_da.sel(rlat=y_values[yindex_on], rlon=x_values[xindex_on]) == 0

            if condition == True:
                x0 = x_values[xindex_on] - cell_size/2
                y0 = y_values[yindex_on] - cell_size/2
                x1, y1 = x0 + cell_size, y0 + cell_size
                # define 4 edges
                rect_edges = [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]] #  [bottom, right, top, left]
                edges.extend(rect_edges)
            if (condition == False) & (resolution == '5km'):
                # There is no overlap between RACMO promice and Freya Qgis
                x0 = x_values[xindex_on] - cell_size/2
                y0 = y_values[yindex_on] - cell_size/2
                x1, y1 = x0 + cell_size, y0 + cell_size
                # define 4 edges
                rect_edges = [[(x0, y0), (x1, y0)], [(x1, y0), (x1, y1)], [(x1, y1), (x0, y1)], [(x0, y1), (x0, y0)]] #  [bottom, right, top, left]
                edges.extend(rect_edges)

        # Deduplicate edges: store as sorted tuples so shared edges collapse
        unique_edges = set()
        for (p1, p2) in edges:
            # sort endpoints to make bottom=top etc.
            edge = tuple(sorted([p1, p2]))
            unique_edges.add(edge)

        # Convert back to list of line segments
        segments = [list(edge) for edge in unique_edges]

        # Choose color based on dataset
        color = {'qgis': 'k', 'promice': 'gray'}.get(dataset_type, 'k')

        # Plot all edges once
        return LineCollection(segments, colors=color, linewidths=1, linestyles='--')

class SquareWithMarker(Rectangle):
    """ Custom patch class that stores marker color.

    """
    def __init__(self, *args, marker_color='black', **kwargs):
        super().__init__(*args, **kwargs)
        self.marker_color = marker_color

class HandlerSquareWithPoint(HandlerBase):
    def create_artists(self, legend, orig_handle,
                    xdescent, ydescent, width, height, fontsize, trans):
        size = min(width, height)
        # Square
        rect = Rectangle(
            (xdescent, ydescent), size, size,
            facecolor=orig_handle.get_facecolor(),
            edgecolor=orig_handle.get_edgecolor(),
            linestyle=orig_handle.get_linestyle(),
            linewidth=orig_handle.get_linewidth(),
            transform=trans
        )

        # Center marker with color from orig_handle.marker_color
        marker_color = getattr(orig_handle, 'marker_color', 'black')  # default black
        cx = xdescent + size / 2
        cy = ydescent + size / 2
        marker = Line2D([cx], [cy], marker='o', color=marker_color, markersize=size/4, linestyle='', transform=trans)
        return [rect, marker]
    # %%

class FreyaInRacmo:
# %%
    def plot_freya_glacier_mask_1km():
        """ Plot model perspective of the study area.

        """
        # %%
        # Freya on qgis
        colors_freya_qgis = [(0, 0, 0), (169, 169, 169)]
        cmap_freya_qgis = fp.make_cmap(colors_freya_qgis, bit=True)
        bounds_freya_qgis = np.array([-0.01, 0.01, 1.5])
        norm_freya_qgis = mpl.colors.BoundaryNorm(bounds_freya_qgis, cmap_freya_qgis.N)
        # Freya on glacier
        colors_freya_racmo = [(255, 255, 255), (0, 139, 0)]
        cmap_freya_racmo = fp.make_cmap(colors_freya_racmo, bit=True)
        bounds_freya_racmo = np.array([-0.01, 0.01, 1.5])
        norm_freya_racmo = mpl.colors.BoundaryNorm(bounds_freya_racmo, cmap_freya_racmo.N)
        # Racmo peripheral glacier
        colors_racmo_peripheral = [(0, 0, 0), (169, 169, 169)]
        cmap_racmo_peripheral = fp.make_cmap(colors_racmo_peripheral, bit=True)
        bounds_racmo_peripheral = np.array([-0.01, 0.01, 1.5])
        norm_racmo_peripheral = mpl.colors.BoundaryNorm(bounds_racmo_peripheral, cmap_racmo_peripheral.N)
        # Freya topography based on racmo
        bounds_topo = [200, 400, 600, 800, 1000, 1200]

        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        mask_file_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/masks/'
        # Freya glacier mask from qgis
        freya_glacier_on_qgis_da = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_qgis_1km.nc")
        # Freya glacier mask from racmo
        freya_glacier_on_racmo_da = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_racmo_1km.nc")

        # Extracting x and y coordinates
        y_values = freya_glacier_on_qgis_da.coords['y'].values
        x_values = freya_glacier_on_qgis_da.coords['x'].values
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(x_values, y_values)

        # Read RACMO files for metadata
        mask_1km = f"{mask_file_path}Icemask_Topo_Iceclasses_lon_lat_average_1km_GrIS.nc"
        racmo_1km_mask_topo = xr.open_dataset(mask_1km)['Topography']

        racmo_1km_mask_promicemask = xr.open_dataset(mask_1km)['Promicemask']
        racmo_1km_mask_promicemask = xr.where((racmo_1km_mask_promicemask == 3), 0, racmo_1km_mask_promicemask)
        racmo_1km_mask_promicemask = xr.where((racmo_1km_mask_promicemask == 2), 1, racmo_1km_mask_promicemask)

        racmo_1km_mask_lat = xr.open_dataset(mask_1km)['LAT']
        racmo_1km_mask_lon = xr.open_dataset(mask_1km)['LON']

        xlims = [693044, 701544] # 702044-500; mid-way of 1-km
        ylims = [-1556596, -1548596]
        # For faster looping change all the value outside of the range to zero
        # Make a mask of "inside the bounding box"
        inside = ((racmo_1km_mask_topo['x'] >= xlims[0]) & (racmo_1km_mask_topo['x'] <= xlims[1] + 1000) &
            (racmo_1km_mask_topo['y'] >= ylims[0] - 1000) & (racmo_1km_mask_topo['y'] <= ylims[1]))

        # Expand mask to 2D by broadcasting
        inside_2d = inside.sel(x=racmo_1km_mask_topo['x']).broadcast_like(racmo_1km_mask_topo) & \
                    inside.sel(y=racmo_1km_mask_topo['y']).broadcast_like(racmo_1km_mask_topo)

        # Set everything outside to NaN
        racmo_1km_mask_topo_subset = racmo_1km_mask_topo.where(inside_2d, np.nan)

        # Plot
        # fig, ax = plt.subplots(figsize=(6.5, 5), dpi=600)
        fig, ax = plt.subplots(figsize=(6, 5), dpi=600)

        # Plot topography
        CS = ax.contour(X, Y, racmo_1km_mask_topo_subset, colors='k', levels=bounds_topo, linewidths=0.5, linestyles='dashed', alpha=0.5)
        ax.clabel(CS, bounds_topo, inline=True, inline_spacing=-1, fmt='%d', fontsize=9, zorder=-1, colors='k')

        # Plot freya on qgis
        # Select x and y index where Freya is present in Qgis
        yindexs_on_qgis, xindexs_on_qgis = np.where(freya_glacier_on_qgis_da == 1)
        for xindex_on_qgis, yindex_on_qgis in zip(xindexs_on_qgis, yindexs_on_qgis):
            if freya_glacier_on_racmo_da.sel(y=y_values[yindex_on_qgis], x=x_values[xindex_on_qgis]) == 0:
                ax.plot(x_values[xindex_on_qgis], y_values[yindex_on_qgis], marker='.', markersize=2, markeredgewidth=1, markeredgecolor='k', color='k')
            #   ax.add_patch(Rectangle(((x_values[xindex_on_qgis] + x_values[xindex_on_qgis] - 1000) / 2, (y_values[yindex_on_qgis] + y_values[yindex_on_qgis] - 1000) / 2), 1000, 1000, facecolor='none', edgecolor='k', linestyle=':', alpha=0.5, zorder=12))

        lc_qgis = _get_edges(resolution='1km', cell_size=1000, xindexs_on=xindexs_on_qgis, yindexs_on=yindexs_on_qgis, x_values=x_values, y_values=y_values, freya_glacier_on_racmo_da=freya_glacier_on_racmo_da, dataset_type='qgis')
        ax.add_collection(lc_qgis)

        # Plot Promice mask in racmo
        # Decrease the runtime by converting everything outside of the study area to nan.
        # Select x and y index where Perpheral Glacier is present in Racmo
        inside = ((racmo_1km_mask_promicemask['x'] >= xlims[0]) & (racmo_1km_mask_promicemask['x'] <= xlims[1] + 1000) &
                    (racmo_1km_mask_promicemask['y'] >= ylims[0] - 1000) & (racmo_1km_mask_promicemask['y'] <= ylims[1]))
        # Expand mask to 2D by broadcasting
        inside_2d = inside.sel(x=racmo_1km_mask_promicemask['x']).broadcast_like(racmo_1km_mask_promicemask) & \
                    inside.sel(y=racmo_1km_mask_promicemask['y']).broadcast_like(racmo_1km_mask_promicemask)
        # Set everything outside to NaN
        racmo_1km_mask_promicemask_subset = racmo_1km_mask_promicemask.where(inside_2d, np.nan)

        yindexs_promice, xindexs_promice = np.where(racmo_1km_mask_promicemask_subset == 1)
        for xindex_promice, yindex_promice in zip(xindexs_promice, yindexs_promice):
            if freya_glacier_on_racmo_da.sel(y=y_values[yindex_promice], x=x_values[xindex_promice]) == 0:
                ax.plot(x_values[xindex_promice], y_values[yindex_promice], marker='.', markersize=2, markeredgewidth=1, markeredgecolor='grey', color='grey')
        #     ax.add_patch(Rectangle(((x_values[xindex_promice] + x_values[xindex_promice] - 1000) / 2, (y_values[yindex_promice] + y_values[yindex_promice] - 1000) / 2),
        #                                 1000, 1000, facecolor='none', edgecolor='grey', linestyle='--', alpha=0.5, zorder=12))#,zorder=12)
        lc_racmo = _get_edges(resolution='1km', cell_size=1000, xindexs_on=xindexs_promice, yindexs_on=yindexs_promice, x_values=x_values, y_values=y_values, freya_glacier_on_racmo_da=freya_glacier_on_racmo_da, dataset_type='promice')
        ax.add_collection(lc_racmo)

        # Plot freya on racmo
        yindexs_on_racmo, xindexs_on_racmo = np.where(freya_glacier_on_racmo_da == 1)
        for xindex_on_racmo, yindex_on_racmo in zip(xindexs_on_racmo, yindexs_on_racmo):
            # ax.add_patch(Rectangle(((x_values[xindex_on_racmo] + x_values[xindex_on_racmo] - 1000) / 2, (y_values[yindex_on_racmo] + y_values[yindex_on_racmo] - 1000) / 2),
            #                            1000, 1000, facecolor='g', edgecolor='none', alpha=0.2))#,zorder=12)
            racmo_1km_mask_topo_point = racmo_1km_mask_topo.sel(y=y_values[yindex_on_racmo], x=x_values[xindex_on_racmo])
            if racmo_1km_mask_topo_point.values > 950:
                edgecolors = 'b'
            elif racmo_1km_mask_topo_point.values < 950:
                edgecolors = 'r'
            ax.plot(x_values[xindex_on_racmo], y_values[yindex_on_racmo], marker='.', markersize=2, markeredgewidth=1, markeredgecolor=edgecolors, color=edgecolors)
            # ax.scatter(x_values[xindex_on_racmo], y_values[yindex_on_racmo], s=100, marker='o', facecolors='none', edgecolors=edgecolors, linestyle='-')

            ax.add_patch(Rectangle(((x_values[xindex_on_racmo] + x_values[xindex_on_racmo] - 1000) / 2, (y_values[yindex_on_racmo] + y_values[yindex_on_racmo] - 1000) / 2),
                                    1000, 1000, facecolor=edgecolors, edgecolor='none', alpha=0.2))#,zorder=12)

        # Plot Freya under investigation
        cell_size = 1000  # m
        # build shapely polygons for selected cells
        polys = []
        for xindex_on_racmo, yindex_on_racmo in zip(xindexs_on_racmo, yindexs_on_racmo):
            x0 = x_values[xindex_on_racmo] - cell_size / 2
            y0 = y_values[yindex_on_racmo] - cell_size / 2
            polys.append(box(x0, y0, x0 + cell_size, y0 + cell_size))
        # merge cells into a single outline
        merged = unary_union(polys)
        # add patch(es) for outline
        if merged.geom_type == 'Polygon':
            ax.add_patch(Polygon(np.array(merged.exterior.coords),
                                facecolor='none', edgecolor='k', linestyle=':', linewidth=0.5))
        else:  # MultiPolygon
            for geom in merged.geoms:
                ax.add_patch(Polygon(np.array(geom.exterior.coords),
                                    facecolor='none', edgecolor='g', linestyle='-', linewidth=1.5, zorder=20))


        # Plot Freya Glacier's Station location
        station_metadata_path = '/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/raw_data/station_data/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        station = 'FRE2 (promice type, tripod)'

        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Find the nearest indices
        iy_min, ix_min = kdtree_fast(racmo_1km_mask_lat.values, racmo_1km_mask_lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])
        # Create a mesh grid for plotting
        ax.plot(X[iy_min, ix_min], Y[iy_min, ix_min], marker='^', ms=16, mew=0.9, mfc='none', mec='g')
        # ax.text(X[iy_min, ix_min] + 200, Y[iy_min, ix_min] + 250, 'FRE', size=14, color='g', path_effects=[pe.withStroke(linewidth=4, foreground='white')], horizontalalignment='right')

        station_elevation_on_racmo = np.int64(np.round(racmo_1km_mask_topo.sel(x=X[iy_min, ix_min], y=Y[iy_min, ix_min]).values, 0))

        # Set x and y limits first otherwise the contour labelling will not work.
        xticks, yticks, xticklabels, yticklabels = _get_xy_limits_plot_freya_glacier_mask(resolution='1km', xlims=xlims, ylims=ylims)
        ax.set_xlim(xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.set_ylim(ylims)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        tick_param = dict(axis='both', direction='out', length=3.5, pad=1, labelsize=9, width=0.5)
        ax.tick_params(**tick_param)

        # [ax.spines[i].set_visible(False) for i in ['top', 'right']]
        [i.set_linewidth(0.5) for i in ax.spines.values()]
        # ax.spines['left'].set_bounds(y_bound[0], y_bound[1])
        # ax_spine_shift = 3 # Adjust this value to control the outward shift of the spines
        # ax.spines['left'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)
        # ax.spines['bottom'].set_bounds(5, 9)
        # ax.spines['bottom'].set_position(('outward', ax_spine_shift))  # smaller than current default (e.g., 1.2)

        # Add legend
        # Create patches with different marker colors
        grid_cell_qgis = SquareWithMarker((0, 0), 1, 1, facecolor='none', edgecolor='k', linestyle='--', linewidth=0.5, marker_color='k')
        grid_cell_promice = SquareWithMarker((0, 0), 1, 1, facecolor='none', edgecolor='grey', linestyle='--', linewidth=0.5, marker_color='gray')
        fre_under_investigation = Patch(lw=0.9, fc='none', ec='green', ls='-', fill=False)
        acc_area = SquareWithMarker((0, 0), 1, 1, facecolor='b', edgecolor='none',  alpha=0.2, marker_color='b')
        abl_area = SquareWithMarker((0, 0), 1, 1, facecolor='r', edgecolor='none',  alpha=0.2, marker_color='r')
        station_loc = Line2D([], [], marker='^', markerfacecolor='none', markeredgecolor='g', markersize=6, linewidth=0, markeredgewidth=0.9)
        elevation = Line2D([], [], color='gray', linewidth=0.9, ls=':')

        fig.legend(
            bbox_to_anchor=(0.97, 0.98),
            handles=[grid_cell_qgis, grid_cell_promice, fre_under_investigation, acc_area, abl_area, station_loc, elevation],
            labels=['FRG (2021) projected in 1-km RACMO', 'Peripheral Glaciers in 1-km RACMO', 'FRG under investigation',
                    r'FRG$_{\mathrm{AA}}$', r'$\mathdefault{FRG_{BA}}$', rf"AWS$_\mathdefault{{FRG}}$ at {station_elevation_on_racmo} m a.s.l", 'Contour'],
            handler_map={SquareWithMarker: HandlerSquareWithPoint()},
            ncol=1, borderaxespad=0, fontsize=9, handletextpad=0.3, labelspacing=0.4, frameon=True,
            facecolor='w', framealpha=0.9, edgecolor='none', handlelength=1.) # handleheight=1.0

        # targets = dict(left=190, right=40, top=30, bottom=90, hspace=0, wspace=0)
        targets = dict(left=198, right=30, top=30, bottom=95, hspace=0, wspace=0)
        # hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Save the plot
        logger.info(f"Saving plot of 'Freya on RACMO mask'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/figures/manuscript_figures')
        fig_name = 'Figure_B1.png'
        fig.savefig(f"{base_output_path}/{fig_name}", dpi=600)
        # %%

    def plot_freya_glacier_mask_5km():
        # %%
        resolution = '5km'
        # Freya on qgis
        colors_freya_qgis = [(0, 0, 0), (169, 169, 169)]
        cmap_freya_qgis = fp.make_cmap(colors_freya_qgis, bit=True)
        bounds_freya_qgis = np.array([-0.01, 0.01, 1.5])
        norm_freya_qgis = mpl.colors.BoundaryNorm(bounds_freya_qgis, cmap_freya_qgis.N)
        # Freya on glacier
        colors_freya_racmo = [(255, 255, 255), (0, 139, 0)]
        cmap_freya_racmo = fp.make_cmap(colors_freya_racmo, bit=True)
        bounds_freya_racmo = np.array([-0.01, 0.01, 1.5])
        norm_freya_racmo = mpl.colors.BoundaryNorm(bounds_freya_racmo, cmap_freya_racmo.N)
        # Racmo peripheral glacier
        colors_racmo_peripheral = [(0, 0, 0), (169, 169, 169)]
        cmap_racmo_peripheral = fp.make_cmap(colors_racmo_peripheral, bit=True)
        bounds_racmo_peripheral = np.array([-0.01, 0.01, 1.5])
        norm_racmo_peripheral = mpl.colors.BoundaryNorm(bounds_racmo_peripheral, cmap_racmo_peripheral.N)
        # Freya topography based on racmo
        bounds_topo = [200, 400, 600, 800, 1000, 1200]

        # Read freya glacier mask on ramco created by function create_freya_glacier_on_qgis in FG0_preprocess_data.py
        mask_file_path = '/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/input_dataset/masks/'
        # Freya glacier mask from qgis
        freya_glacier_on_qgis_da = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_qgis_5km.nc")
        # Freya glacier mask from racmo
        freya_glacier_on_racmo_da = xr.open_dataarray(f"{mask_file_path}freya_glacier_on_racmo_5km.nc")

        # Extracting rlon and rlat coordinates
        y_values = freya_glacier_on_qgis_da.coords['rlat'].values
        x_values = freya_glacier_on_qgis_da.coords['rlon'].values
        # Create a mesh grid for plotting
        X, Y = np.meshgrid(x_values, y_values)

        # Read RACMO files for metadata
        mask_5km = f"{mask_file_path}FGRN055_Masks_5.5km_updated_coordinates.nc"
        racmo_5km_mask_topo = xr.open_dataset(mask_5km)['Topography']

        racmo_5km_mask_promicemask = xr.open_dataset(mask_5km)['Promicemask']
        racmo_5km_mask_promicemask = xr.where((racmo_5km_mask_promicemask == 3), 0, racmo_5km_mask_promicemask)
        racmo_5km_mask_promicemask = xr.where((racmo_5km_mask_promicemask == 2), 1, racmo_5km_mask_promicemask)

        racmo_5km_mask_lat = xr.open_dataset(mask_5km)['lat']
        racmo_5km_mask_lon = xr.open_dataset(mask_5km)['lon']

        # Set x and y limits
        xlims = [4.25 + 0.05, 4.6 - 0.05]
        ylims = [2.75 + 0.05, 3.05]

        # For faster looping change all the value outside of the range to zero
        # Make a mask of "inside the bounding box"
        inside = ((racmo_5km_mask_topo['rlon'] >= xlims[0]) & (racmo_5km_mask_topo['rlon'] <= xlims[1] + 0.05) &
            (racmo_5km_mask_topo['rlat'] >= ylims[0] - 0.05) & (racmo_5km_mask_topo['rlat'] <= ylims[1] + 0.05))
        # Expand mask to 2D by broadcasting
        inside_2d = inside.sel(rlon=racmo_5km_mask_topo['rlon']).broadcast_like(racmo_5km_mask_topo) & \
                    inside.sel(rlat=racmo_5km_mask_topo['rlat']).broadcast_like(racmo_5km_mask_topo)

        # Set everything outside to NaN
        racmo_5km_mask_topo_subset = racmo_5km_mask_topo.where(inside_2d, np.nan)

        # Plot
        fig, ax = plt.subplots(figsize=(5, 5), dpi=600)
        # Set x and y limits first otherwise the contour labelling will not work.
        xticks, yticks, xticklabels, yticklabels = _get_xy_limits_plot_freya_glacier_mask(resolution='5km', xlims=xlims, ylims=ylims)
        ax.set_xlim(xlims)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)

        ax.set_ylim(ylims)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        tick_param = dict(axis='both', direction='inout', length=3.5, pad=1, labelsize=9, width=0.5)
        ax.tick_params(**tick_param)

        # Plot topography
        CS = ax.contour(X, Y, racmo_5km_mask_topo_subset, colors='k', levels=bounds_topo, linewidths=0.5, linestyles='dashed', alpha=0.5)
        ax.clabel(CS, bounds_topo, inline=True, inline_spacing=-1, fmt='%d', fontsize=9, zorder=-1, colors='k')

        # Plot freya on qgis
        # Select x and y index where Freya is present in Qgis
        yindexs_on_qgis, xindexs_on_qgis = np.where(freya_glacier_on_qgis_da == 1)
        for xindex_on_qgis, yindex_on_qgis in zip(xindexs_on_qgis, yindexs_on_qgis):
            ax.plot(x_values[xindex_on_qgis], y_values[yindex_on_qgis], marker='.', markersize=2, markeredgewidth=1, markeredgecolor='k', color='k')

        lc_qgis = _get_edges(resolution='5km', cell_size=0.05, xindexs_on=xindexs_on_qgis, yindexs_on=yindexs_on_qgis, x_values=x_values, y_values=y_values, freya_glacier_on_racmo_da=freya_glacier_on_racmo_da, dataset_type='qgis')
        ax.add_collection(lc_qgis)

        # Plot Promice mask in racmo
        # Select x and y index where Perpheral Glacier is present in Racmo
        inside = ((racmo_5km_mask_promicemask['rlon'] >= xlims[0]) & (racmo_5km_mask_promicemask['rlon'] <= xlims[1] + 0.05) &
        (racmo_5km_mask_promicemask['rlat'] >= ylims[0] - 0.05) & (racmo_5km_mask_promicemask['rlat'] <= ylims[1]))
        # Expand mask to 2D by broadcasting
        inside_2d = inside.sel(rlon=racmo_5km_mask_promicemask['rlon']).broadcast_like(racmo_5km_mask_promicemask) & \
                    inside.sel(rlat=racmo_5km_mask_promicemask['rlat']).broadcast_like(racmo_5km_mask_promicemask)
        # Set everything outside to NaN
        racmo_5km_mask_promicemask_subset = racmo_5km_mask_promicemask.where(inside_2d, np.nan)

        yindexs_promice, xindexs_promice = np.where(racmo_5km_mask_promicemask_subset == 1)
        y_values_promice = []
        x_values_promice = []
        for xindex_promice, yindex_promice in zip(xindexs_promice, yindexs_promice):
            ax.plot(x_values[xindex_promice], y_values[yindex_promice], marker='.', markersize=2, markeredgewidth=1, markeredgecolor='grey', color='grey')
            x_values_promice.append(float(x_values[xindex_promice]))
            y_values_promice.append(float(y_values[yindex_promice]))

        lc_racmo = _get_edges(resolution='5km', cell_size=0.05, xindexs_on=xindexs_promice, yindexs_on=yindexs_promice, x_values=x_values, y_values=y_values, freya_glacier_on_racmo_da=freya_glacier_on_racmo_da, dataset_type='promice')
        ax.add_collection(lc_racmo)

        # Plot Freya Glacier's Station location
        station_metadata_path = '/nas/data/processed/people/sos/zackenberg/northeast_stations_metadata.csv'
        stations_data = pd.read_csv(station_metadata_path, encoding='iso-8859-1') # Check encoding if needed on bash file -i northeast_stations_metadata.csv
        stations_data.index = stations_data.Stations
        stations_data.drop('Stations', axis=1, inplace=True)
        station = 'FRE2 (promice type, tripod)'

        stations_data.loc['lat (N)'] = stations_data.loc['lat (N)'].astype(float)
        stations_data.loc['lon (W)'] = stations_data.loc['lon (W)'].astype(float)

        # Find the nearest indices
        iy_min, ix_min = kdtree_fast(racmo_5km_mask_lat.values, racmo_5km_mask_lon.values, stations_data.loc['lat (N)', station], stations_data.loc['lon (W)', station])

        # Create a mesh grid for plotting
        ax.plot(X[iy_min, ix_min], Y[iy_min, ix_min], marker='^', ms=16, mew=0.9, mfc='none', mec='g')
        station_elevation_on_racmo = np.int64(np.round(racmo_5km_mask_topo.sel(rlon=X[iy_min, ix_min], rlat=Y[iy_min, ix_min]).values, 0))

        # Compute Euclidean distances d=(xi​−xfreya​)2+(yi​−yfreya​)2
        freya_point = [X[iy_min, ix_min], Y[iy_min, ix_min]]
        # Find the closest point
        distances = [ np.sqrt((x - freya_point[0])**2 + (y - freya_point[1])**2)
            for x, y in zip(x_values_promice, y_values_promice)]

        # Get index of closest point
        min_idx = int(np.argmin(distances))
        closest_point = (x_values_promice[min_idx], y_values_promice[min_idx])
        # ax.plot(closest_point[0], closest_point[-1], marker='^', ms=16, mew=0.9, mfc='none', mec='g')

        # ax.scatter(closest_point[0], closest_point[-1], s=100, marker='o', facecolors='none', edgecolors='g', linestyle='-')
        ax.plot(closest_point[0], closest_point[-1], marker='.', markersize=2, markeredgewidth=1, markeredgecolor='g', color='g')

        ax.add_patch(Rectangle(((closest_point[0] + closest_point[0] - 0.05) / 2, (closest_point[-1] + closest_point[-1] - 0.05) / 2),
                                0.05, 0.05, facecolor='none', edgecolor='g', linestyle='-', linewidth=1.5, zorder=20))#,zorder=12)

        closest_point_elevation = np.int64(np.round(racmo_5km_mask_topo.sel(rlon=closest_point[0], rlat=closest_point[-1]).values, 0))
        # ax.text(closest_point[0] + 0.02, closest_point[-1] - 0.02, f"{closest_point_elevation} m ASL", size=9, color='g', path_effects=[pe.withStroke(linewidth=4, foreground='white')], horizontalalignment='right')

        # Change ylims to equally scale x and y
        # ax.set_ylim([2.75, 3.05])

        targets = dict(left=226, right=40, top=30, bottom=90, hspace=0, wspace=0)
        # hook up a function to adjust the borders when the window is resized
        fig.canvas.mpl_connect('resize_event', lambda e: fp.adjust_borders(fig, targets))
        fp.adjust_borders(fig, targets)

        # Add legend
        # Create patches with different marker colors
        grid_cell_qgis = SquareWithMarker((0, 0), 1, 1, facecolor='none', edgecolor='k', linestyle='--', linewidth=0.5, marker_color='k')
        grid_cell_promice = SquareWithMarker((0, 0), 1, 1, facecolor='none', edgecolor='grey', linestyle='--', linewidth=0.5, marker_color='gray')
        fre_under_investigation = SquareWithMarker((0, 0), 1, 1, facecolor='none', edgecolor='g', linestyle='-', linewidth=1, marker_color='g')
        station_loc = Line2D([], [], marker='^', markerfacecolor='none', markeredgecolor='g', markersize=6, linewidth=0, markeredgewidth=0.9)
        elevation = Line2D([], [], color='gray', linewidth=0.9, ls=':')

        fig.legend(
            bbox_to_anchor=(0.54, 0.98),
            handles=[grid_cell_qgis, grid_cell_promice, fre_under_investigation, station_loc, elevation],
            labels=['FG  (2013) projected in 5-km RACMO', 'Peripheral Glaciers in 5-km RACMO', f"FG under investigation at {closest_point_elevation} m ASL",
                    rf"FG$_{{\mathrm{{AWS}}}}$ at {station_elevation_on_racmo} m ASL", 'Contour'],
            handler_map={SquareWithMarker: HandlerSquareWithPoint()},
            ncol=1, borderaxespad=0, fontsize=9, handletextpad=0.3, labelspacing=0.4, frameon=True,
            facecolor='w', framealpha=0.9, edgecolor='none', handlelength=1.) # handleheight=1.0

        # Save the plot
        logger.info(f"Saving plot of 'Freya on RACMO 5-km mask'...")
        base_output_path = Path('/nas/home/sos/third_paper_work/figures/freyagl_svd_analysis_plot/FG0_preprocess_plots-')
        fig_name = 'plot_freya_glacier_mask_5km.png'
        fig.savefig(f"{base_output_path}{fig_name}", dpi=600)

        # %%