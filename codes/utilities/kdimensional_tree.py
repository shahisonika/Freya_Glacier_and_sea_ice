import numpy as np
from matplotlib.patches import Rectangle
from math import pi
from numpy import cos, sin
from scipy.spatial import cKDTree

def kdtree_fast(latvar, lonvar, lat0, lon0):
    """ Find the nearest grid point in a 2D latitude/longitude grid using a k-d tree.

    References:https://github.com/Unidata/python-workshop/blob/fall-2016/notebooks/netcdf-by-coordinates.ipynb
    """
    rad_factor = pi/180.0 # for trignometry, need angles in radians
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor
    ny, nx = latvals.shape
    clat, clon = cos(latvals), cos(lonvals)
    slat, slon = sin(latvals), sin(lonvals)
    # Build kd-tree from big arrays of 3D coordinates
    triples = list(zip(np.ravel(clat * clon), np.ravel(clat * slon), np.ravel(slat)))
    kdt = cKDTree(triples)
    lat0_rad = lat0 * rad_factor
    lon0_rad = lon0 * rad_factor
    clat0, clon0 = cos(lat0_rad), cos(lon0_rad)
    slat0, slon0 = sin(lat0_rad), sin(lon0_rad)
    dist_sq_min, minindex_1d = kdt.query([clat0 * clon0, clat0 * slon0, slat0])
    iy_min, ix_min = np.unravel_index(minindex_1d, latvals.shape)
    return iy_min, ix_min

def kdtree_fast_batch(latvar, lonvar, lat0_list, lon0_list):
    """ Find nearest lat/lon indices for multiple (lat0, lon0) pairs using KDTree.

    Parameters:
        latvar (2D array): Latitude grid
        lonvar (2D array): Longitude grid
        lat0_list (list or array): List of latitudes to locate
        lon0_list (list or array): List of longitudes to locate

    Returns:
        iy_list, ix_list: Lists of y and x indices for nearest points
    """
    rad_factor = pi / 180.0  # degrees to radians
    latvals = latvar[:] * rad_factor
    lonvals = lonvar[:] * rad_factor

    ny, nx = latvals.shape
    clat, clon = np.cos(latvals), np.cos(lonvals)
    slat, slon = np.sin(latvals), np.sin(lonvals)

    # Prepare 3D coordinates for KD-tree
    # Build KD-tree
    points = np.column_stack([(clat * clon).ravel(),
                                (clat * slon).ravel(),
                                    slat.ravel()])
    kdtree = cKDTree(points)

    # Prepare query points
    lat0_rad = np.array(lat0_list) * rad_factor
    lon0_rad = np.array(lon0_list) * rad_factor

    clat0 = np.cos(lat0_rad)
    clon0 = np.cos(lon0_rad)
    slat0 = np.sin(lat0_rad)
    slon0 = np.sin(lon0_rad)

    query_points = np.column_stack((clat0 * clon0, clat0 * slon0, slat0))

    # Query KDTree
    _, indices = kdtree.query(query_points)

    # Convert 1D indices to 2D
    iy_list, ix_list = np.unravel_index(indices, latvals.shape)

    return iy_list, ix_list

# # %%
# # Test single point and batch point queries
# latvar, lonvar = np.meshgrid(racmo_1km_mask.x.values, racmo_1km_mask.y.values)
# lat0_list = list(df_latlon.loc[:, 'y_polar'])
# lon0_list = list(df_latlon.loc[:, 'x_polar'])

# iyb, ixb = kdtree_fast_batch(latvar, lonvar, lat0_list, lon0_list)
# # %%
# cell = 1
# iy, ix = kdtree_fast(latvar, lonvar, df_latlon.loc[cell, 'y_polar'], df_latlon.loc[cell, 'x_polar'])
# print(f"Single point: {iy}, {ix}")
# print(f"Batch points: {iyb[cell]}, {ixb[cell]}")
# # %%