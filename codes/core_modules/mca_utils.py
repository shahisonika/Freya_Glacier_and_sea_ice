#!/usr/bin/env python3

"""
Freya Glacier Analysis Level 1 (FG1) - Maximum Covariance Analysis (MCA) using
QR decomposition and the xeofs package.

Analyze relationship between Greenland Ice Sheet SMB, sea ice, and geopotential height.

Author: Sonika Shahi
Updated: 13 Aug 2025
"""
# %%
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, Iterable, Union
import numpy as np
import xarray as xr
import pandas as pd

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def _nc_compression(da: xr.DataArray) -> dict:
    """Default NetCDF compression encoding for float variables."""
    enc = {}
    for v in [da.name] if da.name else []:
        enc[v] = {'zlib': True, 'complevel': 9}
    return enc

def save_da(da: xr.DataArray, name: str, base_output_path: Path, suffix: str):
    """
    Save an xarray DataArray to NetCDF with proper encoding and directory creation.

    Parameters:
        da: xarray.DataArray to save
        name: variable name for file and DataArray
        base_output_path: directory path to save the file
        suffix: string to append to filename (includes extension)
    """
    da = da.copy()
    da.name = name
    path = base_output_path / f"{name}{suffix}"
    _ensure_parent_dir(path)

    if 'time' in da.indexes:
        if isinstance(da.indexes['time'], pd.MultiIndex): # The multiindex is not supported in netcdf
            da = da.unstack('time')
    da.to_netcdf(path, encoding=_nc_compression(da), mode='w', auto_complex=True)

def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """
    Orthogonal varimax rotation.

    Parameters
    ----------
    Phi : ndarray
        Array of shape (features, modes) to rotate.
    gamma : float, optional
        Kaiser normalization parameter. 1.0 = varimax.
    q : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    Phi_rot : ndarray
        Rotated loadings matrix.
    R : ndarray
        Rotation matrix applied.
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for _ in range(q):
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(
                Phi.T,
                Lambda**3 - (gamma/p) * np.dot(Lambda, np.diag(np.sum(Lambda**2, axis=0)))
            )
        )
        R = np.dot(u, vh)
        d_old, d = d, np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol:
            break
    return np.dot(Phi, R), R

def _normalise_shape(array, nrows, ncols):
    """
    The normalise_shape function crops the array such that an integer number of _nrows_ and _ncols_ fits in its rows and columns.
    """
    h, w = array.shape
    if h % nrows > 0 or w % ncols > 0:
        array = array[:(h // nrows * nrows), :(w // ncols * ncols)]
    return (h // nrows * nrows), (w // ncols * ncols), array

def _coarsen_matrix(array, level):
    """
    The _coarsen_matrix_ function first cuts the input matrix into smaller sections and then
    averages each section, reducing the spatial size.
    It then knits the sections back together, resulting in a matrix which is
    $level^{2}$ times smaller.
    When a field has large scale structure, the values of neighbouring cells do not change drastically.
    This is what autocorrelation means. As such, perhaps neighbouring cells can be averaged together to
    produce a smaller dataset which still faithfully describes the original field.

    """
    h, w, array = _normalise_shape(array, level, level)
    array_reshaped = array.reshape((h // level, level, w // level, level))
    return np.nanmean(array_reshaped, axis=(1, 3))

def coarsen_matrix_3d(array, level):
    """
    The _coarsen_matrix_3d_ function loops over the temporal dimension and coarsens the spatial part.
    In the analysis of spatio-temporal fields, the spatial part of each time period can be coarsened.

    """
    d, h, w = array.shape
    coarsened_matrix_3d = np.empty((d, h // level, w // level))
    for i in range(d):
        coarsened_matrix_3d[i, :, :] = _coarsen_matrix(array[i, :, :], level)
    return coarsened_matrix_3d

def coarsen_matrix_2d(array, level):
    """
    The _coarsen_matrix_2d_ function coarsens the spatial dimensions of a 2D array.
    In the analysis of spatial fields, the spatial part can be coarsened.

    """
    coarsened_matrix_2d = _coarsen_matrix(array, level)
    return coarsened_matrix_2d

def uncoarsen_matrix_3d(array, level):
    return np.repeat(np.repeat(array, level, axis=0), level, axis=1)

def coarsen_coordinates_1d(X, Y, level):
    """ Coarsened coordinates of the variable.

    """

    XX, YY = np.meshgrid(X, Y)

    # Coarsen X coordinate
    X_coarsened_mesh_grid = _coarsen_matrix(XX, level)
    X_coarsened = X_coarsened_mesh_grid[0, :]
    # Coarsen Y coordinate
    Y_coarsened_mesh_grid = _coarsen_matrix(YY, level)
    Y_coarsened = Y_coarsened_mesh_grid[:, 0]

    return X_coarsened, Y_coarsened

# %%