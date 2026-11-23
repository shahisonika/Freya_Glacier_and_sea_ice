#!/usr/bin/env python3
"""
Complex Rotated Maximum Covariance Analysis (MCA) using the xeofs package.

Analyze relationship between Greenland Ice Sheet SMB, sea ice, and geopotential height.

Author: Sonika Shahi
Updated: 27 Oct 2025

Environment: xeofs_env (conda activate xeofs_env on walrus server)
"""
# %%
from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import Literal, Optional, Iterable, Union
# import numpy as np
# import scipy.linalg
# import matplotlib.pyplot as plt
import xarray as xr
import xskillscore as xs
# from sklearn.preprocessing import StandardScaler
import xeofs as xe
from xeofs.cross import HilbertMCA, HilbertMCARotator
import time
from datetime import timedelta
import logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utility helpers
# -------------------------
from mca_data_loader import load_coarsed_mjjas_data, load_coarsed_amjjaso_data, load_coarsed_mjjas_data_detrended, load_coarsed_monthly_mean_data
from mca_utils import save_da

# %%
class ComplexRotatedMCA:
    """Perform Complex Rotated Maximum Covariance Analysis using the xeofs package.

    """
    @staticmethod
    def calc_maximum_covariance_analysis(
        left_var_name: str,
        right_var_name: str,
        start_year: int,
        end_year: int,
        total_mode: int,
        coarsen_level_left: int,
        coarsen_level_right: int,
        clip: Optional[str] = False,
    ):
        """Compute MCA.

        Parameters
        ----------
        left_var_name : str
            Name of the left variable (e.g., 'ci' or 'z0500).
        right_var_name : str
            Name of the right variable (e.g., 'ci' or 'smb').
        start_year : int
            Start year for analysis.
        end_year : int
            End year for analysis.
        total_mode : int
            Number of MCA modes to compute.
        coarsen_level_left : int
            Coarsening level for the left variable.
        coarsen_level_right : int
            Coarsening level for the right variable.
        clip : Optional[str], optional
            Whether to clip the data to a East Greenland only region, by default False.

        """
        # --- Start timer ---
        overall_start = time.time()
        logger.info(f"Starting Complex Rotated MCA for {left_var_name} vs {right_var_name} "
                    f"({start_year}-{end_year}) at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # Output path for saving results
        base_output_path = Path(f"/nas/home/sos/third_paper_work/shahi_et_al_2027_atmos/data/processed_data/complex_xeofs_results")

        # Load datasets
        left_var_da = load_coarsed_mjjas_data(var_name=left_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_left, clip=clip)
        right_var_da = load_coarsed_mjjas_data(var_name=right_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_right, clip=clip)
        # left_var_da = load_coarsed_monthly_mean_data(var_name=left_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_left, clip=clip)
        # right_var_da = load_coarsed_monthly_mean_data(var_name=right_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_right, clip=clip)
        # left_var_da = load_coarsed_amjjaso_data(var_name=left_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_left, clip=clip)
        # right_var_da = load_coarsed_amjjaso_data(var_name=right_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_right, clip=clip)
        # left_var_da = load_coarsed_mjjas_data_detrended(var_name=left_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_left, clip=clip)
        # right_var_da = load_coarsed_mjjas_data_detrended(var_name=right_var_name, start_year=start_year, end_year=end_year, coarsen_level=coarsen_level_right, clip=clip)

        # Hilbert MCA Analysis
        # Initialize the model
        print(f"Initializing Hilbert MCA with {total_mode} modes.")
        hilbert_mca = HilbertMCA(n_modes=total_mode, standardize=True, feature_name=[left_var_name, right_var_name]) # pca_init_rank_reduction=None
        hilbert_mca.fit(left_var_da, right_var_da, dim='time')
        print("Hilbert MCA fitting complete.")

        # # # Load the squared covariance fraction without the rotation applied
        # # # This result was saved to calcualte covariance explained fraction for the unrotated MCA solution,
        # # # which can be used to determine the number of modes to retain for rotation.
        # sq_cov_frac_unrotated = hilbert_mca.squared_covariance_fraction()
        # # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_unrotated.nc"
        # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_monthly_mean_{start_year}to{end_year}_xeofs_unrotated.nc"
        # save_da(da=sq_cov_frac_unrotated, name=sq_cov_frac_unrotated.name, base_output_path=base_output_path, suffix=suffix)

        # # Rotate a solution obtained from ``xe.cross.MCA``.
        # # print("Initializing Hilbert MCA Rotator.")
        # # hilbert_mca_rot = HilbertMCARotator(n_modes=total_mode, power=1) # Varimax rotation

        # The rotation_mode was chosen based on the specific variable pairs to ensure appropriate rotation for the analysis.
        # The values where chosen based on 99% of the variance explained by the first few modes ('sq_cov_frac_unrotated' below),
        # and to ensure that the rotation captures the essential features of the data.
        # if left_var_name == 'ci' and right_var_name == 'smb':
        #     rotation_mode = 5  # For ci vs smb, use 10 modes for rotation
        # elif left_var_name == 'z0500' and right_var_name == 'smb':
        #     rotation_mode = 5  # For z0500 vs smb, use 10 modes for rotation
        # elif left_var_name == 'z0500' and right_var_name == 'ci':
        #     rotation_mode = 3  # For z0500 vs ci, use 10 modes for rotation

        # rotation_mode = total_mode  # Default to total_mode if no specific case matches
        rotation_mode = 3  # Default to total_mode if no specific case matches

        # for power in [1, 2]:  # Test both Varimax (power=1) orthogonal and promax (p=2) oblique rotation
        # for power in [1, 2]:  # Test both Varimax (power=1) orthogonal and promax (p=2) oblique rotation
        for power in [3, ]:  # Test both Varimax (power=1) orthogonal and promax (p=2) oblique rotation
            print(f"Initializing Hilbert MCA Rotator with rotation_mode={rotation_mode} and power={power}.")
            hilbert_mca_rot = HilbertMCARotator(n_modes=rotation_mode, power=power)
            hilbert_mca_rot.fit(hilbert_mca)
            print(f"Hilbert MCA rotation complete with power={power}.")

            # Load results
            print("Loading Hilbert MCA rotation results.")
            # singular_vectors = hilbert_mca_rot.components()
            component_amplitudes = hilbert_mca_rot.components_amplitude()
            component_phases = hilbert_mca_rot.components_phase()
            scores = hilbert_mca_rot.scores()
            # scores_normalized = hilbert_mca_rot.scores(normalized=True)
            # hom_pats, hom_pats_pvals = hilbert_mca_rot.homogeneous_patterns()
            # het_pats, het_pats_pvals = hilbert_mca_rot.heterogeneous_patterns()
            sq_cov_frac = hilbert_mca_rot.squared_covariance_fraction()
            # Reconstruct the original data from the scores and components
            # reconstructed_vars = hilbert_mca_rot.inverse_transform(X=scores[0], Y=scores[1])

            if clip == True:
                # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_east_greenland.nc"
                suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_amjjaso_{start_year}to{end_year}_xeofs_east_greenland.nc"
            else:
                if power == 1:
                    suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_varimax_total_mode_{total_mode}_rotation_mode_{rotation_mode}.nc"
                    # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_varimax_{total_mode}_{rotation_mode}.nc"
                    # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_monthly_mean_{start_year}to{end_year}_xeofs_varimax.nc"
                elif power > 1:
                    suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_promax_{power}_total_mode_{total_mode}_rotation_mode_{rotation_mode}.nc"
                    # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_promax_total_mode_{total_mode}_rotation_mode_{rotation_mode}.nc"
                    # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_promax_{total_mode}_{rotation_mode}.nc"
                    # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_monthly_mean_{start_year}to{end_year}_xeofs_promax.nc"
                # Original suffixes for saving other outputs (commented out):
                # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs.nc"
                # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_amjjaso_{start_year}to{end_year}_xeofs.nc"
                # suffix = f"_{left_var_name}_{coarsen_level_left}_vs_{right_var_name}_{coarsen_level_right}_mjjas_{start_year}to{end_year}_xeofs_detrended.nc"

            # logger.info('Saving singular vectors...')
            # save_da(da=singular_vectors[0], name='left_components', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=singular_vectors[1], name='right_components', base_output_path=base_output_path, suffix=suffix)

            logger.info('Saving component amplitudes...')
            save_da(da=component_amplitudes[0], name='left_component_amplitudes', base_output_path=base_output_path, suffix=suffix)
            save_da(da=component_amplitudes[1], name='right_component_amplitudes', base_output_path=base_output_path, suffix=suffix)

            logger.info('Saving component phases...')
            save_da(da=component_phases[0], name='left_component_phases', base_output_path=base_output_path, suffix=suffix)
            save_da(da=component_phases[1], name='right_component_phases', base_output_path=base_output_path, suffix=suffix)

            logger.info('Saving scores...')
            save_da(da=scores[0], name='left_scores', base_output_path=base_output_path, suffix=suffix)
            save_da(da=scores[1], name='right_scores', base_output_path=base_output_path, suffix=suffix)

            # logger.info('Saving normalized scores...')
            # save_da(da=scores_normalized[0], name='left_scores_normalized', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=scores_normalized[1], name='right_scores_normalized', base_output_path=base_output_path, suffix=suffix)

            logger.info('Saving squared covariance fraction...')
            # save_da(da=sq_cov_frac_unrotated, name=sq_cov_frac_unrotated.name, base_output_path=base_output_path, suffix=suffix)
            save_da(da=sq_cov_frac, name=sq_cov_frac.name, base_output_path=base_output_path, suffix=suffix)

            # logger.info('Saving reconstructed variables...')
            # save_da(da=reconstructed_vars[0], name='left_reconstructed', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=reconstructed_vars[1], name='right_reconstructed', base_output_path=base_output_path, suffix=suffix)

            # logger.info('Saving heterogeneous correlations (Pearson)...')
            # for het in het_pats: save_da(da=het, name=f"pearson_{het.name}", base_output_path=base_output_path, suffix=suffix)
            # for hetp in het_pats_pvals: save_da(da=hetp, name=f"pearson_{hetp.name}", base_output_path=base_output_path, suffix=suffix)

            # logger.info('Calculating heterogeneous correlations (Spearman)...')
            # left_het_spearman  = xs.spearman_r(left_var_da,  scores[1], dim='time').rename('spearman_left_heterogeneous_patterns')
            # left_het_spearman_pval = xs.spearman_r_p_value(left_var_da,  scores[1], dim='time').rename('spearman_pvalues_of_left_heterogeneous_patterns')
            # right_het_spearman = xs.spearman_r(right_var_da, scores[0], dim='time').rename('spearman_right_heterogeneous_patterns')
            # right_het_spearman_pval = xs.spearman_r_p_value(right_var_da, scores[0], dim='time').rename('spearman_pvalues_of_right_heterogeneous_patterns')

            # save_da(da=left_het_spearman, name='spearman_left_heterogeneous_patterns', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=left_het_spearman_pval, name='spearman_pvalues_of_left_heterogeneous_patterns', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=right_het_spearman, name='spearman_right_heterogeneous_patterns', base_output_path=base_output_path, suffix=suffix)
            # save_da(da=right_het_spearman_pval, name='spearman_pvalues_of_right_heterogeneous_patterns', base_output_path=base_output_path, suffix=suffix)

            # --- End timer ---
            total_time = timedelta(seconds=time.time() - overall_start)
            logger.info(f"Complex Rotated MCA completed for {left_var_name} vs {right_var_name} "
                        f"in {total_time} (finished {time.strftime('%Y-%m-%d %H:%M:%S')})")
        # %%

# # Uncomment arguments for testing
# left_var_name = 'ci'
# right_var_name = 'smb'
# start_year = 1990
# end_year = 1990
# total_mode = 3
# coarsen_level_left = 0
# coarsen_level_right = 0
# clip = True

# %%
# %%
# Testing reconstruction at a specific point (e.g., Freya Glacier location)
# freya_point_X, freya_point_Y = 4.4, 3.0000000000000004
# left_da_freya = left_var_da.sel(rlon=freya_point_X, rlat=freya_point_Y, method='nearest')
# left_reconstructed_at_freya = reconstructed_vars[0].sel(rlon=freya_point_X, rlat=freya_point_Y, method='nearest')

# right_da_freya = right_var_da.sel(rlon=freya_point_X, rlat=freya_point_Y, method='nearest')
# right_reconstructed_at_freya = reconstructed_vars[1].sel(rlon=freya_point_X, rlat=freya_point_Y, method='nearest')

# fig, ax = plt.subplots(2, 1, figsize=(6, 12))
# ax[0].plot(left_da_freya.coords['time'], left_da_freya.values, label='Original Left Data (Freya)', color='red', ls=':', alpha=0.5)
# ax[0].plot(left_reconstructed_at_freya.coords['time'], left_reconstructed_at_freya.real.values, label='Reconstructed Left Data (Freya)', color='darkred', alpha=0.5)
# ax[1].plot(right_da_freya.coords['time'], right_da_freya.values, label='Original Right Data (Freya)', color='blue', ls=':', alpha=0.5)
# ax[1].plot(right_reconstructed_at_freya.coords['time'], right_reconstructed_at_freya.real.values, label='Reconstructed Right Data (Freya)', color='darkblue', alpha=0.5)


# %%
# # Testing saving and loading complex DataArray
# singular_vectors[0].to_netcdf('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/test_left_components_try1.nc', auto_complex=True)
# da = xr.open_dataarray('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/test_left_components_try1.nc')

# def reconstruct_complex(da):
#     if isinstance(da.dtype, np.dtype) and da.dtype.names == ('r', 'i'):
#         return xr.DataArray(
#             da.data['r'] + 1j * da.data['i'],
#             coords=da.coords,
#             dims=da.dims,
#             attrs=da.attrs)
#     return da

# # Example usage:
# singular_vectors_complex = reconstruct_complex(da)
# singular_vectors_complex.real.sel(mode=1).plot()

# Non complex number saving test
# sq_cov_frac.to_netcdf('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/test1.nc', auto_complex=True)
# da = xr.open_dataarray('/nas/data/processed/people/sos/third_paper_data/freyagl_svd_analysis/using_complex_xeofs/test1.nc')
