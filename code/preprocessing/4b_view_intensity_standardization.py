# File: 4b_view_intensity_standardization.py
# Date: 01/23/2024
# Author: Lawrence Chillrud <chili@u.northwestern.edu>
# Description: Inspects the results of intensity standardization on the MRI scans more manually.

#--------------------------#
####      CONTENTS      ####
#--------------------------#
# N. Notes
# 0. Package imports

#--------------------------#
####      N. NOTES      ####
#--------------------------#
# This script is meant to view the results of intensity standardization on the MRI scans in 
# a little more detail than the before and after histograms from script 4a.
# Plots are meant to be viewed in VS Code's interactive window, and then saved from there.
#
# This script relies on the following file(s) as inputs:
#   * data/preprocessing/output/4_INTENSITY_STANDARDIZED/*/*_Brainlab/*/*.nii.gz
#   * data/preprocessing/output/3_N4_BIAS_FIELD_CORRECTED/*/*_Brainlab/*/*.nii.gz

#%%------------------------#
#### 0. PACKAGE IMPORTS ####
#--------------------------#
from utils import setup, lsdir, read_example_mri, plot_histogram
import matplotlib.pyplot as plt
import numpy as np

setup()

def view_intensity_standardization(data_dir='data/preprocessing/output', scan_type='AX_3D_T1_POST', subjects_to_plot=None, num_subjects=4, cmap='gray', fig_height=6):
    """
    Author: Lawrence Chillrud
    
    This function is meant to view the results of intensity standardization on the MRI scans in
    a little more detail than the before and after histograms from script 4a.
    Plots are meant to be viewed in VS Code's interactive window, and then saved from there.

    Parameters
    ----------
    data_dir : str, optional
        The directory where the data is stored. The default is 'data/preprocessing/output'.
    scan_type : str, optional
        The scan type to plot. The default is 'AX_3D_T1_POST'.
    subjects_to_plot : list, optional
        A list of subjects to plot. The default is None, in which case subjects are chosen randomly.
    num_subjects : int, optional
        The number of subjects to plot. The default is 4. If subjects_to_plot is not None, then this
        parameter is ignored.
    cmap : str, optional
        The colormap to use. The default is 'gray'.
    fig_height : int, optional
        The height of the figure. The default is 6.
    """
    before_dir = f'{data_dir}/3_N4_BIAS_FIELD_CORRECTED'
    after_dir = f'{data_dir}/4_INTENSITY_STANDARDIZED'
    subjects = lsdir(after_dir)
    subjects = [s for s in subjects if f'{s}_Brainlab' in lsdir(f'{after_dir}/{s}')]

    if subjects_to_plot is None:
      # pick randomly num_subjects from subjects so long as they have a scan of type scan_type
      all_scans = [' '.join(lsdir(f'{after_dir}/{subject}/{subject}_Brainlab')) for subject in subjects]
      subjects = [s for i, s in enumerate(subjects) if scan_type in all_scans[i]]
      subjects_to_plot = sorted(np.random.choice(subjects, num_subjects, replace=False))
      # print("StP: ", subjects_to_plot)
    else:
      num_subjects = len(subjects_to_plot)
    
    arr_before = []
    arr_after = []
    for i, subject in enumerate(subjects_to_plot):
      session = f'{subject}_Brainlab'
      scans = lsdir(f'{after_dir}/{subject}/{session}')
      # print("Scans: ", scans)
      scan = [s for s in scans if s.endswith(scan_type)][0]
      before = read_example_mri(before_dir, subject, session, f'{scan}', ants=True)
      arr_before.append(before.numpy())
      after = read_example_mri(after_dir, subject, session, f'{scan}', ants=True)
      arr_after.append(after.numpy())
    
    before_min = min([arr.min() for arr in arr_before])
    after_min = min([arr.min() for arr in arr_after])
    before_max = max([arr.max() for arr in arr_before])
    after_max = max([arr.max() for arr in arr_after])

    colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    fig, axs = plt.subplots(2, num_subjects + 1, figsize=(fig_height*(num_subjects + 1), fig_height*2))
    fig.suptitle(f'{scan_type}: Intensity Standardization Before vs. After', fontsize=36, y=1.05)
    for i, subject in enumerate(subjects_to_plot):
      session = f'{subject}_Brainlab'
      scans = lsdir(f'{after_dir}/{subject}/{session}')
      # print("Scans: ", scans)
      scan = [s for s in scans if s.endswith(scan_type)][0]
      before = read_example_mri(before_dir, subject, session, f'{scan}', ants=True)
      after = read_example_mri(after_dir, subject, session, f'{scan}', ants=True)

      slice = before.shape[0] // 2
      axs[0, i].set_title(f'{subject} Before', fontsize=24)
      axs[0, i].set_xticks([])
      axs[0, i].set_yticks([])
      for spine in axs[0, i].spines.values():
        spine.set_edgecolor(colours[i])
        spine.set_linewidth(8)
      im1 = axs[0, i].imshow(before[slice, :, :], cmap=cmap, vmin=before_min, vmax=before_max)
      fig.colorbar(im1, ax=axs[0, i], orientation='vertical', fraction=0.046, pad=0.04)

      slice = after.shape[0] // 2
      axs[1, i].set_title(f'{subject} After', fontsize=24)
      axs[1, i].set_xticks([])
      axs[1, i].set_yticks([])
      for spine in axs[1, i].spines.values():
        spine.set_edgecolor(colours[i])
        spine.set_linewidth(8)
      im2 = axs[1, i].imshow(after[slice, :, :], cmap=cmap, vmin=after_min, vmax=after_max)
      fig.colorbar(im2, ax=axs[1, i], orientation='vertical', fraction=0.046, pad=0.04)

    axs[0, num_subjects].set_title('Before', fontsize=24)
    for i, image in enumerate(arr_before):
      _ = plot_histogram(image, ax=axs[0, num_subjects], alpha=0.8, label=subjects_to_plot[i])

    axs[1, num_subjects].set_title('After', fontsize=24)
    for i, image in enumerate(arr_after):
      _ = plot_histogram(image, ax=axs[1, num_subjects], alpha=0.8, label=subjects_to_plot[i])

    plt.tight_layout()
    plt.show()

view_intensity_standardization()
# %%