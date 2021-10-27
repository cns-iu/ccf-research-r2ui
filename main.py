#!/usr/local/bin/python3
"""
Author: Yashvardhan Jain (github.com/J-Yash)
"""

import os
import pickle
from datetime import datetime
from time import time

import ants
import dipy
import imutils
import numpy as np
import psutil
import scipy
from dipy.align import (affine, affine_registration, center_of_mass,
                        register_dwi_to_template, rigid, translation)
from dipy.align.imaffine import (AffineMap, AffineRegistration,
                                 MutualInformationMetric,
                                 transform_centers_of_mass)
from dipy.align.imwarp import (DiffeomorphicMap,
                               SymmetricDiffeomorphicRegistration)
from dipy.align.metrics import CCMetric
from dipy.align.transforms import (AffineTransform3D, RigidTransform3D,
                                   TranslationTransform3D)
from dipy.core.gradients import gradient_table
from dipy.data import fetch_stanford_hardi
from dipy.data.fetcher import fetch_syn_data
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti
from dipy.viz import regtools
from skimage.io import imread, imsave
from skimage.metrics import hausdorff_distance as h_dis
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as s_sim
from skimage.util import crop, pad

import r2ui
from r2ui.utils import image_read, get_image_crop
from r2ui.registration import sliding_registration

def run_registration(data_filepath, results_filepath, pickle_filepath, block_size=50):
    # Open a file to write results
    f = open(results_filepath, "a+")
    print(f"+++++ NEW EXPERIMENT +++++ : Started at: {datetime.now()}", file=f)

    #print(f"ITK Threads: {num_threads}", file=f)
    # Load Data
    PATH = data_filepath
    print(f"Data file used: {PATH}", file=f)
    reference_image = image_read(PATH, plugin='tifffile')

    print(reference_image, file=f)

    # Get central patch (tissue block)
    crop_size = block_size
    x1_ref = 85
    y1_ref = 311
    z1_ref = 372
    moving_image = get_image_crop(reference_image, x1_ref, y1_ref, z1_ref, crop_size)
    print(f"Tissue block size: {crop_size}", file=f)
    print(f"Position of cropped image(tissue block): ({x1_ref}, {y1_ref}, {z1_ref})", file=f)

    
    print(f"Starting Sliding Registration", file=f)
    metrics, reg_dict = sliding_registration(reference_image,moving_image, crop_size, x1_ref, y1_ref, z1_ref, f)
    

    metrics_sorted_jac = sorted(metrics, key=lambda x: x['jac'], reverse=False)
    print(f"Best Position based on Jac: {metrics_sorted_jac[0]}", file=f)

    metrics_sorted_ssim = sorted(metrics, key=lambda x: x['ssim'][3], reverse=True)
    print(f"Best Position based on SSIM: {metrics_sorted_ssim[0]}", file=f)

    # Pickling metrics dict and image dict
    with open(pickle_filepath, 'wb') as handle:
        pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"+++++ END OF EXPERIMENT +++++ : End at {datetime.now()}", file=f)

    f.close()

if __name__ == "__main__":

    block_size = 50
    #reg_design = 2
    data = "tif"

    print(f"Running KPMP {data}_{block_size}")

    data_filepath = f'../data/kpmp_data.tif'
    res_filepath = f"../exp_results/dipy_results_{block_size}_{data}_gpu.txt" # blocksize_mode_data.txt
    pickle_filepath = f'../exp_results/dipy_all_metrics_{block_size}__{data}_gpu.pickle'
    
    run_registration(data_filepath, res_filepath, pickle_filepath, block_size)

    """block_size = 50
    #reg_design = 3
    #data = "stl_all_surface"

    print(f"Running KPMP {data}_{block_size}")

    #data_filepath = f'./Allen_brain_data/sym_b/{data}.nii'
    res_filepath = f"./exp_results/kpmp/results_{block_size}_{reg_design}_{data}.txt" # blocksize_mode_data.txt
    pickle_filepath = f'./exp_results/kpmp/all_metrics_{block_size}_{reg_design}_{data}.pickle'
    
    run_registration(data_filepath, res_filepath, pickle_filepath, block_size, mode=reg_design)

    ##
    block_size = 100
    #reg_design = 3
    #data = "pd"

    print(f"Running KPMP {data}_{block_size}")

    #data_filepath = f'./Allen_brain_data/sym_b/{data}.nii'
    res_filepath = f"./exp_results/kpmp/results_{block_size}_{reg_design}_{data}.txt" # blocksize_mode_data.txt
    pickle_filepath = f'./exp_results/kpmp/all_metrics_{block_size}_{reg_design}_{data}.pickle'
    
    run_registration(data_filepath, res_filepath, pickle_filepath, block_size, mode=reg_design)"""

    print("COMPLETE!!!")

