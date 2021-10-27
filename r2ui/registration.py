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
from r2ui.metrics import calculate_global_warping, calculate_metrics

import cupy as cp
from cudipy.align.imwarp import SymmetricDiffeomorphicRegistration as cp_SymmetricDiffeomorphicRegistration
from cudipy.align.metrics import CCMetric as cp_CCMetric


def register_affine(static, moving, static_affine, moving_affine, f):
    #level_iters = [10000, 1000, 100]
    #sigmas = [3.0, 1.0, 0.0]
    #factors = [4, 2, 1]

    level_iters = [1]
    sigmas = [0.0]
    factors = [1]

    pipeline = [center_of_mass, translation, rigid, affine]

    xformed_img, reg_affine = affine_registration(
                                                moving,
                                                static,
                                                moving_affine=moving_affine,
                                                static_affine=static_affine,
                                                nbins=32,
                                                sampling_proportion = None,
                                                metric='MI',
                                                pipeline=pipeline,
                                                level_iters=level_iters,
                                                sigmas=sigmas,
                                                factors=factors)
    
    return xformed_img, reg_affine

def register_diffeomorphic(static, moving, static_affine, moving_affine, reg_affine, f):
    metric_syn = CCMetric(3)
    level_iters_syn = [10, 10, 5]
    
    sdr = SymmetricDiffeomorphicRegistration(metric_syn, level_iters_syn)
    
    mapping = sdr.optimize(static, moving, static_affine, moving_affine, reg_affine)
    warped_moving = mapping.transform(moving)

    mapping = mapping.get_forward_field()
    
    return mapping, warped_moving

def register_diffeomorphic_gpu(static, moving, static_affine, moving_affine, reg_affine, f):
    metric_syn = cp_CCMetric(3)
    level_iters_syn = [10, 10, 5]
    
    sdr = cp_SymmetricDiffeomorphicRegistration(metric_syn, level_iters_syn)
    
    static = cp.asarray(static)
    moving = cp.asarray(moving)
    static_affine = cp.asarray(static_affine)
    moving_affine = cp.asarray(moving_affine)
    reg_affine = cp.asarray(reg_affine)

    mapping = sdr.optimize(static, moving, static_affine, moving_affine, reg_affine)
    warped_moving = mapping.transform(moving)

    mapping = cp.asnumpy(mapping.get_forward_field())
    warped_moving = cp.asnumpy(warped_moving)
    
    return mapping, warped_moving

def register_image(static, moving, f):
    print("Register Image")
    static_affine =  np.eye(4)
    moving_affine =  np.eye(4)

    start_time_1 = time()
    xformed_img, reg_affine = register_affine(static, moving, static_affine, moving_affine, f)
    end_time_1 = time()
    time_affine = end_time_1 - start_time_1
    print(f"Affine Registration time: {time_affine} seconds", file=f)
    print(f"Affine Registration time: {time_affine} seconds")

    start_time_2 = time()
    mapping, warped_moving = register_diffeomorphic_gpu(static, moving, static_affine, moving_affine, reg_affine, f)
    end_time_2 = time()
    time_diffeomorphic = end_time_2 - start_time_2
    print(f"Diffeomorphic Registration time: {time_diffeomorphic} seconds", file=f)
    print(f"Diffeomorphic Registration time: {time_diffeomorphic} seconds")
    
    print(f" Total Single Registration time: {time_affine + time_diffeomorphic} seconds", file=f)
    print(f" Total Single Registration time: {time_affine + time_diffeomorphic} seconds")
    return mapping, warped_moving

def sliding_registration(reference_image, moving_image, patch_size, x1_ref, y1_ref, z1_ref, f):
    print("Start Sliding Registration")
    start_time = time()
    image_dims = [reference_image.shape[0], reference_image.shape[1], reference_image.shape[2]]
    jac_all = []
    reg_dict = {}
    jac_pos = []
    reg_dict_pos = {}
    num = 1
    delta = 3
    
    max_rotation_angle = 3 # in degrees
    mi_np = moving_image #.numpy()
    moving_rotated_max_x = imutils.rotate_bound(mi_np[:,:,0], max_rotation_angle)
    moving_rotated_max_y = imutils.rotate_bound(mi_np[0,:,:], max_rotation_angle)
    moving_rotated_max_z = imutils.rotate_bound(mi_np[:,0,:], max_rotation_angle)

    x_diff = max(abs(moving_rotated_max_x.shape[0] - mi_np.shape[0]),  abs(moving_rotated_max_z.shape[0] - mi_np.shape[0]))
    y_diff = max(abs(moving_rotated_max_x.shape[1] - mi_np.shape[1]),  abs(moving_rotated_max_y.shape[0] - mi_np.shape[1]))
    z_diff = max(abs(moving_rotated_max_y.shape[1] - mi_np.shape[2]),  abs(moving_rotated_max_z.shape[1] - mi_np.shape[2]))

    max_diff = max(x_diff, y_diff, z_diff)
    padding_size = -(max_diff // -2) # trick for ceil division

    # pad both images
    moving_padded_np = pad(mi_np, ((padding_size, padding_size),
                                               (padding_size, padding_size), 
                                               (padding_size, padding_size)),
                                 'constant', constant_values=((0,0),(0,0),(0,0)))
    
    print("Start Position Search")
    for i in range(x1_ref - delta, x1_ref + delta + 1): 
        for j in range(y1_ref - delta, y1_ref + delta + 1): 
            for k in range(z1_ref - delta, z1_ref + delta + 1):
                num += 1
                fixed = get_image_crop(reference_image, i, j, k, patch_size)
                mapping, warped_moving = register_image(fixed, moving_image, f)
                
                #if len(reg['fwdtransforms']) != 0:
                jac = calculate_global_warping(fixed, mapping)
                jac_pos.append((jac, i, j, k))
                reg_dict_pos[(i,j, k)] = (warped_moving,fixed)
    print("End Position Search")
                
    jac_pos.sort(key=lambda x:x[0])
    jac_best = jac_pos[0]
    fixed_best = reg_dict_pos[(jac_best[1], jac_best[2], jac_best[3])][1]
                
    fixed_np = fixed_best #.numpy()
    fixed_padded_np = pad(fixed_np, ((padding_size,padding_size),(padding_size,padding_size),(padding_size,padding_size)),
                     'constant', constant_values=((0,0),(0,0),(0,0)))
                
    print("Start Orientation Search")
    for theta1 in range(-max_rotation_angle, max_rotation_angle+1): # -degrees to +degrees
        moving_rotated1 = scipy.ndimage.rotate(moving_padded_np,angle=theta1,axes=(0,1), reshape=False)
        for theta2 in range(-max_rotation_angle, max_rotation_angle+1): # -degrees to +degrees
            moving_rotated2 = scipy.ndimage.rotate(moving_rotated1,angle=theta2,axes=(1,2), reshape=False)
            for theta3 in range(-max_rotation_angle, max_rotation_angle+1): # -degrees to +degrees
                num += 1
                moving_rotated3 = scipy.ndimage.rotate(moving_rotated2,angle=theta3,axes=(0,2), reshape=False)

                moving_image_to_register = moving_rotated3
                fixed_image_to_register = fixed_padded_np
                mapping, warped_moving = register_image(fixed_image_to_register, moving_image_to_register, f)
                #if len(reg['fwdtransforms']) != 0:
                jac = calculate_global_warping(fixed_image_to_register, mapping)
                
                ssim1, mutualinfo1, psnrval1, hdis1, mseval1, nrmseval1 = calculate_metrics(fixed_image_to_register, fixed_image_to_register, f)
                ssim2, mutualinfo2, psnrval2, hdis2, mseval2, nrmseval2 = calculate_metrics(fixed_image_to_register, moving_image_to_register, f)
                ssim3, mutualinfo3, psnrval3, hdis3, mseval3, nrmseval3 = calculate_metrics(moving_image_to_register, warped_moving, f)
                ssim4, mutualinfo4, psnrval4, hdis4, mseval4, nrmseval4 = calculate_metrics(fixed_image_to_register, warped_moving, f)

                metrics = {'jac': jac, 
                        'pos': (jac_best[1], jac_best[2], jac_best[3], theta1, theta2, theta3), 
                        'ssim': (ssim1, ssim2, ssim3, ssim4), 
                        'mi': (mutualinfo1,mutualinfo2, mutualinfo3, mutualinfo4),
                        'psnr': (psnrval1, psnrval2, psnrval3, psnrval4), 
                        'hdis': (hdis1, hdis2, hdis3, hdis4), 
                        'mse': (mseval1, mseval2, mseval3, mseval4),
                        'nrmse': (nrmseval1, nrmseval2, nrmseval3, nrmseval4)}

                jac_all.append(metrics)
                
                reg_dict[(jac_best[1], jac_best[2], jac_best[3], theta1, theta2, theta3)] = (warped_moving,fixed_image_to_register)
    print("End Orientation Search")
    end_time = time()
    print(f"Sliding Registration (fastest dense) time: {end_time - start_time} seconds", file=f)
    print("End Sliding Registration")
    return jac_all, reg_dict