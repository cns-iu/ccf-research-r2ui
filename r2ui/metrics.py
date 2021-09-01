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


def calculate_global_warping(static, mapping):
    fwd = mapping.get_forward_field()
    fwd_nii = ants.from_numpy(fwd, has_components=True)
    static_f = static.astype(float)
    static_nii = ants.from_numpy(static_f)
    ants.image_write(fwd_nii,'/tmp/warped_fwd_transform.nii.gz')
    jac = ants.create_jacobian_determinant_image(static_nii,"/tmp/warped_fwd_transform.nii.gz",1,0)
    jac_val = np.sum(np.abs(jac.numpy()))
    
    return jac_val

def calculate_metrics(static, moving, f):
    start_time = time()
    fi = static #.clone('float').numpy()
    mi = moving #.clone('float').numpy()
    ssim = s_sim(fi, mi)
    mutualinfo = 0.0 #ants.image_mutual_information(static, moving)
    psnrval = psnr(fi,mi, data_range=255.0)
    hdis = h_dis(fi, mi)
    mseval = mse(fi, mi)
    nrmseval = nrmse(fi, mi)
    end_time = time()
    print(f"Calculate metrics time: {end_time - start_time} seconds", file=f)
    return ssim, mutualinfo, psnrval, hdis, mseval, nrmseval