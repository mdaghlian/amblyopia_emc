#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import os
opj = os.path.join
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from amb_scripts.load_saved_info import *
from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.plot_functions import *
from dag_prf_utils.utils import *
from dag_prf_utils.mesh_maker import *
from dag_prf_utils.cmap_functions import *

fs_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/freesurfer'

sub = sys.argv[1]

n_verts = dag_load_nverts(sub=sub, fs_dir=fs_dir)
total_num_vx = np.sum(n_verts)    

# Else look for rois in subs freesurfer label folder
roi_dir = opj(fs_dir, sub, 'label')    
b14pol_str = 'benson14_angle-0001.label'
b14pol_file = {}
b14pol_file['lh'] = dag_find_file_in_folder([b14pol_str, 'lh'], roi_dir, recursive=True)    
b14pol_file['rh'] = dag_find_file_in_folder([b14pol_str, 'rh'], roi_dir, recursive=True)    

b14ecc_str = 'benson14_eccen-0001.label'
b14ecc_file = {}
b14ecc_file['lh'] = dag_find_file_in_folder([b14ecc_str, 'lh'], roi_dir, recursive=True)    
b14ecc_file['rh'] = dag_find_file_in_folder([b14ecc_str, 'rh'], roi_dir, recursive=True)    



LR_pol = []
LR_ecc = []
for i,hemi in enumerate(['lh', 'rh']):
    with open(b14pol_file[hemi]) as f:
        contents = f.readlines()            
    idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
    idx_int = [int(idx_str[i]) for i in range(len(idx_str))]
    val_str = [contents[i].split(' ')[-1].split('\n')[0] for i in range(2,len(contents))]
    val_float = [float(val_str[i]) for i in range(len(val_str))]
    this_pol = np.zeros(n_verts[i]) *np.nan
    this_pol[idx_int] = val_float
    LR_pol.append(this_pol)

    # ECC
    with open(b14ecc_file[hemi]) as f:
        contents = f.readlines()            
    idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
    idx_int = [int(idx_str[i]) for i in range(len(idx_str))]
    val_str = [contents[i].split(' ')[-1].split('\n')[0] for i in range(2,len(contents))]
    val_float = [float(val_str[i]) for i in range(len(val_str))]
    this_ecc = np.zeros(n_verts[i]) *np.nan
    this_ecc[idx_int] = val_float
    LR_ecc.append(this_ecc)    

LR_pol = np.concatenate(LR_pol)
LR_ecc = np.concatenate(LR_ecc)

#
LR_mask = np.round(LR_pol, decimals=5) != 0.00000
LR_mask &= LR_ecc<=5
print(LR_mask.mean())

fs = FSMaker(sub, fs_dir)
fs.add_surface(
    data=LR_pol,
    data_mask = LR_mask,
    vmin = 0, vmax=180, # min and max values of polar anlge 
    cmap = 'pol_simple', # using hsv for polar angle, can use something else...
    cmap_nsteps=100,        
    surf_name='b14pol_ref',)

fs.add_surface(
    data=LR_ecc,
    data_mask = LR_mask,
    vmin = 0, vmax=5, # min and max values of polar anlge 
    cmap = 'ecc', # using hsv for polar angle, can use something else...
    cmap_nsteps=100,        
    surf_name='b14ecc_ref',)

fs.open_fs_surface(
    fs.surf_list,
    roi_list=['b14_V1', 'b14_V2', 'b14_V3.', 'b14_V3a', 'b14_V3b', 'b14_LO1', 'b14_LO2'],    
    roi_col_spec = "white",
    )

    