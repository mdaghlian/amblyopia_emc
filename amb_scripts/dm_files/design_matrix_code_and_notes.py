#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
Files in dm_files:

These are the same for all runs & subjects

*** CSF relevant info: ***
> TR=1.5; Number of scans=214
> 2 relevant files:
>> copied from 'sourcedata/sub-0#/BEH/CSF#E/***.mat' [# specifics doesn't matter, all the same]
params.sf_vect          -> saved as 'sf_vect.mat'
params.contrasts_vect   -> saved as 'contrasts_vect.mat'
> sf_vect.mat                   list of spatial frequencies, for each TR (214 values)
> contrasts_vect.mat            list of contrast values, for each TR (214 values)

*** PRF relevant info ***
> TR=1.5; Number of scans = 224
>> copied from 'sourcedata/sub-0#/BEH/pRF#E/***.mat' [# specifics doesn't matter, all the same]
stimulus.seq            -> saved as 'prf_seq.mat'
stimulus.seqtiming      -> saved as 'prf_seqtiming.mat'
stimulus.images         -> saved as 'prf_images.mat'

> prf_images.mat        -> 1080 x 1080 x 801, different frames/textures for PRF bar
> prf_seq               -> order of the frames presented (each is an index for prf_images) 
> prf_seqtiming         -> the timing for each frame (in prf_seq)

This script is to create the PRF design matrix
[1] Only select every 30 frames of prf_images (only the images for each TR) (1.5/0.05=30)
[2] New dm is 1080 x 1080 x 224
[3] Binarize (0,1)
[4] Downsample 
[5] Save as prf_design_matrix.npy

'''

import numpy as np
import os 
import scipy.io
from linescanning.utils import resample2d

prf_images = np.squeeze(scipy.io.loadmat('prf_images.mat')['prf_images'])           # 1080 x 1080 x 801, different frames/textures for PRF bar
prf_seq = np.squeeze(scipy.io.loadmat('prf_seq.mat')['prf_seq'])                    # order of the frames presented (each is an index for prf_images) 
prf_seqtiming = np.squeeze(scipy.io.loadmat('prf_seqtiming.mat')['prf_seqtiming'])  # the timing for each frame (in prf_seq)

n_pix = 100 # NUMBER OF PIXELS FOR DOWNSAMPLED DM
total_n_frames = prf_seq.shape[0]
relevant_seq_id = np.arange(0,total_n_frames, 30)
relevant_frames_id = prf_seq[relevant_seq_id] - 1 # *** -1 because the seq was written in matlab
hires_tr_images = prf_images[:,:,relevant_frames_id]
hires_binary_dm = (hires_tr_images!=128)*1.0 #keep as float
prf_design_matrix = resample2d(hires_binary_dm, n_pix) # Downsample

np.save('prf_design_matrix', prf_design_matrix)


'''
Now to make the CSF design matrix
'''