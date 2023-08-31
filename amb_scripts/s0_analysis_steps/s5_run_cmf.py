#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S4 *********
Bayesian prf fittting all models
'''

import getopt

import numpy as np
import os
import sys
import warnings
import yaml
import pickle
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')
opj = os.path.join

from amb_scripts.load_saved_info import *
from dag_prf_utils.utils import *
from dag_prf_utils.prfpy_functions import *
from dag_prf_utils.pyctx import *

import multiprocessing

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'
settings_path = get_yml_settings_path()

def main(argv):

    """
---------------------------------------------------------------------------------------------------

Fit the real time series using prf model
- ROI 
- Fitter
- Design matrix

Args:
    -s (--sub=)         e.g., 01
    -n (--ses=)
    -t (--task=)        
    --ow                overwrite
Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # ALWAYS
    verbose = True
    prf_out = 'cmf_est'    
    clip_start = 0
    model = 'gauss'
    roi_fit = 'all'
    th = {'min-rsq':.1, 'max-ecc':5} # threshold 
    closest_x = 10
    min_neighbours = 6
    n_threads = 10
    # Specify
    sub = None
    ses = None
    task = None
    ow = False

    try:
        opts = getopt.getopt(argv,"h:s:n:t:m:r:",[
            "help=", "sub=","ses=", "task=", "ow"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            sub = dag_hyphen_parse('sub', arg)
        elif opt in ("-n", "--ses"):
            ses = dag_hyphen_parse('ses', arg)            
        elif opt in ("-t", "--task"):
            task = arg 
        elif opt in ("--ow"):
            ow = True

    if len(argv) < 1:
        print("NOT ENOUGH ARGUMENTS SPECIFIED")
        print(main.__doc__)
        sys.exit()

    prf_dir = opj(derivatives_dir, prf_out)

    # CREATE THE DIRECTORY TO SAVE THE PRFs
    if not os.path.exists(prf_dir): 
        os.mkdir(prf_dir)
    if not os.path.exists(opj(prf_dir, sub)): 
        os.mkdir(opj(prf_dir, sub))
    if not os.path.exists(opj(prf_dir, sub, ses)): 
        os.mkdir(opj(prf_dir, sub, ses))    

    outputdir = opj(prf_dir, sub, ses)
    out = f"{sub}_gauss_{task}-cmf"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD FITS
    n_vox = amb_load_nverts(sub)
    prf_data = amb_load_prf_params(
        sub=sub,
        task_list=task,
        model_list=model, 
        ses=ses    
        )[task][model]
    prf_obj = Prf1T1M(prf_data, model)
    
    vx_mask = prf_obj.return_vx_mask(th) # mask
    # ****************************************************


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   LOAD GDIST
    gdist_obj = PyctxSurf(sub=sub) # Load gdist (to calculate geodistance)

    # ************************************************************************
    print('Starting CMF calculation...')
    start = time.time()
    # mdist_prf -> mean distance (deg) to surrounding prfs...
    mdist_prf = np.zeros(gdist_obj.total_n_vx) * np.nan
    # mdist_ctx -> mean distance (mm) to surrounding prfs...
    mdist_ctx = np.zeros(gdist_obj.total_n_vx) * np.nan
    # cmf -> cortical magnification (mdiff_ctx / mdiff_prf)
    cmf = np.zeros(gdist_obj.total_n_vx) * np.nan
    
    # Get the vx mask
    vx_mask = prf_obj.return_vx_mask(th)
    # Calculate the distance of 10 closest vx, & their indices
    hemi_dists = gdist_obj.calculate_gdists(vx_mask,closest_x, n_threads=n_threads)
    print('Done geodesic calculation')
    # Put them togther in one list
    close_val = []
    close_ivx = []
    for hemi in ['lh', 'rh']:
        close_val += [i['close_val'] for i in hemi_dists[hemi]]
        close_ivx += [i['close_ivx'] for i in hemi_dists[hemi]]

    # Now loop through all the relevant vx
    ivx_list = np.where(vx_mask)[0]
    for i,ivx in enumerate(ivx_list):
        this_close_val = close_val[i]
        this_close_ivx = close_ivx[i]

        # Mask out those close vertices which don't meet the original th
        this_close_mask = vx_mask[this_close_ivx]
        if this_close_mask.sum()<min_neighbours: # only continue if we have enough valid neighbours
            continue
        this_close_val = this_close_val[this_close_mask]
        this_close_ivx = this_close_ivx[this_close_mask]        
        # target x,y
        tx,ty = prf_obj.pd_params['x'][ivx],prf_obj.pd_params['y'][ivx]
        # neighbour x,y
        sx,sy = prf_obj.pd_params['x'][this_close_ivx],prf_obj.pd_params['y'][this_close_ivx]            
        mdist_prf[ivx] = np.mean(np.sqrt((tx-sx)**2 + (ty-sy)**2))
        mdist_ctx[ivx] = np.mean(this_close_val)

        cmf[ivx] = mdist_ctx[ivx] / mdist_prf[ivx]

    elapsed = (time.time() - start)
    print(f'CMF calculation took {timedelta(seconds=elapsed)}')
    print(f'Percent of vx inside vx mask, with enough neigbours: {(np.isnan(mdist_prf[vx_mask])==0).mean()*100 : .3f}')
    # Save everything
    cmf_info = {}
    cmf_info['cmf'] = cmf
    cmf_info['mdist_ctx'] = mdist_ctx
    cmf_info['mdist_prf'] = mdist_prf
    pkl_file = opj(outputdir, f'{out}.pkl')
    f = open(pkl_file, "wb")
    pickle.dump(cmf_info, f)
    f.close()    
if __name__ == "__main__":
    main(sys.argv[1:])
