#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S4 *********
Bayesian prf fittting all models
'''

import getopt

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel,CSS_Iso2DGaussianModel, DoG_Iso2DGaussianModel, Norm_Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter, CSS_Iso2DGaussianFitter, DoG_Iso2DGaussianFitter,Norm_Iso2DGaussianFitter

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

from prfpy_bayes.utils import *
from prfpy_bayes.prf_bayes import *

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
    -m (--model=)       e.g., norm, css, dog
    -s (--sub=)         e.g., 01
    -n (--ses=)
    -t (--task=)        
    -r (--roi_fit=)     all, V1_exvivo
    --n_walkers         number of walkers
    --n_steps           number of steps
    --nr_jobs           number of jobs (number of voxels to run per time)
    --ow                overwrite
Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # ALWAYS
    verbose = True
    prf_out = 'bayes_prf'    
    clip_start = 0

    # Specify
    sub = None
    ses = None
    task = None
    model = None
    roi_fit = None
    n_steps = None
    n_walkers = None

    nr_jobs = 1
    ow = False

    try:
        opts = getopt.getopt(argv,"h:s:n:t:m:r:",[
            "help=", "sub=","ses=", "task=", "model=", "roi_fit=", "prf_out=","nr_jobs=",
            "n_steps=", "n_walkers=", "ow"])[0]
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
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("--prf_out"):
            prf_out = arg        
        elif opt in ("--nr_jobs"):
            nr_jobs = int(arg)            
        elif opt in ("--n_walkers"):
            n_walkers = int(arg)            
        elif opt in ("--n_steps"):
            n_steps = int(arg)            
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
    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-bayes_fits"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    # load basic settings from the yml file
    prf_settings_file = settings_path#opj(os.path.dirname(os.path.abspath(__file__)), 's0_prf_analysis.yml') 
    with open(prf_settings_file) as f:
        prf_settings = yaml.safe_load(f)    
    # Add important info to settings
    prf_settings['sub'] = sub
    prf_settings['task'] = task
    prf_settings['model'] = model
    prf_settings['roi_fit'] = roi_fit
    prf_settings['nr_jobs'] = nr_jobs
    prf_settings['ses'] = ses
    prf_settings['verbose'] = verbose
    prf_settings['out'] = out 
    # ****************************************************

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD TIME SERIES & MASK THE ROI   
    num_vx = np.sum(amb_load_nverts(sub=sub))
    tc_data = amb_load_real_tc(sub=sub, ses=ses, task_list=task, clip_start=clip_start)[task]
    roi_mask = amb_load_roi(sub=sub, label=roi_fit)
    # Are we limiting the fits to an roi?
    num_vx_in_roi = roi_mask.sum()
    print(f'Fitting {roi_fit} {num_vx_in_roi} voxels out of {num_vx} voxels in total')
    print('Removing irrelevant times courses')
    print('These will be added in later, as zero fits in the parameters')
    zero_pad = False
    tc_data = mask_time_series(ts=tc_data, mask=roi_mask, zero_pad=zero_pad)
    # ************************************************************************


    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<   LOAD DESIGN MATRIX   
    design_matrix = amb_load_dm('prf')['prf']
    prf_stim = PRFStimulus2D(
        screen_size_cm=prf_settings['screen_size_cm'],          # Distance of screen to eye
        screen_distance_cm=prf_settings['screen_distance_cm'],  # height of the screen (i.e., the diameter of the stimulated region)
        design_matrix=design_matrix,                            # dm (npix x npix x time_points)
        TR=prf_settings['TR'],                                  # TR
        )   
    max_eccentricity = prf_stim.screen_size_degrees/2 # It doesn't make sense to look for PRFs which are outside the stimulated region 
    # ************************************************************************
    
    # Check for old parameters:
    old_params = dag_find_file_in_folder([sub, model, task, roi_fit], outputdir, exclude='.txt', return_msg=None)
    if (old_params is not None) and (not ow):
        print(f'Already done {old_params}, not overwriting')
        sys.exit()        
    elif (old_params is not None) and ow:
        print('Old params exists, but overwriting')
    
    print('Starting MCMC bayesian fitting...')
    print(f'Num walkers = {n_walkers}, n_steps={n_steps}, n_vox at a time = {nr_jobs}')
    
    bprf_kwargs = {}
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE MODEL & fitter
    if model=='gauss': # ******************************** GAUSS
        extra_bounds = []
        bprf_kwargs['init_walker_method'] = 'gauss_ball'
        bprf_kwargs['init_walker_ps'] = [0,0,1,0.01,0,1,0] # Reasonable starting point...
    elif model=='norm': # ******************************** NORM
        bprf_kwargs['init_walker_method'] = 'gauss_ball'
        # Reasonable starting point...
        bprf_kwargs['init_walker_ps'] = [
            0,0,1,      #x,y,size_1
            10,0,1,     #amp_1,bold_baseline,amp_2,
            10,50,100,  #size_2,b_val,d_val
            1,0         #hrf_deriv, hrf_disp
            ] # 
        extra_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            (prf_settings['norm']['neural_baseline_bound']),        # neural baseline (b) 
            (prf_settings['norm']['surround_baseline_bound']),      # surround baseline (d)
            ] 
    elif model=='dog': # ******************************** DOG
        extra_bounds = [
            (prf_settings['prf_ampl']),                             # surround amplitude
            (1e-1, max_eccentricity*6),                             # surround size
            ]

    elif model=='css': # ******************************** CSS
        extra_bounds = [
            (prf_settings['css']['css_exponent_bound']),  # css exponent 
            ]
    # Combine the bounds 
    # first create the standard bounds
    standard_bounds = [
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
        (1e-1, max_eccentricity*3),                             # prf size bounds
        (prf_settings['prf_ampl']),                             # prf amplitude
        (prf_settings['bold_bsl']),                             # bold baseline (fixed)
    ]    
    # & the hrf bounds. these will be overwritten later by the vx wise hrf parameters
    # ( inherited from previous fits)
    hrf_bounds = [
        (prf_settings['hrf']['deriv_bound']),                   # hrf_1 bound
        (prf_settings['hrf']['disp_bound']),                    # hrf_2 bound
    ]
    bounds = standard_bounds.copy() + extra_bounds.copy() + hrf_bounds.copy()
    bprf_kwargs['gauss_ball_jitter'] = .1
    # ************************************************************************
    bprf = BayesPRF(
        model=model, 
        prfpy_stim=prf_stim,
        **bprf_kwargs) # BPRF model
    bprf.add_priors_from_bounds(bounds)
    bprf.prep_info()

    print(f'Initialised: {bprf.init_walker_method}')
    if bprf.init_walker_method == 'gauss_ball':
        print(f'Jitter about: {bprf.init_walker_ps}')
        print(f'with jitter {bprf.gauss_ball_jitter}')        
    
    print(f'Fixed parameters: {bprf.fix_p_list}')
    print(f'Fitting parameters: {bprf.fit_p_list}')

    i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'Starting bayes fit for {model} {i_start_time}, doing {nr_jobs} voxels at a time...')
    start = time.time()

    batch_size = nr_jobs
    total_n = tc_data.shape[0] 
    # Loop through the batches of "vx" values
    samples_pvx = []
    for batch_start in range(0, total_n, batch_size):
        batch_end = min(batch_start + batch_size, total_n)
        # Create a process pool based on the number of CPUs specified
        # Process each "vx" in the batch in parallel
        batch_vx = range(batch_start, batch_end)
        with multiprocessing.Pool(processes=nr_jobs) as pool:
            batch_results = pool.starmap(
                process_ivx, 
                [(ivx, tc_data[ivx,:], bprf, n_walkers, n_steps) for ivx in batch_vx]
                )
        # Close the pool and wait for all tasks to complete
        pool.close()
        pool.join()

        # Collect the results from this batch separately and store in batch_samples_pvx
        batch_samples_pvx = []
        batch_samples_pvx.extend(batch_results)

        # Append the results from this batch to the main samples_pvx list
        samples_pvx.extend(batch_samples_pvx)

    i_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'End bayes fit {i_end_time}')           
    elapsed = (time.time() - start)
    print(f'Finished bayes fit {i_end_time}')
    print(f'Took {timedelta(seconds=elapsed)}')
    # # Print mean rsq:
    # Print mean rsq:
    best_rsq = np.zeros(len(samples_pvx)) * np.nan
    for i,this_sample in enumerate(samples_pvx):
        if len(this_sample['rsq'])>0:
            best_rsq[i] = np.nanmax(this_sample['rsq'])
    id_not_nan = np.isnan(best_rsq)==0
    print(f'n not nan ={id_not_nan.sum()}')
    print(f'm rsq ={best_rsq[id_not_nan].mean()}') 
    # Save everything
    bparams = {}
    bparams['settings'] = prf_settings
    bparams['bounds'] = bounds
    bparams['samples'] = samples_pvx
    bparams['bprf_kwargs'] = bprf_kwargs
    bparams['roi_mask'] = roi_mask
    bparams['start_time'] = i_start_time
    bparams['end_time'] = i_end_time
    pkl_file = opj(outputdir, f'{out}_desc-bayes-prf_params.pkl')
    f = open(pkl_file, "wb")
    pickle.dump(bparams, f)
    f.close()    
if __name__ == "__main__":
    main(sys.argv[1:])
