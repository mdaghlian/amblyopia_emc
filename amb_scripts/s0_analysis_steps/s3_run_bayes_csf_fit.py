#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import getopt

import numpy as np
from datetime import datetime, timedelta
import time
import os
import sys
import warnings
import pickle

warnings.filterwarnings('ignore')
opj = os.path.join
from prfpy.model import CSenFModel

from amb_scripts.load_saved_info import *
# from amb_scripts.utils import *
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

Fit the real time series using the gaussian and normalisation model
- Specify task [AS0,AS1,AS2]
- ROI 
- Fitter
- Design matrix

Args:
    -s (--sub=)         e.g., 01
    -n (--ses=)
    -t (--task=)        CSFLE,CSFRE
    -r (--roi_fit=)     all, V1_exvivo
    --n_walkers         number of walkers
    --n_steps           number of steps
    --nr_jobs           number of jobs (number of voxels to run per time)
    --ow                overwrite
Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # Always    
    verbose = True
    csf_out = 'bayes_csf'    
    model = 'bayes_csf'
    clip_start = 0

    # Specify
    sub = None
    ses = None
    task = None
    roi_fit = None
    n_steps = None
    n_walkers = None

    nr_jobs = 1
    ow = False
    try:
        opts = getopt.getopt(argv,"h:s:n:t:r:",[
            "help=", "sub=","ses=", "task=", "roi_fit=", "csf_out=","nr_jobs=",
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
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("--csf_out"):
            csf_out = arg        
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
        
    csf_dir = opj(derivatives_dir, csf_out)
    if not os.path.exists(csf_dir): 
        os.mkdir(csf_dir)    
    # CREATE THE DIRECTORY TO SAVE THE PRFs
    if not os.path.exists(opj(csf_dir, sub)): 
        os.mkdir(opj(csf_dir, sub))
    if not os.path.exists(opj(csf_dir, sub, ses)): 
        os.mkdir(opj(csf_dir, sub, ses))    

    outputdir = opj(csf_dir, sub, ses)
    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-bayes_fits"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    # Update specific parameters...
    settings['sub'] = sub
    settings['task'] = task
    settings['model'] = model
    settings['roi_fit'] = roi_fit
    settings['nr_jobs'] = nr_jobs
    settings['ses'] = ses
    settings['verbose'] = verbose
    settings['out'] = out 
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
    csf_dm = amb_load_dm(['sf_vect', 'c_vect'])
    sf_vect = csf_dm['sf_vect'][clip_start::]
    c_vect = csf_dm['c_vect'][clip_start::]

    # Number of stimulus types:
    u_sfs = np.sort(list(set(sf_vect))) # unique SFs
    u_sfs = u_sfs[u_sfs>0]
    u_con = np.sort(list(set(c_vect)))
    u_con = u_con[u_con>0]
    CSF_stim = CSenFStimulus(
        SFs = u_sfs,
        CONs = u_con,
        SF_seq=sf_vect,
        CON_seq = c_vect,
        TR=settings['TR'],
    )
    # ************************************************************************

    
    # Check for old parameters:
    old_params = dag_find_file_in_folder([sub, model, task, roi_fit], outputdir, return_msg=None)
    if (old_params is not None) and (not ow):
        print(f'Already done {old_params}, not overwriting')
        sys.exit()        
    elif (old_params is not None) and ow:
        print('Old params exists, but overwriting')
    
    print('Starting MCMC bayesian fitting...')
    print(f'Num walkers = {n_walkers}, n_steps={n_steps}, n_vox at a time = {nr_jobs}')
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE MODEL & fitter
    bounds = [
        (settings['csf_bounds']['width_r']),     # width_r
        (settings['csf_bounds']['sf0']),     # sf0
        (settings['csf_bounds']['maxC']),    # maxC
        (settings['csf_bounds']['width_l']),     # width_l
        (settings['csf_bounds']['beta']),   # beta
        (settings['csf_bounds']['baseline']),      # baseline
        (settings['hrf']['deriv_bound']),
        (settings['hrf']['disp_bound'])
        ]
    bprf_kwargs = {}
    bprf_kwargs['init_walker_method'] = 'gauss_ball'
    # Reasonable starting point...
    bprf_kwargs['init_walker_ps'] = [
        1.5, 2, 40, # width_r, sf0, maxC
        0.447,1,0,  # width_l, amp_1, bold_baseline
        1,0         # hrf_deriv, hrf_disp
        ] 
    bprf_kwargs['gauss_ball_jitter'] = 1 # 
    # bprf_kwargs['gauss_ball_jitter'] = .1*np.array([1, 2, 52, 6, 1]) # based on std
    CSF_bprf = BayesPRF(
        model='csf', 
        prfpy_stim=CSF_stim, 
        **bprf_kwargs
        ) # BPRF model
    CSF_bprf.add_priors_from_bounds(bounds)
    CSF_bprf.prep_info()
    # set jitter proportional to bounds...
    # pjitter = 1

    # gball_jitter = []
    # # for p in CSF_bprf.fit_p_list:
    # #     gball_jitter.append(pjitter * (CSF_bprf.bounds[p][1] - CSF_bprf.bounds[p][0]))
    # gball_jitter = pjitter * np.array([1.5, 2, 40, 1, 1])
    # bprf_kwargs['gauss_ball_jitter'] = gball_jitter
    # CSF_bprf.gauss_ball_jitter = gball_jitter
    # # CSF_bprf.gauss_ball_jitter = 10
    # ***
    print(f'Initialised: {CSF_bprf.init_walker_method}')
    if CSF_bprf.init_walker_method == 'gauss_ball':
        print(f'Jitter about: {CSF_bprf.init_walker_ps}')
        print(f'with jitter {CSF_bprf.gauss_ball_jitter}')
    print(f'Fixed parameters: {CSF_bprf.fix_p_list}')
    print(f'Fitting parameters: {CSF_bprf.fit_p_list}')


    i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'Starting bayes fit {i_start_time}, doing {nr_jobs} voxels at a time...')
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
                [(ivx, tc_data[ivx,:], CSF_bprf, n_walkers, n_steps) for ivx in batch_vx]
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
    bparams['settings'] = settings
    bparams['bounds'] = bounds
    bparams['samples'] = samples_pvx
    bparams['bprf_kwargs'] = bprf_kwargs
    bparams['roi_mask'] = roi_mask
    bparams['init_walker_ps'] = roi_mask
    bparams['start_time'] = i_start_time
    bparams['end_time'] = i_end_time    
    pkl_file = opj(outputdir, f'{out}_desc-bayes-csf_params.pkl')
    f = open(pkl_file, "wb")
    pickle.dump(bparams, f)
    f.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])



