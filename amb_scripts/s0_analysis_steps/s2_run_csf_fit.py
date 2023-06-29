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
from joblib import parallel_backend
warnings.filterwarnings('ignore')
opj = os.path.join
from prfpy.model import CSenFModel
from prfpy.fit import CSenFFitter

from amb_scripts.load_saved_info import *
# from amb_scripts.utils import *
from dag_prf_utils.utils import *
from dag_prf_utils.prfpy_functions import *

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
    -t (--task=)        CSFLE,CSFRE
    -r (--roi_fit=)     all, V1_exvivo
    --nr_jobs           number of jobs
    --verbose
    --tc                Which fitter using? tc, lgbfs
    --bgfs
    --hrf                         
    --ow                overwrite
Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # Always    
    verbose = True
    csf_out = 'csf'    
    model = 'csf'
    clip_start = 0

    # Specify
    sub = None
    ses = None
    task = None
    roi_fit = None
    constraints = None
    fit_hrf = False
    nr_jobs = 1
    ow = False
    try:
        opts = getopt.getopt(argv,"h:s:n:t:r:",[
            "help=", "sub=","ses=", "task=", "roi_fit=", "csf_out=","nr_jobs=",
            "verbose", "tc", "bgfs", "hrf", "ow"])[0]
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
        elif opt in ("--verbose"):
            verbose = True
        elif opt in ("--nr_jobs"):
            nr_jobs = int(arg)            
        elif opt in ("--tc"):
            constraints = 'tc'
        elif opt in ("--bgfs"):
            constraints = 'bgfs'
        elif opt in ("--hrf"):
            fit_hrf = True
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
    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-fits"

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD SETTINGS
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    # Update specific parameters...
    settings['sub'] = sub
    settings['task'] = task
    settings['model'] = model
    settings['roi_fit'] = roi_fit
    settings['nr_jobs'] = nr_jobs
    settings['constraints'] = constraints
    settings['ses'] = ses
    settings['fit_hrf'] = fit_hrf
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

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE MODEL & fitter
    # Model 
    CSF_mod = CSenFModel(
        stimulus=CSF_stim,
        hrf=settings['hrf']['pars'],
    )
    CSF_fit = CSenFFitter(
        data=tc_data,
        model=CSF_mod,
        fit_hrf=settings['fit_hrf'],
        n_jobs=settings['nr_jobs'],
        verbose=verbose,
    )
    # ************************************************************************
    
    # Check for old parameters:
    old_grid_params = dag_find_file_in_folder([sub, model, task, roi_fit, 'grid'], outputdir, return_msg=None)
    if old_grid_params is None:
        # -> grid is faster than the iter, so we may have the 'all' fit already...
        # -> check for this and use it if appropriate
        old_grid_params = dag_find_file_in_folder([sub, model, task, 'all', 'grid'], outputdir, return_msg=None)
    csf_idx = print_p()['csf']
    if (old_grid_params is None) or ow:
        print('Not done grid fit - doing that now')
        g_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f'Starting grid {g_start_time}')
        start = time.time()
        # 
        width_r_grid   = np.linspace(
            settings['csf_bounds']['width_r'][0],
            settings['csf_bounds']['width_r'][1],
            settings['csf_grid_nr'] )     
        sf0_grid       = np.linspace(
            settings['csf_bounds']['sf0'][0],
            settings['csf_bounds']['sf0'][1],
            settings['csf_grid_nr'] ) 
        maxC_grid      = np.linspace(
            settings['csf_bounds']['maxC'][0],
            settings['csf_bounds']['maxC'][1],
            settings['csf_grid_nr'] ) # BOOST
        width_l_grid   = np.array(settings['csf_bounds']['width_l'][0])
            # settings['csf_bounds']['width_l'][0],
            # settings['csf_bounds']['width_l'][1],
            # settings['grid_nr'] )

        # We can also fit the hrf in the same way (specifically the derivative)
        # -> make a grid between 0-10 (see settings file)
        if fit_hrf:
            hrf_1_grid = np.linspace(
                settings['hrf']['deriv_bound'][0], 
                settings['hrf']['deriv_bound'][1], 
                settings['csf_grid_nr'] )
            # We generally recommend to fix the dispersion value to 0
            hrf_2_grid = np.array([0.0])        
        else:
            hrf_1_grid = None
            hrf_2_grid = None
        csf_grid_bounds = [settings['csf_bounds']['beta']]
        # Start grid fit
        CSF_fit.grid_fit(
            width_r_grid    = width_r_grid,
            sf0_grid        = sf0_grid,
            maxC_grid       = maxC_grid,
            width_l_grid    = width_l_grid,
            hrf_1_grid      = hrf_1_grid,
            hrf_2_grid      = hrf_2_grid,
            verbose         = True,
            n_batches=settings['nr_jobs'],
            fixed_grid_baseline=settings['fixed_grid_baseline'],
            grid_bounds=csf_grid_bounds, 
        )
        elapsed = (time.time()-start)

        # Proccess the fit parameters... (make the shape back to normals )
        CSF_fit.gridsearch_params = dag_filter_for_nans(CSF_fit.gridsearch_params)            
        g_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        elapsed = (time.time() - start)
        
        # Stuff to print:         
        print(f'Finished grid {g_end_time}')
        print(f'Took {timedelta(seconds=elapsed)}')
        vx_gt_rsq_th = CSF_fit.gridsearch_params[:,-1]>settings['rsq_threshold']
        nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
        mean_vx_gt_rsq_th = np.mean(CSF_fit.gridsearch_params[vx_gt_rsq_th,-1])
        print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')


        # Save everything as a pickle...
        grid_pkl_file = opj(outputdir, f'{out}_stage-grid_desc-csf_params.pkl')
        # Put them in the correct format to save
        if zero_pad:
            grid_pars_to_save = CSF_fit.gridsearch_params
        else:
            grid_pars_to_save = process_prfpy_out(CSF_fit.gridsearch_params, roi_mask) 
        grid_dict = {}
        grid_dict['pars'] = grid_pars_to_save
        grid_dict['settings'] = settings
        grid_dict['start_time'] = g_start_time
        grid_dict['end_time'] = g_end_time
        f = open(grid_pkl_file, "wb")
        pickle.dump(grid_dict, f)
        f.close()

    else:
        print('Loading old grid parameters')
        g_params = load_params_generic(old_grid_params)
        # Apply the mask 
        if not zero_pad:
            g_params = g_params[roi_mask,:]
        CSF_fit.gridsearch_params = g_params        

    print(f'Mean rsq = {CSF_fit.gridsearch_params[:,-1].mean():.3f}')
    # ************************************************************************    

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DO ITERATIVE FIT
    iter_check = dag_find_file_in_folder([sub, model, task, roi_fit, 'iter', constraints], outputdir, return_msg=None)
    if (iter_check is not None) and (not ow):
        print(f'Already done {iter_check}')
        sys.exit()        

    bounds = [
        (settings['csf_bounds']['width_r']),     # width_r
        (settings['csf_bounds']['sf0']),     # sf0
        (settings['csf_bounds']['maxC']),    # maxC
        (settings['csf_bounds']['width_l']),     # width_l
        (settings['csf_bounds']['beta']),   # beta
        (settings['csf_bounds']['baseline']),      # baseline
    ]
    # if fit_hrf:
    bounds += [
        (settings['hrf']['deriv_bound']),
        (settings['hrf']['disp_bound'])]

    # Constraints determines which scipy fitter is used
    # -> can also be used to make certain parameters interdependent (e.g. size depening on eccentricity... not normally done)
    if settings['constraints']=='tc':
        csf_constraints = []   # uses trust-constraint (slower, but moves further from grid
    elif settings['constraints']=='bgfs':
        csf_constraints = None # uses l-BFGS (which is faster)

    i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'Starting iter {i_start_time}, constraints = {csf_constraints}')
    start = time.time()

    CSF_fit.iterative_fit(
        rsq_threshold = settings['rsq_threshold'],            
        verbose = False,
        bounds = bounds,
        constraints = csf_constraints,
        xtol=float(settings['xtol']),   
        ftol=float(settings['ftol']),           
        )

    # Fiter for nans
    CSF_fit.iterative_search_params = dag_filter_for_nans(CSF_fit.iterative_search_params)    
    i_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'End iter {i_end_time}')           
    elapsed = (time.time() - start)
    print(f'Finished iter {i_end_time}')
    print(f'Took {timedelta(seconds=elapsed)}')
    vx_gt_rsq_th = CSF_fit.iterative_search_params[:,-1]>settings['rsq_threshold']
    nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
    mean_vx_gt_rsq_th = np.mean(CSF_fit.iterative_search_params[vx_gt_rsq_th,-1]) 
    print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')

    # CREATE PREDICTIONS
    preds = CSF_mod.return_prediction(*list(CSF_fit.iterative_search_params[:,:-1].T))
    preds = dag_filter_for_nans(preds)
    # *************************************************************


    # Save everything as a pickle...
    if zero_pad:
        iter_pars_to_save = CSF_fit.iterative_search_params
        preds_to_save = preds
    else:
        iter_pars_to_save = process_prfpy_out(CSF_fit.iterative_search_params, roi_mask)
        preds_to_save = process_prfpy_out(preds, roi_mask)
    print(iter_pars_to_save.shape)
    iter_pkl_file = opj(outputdir, f'{out}_stage-iter_constr-{constraints}_desc-csf_params.pkl')
    iter_dict = {}
    iter_dict['pars'] = iter_pars_to_save
    iter_dict['settings'] = settings
    iter_dict['preds'] = preds_to_save
    iter_dict['start_time'] = i_start_time
    iter_dict['end_time'] = i_end_time
    f = open(iter_pkl_file, "wb")
    pickle.dump(iter_dict, f)
    f.close()
    
if __name__ == "__main__":
    main(sys.argv[1:])
