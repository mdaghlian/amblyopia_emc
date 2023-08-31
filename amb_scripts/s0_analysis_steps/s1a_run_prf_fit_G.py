#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S1 *********
run_prf_fit_G
> Fit the gaussian model on the data. 
> How to do the HRF? Different for each eye? 
> Fix for one? Mean?
...
'''

import getopt

from prfpy.stimulus import PRFStimulus2D
from prfpy.model import Iso2DGaussianModel
from prfpy.fit import Iso2DGaussianFitter

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

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'
settings_path = get_yml_settings_path()

def main(argv):

    """
---------------------------------------------------------------------------------------------------

Fit the real time series using the gaussian 
- ROI 
- Fitter
- Design matrix

Args:
    -s (--sub=)         e.g., 01
    -n (--ses=)
    -t (--task=)        pRFLE,pRFRE
    -r (--roi_fit=)     e.g., all, V1_exvivo
    --nr_jobs           number of jobs
    --verbose
    --tc                
    --bgfs

Example:


---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # ALWAYS
    model = 'gauss'    
    verbose = True
    prf_out = 'prf'    

    # Specify
    sub = None
    ses = None
    task = None
    fit_hrf = False
    roi_fit = None
    constraints = None
    nr_jobs = None


    try:
        opts = getopt.getopt(argv,"qp:s:t:m:n:r:d:c:",[
            "help=", "sub=", "task=", "ses=", "roi_fit=", "nr_jobs=", "hrf"            
            "tc", "bgfs"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            print(main.__doc__)
            sys.exit()

        elif opt in ("-s", "--sub"):
            sub = dag_hyphen_parse('sub', arg)
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-n", "--ses"):
            ses = dag_hyphen_parse('ses', arg)
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("--nr_jobs"):
            nr_jobs = int(arg)            
        elif opt in ("--tc"):
            constraints = "tc"
        elif opt in ("--bgfs"):
            constraints = "bgfs"
        elif opt in ("--hrf"):
            fit_hrf = True

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
    out = f"{sub}_{dag_hyphen_parse('model', model)}_{dag_hyphen_parse('roi', roi_fit)}_{task}-fits"

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
    prf_settings['constraints'] = constraints
    prf_settings['ses'] = ses
    prf_settings['task'] = task
    prf_settings['fit_hrf'] = fit_hrf
    prf_settings['verbose'] = verbose
    prf_settings['prf_out'] = out 
    # ****************************************************

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< LOAD TIME SERIES & MASK THE ROI   
    num_vx = np.sum(amb_load_nverts(sub=sub))
    m_prf_tc_data = amb_load_real_tc(sub=sub, task_list=task, ses=ses)[task]
    roi_mask = amb_load_roi(sub=sub, label=roi_fit)
    # Are we limiting the fits to an roi?
    num_vx_in_roi = roi_mask.sum()
    print(f'Fitting {roi_fit} {num_vx_in_roi} voxels out of {num_vx} voxels in total')
    print('Removing irrelevant times courses')
    print('These will be added in later, as zero fits in the parameters')
    zero_pad = False
    m_prf_tc_data = mask_time_series(ts=m_prf_tc_data, mask=roi_mask, zero_pad=zero_pad)
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
    
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< CREATE GAUSSIAN MODEL & fitter   
    gg = Iso2DGaussianModel(
        stimulus=prf_stim,                                  # The stimulus we made earlier
        hrf=prf_settings['hrf']['pars'],                    # These are the parameters for the HRF that we normally use at Spinoza (with 7T data). (we can fit it, this will be done later...)
        normalize_RFs=prf_settings['normalize_RFs'],        # Normalize the volume of the RF (so that RFs w/ different sizes have the same volume. Generally not needed, as this can be solved using the beta values i.e.,amplitude)
        )
    gf = Iso2DGaussianFitter(
        data=m_prf_tc_data,             # time series
        model=gg,                       # model (see above)
        n_jobs=prf_settings['nr_jobs'], # number of jobs to use in parallelization 
        )    
    # ************************************************************************



    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< IF NOT DONE - DO GRID FIT
    grid_gauss = dag_find_file_in_folder([sub, model, task, roi_fit, 'gauss', 'grid'], outputdir, return_msg=None)
    if grid_gauss is None:
        # -> grid is faster than the iter, so we may have the 'all' fit already...
        # -> check for this and use it if appropriate
        grid_gauss = dag_find_file_in_folder([sub, model, task, 'all', 'gauss', 'grid'], outputdir, return_msg=None)
    gauss_idx = prfpy_params_dict()['gauss']
    if grid_gauss is None:
        print('Not done grid fit - doing that now')
        g_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f'Starting grid {g_start_time}')
        start = time.time()
        #        
        grid_nr = prf_settings['grid_nr'] # Size of the grid (i.e., number of possible PRF models). Higher number means that the grid fit will be more exact, but take longer...
        eccs    = max_eccentricity * np.linspace(0.25, 1, grid_nr)**2 # Squared because of cortical magnification, more efficiently tiles the visual field...
        sizes   = max_eccentricity * np.linspace(0.1, 1, grid_nr)**2  # Possible size values (i.e., sigma in gaussian model) 
        polars  = np.linspace(0, 2*np.pi, grid_nr)              # Possible polar angle coordinates

        # We can also fit the hrf in the same way (specifically the derivative)
        if fit_hrf:
            # -> make a grid between 0-10 (see settings file)
            hrf_1_grid = np.linspace(prf_settings['hrf']['deriv_bound'][0], prf_settings['hrf']['deriv_bound'][1], int(grid_nr/2))
            # We generally recommend to fix the dispersion value to 0
            hrf_2_grid = np.array([0.0])        
        else:
            hrf_1_grid = None
            hrf_2_grid = None            

        gauss_grid_bounds = [prf_settings['prf_ampl']] 


        gf.grid_fit(
            ecc_grid=eccs,
            polar_grid=polars,
            size_grid=sizes,
            hrf_1_grid=hrf_1_grid,
            hrf_2_grid=hrf_2_grid,
            verbose=True,
            n_batches=prf_settings['nr_jobs'],                          # The grid fit is performed in parallel over n_batches of units.Batch parallelization is faster than single-unit parallelization and of sequential computing.
            fixed_grid_baseline=prf_settings['fixed_grid_baseline'],    # Fix the baseline? This makes sense if we have fixed the baseline in preprocessing
            grid_bounds=gauss_grid_bounds
            )
        # Proccess the fit parameters... (make the shape back to normals )
        gf.gridsearch_params = dag_filter_for_nans(gf.gridsearch_params)            
        g_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
        elapsed = (time.time() - start)
        
        # Stuff to print:         
        print(f'Finished grid {g_end_time}')
        print(f'Took {timedelta(seconds=elapsed)}')
        vx_gt_rsq_th = gf.gridsearch_params[:,-1]>prf_settings['rsq_threshold']
        nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
        mean_vx_gt_rsq_th = np.mean(gf.gridsearch_params[vx_gt_rsq_th,-1])
        print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')


        # Save everything as a pickle...
        grid_pkl_file = opj(outputdir, f'{out}_stage-grid_desc-prf_params.pkl')
        # Put them in the correct format to save
        if zero_pad:
            grid_pars_to_save = gf.gridsearch_params
        else:
            grid_pars_to_save = process_prfpy_out(gf.gridsearch_params, roi_mask) 
        grid_dict = {}
        grid_dict['pars'] = grid_pars_to_save
        grid_dict['settings'] = prf_settings
        grid_dict['start_time'] = g_start_time
        grid_dict['end_time'] = g_end_time
        f = open(grid_pkl_file, "wb")
        pickle.dump(grid_dict, f)
        f.close()

    else:
        print('Loading old grid parameters')
        g_params = load_params_generic(grid_gauss)
        # Apply the mask 
        if not zero_pad:
            g_params = g_params[roi_mask,:]
        gf.gridsearch_params = g_params        

    print(f'Mean rsq = {gf.gridsearch_params[:,-1].mean():.3f}')
    # ************************************************************************
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DO ITERATIVE FIT
    iter_check = dag_find_file_in_folder([out, 'gauss', 'iter', constraints], outputdir, return_msg=None)
    if iter_check is not None:
        print(f'Already done {iter_check}')
        sys.exit()        

    gauss_bounds = [
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # x bound
        (-1.5*max_eccentricity, 1.5*max_eccentricity),          # y bound
        (1e-1, max_eccentricity*3),                             # prf size bounds
        (prf_settings['prf_ampl'][0],prf_settings['prf_ampl'][1]),      # prf amplitude
        (prf_settings['bold_bsl'][0],prf_settings['bold_bsl'][1]),      # bold baseline (fixed)
    ]

    gauss_bounds += [
        (prf_settings['hrf']['deriv_bound'][0], prf_settings['hrf']['deriv_bound'][1]), # hrf_1 bound
        (prf_settings['hrf']['disp_bound'][0],  prf_settings['hrf']['disp_bound'][1]), # hrf_2 bound
        ]

    # Constraints determines which scipy fitter is used
    # -> can also be used to make certain parameters interdependent (e.g. size depening on eccentricity... not normally done)
    if prf_settings['constraints']=='tc':
        g_constraints = []   # uses trust-constraint (slower, but moves further from grid
    elif prf_settings['constraints']=='bgfs':
        g_constraints = None # uses l-BFGS (which is faster)

    i_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'Starting iter {i_start_time}, constraints = {g_constraints}')
    start = time.time()

    gf.iterative_fit(
        rsq_threshold=prf_settings['rsq_threshold'],    # Minimum variance explained. Puts a lower bound on the quality of PRF fits. Any fits worse than this are thrown away...     
        verbose=False,
        bounds=gauss_bounds,       # Bounds (on parameters)
        constraints=g_constraints, # Constraints
        xtol=float(prf_settings['xtol']),     # float, passed to fitting routine numerical tolerance on x
        ftol=float(prf_settings['ftol']),     # float, passed to fitting routine numerical tolerance on function
        )

    # Fiter for nans
    gf.iterative_search_params = dag_filter_for_nans(gf.iterative_search_params)    
    i_end_time = datetime.now().strftime('%Y-%m-%d_%H-%M')
    print(f'End iter {i_end_time}')           
    elapsed = (time.time() - start)
    print(f'Finished iter {i_end_time}')
    print(f'Took {timedelta(seconds=elapsed)}')
    vx_gt_rsq_th = gf.iterative_search_params[:,-1]>prf_settings['rsq_threshold']
    nr_vx_gt_rsq_th = np.mean(vx_gt_rsq_th) * 100
    mean_vx_gt_rsq_th = np.mean(gf.iterative_search_params[vx_gt_rsq_th,-1]) 
    print(f'Percent of vx above rsq threshold: {nr_vx_gt_rsq_th}. Mean rsq for threshold vx {mean_vx_gt_rsq_th}')

    # CREATE PREDICTIONS
    preds = gg.return_prediction(*list(gf.iterative_search_params[:,:-1].T))
    preds = dag_filter_for_nans(preds)
    # *************************************************************


    # Save everything as a pickle...
    if zero_pad:
        iter_pars_to_save = gf.iterative_search_params
        preds_to_save = preds
    else:
        iter_pars_to_save = process_prfpy_out(gf.iterative_search_params, roi_mask)
        preds_to_save = process_prfpy_out(preds, roi_mask)    
    iter_pkl_file = opj(outputdir, f'{out}_stage-iter_constr-{constraints}_desc-prf_params.pkl')
    iter_dict = {}
    iter_dict['pars'] = iter_pars_to_save
    iter_dict['settings'] = prf_settings
    iter_dict['preds'] = preds_to_save
    iter_dict['start_time'] = i_start_time
    iter_dict['end_time'] = i_end_time
    f = open(iter_pkl_file, "wb")
    pickle.dump(iter_dict, f)
    f.close()



if __name__ == "__main__":
    main(sys.argv[1:])
