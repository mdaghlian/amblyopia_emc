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
from prfpy.model import CSFModel
from prfpy.fit import CSFFitter

from amb_scripts.load_saved_info import *
from amb_scripts.utils import *

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'
csf_setting_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/code/amb_code/amb_scripts/data_prf_fits_scripts'#opj(os.path.dirname(os.path.realpath(__file__)))

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

    --verbose
    --tc                Which fitter using? tc, lgbfs
    --bgfs                         
    --hrf               Fit HRF?

Example:


---------------------------------------------------------------------------------------------------
    """
    
    sub = None
    ses = 'ses-1'
    task = None
    roi_fit = None
    fit_hrf = False
    verbose = True
    constraints = None
    fit_type = 'bgfs'
    csf_out = 'amb-csf'    

    try:
        opts = getopt.getopt(argv,"qp:s:t:m:r:d:c:",[
            "help=", "sub=", "task=", "roi_fit=", 
            "dm_fit=", "constraints=", "csf_out=", "verbose", 
            "tc", "bgfs", "hrf"])[0]
    except getopt.GetoptError:
        qprint(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            qprint(main.__doc__)
            sys.exit()

        elif opt in ("-s", "--sub"):
            sub = hyphen_parse('sub', arg)
        elif opt in ("-t", "--task"):
            task = arg #hyphen_parse('task', arg)
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("--csf_out"):
            csf_out = arg        
        elif opt in ("--verbose"):
            verbose = True
        elif opt in ("--tc"):
            constraints = [] # tc
            fit_type = 'tc'
        elif opt in ("--bgfs"):
            constraints = None # bgfs           
            fit_type = 'bgfs'
        elif opt in ("--hrf"):
            fit_hrf = True

    if len(argv) < 1:
        qprint("NOT ENOUGH ARGUMENTS SPECIFIED")
        qprint(main.__doc__)
        sys.exit()
    # Load settings
    settings_path = opj(csf_setting_dir, 'csf_fit_settings.yml')
    with open(settings_path) as f:
        settings = yaml.safe_load(f)
    # Update specific parameters...
    settings['fit_hrf'] = fit_hrf
    settings['constraints'] = constraints
    settings['sub'] = sub
    settings['ses'] = ses
    settings['task'] = task

    # 
    qprint(f'data{sub}_{task}_{roi_fit}')
    csf_dir = opj(derivatives_dir, csf_out)
    if not os.path.exists(csf_dir): 
        os.mkdir(csf_dir)    
    # CREATE THE DIRECTORY TO SAVE THE PRFs
    if not os.path.exists(opj(csf_dir, sub)): 
        os.mkdir(opj(csf_dir, sub))
    if not os.path.exists(opj(csf_dir, sub, ses)): 
        os.mkdir(opj(csf_dir, sub, ses))    

    outputdir = opj(csf_dir, sub, ses)

    # LOAD THE RELEVANT TIME COURSES AND DESIGN MATRICES    
    tc_data = amb_load_real_tc(sub=sub, task_list=task)[task].T
    # Are we limiting the fits to an roi?

    qprint(f'Fitting {roi_fit} ')
    roi_mask = amb_load_roi(sub=sub, roi=roi_fit)
    if roi_fit=='all':
        qprint('fitting ALL voxels')
        if constraints=='tc':
            qprint('warning - tc for full brain is too long...')            
    else:
        # initialize empty array and only keep the timecourses from label; keeps the original dimensions for simplicity sake! You can always retrieve the label indices with linescanning.optimal.SurfaceCalc
        empty = np.zeros_like(tc_data)

        # insert timecourses 
        lbl_true = np.where(roi_mask == True)[0]
        empty[:,lbl_true] = tc_data[:,lbl_true]

        # overwrite m_prf_tc_data
        tc_data = empty.copy()

    # Design matrix
    CSF_stim = amb_load_prfpy_stim('CSF')

    # Model 
    CSF_mod = CSFModel(
        stimulus=CSF_stim,
        # hrf=settings['hrf'],        
    )
    # Fitter
    CSF_fit = CSFFitter(
        data=tc_data.T,
        model=CSF_mod,
        fit_hrf=settings['fit_hrf'],
        xtol=settings['xtol'],
        ftol=settings['ftol'],
        verbose=verbose,
    )

    out = f"{sub}_{ses}_{task}_{hyphen_parse('roi', roi_fit)}_data-fits"
    # Check for old parameters:
    old_grid_params = None#utils.get_file_from_substring([out, 'grid'], outputdir, return_msg=None)
    old_iter_params = None#utils.get_file_from_substring([out, 'iter'], outputdir, return_msg=None)
    # if old_iter_params != None:
    #     qprint('Iter params already exist - exiting')
    #     return    
    if old_grid_params == None:
        qprint('Grid params not found. Running grid fit')
        width_r_grid   = np.linspace(
            settings['csf_bounds']['width_r'][0],
            settings['csf_bounds']['width_r'][1],
            settings['grid_nr'] )     
        sf0_grid       = np.linspace(
            settings['csf_bounds']['sf0'][0],
            settings['csf_bounds']['sf0'][1],
            settings['grid_nr'] ) 
        maxC_grid      = np.linspace(
            settings['csf_bounds']['maxC'][0],
            settings['csf_bounds']['maxC'][1],
            settings['grid_nr'] ) 
        width_l_grid   = np.array(settings['csf_bounds']['width_l'][0])
            # settings['csf_bounds']['width_l'][0],
            # settings['csf_bounds']['width_l'][1],
            # settings['grid_nr'] )             
        with parallel_backend('threading', n_jobs=1):
            import mkl
            mkl.set_num_threads(8)
            start = time.time()
            CSF_fit.grid_fit(
                width_r_grid    = width_r_grid,
                sf0_grid        = sf0_grid,
                maxC_grid       = maxC_grid,
                width_l_grid    = width_l_grid,
                # fixed_grid_baseline=False,
                grid_bounds=[
                tuple(settings['csf_bounds']['beta']), 
                # tuple(settings['csf_bounds']['baseline']), 
                ],
            )
            elapsed = (time.time()-start)

        grid_params = CSF_fit.gridsearch_params
        grid_params_gt_rsq = grid_params[:,-1]>settings['rsq_threshold']

        mean_rsq = np.mean(grid_params[grid_params_gt_rsq, -1])
                
        # verbose stuff
        start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
        nr = np.sum(grid_params_gt_rsq)
        total = grid_params.shape[0]
        qprint(f"Completed CSF gridfit at {start_time}. Vx above {settings['rsq_threshold']}: {nr}/{total}")
        qprint(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{settings['rsq_threshold']}: {round(mean_rsq,2)}")
        
        # SAVE INTO PKL FILE (SAME STYLE AS JH)
        pkl_file = opj(outputdir,f'{out}_stage-grid_desc-csf-params.pkl')
        out_dict = {}
        out_dict['pars'] = CSF_fit.gridsearch_params
        out_dict['predictions'] = CSF_mod.return_prediction(*list(CSF_fit.gridsearch_params[:,0:6].T))
        out_dict['settings'] = settings

        qprint(f'Save CSF grid params in {pkl_file}')
        f = open(pkl_file, 'wb')
        pickle.dump(out_dict,f)
        f.close()

        old_grid_params = grid_params
    else:
        old_grid_params = amb_load_prf_params(
            sub=sub, task_list=task, model_list='CSF', roi_fit=roi_fit, fit_stage='grid'
            )[task]['CSF']
        
    # 
    qprint(f'Beginning iterative fit, using {fit_type}')    
    bounds = [
        tuple(settings['csf_bounds']['width_r']),     # width_r
        tuple(settings['csf_bounds']['sf0']),     # sf0
        tuple(settings['csf_bounds']['maxC']),    # maxC
        tuple(settings['csf_bounds']['width_l']),     # width_l
        (0, 1000),#tuple(settings['csf_bounds']['beta']),   # beta
        (-10, 10),#tuple(settings['csf_bounds']['baseline']),      # baseline
    ]

    with parallel_backend('threading', n_jobs=1):
        import mkl
        mkl.set_num_threads(8)
        start = time.time()
        CSF_fit.iterative_fit(
            starting_params = old_grid_params,
            bounds = bounds,
            constraints = constraints,
            rsq_threshold = settings['rsq_threshold'],            
        )
        elapsed = (time.time()-start)

    iter_params = CSF_fit.iterative_search_params
    iter_params_gt_rsq = iter_params[:,-1]>settings['rsq_threshold']

    mean_rsq = np.mean(iter_params[iter_params_gt_rsq, -1])
            
    # verbose stuff
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    nr = np.sum(iter_params_gt_rsq)
    total = iter_params.shape[0]
    qprint(f"Completed CSF iterfit at {start_time}. Vx above {settings['rsq_threshold']}: {nr}/{total}")
    qprint(f"Iterfit took {timedelta(seconds=elapsed)} | Mean rsq>{settings['rsq_threshold']}: {round(mean_rsq,2)}")
    
    # SAVE INTO PKL FILE (SAME STYLE AS JH)
    pkl_file = opj(outputdir,f'{out}_stage-iter_desc-csf-params.pkl')
    out_dict = {}
    out_dict['pars'] = CSF_fit.iterative_search_params
    out_dict['predictions'] = CSF_mod.return_prediction(*list(CSF_fit.iterative_search_params[:,0:6].T))
    out_dict['settings'] = settings

    qprint(f'Save CSF iter params in {pkl_file}')
    f = open(pkl_file, 'wb')
    pickle.dump(out_dict,f)
    f.close()

if __name__ == "__main__":
    main(sys.argv[1:])
