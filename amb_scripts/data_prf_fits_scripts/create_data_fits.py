#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import ast
import getopt
from linescanning import (
    prf,
    utils,
    # dataset
)
import numpy as np
import nibabel as nb
import os
from scipy import io
import sys
import warnings
import json
import pickle
from joblib import parallel_backend
warnings.filterwarnings('ignore')
opj = os.path.join

from pfa_scripts.load_saved_info import get_design_matrix_npy, get_real_tc, get_roi, get_number_of_vx
from pfa_scripts.utils import hyphen_parse

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")

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
    -t (--task=)        AS0,AS1,AS2
    -m (--model=)       gauss or norm
    -r (--roi_fit=)     all, core
    -d (--dm_fit=)      are we fitting using the actual dm? 'standard'
                        Or using ignoring the scotoma? 'dm-AS0'

    --verbose
    --tc                
    --bgfs                         

Example:


---------------------------------------------------------------------------------------------------
    """
    
    sub = None
    ses = 'ses-1'
    task = None
    model = None
    roi_fit = None
    dm_fit = None
    fit_hrf = False
    verbose = True
    constraints = None
    prf_out = 'prf'    


    try:
        opts = getopt.getopt(argv,"qp:s:t:m:r:d:c:",[
            "help=", "sub=", "task=", "model=", "roi_fit=", 
            "dm_fit=", "constraints=", "prf_out=", "verbose", 
            "tc", "bgfs", "hrf"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            print(main.__doc__)
            sys.exit()

        elif opt in ("-s", "--sub"):
            sub = hyphen_parse('sub', arg)

        elif opt in ("-t", "--task"):
            task = hyphen_parse('task', arg)
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("-d", "--dm_fit"):
            dm_fit = arg
        elif opt in ("--prf_out"):
            prf_out = arg        
        elif opt in ("--verbose"):
            verbose = True
        elif opt in ("--tc"):
            constraints = "tc"
        elif opt in ("--bgfs"):
            constraints = "bgfs"            
        elif opt in ("-c", "--constraints"):
            constraints = arg.split(',')
        elif opt in ("--hrf"):
            fit_hrf = True

    if len(argv) < 1:
        print("NOT ENOUGH ARGUMENTS SPECIFIED")
        print(main.__doc__)
        sys.exit()

    prf_dir = opj(derivatives_dir, prf_out)
    # Determine the labels - needed for saving the file
    print(f'data{sub}_{task}_{model}_{roi_fit}_{dm_fit}')
    # CREATE THE DIRECTORY TO SAVE THE PRFs
    if not os.path.exists(opj(prf_dir, sub)): 
        os.mkdir(opj(prf_dir, sub))
    if not os.path.exists(opj(prf_dir, sub, ses)): 
        os.mkdir(opj(prf_dir, sub, ses))    

    outputdir = opj(prf_dir, sub, ses)

    # LOAD THE RELEVANT TIME COURSES AND DESIGN MATRICES    
    num_vx = get_number_of_vx(sub=sub)
    m_prf_tc_data = get_real_tc(sub=sub, task_list=task)[task].T
    # Are we limiting the fits to an roi?
    if roi_fit=='all':
        print('fitting ALL voxels')
        if constraints=='tc':
            print('warning - tc for full brain is too long...')            
        pass
        roi_mask = get_roi(sub=sub, label=roi_fit)

    elif roi_fit=='core-vis':
        print('Fitting core visual regions ')
        roi_list = ['v1', 'v2', 'v3','v3ab', 'v4', 'LO','TO', 'lowerIPS', 'upperIPS']
        roi_mask = get_roi(sub=sub, label=roi_list)
        
    else: 
        print('Fitting {roi_fit} ')
        roi_mask = get_roi(sub=sub, label=roi_fit)    

    # initialize empty array and only keep the timecourses from label; keeps the original dimensions for simplicity sake! You can always retrieve the label indices with linescanning.optimal.SurfaceCalc
    empty = np.zeros_like(m_prf_tc_data)

    # insert timecourses 
    lbl_true = np.where(roi_mask == True)[0]
    empty[:,lbl_true] = m_prf_tc_data[:,lbl_true]

    # overwrite m_prf_tc_data
    m_prf_tc_data = empty.copy()    

    # Check mask is the correct shape
    assert roi_mask.shape[0]==num_vx
    assert m_prf_tc_data.shape[-1]==num_vx

    if not 'AS0' in dm_fit:
        design_matrix = get_design_matrix_npy([task])[task] 
    else:
        # ALWAYS USING TASK AS0
        design_matrix = get_design_matrix_npy(['task-AS0'])['task-AS0'] 
        
    out = f"{sub}_{ses}_{task}_{hyphen_parse('roi', roi_fit)}_{hyphen_parse('dm', dm_fit)}_data-fits"
    # Check for old parameters:
    if model=='gauss':
        old_grid_gauss = utils.get_file_from_substring([out, 'gauss', 'grid'], outputdir, return_msg=None)
        old_params = old_grid_gauss
    elif model=='norm':
        old_iter_gauss = utils.get_file_from_substring([out, 'gauss', 'iter'], outputdir, return_msg=None)
        old_params = old_iter_gauss
    # check_norm_iter = utils.get_file_from_substring([out, 'norm', 'iter'], outputdir, return_msg=None)
    # if check_norm_iter!=None:
    #     print('Already fit norm model!')
    #     return
    with parallel_backend('threading', n_jobs=1):
        import mkl
        mkl.set_num_threads(8)
        
        # stage 1 - no HRF
        stage1 = prf.pRFmodelFitting(
            m_prf_tc_data.T, 
            design_matrix=design_matrix, 
            TR=1.5, 
            model=model, 
            stage='iter', 
            verbose=True, 
            output_dir=outputdir,
            output_base=out,
            write_files=True,
            fit_hrf=False,
            fix_bold_baseline=True,
            old_params=old_params, #old_iter_gauss,
            constraints=constraints,
            save_grid=True,
            use_grid_bounds=False,
            nr_jobs=1)
        stage1.fit()

        # stage2 - fit HRF after initial iterative fit
        if fit_hrf:

            previous_fitter = f"{model}_fitter"
            if not hasattr(stage1, previous_fitter):
                raise ValueError(f"fitter does not have attribute {previous_fitter}")

            # add tag to output to differentiate between HRF=false and HRF=true
            out += "_hrf-true"

            # initiate fitter object with previous fitter
            stage2 = prf.pRFmodelFitting(
                m_prf_tc_data.T, 
                design_matrix=stage1.design, 
                TR=stage1.TR, 
                model=model, 
                stage='iter', 
                verbose=stage1.verbose,
                fit_hrf=True,
                output_dir=stage1.output_dir,
                output_base=out,
                write_files=True,                                
                previous_gaussian_fitter=stage1.previous_fitter,
                fix_bold_baseline=True,
                constraints=constraints,
                save_grid=True,
                use_grid_bounds=False,
                nr_jobs=1)

            stage2.fit()


if __name__ == "__main__":
    main(sys.argv[1:])
