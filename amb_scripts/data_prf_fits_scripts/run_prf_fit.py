#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import getopt
from linescanning import (
    prf,
    utils,
)
import numpy as np
import os
import sys
import warnings
from joblib import parallel_backend
warnings.filterwarnings('ignore')
opj = os.path.join

from amb_scripts.load_saved_info import *

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'

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
    -t (--task=)        pRFLE,pRFRE,CSFLE,CSFRE
    -m (--model=)       gauss or norm
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
    model = None
    roi_fit = None
    fit_hrf = False
    verbose = True
    constraints = None
    prf_out = 'amb-prf'    

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
            task = arg #hyphen_parse('task', arg)
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
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

    # Determine the labels - needed for saving the file
    print(f'data{sub}_{task}_{model}_{roi_fit}')
    prf_dir = opj(derivatives_dir, prf_out)
    if not os.path.exists(prf_dir): 
        os.mkdir(prf_dir)    
    # CREATE THE DIRECTORY TO SAVE THE PRFs
    if not os.path.exists(opj(prf_dir, sub)): 
        os.mkdir(opj(prf_dir, sub))
    if not os.path.exists(opj(prf_dir, sub, ses)): 
        os.mkdir(opj(prf_dir, sub, ses))    

    outputdir = opj(prf_dir, sub, ses)

    # LOAD THE RELEVANT TIME COURSES AND DESIGN MATRICES    
    tc_data = amb_load_real_tc(sub=sub, task_list=task)[task].T
    # Are we limiting the fits to an roi?

    print('Fitting {roi_fit} ')
    roi_mask = amb_load_roi(sub=sub, roi=roi_fit)
    if roi_fit=='all':
        print('fitting ALL voxels')
        if constraints=='tc':
            print('warning - tc for full brain is too long...')            
        pass

    # initialize empty array and only keep the timecourses from label; keeps the original dimensions for simplicity sake! You can always retrieve the label indices with linescanning.optimal.SurfaceCalc
    empty = np.zeros_like(tc_data)

    # insert timecourses 
    lbl_true = np.where(roi_mask == True)[0]
    empty[:,lbl_true] = tc_data[:,lbl_true]

    # overwrite m_prf_tc_data
    tc_data = empty.copy()

    # Design matrix
    design_matrix = amb_load_dm('prf')['prf']        
    out = f"{sub}_{ses}_{task}_{hyphen_parse('roi', roi_fit)}_data-fits"
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
            tc_data.T, 
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
                tc_data.T, 
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
