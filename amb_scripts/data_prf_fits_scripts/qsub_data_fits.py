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
warnings.filterwarnings('ignore')
opj = os.path.join

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")

def main(argv):

    """
---------------------------------------------------------------------------------------------------

Submit fitting jobs (gauss and norm) for the real data. 
Will loop through subjects and tasks
Need to specify:

Args:
    -r (--roi_fit=)     all, core, or specific roi (which voxels are we going to fit?)
    -d (--dm_fit=)      are we fitting using the actual dm? 'standard'
                        Or using ignoring the scotoma? 'AS0'. If AS0, do not fit the AS0 timeseries (this is the same)
    --prf_out           Where files are output to
    --tc                
    --bgfs                         

Example:


---------------------------------------------------------------------------------------------------
    """
    
    sub_list = ['sub-02']#, 'sub-01', 'sub-03']                        # -> subjects 
    task_list = ['task-AS0', 'task-AS1', 'task-AS2']       # -> tasks 
    model = 'norm'
    ses = 'ses-1'    
    roi_fit = None
    dm_fit = None
    fitter_type = None
    prf_out = 'prf'
    prf_out_flag = ''
    hrf_flag = ''

    try:
        opts = getopt.getopt(argv,"qp:r:d:c:m:",["help=", "model=", "roi_fit=", "dm_fit=", "prf_out=", "constraints=", "tc", "bgfs", "hrf"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            print(main.__doc__)
            sys.exit()
        elif opt in ("-m", "--model"):
            model = arg            
        elif opt in ("-r", "--roi_fit"):
            roi_fit = arg
        elif opt in ("-d", "--dm_fit"):
            dm_fit = arg    
            if 'AS0' in dm_fit:
                print('Do not run AS0 again...')
                task_list = ['task-AS1', 'task-AS2']
        elif opt in ("--prf_out"):
            prf_out = arg
        elif opt in ("--tc"):
            fitter_type = "--tc"
        elif opt in ("--bgfs"):
            fitter_type = "--bgfs"   
        elif opt in ("-c", "--constraints"):
            fitter_type = f'--constraints {arg}'
        elif opt in ("--hrf"):
            hrf_flag = "--hrf"

    if len(argv) < 1:
        print("NOT ENOUGH ARGUMENTS SPECIFIED")
        print(main.__doc__)
        sys.exit()

    prf_dir = opj(derivatives_dir, prf_out)

    # Check if  exists 
    if not os.path.exists(prf_dir): # Does xparams_dir exist? No? -> make it
        os.mkdir(prf_dir)

    # ************ LOOP THROUGH SUBJECTS ***************
    for sub in sub_list:
        # CREATE THE DIRECTORY TO SAVE THE PRFs
        if not os.path.exists(opj(prf_dir, sub)): 
            os.mkdir(opj(prf_dir, sub))
        if not os.path.exists(opj(prf_dir, sub, ses)): 
            os.mkdir(opj(prf_dir, sub, ses))    

        this_dir = opj(prf_dir, sub, ses)
        
        # ************ LOOP THROUGH TASKS ***************
        for task in task_list:
            prf_job_name = f'data{sub}_{task}_{model}_{roi_fit}_{dm_fit}'            
            # remove the 
            job=f"qsub -q verylong.q@jupiter -pe smp 2 -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
            # job="python"

            script_path = opj(os.path.dirname(__file__),'create_data_fits.py')
            script_args = f"--sub {sub} --task {task} --model {model} --roi_fit {roi_fit} --dm_fit {dm_fit} --prf_out {prf_out} {fitter_type} {hrf_flag} --verbose"
            # print(f'{job} {script_path} {script_args}')
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()
if __name__ == "__main__":
    main(sys.argv[1:])
