#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''

'''

import os
import sys
opj = os.path.join
from dag_prf_utils.utils import dag_get_cores_used
import time

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
prf_dir = opj(derivatives_dir, 'cmf_est')
if not os.path.exists(prf_dir): 
    os.mkdir(prf_dir)

sub_list = ['sub-01', 'sub-02']
task_list = ['pRFLE', 'pRFRE']
ses_list = ['ses-1', 'ses-2']




# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    # ************ LOOP THROUGH SESSIONS ***************
    for ses in ses_list:    
        if not os.path.exists(opj(prf_dir, sub)): 
            os.mkdir(opj(prf_dir, sub))
        if not os.path.exists(opj(prf_dir, sub, ses)): 
            os.mkdir(opj(prf_dir, sub, ses))        
        this_dir = opj(prf_dir, sub, ses)

        # ************ LOOP THROUGH TASKS ***************
        for task in task_list:
            prf_job_name = f's{sub[-1]}n{ses[-1]}t{task}-cmf'            
            # remove the 
            job=f"qsub -q short.q@jupiter -pe smp 1 -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
            # job="python"

            script_path = opj(os.path.dirname(__file__),'s5_run_cmf.py')
            script_args  = f"--sub {sub} --ses {ses} --task {task}"

            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()