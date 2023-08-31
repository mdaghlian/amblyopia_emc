#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S2 *********
run_csf_fit
> Fit the gaussian model on the data. 
> How to do the HRF? Different for each eye? 
> Fix for one? Mean?
...
'''

import os
import sys
from dag_prf_utils.utils import dag_get_cores_used
import time
opj = os.path.join

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
csf_dir = opj(derivatives_dir, 'bayes_csf')
if not os.path.exists(csf_dir): 
    os.mkdir(csf_dir)

sub_list = ['sub-02']#['sub-01', 'sub-02']
task_list = ['CSFLE', 'CSFRE']
ses_list = ['ses-1', 'ses-2']

roi_fit = 'V1_exvivo'
nr_jobs = 20
n_walkers = 50
n_steps = 250


# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    # ************ LOOP THROUGH SESSIONS ***************
    for ses in ses_list:    
        if not os.path.exists(opj(csf_dir, sub)): 
            os.mkdir(opj(csf_dir, sub))
        if not os.path.exists(opj(csf_dir, sub, ses)): 
            os.mkdir(opj(csf_dir, sub, ses))        
        this_dir = opj(csf_dir, sub, ses)
        
        # ************ LOOP THROUGH TASKS ***************
        for task in task_list:
            prf_job_name = f'C{sub}_{task}_{roi_fit}'            
            # remove the 
            job=f"qsub -q short.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
            # job="python"

            script_path = opj(os.path.dirname(__file__),'s3_run_bayes_csf_fit.py')
            script_args  = f"--sub {sub} --ses {ses} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} "
            script_args += f"--n_walkers {n_walkers} --n_steps {n_steps} --ow"
            n_cores_used = dag_get_cores_used()
            while n_cores_used>50:
                time.sleep(60*5)
                n_cores_used = dag_get_cores_used()
            print(f'{job} {script_path} {script_args}')
            os.system(f'{job} {script_path} {script_args}')
        # sys.exit()
