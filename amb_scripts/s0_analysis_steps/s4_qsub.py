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
prf_dir = opj(derivatives_dir, 'bayes_prf')
if not os.path.exists(prf_dir): 
    os.mkdir(prf_dir)

sub_list = ['sub-02']#, 'sub-02']
model_list = ['gauss']#['gauss', 'norm']
task_list = ['pRFLE']#, 'pRFRE']
ses_list = ['ses-1']#, 'ses-2']

roi_fit = 'demo-100'
nr_jobs = 25
n_walkers = 15
n_steps = 500


# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    # ************ LOOP THROUGH SESSIONS ***************
    for ses in ses_list:    
        if not os.path.exists(opj(prf_dir, sub)): 
            os.mkdir(opj(prf_dir, sub))
        if not os.path.exists(opj(prf_dir, sub, ses)): 
            os.mkdir(opj(prf_dir, sub, ses))        
        this_dir = opj(prf_dir, sub, ses)

        # ************ LOOP THROUGH MODELS ***************
        for model in model_list:
            # ************ LOOP THROUGH TASKS ***************
            for task in task_list:
                prf_job_name = f's{sub[-1]}n{ses[-1]}{model}{task}{roi_fit}'            
                # remove the 
                job=f"qsub -q short.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
                job="python"

                script_path = opj(os.path.dirname(__file__),'s4_run_bayes_prf_fit.py')
                script_args  = f"--sub {sub} --ses {ses} --task {task} --model {model} --roi_fit {roi_fit} --nr_jobs {nr_jobs} "
                script_args += f"--n_walkers {n_walkers} --n_steps {n_steps} --ow" # --ow
                # n_cores_used = dag_get_cores_used()
                # while n_cores_used>50:
                #     time.sleep(60*5)
                #     n_cores_used = dag_get_cores_used()
                # print(f'{job} {script_path} {script_args}')
                os.system(f'{job} {script_path} {script_args}')
                sys.exit()