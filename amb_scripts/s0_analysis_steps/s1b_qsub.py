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

import os
import sys
opj = os.path.join

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
prf_dir = opj(derivatives_dir, 'prf')

sub_list = ['sub-03'] # 'sub-01', 'sub-02', 'sub-03']
task_list = ['pRFLE', 'pRFRE']
model_list  = ['gauss']
ses_list = ['ses-1', 'ses-2']

roi_fit = 'all'
constraint = '--bgfs'
nr_jobs = 2
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
            # ************ LOOP THROUGH TASKS ***************
            for model in model_list:        
                prf_job_name = f'X{sub}_{ses}_{model}_{task}_{roi_fit}'            
                # remove the 
                job=f"qsub -q short.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
                job="python"

                script_path = opj(os.path.dirname(__file__),'s1b_run_prf_fit_X.py')
                script_args = f"--sub {sub} --ses {ses} --task {task} --model {model} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --ow"
                # print(f'{job} {script_path} {script_args}')
                os.system(f'{job} {script_path} {script_args}')
                # sys.exit()