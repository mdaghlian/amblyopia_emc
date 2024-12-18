#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S1 *********
run_prf_fit_G
'''

import os
import sys
opj = os.path.join

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
prf_dir = opj(derivatives_dir, 'prf')

sub_list = ['sub-01', 'sub-02', 'sub-03'] 
task_list = ['pRFLE', 'pRFRE']
ses_list = ['ses-1', 'ses-2',]

roi_fit = 'all'
constraint = '--bgfs'
hrf_flag = ''
nr_jobs = 20
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
            prf_job_name = f'G{sub}_{task}_{ses}_{roi_fit}'            
            # remove the 
            job=f"qsub -q short.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
            # job="python"

            script_path = opj(os.path.dirname(__file__),'s1a_run_prf_fit_G.py')
            script_args = f"--sub {sub} --task {task} --ses {ses} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} {hrf_flag}"
            # print(f'{job} {script_path} {script_args}')
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()