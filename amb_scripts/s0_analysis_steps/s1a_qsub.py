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

sub_list = ['sub-01']#, 'sub-02']
task_list = ['pRFLE', 'pRFRE']

roi_fit = 'all'
constraint = '--bgfs'
nr_jobs = 5
ses = 'ses-1'
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    if not os.path.exists(opj(prf_dir, sub)): 
        os.mkdir(opj(prf_dir, sub))
    if not os.path.exists(opj(prf_dir, sub, ses)): 
        os.mkdir(opj(prf_dir, sub, ses))        
    this_dir = opj(prf_dir, sub, ses)
    
    # ************ LOOP THROUGH TASKS ***************
    for task in task_list:
        prf_job_name = f'G{sub}_{task}_{roi_fit}'            
        # remove the 
        job=f"qsub -q verylong.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {prf_job_name} -o {prf_job_name}.txt"
        # job="python"

        script_path = opj(os.path.dirname(__file__),'s1a_run_prf_fit_G.py')
        script_args = f"--sub {sub} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} --ow"
        # print(f'{job} {script_path} {script_args}')
        os.system(f'{job} {script_path} {script_args}')
        # sys.exit()