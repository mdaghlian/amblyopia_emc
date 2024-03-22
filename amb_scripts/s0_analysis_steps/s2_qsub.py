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
opj = os.path.join
from dag_prf_utils.utils import dag_get_cores_used
import time


source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
csf_dir = opj(derivatives_dir, 'csf')
if not os.path.exists(csf_dir):
    os.mkdir(csf_dir)
sub_list = ['sub-03'] #, 'sub-01']#['sub-01', 'sub-02']
task_list = ['CSFLE', 'CSFRE']
ses_list = ['ses-1', ] # 'ses-2']

roi_fit = 'all'
constraint = '--bgfs'
hrf = ''
nr_jobs = 25
# model = 'straight'
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

            script_path = opj(os.path.dirname(__file__),'s2_run_csf_fit.py')
            # script_path = opj(os.path.dirname(__file__),'s2b_run_different_csf_fit.py')
            # script_args = f"--sub {sub} --ses {ses} --task {task}  --roi_fit {roi_fit} --model {model} --nr_jobs {nr_jobs} {constraint} {hrf}  --ow"
            script_args = f"--sub {sub} --ses {ses} --task {task}  --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} {hrf}  --ow"
            # n_cores_used = dag_get_cores_used()
            # while n_cores_used>60:
            #     time.sleep(60*5)
            #     n_cores_used = dag_get_cores_used()            
            # print(f'{job} {script_path} {script_args}')
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()