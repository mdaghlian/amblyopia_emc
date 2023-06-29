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

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
csf_dir = opj(derivatives_dir, 'csf')

sub_list = ['sub-01', 'sub-02']
task_list = ['CSFLE', 'CSFRE']
ses_list = ['ses-1']

roi_fit = 'all'
constraint = '--bgfs'
hrf = ''
nr_jobs = 5

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
            script_args = f"--sub {sub} --ses {ses} --task {task} --roi_fit {roi_fit} --nr_jobs {nr_jobs} {constraint} {hrf}  --ow"
            # print(f'{job} {script_path} {script_args}')
            os.system(f'{job} {script_path} {script_args}')
            # sys.exit()