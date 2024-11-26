#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

'''
********* S0 *********
Submit the jobs to make the psc timecourses
'''

import os
import sys
opj = os.path.join

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")

sub_list = ['sub-01', 'sub-02', 'sub-03']
task_list = ['pRFLE', 'pRFRE', 'CSFLE', 'CSFRE']
ses_list = [ 'ses-1', 'ses-2']

nr_jobs = 1
# ************ LOOP THROUGH SUBJECTS ***************
for sub in sub_list:
    # ************ LOOP THROUGH SESSIONS ***************
    for ses in ses_list:
        psc_tc_dir = opj(derivatives_dir, 'psc_tc')
        if not os.path.exists(psc_tc_dir): # 
            os.mkdir(psc_tc_dir)
        if not os.path.exists(opj(psc_tc_dir, sub)): # 
            os.mkdir(opj(psc_tc_dir, sub))
        if not os.path.exists(opj(psc_tc_dir, sub, ses)): # 
            os.mkdir(opj(psc_tc_dir, sub, ses))
        this_dir = opj(psc_tc_dir, sub, ses)
        # ************ LOOP THROUGH TASKS ***************
        for task in task_list:
            job_name = f'PSC{sub}_{ses}_{task}'            
            job=f"qsub -q short.q@jupiter -pe smp {nr_jobs} -wd {this_dir} -N {job_name} -o {job_name}.txt"
            script_path = opj(os.path.dirname(__file__),'s0_make_psc_tc.py')
            script_args = f"--sub {sub} --ses {ses} --task {task}"
            os.system(f'{job} {script_path} {script_args}')