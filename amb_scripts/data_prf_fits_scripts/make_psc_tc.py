#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V


import numpy as np  
import nibabel as nb
import os
import linescanning.utils as lsutils
opj = os.path.join

from amb_scripts.load_saved_info import *

sub = sys.argv[1]
ses = 'ses-1'

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'
freesurfer_dir = opj(derivatives_dir, 'freesurfer')
task_list = ['CSFLE', 'CSFRE']#['pRFLE', 'pRFRE', 'CSFLE', 'CSFRE']

psc_tc_dir = opj(derivatives_dir, 'psc_tc')
if not os.path.exists(psc_tc_dir): # 
    os.mkdir(psc_tc_dir)
if not os.path.exists(opj(psc_tc_dir, sub)): # 
    os.mkdir(opj(psc_tc_dir, sub))
if not os.path.exists(opj(psc_tc_dir, sub, ses)): # 
    os.mkdir(opj(psc_tc_dir, sub, ses))

space       = 'fsnative'
file_ending = "desc-denoised_bold.npy"
inputdir    = opj(derivatives_dir, 'pybest', sub, ses, 'unzscored')
outputdir   = opj(psc_tc_dir, sub, ses)



# Load design matrices     
dm = amb_load_dm(['prf', 'sf_vect'])
# Function to id the baseline period from DM
# -> we want TRs without stimulation, which are not right at the start (cut_vols)
# -> also not just after a bar (n_trs)
def baseline_from_dm(dm, n_trs=7, cut_vols=5):
    shifted_dm = np.zeros_like(dm)
    if len(dm.shape)==3:
        shifted_dm[..., n_trs:] = dm[..., :-n_trs]
        base_bool = (np.sum(dm, axis=(0, 1)) == 0) & (np.sum(shifted_dm, axis=(0, 1)) == 0)
    else:
        shifted_dm[n_trs:] = dm[:-n_trs]
        base_bool = (dm== 0) & (shifted_dm== 0)
    
    base_bool[0:cut_vols] = 0
    base_time = np.where(base_bool)[0]
    return base_time

for task in task_list:
    search_for = [sub, ses, task, space, file_ending, 'run-']
    files = lsutils.get_file_from_substring(search_for, inputdir)
    print("Loading in data", flush=True)
    for ff in files:
        print(f" {ff}", flush=True)    
    # convert to psc according to baseline (as per Serge Dumoulin).
    print("Converting to percent signal change and fix baseline", flush=True)
    
    # -> what is the best way to get baselines? 
    # default baseline is 19 volumes (from Serge)
    # Periods without stim, for at least 7 TR, also not the 1st 5 TRs (i.e., cut_vol) 
    if 'pRF' in task:
        baseline = baseline_from_dm(dm['prf'], n_trs=7, cut_vols=5)
    elif 'CSF' in task:
        baseline = baseline_from_dm(dm['sf_vect'], n_trs=14, cut_vols=5)
        
    # get unique run-IDs
    run_ids = []
    for ii in files:
        run_ids.append(lsutils.split_bids_components(ii)["run"])

    run_ids = np.unique(np.array(run_ids))
    # chunk into L/R pairs
    hemi_pairs = []

    for run in run_ids:
        pair = []
        for ii in ["L","R"]:
            pair.append(lsutils.get_file_from_substring([f"run-{run}_", f"hemi-{ii}"], files))

        hemi_pairs.append(pair)

    # load them in
    tc_data = []
    for pair in hemi_pairs:        
        hemi_data = [lsutils.percent_change(np.load(pair[ix]), 0, baseline=baseline) for ix in range(len(pair))]        
        tc_data.append(np.hstack(hemi_data))

    # take median of data
    m_tc_data = np.median(np.array(tc_data), 0)

    if space == "fsnative":
        # vertices per hemi
        n_verts = [ii.shape[-1] for ii in hemi_data]

        # check if this matches with FreeSurfer surfaces
        n_verts_fs = []
        for i in ['lh', 'rh']:
            surf = opj(freesurfer_dir, sub, 'surf', f'{i}.white')
            verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
            n_verts_fs.append(verts)

        if n_verts_fs != n_verts:
            raise ValueError(f"Mismatch between number of vertices in pRF-analysis ({n_verts}) and FreeSurfer ({n_verts_fs})..?\nYou're probably using an older surface reconstruction. Check if you've re-ran fMRIprep again with new FreeSurfer-segment")

    # save files
    
    print("Saving averaged data", flush=True)
    out = f"{sub}_{ses}_{task}"    
    np.save(opj(outputdir, f'{out}_hemi-LR_desc-avg_bold.npy'),m_tc_data)
    np.save(opj(outputdir, f'{out}_hemi-L_desc-avg_bold.npy'), m_tc_data[:,:n_verts[0]])
    np.save(opj(outputdir, f'{out}_hemi-R_desc-avg_bold.npy'), m_tc_data[:,n_verts[0]:])