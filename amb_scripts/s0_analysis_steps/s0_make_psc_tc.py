#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import getopt

import numpy as np
import os
import sys
import warnings
import yaml
import pickle
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')
opj = os.path.join

import numpy as np  
import nibabel as nb
import os
import linescanning.utils as lsutils
opj = os.path.join

from amb_scripts.load_saved_info import *
from dag_prf_utils.utils import *

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'
freesurfer_dir = opj(derivatives_dir, 'freesurfer')
settings_path = get_yml_settings_path()

def main(argv):

    """
---------------------------------------------------------------------------------------------------

Convert the pybest outputs to psc (average over runs)

Args:
    -s (--sub=)         e.g., 01
    -n (--ses=)
    -t (--task=)        pRFLE,pRFRE,CSFRE,CSFLE



---------------------------------------------------------------------------------------------------
    """
    print('\n\n')
    # Specify
    sub = None
    ses = None
    task = None

    try:
        opts = getopt.getopt(argv,"h:s:n:t:",[
            "help=", "sub=", "ses=", "task="])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-q':
            print(main.__doc__)
            sys.exit()

        elif opt in ("-s", "--sub"):
            sub = dag_hyphen_parse('sub', arg)
        elif opt in ("-n", "--ses"):
            ses = dag_hyphen_parse('ses', arg)
        elif opt in ("-t", "--task"):
            task = arg

    if len(argv) < 1:
        print("NOT ENOUGH ARGUMENTS SPECIFIED")
        print(main.__doc__)
        sys.exit()

    psc_tc_dir = opj(derivatives_dir, 'psc_tc')
    if not os.path.exists(psc_tc_dir): # 
        os.mkdir(psc_tc_dir)
    if not os.path.exists(opj(psc_tc_dir, sub)): # 
        os.mkdir(opj(psc_tc_dir, sub))
    if not os.path.exists(opj(psc_tc_dir, sub, ses)): # 
        os.mkdir(opj(psc_tc_dir, sub, ses))
    out = f"{sub}_{ses}_{task}"

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

    search_for = [sub, ses, task, space, file_ending, 'run-']
    files = dag_find_file_in_folder(search_for, inputdir)
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
    if (sub=='sub-02') & ('CSF' in task):
        print('ONLY DOING RUNS 1 & 2 & 3')
        run_ids = ['1', '2', '3']
    # chunk into L/R pairs
    hemi_pairs = []

    for run in run_ids:
        pair = []
        for ii in ["L","R"]:
            pair.append(dag_find_file_in_folder([f"run-{run}_", f"hemi-{ii}"], files))

        hemi_pairs.append(pair)

    # load them in
    tc_data = []
    mepi_data = []
    for pair in hemi_pairs:        
        hemi_data = [lsutils.percent_change(np.load(pair[ix]), 0, baseline=baseline) for ix in range(len(pair))]                
        tc_data.append(np.hstack(hemi_data))
        mepi = [np.mean(np.load(pair[ix]), axis=0) for ix in range(len(pair))]        
        mepi_data.append(np.hstack(mepi))
    mepi_data = np.array(mepi_data)
    # SAVE THE MEAN EPI
    np.save(opj(outputdir, f'{out}_hemi-LR_desc-mepi.npy'),mepi_data)

    tc_data = np.array(tc_data)
    
    # Find correlation between 2 halves of the runs    
    half_1 = tc_data.shape[0] // 2
    h1_ts = np.mean(tc_data[:half_1,:,:], axis=0)
    h2_ts = np.mean(tc_data[half_1:,:,:], axis=0)
    print(h1_ts.shape)
    print(h2_ts.shape)

    std_mask     = h1_ts.std(axis=0) != 0
    std_mask    &= h2_ts.std(axis=0) != 0
    std_idx = np.where(std_mask)[0]
    # run_correlation = np.zeros(h1_ts.shape[-1])
    # i_count = 0
    # for i in std_idx:
    #     run_correlation[i] = np.corrcoef(h1_ts[:, i], h2_ts[:, i])[0,1]
    #     i_count += 1
    #     if i_count % 5000 == 0:
    #         print(f'Calculating correlation for voxel {i_count} of {h1_ts.shape[0]}')

    # print(f'Run correlation: {run_correlation}')
    # # -> save it
    # np.save(opj(outputdir, f'{out}_hemi-LR_desc-run_corr.npy'),run_correlation)

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
    np.save(opj(outputdir, f'{out}_hemi-LR_desc-avg_bold.npy'),m_tc_data)
    np.save(opj(outputdir, f'{out}_hemi-L_desc-avg_bold.npy'), m_tc_data[:,:n_verts[0]])
    np.save(opj(outputdir, f'{out}_hemi-R_desc-avg_bold.npy'), m_tc_data[:,n_verts[0]:])


if __name__ == "__main__":
    main(sys.argv[1:])