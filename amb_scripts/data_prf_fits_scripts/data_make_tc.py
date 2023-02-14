#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import ast
import getopt
from linescanning import (
    prf,
    utils,
    # dataset
)
import numpy as np
import nibabel as nb
import os
from scipy import io
import sys
import warnings
import json
import pickle
warnings.filterwarnings('ignore')
opj = os.path.join

from pfa_scripts.load_saved_info import get_design_matrix_npy, get_real_tc, get_roi
from pfa_scripts.utils import hyphen_parse

source_data_dir = os.getenv("DIR_DATA_SOURCE")
derivatives_dir = os.getenv("DIR_DATA_DERIV")
prf_dir = opj(derivatives_dir, 'prf')

# Check if xpred_fit_dir exists 
if not os.path.exists(prf_dir): # Does xparams_dir exist? No? -> make it
    os.mkdir(prf_dir)

def main(argv):

    """
---------------------------------------------------------------------------------------------------
based on call_prf (from JH), edited by MD

Wrapper to get the time series data & design matrices. 
design matrix based on a specified path to screenshots as outputted by the pRF-experiment script.
If you're not running that particular experiment, you'll need to create a design matrix yourself.
Assumes the input is BIDS compliant, as it will extract certain features from the filenames.
Currently compatible with output from pybest (ending with "*desc-denoised_bold.npy") and from fMRI-
Prep (ending with "*bold.func.gii"). It will throw an error if neither of these conditions are met.
If you have a different case (e.g., nifti's), please open an issue so we can deal with that. We'll 
select files from the input directory based on 'space-' (check spinoza_setup)/'run-'/'task-IDs. They
are then paired into left+right hemisphere. Finally, the median over all runs is calculated and in-
serted in the model-fitting object.

Arguments:
    -s|--sub    <sub number>        number of subject's FreeSurfer directory from which you can 
                                    omit "sub-" (e.g.,for "sub-001", enter "001").
    -t|--task   <task name>         name of the experiment performed (e.g., "2R")

Options:                                  
    -v|--verbose    print some stuff to a log-file
    --overwrite     If specified, we'll overwrite existing Gaussian parameters. If not, we'll look
                    for a file with ['model-gauss', 'stage-iter', 'params.npy'] in *outputdir* and,
                    if it exists, inject it in the normalization model (if `model=norm`)  

Example:


---------------------------------------------------------------------------------------------------
"""

    sub         = None
    task        = None
    ses         = 'ses-1'
    psc         = True
    space       = 'fsnative'
    verbose     = True
    file_ending = "desc-denoised_bold.npy"
    cut_vols    = 0

    try:
        opts = getopt.getopt(argv,"ghs:n:t:",["help", "sub=", "ses=", "task=", "space=", "verbose", "file_ending=", "zscore", "overwrite", "raw", "cut_vols="])[0]
    except getopt.GetoptError:
        print("ERROR while reading arguments; did you specify an illegal argument?")
        print(main.__doc__)
        sys.exit(2)
    
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            sub = hyphen_parse('sub', arg)
        elif opt in ("-n", "--ses"):
            ses = hyphen_parse('ses', str(arg))
        elif opt in ("-t", "--task"):
            task = hyphen_parse('task', arg)
        elif opt in ("-u", "--space"):
            space = arg
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("--file_ending"):
            file_ending = arg
        elif opt in ("--zscore"):
            psc = False
        elif opt in ("--raw"):
            psc = False            
        elif opt in ("--cut_vols"):
            cut_vols = int(arg)

    if len(argv) < 2:
        print(main.__doc__)
        sys.exit()

    if not os.path.exists(opj(prf_dir, sub)): 
        os.mkdir(opj(prf_dir, sub))
    if not os.path.exists(opj(prf_dir, sub, ses)): 
        os.mkdir(opj(prf_dir, sub, ses))    

    outputdir   = opj(prf_dir, sub, ses)
    inputdir    = opj(derivatives_dir, 'pybest', sub, ses, 'unzscored')

    out = f"{sub}"
    if ses != None:
        out += f"_{ses}"
    
    if task != None:
        out += f"_{task}"

    # search for space-/task-/ and file ending; add run as well to avoid the concatenated version being included
    # no space means native BOLD
    search_for = ["run-", task, file_ending]
    if space != None:
        search_for += [f"space-{space}"]

    files = utils.get_file_from_substring(search_for, inputdir)
    
    if verbose:
        print("Loading in data", flush=True)
        for ff in files:
            print(f" {ff}", flush=True)
    
    # convert to psc according to baseline (as per Serge Dumoulin).
    if psc:
        if verbose:
            print("Converting to percent signal change and fix baseline", flush=True)
        # default baseline is 19 volumes
        baseline = 19-cut_vols
        
    # get unique run-IDs
    run_ids = []
    for ii in files:
        run_ids.append(utils.split_bids_components(ii)["run"])

    run_ids = np.unique(np.array(run_ids))

    # chunk into L/R pairs
    hemi_pairs = []

    for run in run_ids:
        pair = []
        for ii in ["L","R"]:
            pair.append(utils.get_file_from_substring([f"run-{run}_", f"hemi-{ii}"], files))

        hemi_pairs.append(pair)

    # load them in
    prf_tc_data = []
    for pair in hemi_pairs:
        
        if psc:
            hemi_data = [utils.percent_change(np.load(pair[ix]), 0, baseline=baseline) for ix in range(len(pair))]
        else:
            hemi_data = [np.load(pair[ix]) for ix in range(len(pair))]
        
        prf_tc_data.append(np.hstack(hemi_data))

    # take median of data
    m_prf_tc_data = np.median(np.array(prf_tc_data), 0)


    if space == "fsnative":
        # vertices per hemi
        n_verts = [ii.shape[-1] for ii in hemi_data]

        # check if this matches with FreeSurfer surfaces
        n_verts_fs = []
        for i in ['lh', 'rh']:
            surf = opj(os.environ.get('SUBJECTS_DIR'), sub, 'surf', f'{i}.white')
            verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
            n_verts_fs.append(verts)

        if n_verts_fs != n_verts:
            raise ValueError(f"Mismatch between number of vertices in pRF-analysis ({n_verts}) and FreeSurfer ({n_verts_fs})..?\nYou're probably using an older surface reconstruction. Check if you've re-ran fMRIprep again with new FreeSurfer-segment")

    # cut volumes at the beginning of the timeseries. Also subtract number of volumes from baseline
    if verbose:
        print(f"Cutting {cut_vols} from beginning of timeseries", flush=True)
    
    m_prf_tc_data = m_prf_tc_data[cut_vols:,:]
    # design = design_matrix[list(design_matrix.keys())[-1]][...,cut_vols:]

    # save files
    if verbose:
        print("Saving averaged data", flush=True)
        
    np.save(opj(outputdir, f'{out}_hemi-LR_desc-avg_bold.npy'), m_prf_tc_data)
    np.save(opj(outputdir, f'{out}_hemi-L_desc-avg_bold.npy'), m_prf_tc_data[:,:n_verts[0]])
    np.save(opj(outputdir, f'{out}_hemi-R_desc-avg_bold.npy'), m_prf_tc_data[:,n_verts[0]:])

if __name__ == "__main__":
    main(sys.argv[1:])
