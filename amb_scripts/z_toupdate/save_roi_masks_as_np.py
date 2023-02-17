#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import cortex
import numpy as np
from pfa_scripts.load_saved_info import get_number_of_vx
from pfa_scripts.utils import get_roi_idx_from_dot_label
import linescanning.utils as lsutils
import os
opj = os.path.join
import nibabel as nb
import sys

prf_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives/prf/'
freesurfer_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives/freesurfer/'
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/pilot1/derivatives/'

sub = sys.argv[1]

# [1] All the ones in the standard labels bit...
numpy_roi_idx = opj(derivatives_dir, 'numpy_roi_idx')
if not os.path.exists(numpy_roi_idx):
    os.mkdir(numpy_roi_idx)
if not os.path.exists(opj(numpy_roi_idx, sub)):
    os.mkdir(opj(numpy_roi_idx, sub))

roi_label_dir = opj(freesurfer_dir, sub, 'label')
num_vx = get_number_of_vx(sub)
list_of_roi_files = lsutils.get_file_from_substring(['.thresh', '.label'], roi_label_dir)
list_of_rois = [list_of_roi_files[i].split('/')[-1] for i in range(len(list_of_roi_files))] # Select only the file name
# remove the lh, rh, and .label stuff...
list_of_rois = [list_of_rois[i].replace('lh.','') for i in range(len(list_of_roi_files))]
list_of_rois = [list_of_rois[i].replace('rh.','') for i in range(len(list_of_roi_files))]
list_of_rois = [list_of_rois[i].replace('.label','') for i in range(len(list_of_roi_files))]
list_of_rois = list(set(list_of_rois)) # remove duplicates
for roi in list_of_rois:
    roi_label_to_save = opj(numpy_roi_idx, sub, f"{roi.replace('.thresh','')}.npy")
    roi_where = cortex.freesurfer.get_label(sub,label=roi)[0]
    roi_idx = np.zeros(num_vx, dtype=bool)
    roi_idx[roi_where] = True
    np.save(roi_label_to_save, roi_idx)

# Now do it for any *custom* rois in this folder:
list_of_roi_files = lsutils.get_file_from_substring(['custom', '.label'], roi_label_dir)
list_of_rois = [list_of_roi_files[i].split('/')[-1] for i in range(len(list_of_roi_files))] # Select only the file name
# remove the lh, rh, and .label stuff...
list_of_rois = [list_of_rois[i].replace('lh.','') for i in range(len(list_of_roi_files))]
list_of_rois = [list_of_rois[i].replace('rh.','') for i in range(len(list_of_roi_files))]
list_of_rois = [list_of_rois[i].replace('.label','') for i in range(len(list_of_roi_files))]
list_of_rois = list(set(list_of_rois)) # remove duplicates
for roi in list_of_rois:
    roi_label_to_save = opj(numpy_roi_idx, sub, f"{roi}.npy")
    try:
        roi_where = cortex.freesurfer.get_label(sub,label=roi)[0]
        roi_idx = np.zeros(num_vx, dtype=bool)
        roi_idx[roi_where] = True
        np.save(roi_label_to_save, roi_idx)
    except:
        pass


# Now do the custom ROI files - attn_rois

attn_roi_dir = opj(roi_label_dir, 'attn_rois')
if os.path.exists(attn_roi_dir):
    list_of_roi_files = lsutils.get_file_from_substring(['custom', '.label'], attn_roi_dir)
    list_of_rois = [list_of_roi_files[i].split('/')[-1] for i in range(len(list_of_roi_files))] # Select only the file name
    # remove the lh, rh, and .label stuff...
    list_of_rois = [list_of_rois[i].replace('lh.','') for i in range(len(list_of_roi_files))]
    list_of_rois = [list_of_rois[i].replace('rh.','') for i in range(len(list_of_roi_files))]
    list_of_rois = [list_of_rois[i].replace('.label','') for i in range(len(list_of_roi_files))]
    list_of_rois = list(set(list_of_rois)) # remove duplicates
    for roi in list_of_rois:
        roi_label_to_save = opj(numpy_roi_idx, sub, f"{roi}.npy")

        LR_bool = []
        for ih in ['lh.', 'rh.']:
            # first 2 lines are not useful (look something like this):
            # #!ascii label  , from subject sub-005 vox2ras=TkReg\n
            # 5838\n
            hemi_roi_file = opj(attn_roi_dir, f'{ih}{roi}.label')
            surf = opj(os.environ.get('SUBJECTS_DIR'), sub, 'surf', f'{ih}white')
            verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0] # Number of vertices in hemi
            try:
                with open(hemi_roi_file) as f:
                    contents = f.readlines()

                idx_str = [contents[i].split(' ')[0] for i in range(2,len(contents))]
                idx_int = [int(idx_str[i]) for i in range(len(idx_str))]

            except:
                idx_int = np.zeros(verts, dtype=bool)
                print(f'could not find {ih} for {roi}, assuming none')

            this_bool = np.zeros(verts, dtype=bool)
            this_bool[idx_int] = True

            LR_bool.append(this_bool)
        LR_bool = np.concatenate(LR_bool)    
        np.save(roi_label_to_save, LR_bool)