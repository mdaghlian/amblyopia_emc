import numpy as np
import scipy.io
import os
import sys
import yaml
import pickle
opj = os.path.join

import nibabel as nb
from prfpy.stimulus import PRFStimulus2D, CSenFStimulus

# import linescanning.utils as lsutils
import pandas as pd
# from .utils import print_p
# from collections import defaultdict as dd
# import cortex

# from .utils import hyphen_parse, coord_convert
from dag_prf_utils.utils import *

source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#os.getenv("DIR_DATA_SOURCE")
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'#os.getenv("DIR_DATA_DERIV")
freesurfer_dir = opj(derivatives_dir, 'freesurfer')
default_prf_dir = opj(derivatives_dir, 'prf')
dm_dir = opj(os.path.dirname(os.path.realpath(__file__)), 'dm_files' )
psc_tc_dir = opj(derivatives_dir, 'psc_tc')
qCSF_dir = opj(derivatives_dir, 'qCSF')

def get_yml_settings_path():
    yml_path = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/code/amb_code/amb_scripts/amb_fit_settings.yml'
    return yml_path

def amb_load_fit_settings(sub, task_list, model_list, **kwargs):
    fit_settings = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='settings', **kwargs)
    return fit_settings

def amb_load_pred_tc(sub, task_list, model_list, **kwargs):
    pred_tc = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='predictions', **kwargs)
    return pred_tc

def amb_load_prf_params(sub, task_list, model_list, **kwargs):
    prf_params = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, key='pars', **kwargs)
    return prf_params

def amb_load_pkl_key(sub, task_list, model_list, key, **kwargs):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(model_list, list):
        model_list = [model_list]        
    
    prf_dict = {}
    for task in task_list:
        prf_dict[task] = {}
        for model in model_list:
            this_pkl = amb_load_pkl(sub=sub, task=task, model=model, **kwargs)
            prf_dict[task][model] = this_pkl[key]
    return prf_dict

def amb_load_pkl(sub, task, model, **kwargs):
    '''
    linescanning toolbox nicely saves everything into a pickle
    this will load the correct pickle associated with the correct, sub, ses, model and task
    roi_fit specifies which fitting run was used.  
    '''    
    if 'pRF' in task:
        # amb_prf_dir = opj(derivatives_dir, 'amb-prf')
        amb_prf_dir = opj(derivatives_dir, 'prf')
    else:
        # amb_prf_dir = opj(derivatives_dir, 'amb-csf')
        amb_prf_dir = opj(derivatives_dir, 'csf')


    dir_to_search = opj(amb_prf_dir, sub, 'ses-1')
    include = kwargs.get("include", []) # any extra details to search for in file name
    exclude = kwargs.get("exclude", []) # any extra details to search for in file name
    roi_fit = kwargs.get('roi_fit', 'all')    
    fit_stage = kwargs.get('fit_stage', 'iter')

    # the folder specified in "dir_to_search" will contain several files
    # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
    # Now we need to find the relevant file, by filtering for key terms (see included)
    include += [sub, model, task, roi_fit, fit_stage] # Make sure we get the correct model and task (& subject)    
    exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

    data_path = dag_find_file_in_folder(filt=include, path=dir_to_search, exclude=exclude)
    if isinstance(data_path, list):
        print(f'Error, more than 1 match ({len(data_path)} files)')
        sys.exit()

    pkl_file = open(data_path,'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()     

    return data    

def amb_load_real_tc(sub, task_list, clip_start=0):
    if not isinstance(task_list, list):
        task_list = [task_list]
    this_dir = opj(psc_tc_dir, sub, 'ses-1')
    real_tc = {}
    for task in task_list:
        real_tc_file = dag_find_file_in_folder([task, 'hemi-LR_desc-avg_bold'], this_dir)
        if isinstance(real_tc_file, list):
            print(f'Error, more than 1 match ({len(real_tc_file)} files)')
            sys.exit()
        unclipped = np.load(real_tc_file).T        
        real_tc[task] = np.copy(unclipped[:,clip_start::])

    return real_tc

def amb_load_real_tc_run(sub, task_list, run_list):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(run_list, list):
        run_list=[run_list]
    unz_dir = opj(derivatives_dir, 'pybest', sub, 'ses-1', 'unzscored')
    real_tc = {}
    for task in task_list:
        real_tc[task] = []
        for run in run_list:
            LH_real_tc_file = dag_find_file_in_folder(
                [task, f'run-{run}', 'fsnative', 'hemi-R_desc-denoised_bold'], unz_dir)
            LH_tc = np.load(LH_real_tc_file)
            RH_real_tc_file = lsutils.get_file_from_substring(
                [task, f'run-{run}', 'fsnative', 'hemi-R_desc-denoised_bold'], unz_dir)
            RH_tc = np.load(RH_real_tc_file)
            real_tc[task].append(np.concatenate([LH_tc, RH_tc], axis=1).T)

    return real_tc


def amb_load_dm(dm_types):
    
    if not isinstance(dm_types, list):
        dm_types = [dm_types]
    
    dm = {}
    for dm_type in dm_types:
        if not dm_type in ['sf_vect', 'c_vect', 'prf', 'csf']:
            print(f'Could not find dm type {dm_type}')        
            print(f'Must be - sf_vect, c_vect, prf')
            sys.exit()
        if dm_type == 'sf_vect'        :
            dm[dm_type] = np.squeeze(scipy.io.loadmat(opj(dm_dir, 'sf_vect.mat'))['sf_vect'])
        elif dm_type == 'c_vect':
            dm[dm_type] = np.squeeze(scipy.io.loadmat(opj(dm_dir, 'contrasts_vect.mat'))['contrasts_vect'])
        elif dm_type == 'prf':
            dm[dm_type] = np.load(opj(dm_dir, 'prf_design_matrix.npy'))

    return dm

def amb_load_prfpy_stim(dm_type='pRF', clip_start=0):
    if dm_type=='pRF':
        screen_info_path = opj(dm_dir, 'screen_info.yml')
        with open(screen_info_path) as f:
            screen_info = yaml.safe_load(f)

        dm_prf = amb_load_dm('prf')['prf'][:,:,clip_start::]    
        prfpy_stim = PRFStimulus2D(
            screen_size_cm    =screen_info['screen_size_cm'],
            screen_distance_cm=screen_info['screen_distance_cm'],
            design_matrix=dm_prf, 
            axis=0,
            TR=screen_info['TR']
            )
    elif dm_type=='CSF':
        csf_dm = amb_load_dm(['sf_vect', 'c_vect'])
        sf_vect = csf_dm['sf_vect'][clip_start::]
        c_vect = csf_dm['c_vect'][clip_start::]

        # Number of stimulus types:
        u_sfs = np.sort(list(set(sf_vect))) # unique SFs
        u_sfs = u_sfs[u_sfs>0]
        u_con = np.sort(list(set(c_vect)))
        u_con = u_con[u_con>0]
        prfpy_stim = CSenFStimulus(
            SFs = u_sfs,#,
            CONs = u_con,
            SF_seq=sf_vect,
            CON_seq = c_vect,
            TR=1.5,
        )


    return prfpy_stim    
def amb_load_qcsf(sub, eye_list, ses='ses-1'):
    if not isinstance(eye_list, list):
        eye_list = [eye_list]
    this_dir = opj(qCSF_dir, sub, ses)
    qCSF_info = {}
    for eye in eye_list:        
        qCSF_file = dag_find_file_in_folder([f'eye-{eye}', 'struct', '.mat'], this_dir)
        if isinstance(qCSF_file, list):
            print(f'Error, more than 1 match ({len(qCSF_file)} files)')
            sys.exit()
        mat_struct = scipy.io.loadmat(qCSF_file).get('qCSF_struct')
        qCSF_info[eye] = mat_struct_to_python_dict(mat_struct)
    
    
    return qCSF_info

def mat_struct_to_python_dict(mat_struct):
    # mat_struct = scipy.io.loadmat(file/path).get('entry_of_interest')
    recordarr = np.rec.array(mat_struct)
    py_dict = {}
    myList = []
    for field in recordarr.dtype.names: #iterates through field names of numpy array
        for array in recordarr[field]: #iterates through the array of each numpy array                                    
            for value in array:
                myList.append(np.squeeze(value.flatten()))
                #print(np.squeeze(value.flatten()[0]))
        
        # print(np.array(myList).shape)
        py_dict[field] = np.squeeze(np.array(myList))
        myList = []

    return py_dict
def amb_load_nverts(sub):
    n_verts = []
    for i in ['lh', 'rh']:
        surf = opj(freesurfer_dir, sub, 'surf', f'{i}.white')
        verts = nb.freesurfer.io.read_geometry(surf)[0].shape[0]
        n_verts.append(verts)
    return n_verts

def amb_load_roi(sub, label, **kwargs):
    '''
    Return a boolean array of voxels included in the specified roi
    array is vector with each entry corresponding to a point on the subjects cortical surface
    (Note this is L & R hemi combined)

    roi can be a list (in which case more than one is included)
    roi can also be exclusive (i.e., everything *but* x)

    TODO - conjunctive statements (not)
    '''
    roi_idx = dag_load_roi(sub=sub, roi=label, fs_dir=opj(derivatives_dir, 'freesurfer'), **kwargs)

    return roi_idx    

# def 

# def load_prf_data(sub, ses, model, task, dir_to_search, roi_fit='all', dm_fit='standard', **kwargs):
#     '''
#     linescanning toolbox nicely saves everything into a pickle
#     this will load the correct pickle associated with the correct, sub, ses, model and task
#     roi_fit specifies which fitting run was used.  
#     I also ran the fitting using only the V1 voxels. 
#     This was to speed up the fitting, so that I could use the trust constrained on the normalization model too
#     BUT - this gives a problem, when we search for the parameter file, it now comes up with two matches; e.g., 
#     >> sub-01_ses-1_task-AS0_model-norm_stage-iter_desc-prf_params.pkl
#     >> sub-01_ses-1_task-AS0_roi-V1_model-norm_stage-iter_desc-prf_params.pkl

#     So to solve this, I am doing this...
#     >> adding roi_fit='all' to the default...    
#     '''
#     fit_stage = kwargs.get('fit_stage', 'iter')
#     include = kwargs.get("include", [])
#     exclude = kwargs.get("exclude", [])
    
#     # Caught exceptions - now lets continue...
#     # the folder specified in "dir_to_search" will contain several files
#     # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
#     # Now we need to find the relevant file, by filtering for key terms (see included)
#     include += [sub, model, task, roi_fit, fit_stage] # Make sure we get the correct model and task (& subject)    
#     exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

#     data_path = utils.get_file_from_substring(filt=include, path=dir_to_search, exclude=exclude)

#     # If the exclude is used, it returns a list of length one
#     if isinstance(data_path, list) and (len(data_path)==1): # check whether this happened 
#         data_path = data_path[0] # (we only want to do this if the list is length 1 - we want errors if there are more than 1 possible matches...)         

#     if '.npy' in data_path:
#         # Load numpy data
#         data = np.load(data_path)
#     elif '.pkl' in data_path:
#         pkl_file = open(data_path,'rb')
#         data = pickle.load(pkl_file)
#         pkl_file.close()     

#     return data

# def get_fit_settings(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
#     '''
#     This will get the fitting settings stored in the pickle file associated with this model & task
    
#     '''
#     if isinstance(task_list, str):
#         task_list = [task_list]
#     if isinstance(model_list, str):
#         model_list = [model_list]

#     fit_settings  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)
#         fit_settings[task] = {}
#         for model in model_list:   
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
#             fit_settings[task][model] = this_pkl['settings']
#     return fit_settings

# def get_model_params(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard', fit_stage='iter'):
#     # Turn any strings into lists
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     if not isinstance(model_list, list):
#         model_list = [model_list]

#     model_params  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)

#         model_params[task] = {}
#         for model in model_list:
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit, fit_stage=fit_stage)
#             model_params[task][model] = this_pkl['pars']            
#     return model_params

# def get_number_of_vx(sub):
#     # Do this by loading the ROI mask for v1
#     try:
#         roi_idx = get_roi(sub, label='V1')
#         num_vx = roi_idx.shape[0]
    
#     except:
#         real_tc = get_real_tc(sub, 'task-AS0')['task-AS0']
#         num_vx = real_tc.shape[0]
    
#     return num_vx

# def get_roi(sub, label):    
#     if label=='all':
#         num_vx = get_number_of_vx(sub)
#         roi_idx = np.ones(num_vx, dtype=bool)
#         return roi_idx
    
#     if not isinstance(label, list):
#         label = [label]
    
#     roi_idx = []
#     for this_label in label:
#         if "not" in this_label:
#             this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label.split("-")[-1]}.npy')        
#             this_roi_idx = np.load(this_roi_file)
#             this_roi_idx = this_roi_idx==0
#         else:
#             this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label}.npy')
#             this_roi_idx = np.load(this_roi_file)
        
#         roi_idx.append(this_roi_idx)
#     roi_idx = np.vstack(roi_idx)
#     roi_idx = roi_idx.any(0)
#     return roi_idx

# def get_real_tc(sub, task_list, prf_dir=default_prf_dir):
#     if isinstance(task_list, str):
#         task_list = [task_list]

#     real_tc  = {}
#     for task in task_list:'            
#         this_dir = opj(prf_dir, sub, ses)
#         real_tc_path = utils.get_file_from_substring([sub, ses, task, 'hemi-LR', 'desc-avg_bold'], this_dir, exclude='roi')
#         if isinstance(real_tc_path, list) and (len(real_tc_path)==1):
#             real_tc_path = real_tc_path[0]
#         real_tc[task] = np.load(real_tc_path).T

#     return real_tc

# def get_pred_tc(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
#     # Turn any strings into lists
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     if not isinstance(model_list, list):
#         model_list = [model_list]

#     pred_tc  = {}
#     for task in task_list:
#         if "AS" in task:
#             ses='ses-1'
#         elif "2R" in task:
#             ses='ses-2'            
#         this_dir = opj(prf_dir, sub, ses)

#         pred_tc[task] = {}
#         for model in model_list:
#             this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
#             pred_tc[task][model] = this_pkl['predictions']            

#     return pred_tc

# def get_design_matrix_npy(task_list, prf_dir=default_prf_dir):
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     this_dir = opj(prf_dir)
#     dm_npy  = {}    
#     for task in task_list:
#         dm_path = utils.get_file_from_substring(['design', task], this_dir)        
#         dm_npy[task] = scipy.io.loadmat(dm_path)['stim']

#     return dm_npy

# def get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir):
#     if not isinstance(task_list, list):
#         task_list = [task_list]
#     dm_npy = get_design_matrix_npy(task_list, prf_dir=prf_dir)
#     model_list = ['gauss']     # stimulus settings are the same for both norm & gauss models  (so only use gauss) 
#     stim_settings = get_fit_settings(sub,task_list, model_list=model_list, prf_dir=prf_dir)
#     prfpy_stim = {}
#     for task in task_list:
#         prfpy_stim[task] = PRFStimulus2D(
#             screen_size_cm=stim_settings[task][model_list[0]]['screen_size_cm'],
#             screen_distance_cm=stim_settings[task][model_list[0]]['screen_distance_cm'],
#             design_matrix=dm_npy[task], 
#             axis=0,
#             TR=stim_settings[task][model_list[0]]['TR']
#             )    
#     return prfpy_stim



# # # ********************
# # def load_params_generic(params_file, load_all=False, load_var=[]):
# #     """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

# #     if isinstance(params_file, str):
# #         if params_file.endswith('npy'):
# #             params = np.load(params_file)
# #         elif params_file.endswith('pkl'):
# #             with open(params_file, 'rb') as input:
# #                 data = pickle.load(input)
            
# #             if len(load_var)==1:
# #                 params = data[load_var[0]]
# #             elif len(load_var)>1:
# #                 params = {}
# #                 # Load the specified variables
# #                 for this_var in load_var:
# #                     params[this_var] = data[this_var]
# #             elif load_all:
# #                 params = {}
# #                 for this_var in data.keys():
# #                     params[this_var] = data[this_var]
# #             else:
# #                 params = data['pars']

# #     elif isinstance(params_file, np.ndarray):
# #         params = params_file.copy()
# #     elif isinstance(params_file, pd.DataFrame):
# #         dict_keys = list(params_file.keys())
# #         if not "hemi" in dict_keys:
# #             # got normalization parameter file
# #             params = np.array((params_file['x'][0],
# #                                 params_file['y'][0],
# #                                 params_file['prf_size'][0],
# #                                 params_file['A'][0],
# #                                 params_file['bold_bsl'][0],
# #                                 params_file['B'][0],
# #                                 params_file['C'][0],
# #                                 params_file['surr_size'][0],
# #                                 params_file['D'][0],
# #                                 params_file['r2'][0]))
# #         else:
# #             raise NotImplementedError()
# #     else:
# #         raise ValueError(f"Unrecognized input type for '{params_file}'")

# #     return params


