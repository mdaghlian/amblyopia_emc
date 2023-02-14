import numpy as np
import scipy.io
from linescanning import utils
import yaml
import pickle
import os
from prfpy.stimulus import PRFStimulus2D
import pandas as pd
from collections import defaultdict as dd
def dd_func():
    return "Not present"
# import cortex
opj = os.path.join


source_data_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/sourcedata'#os.getenv("DIR_DATA_SOURCE")
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'#os.getenv("DIR_DATA_DERIV")
default_prf_dir = opj(derivatives_dir, 'prf')

def load_prf_data(sub, ses, model, task, dir_to_search, roi_fit='all', dm_fit='standard', **kwargs):
    '''
    linescanning toolbox nicely saves everything into a pickle
    this will load the correct pickle associated with the correct, sub, ses, model and task
    roi_fit specifies which fitting run was used.  
    I also ran the fitting using only the V1 voxels. 
    This was to speed up the fitting, so that I could use the trust constrained on the normalization model too
    BUT - this gives a problem, when we search for the parameter file, it now comes up with two matches; e.g., 
    >> sub-01_ses-1_task-AS0_model-norm_stage-iter_desc-prf_params.pkl
    >> sub-01_ses-1_task-AS0_roi-V1_model-norm_stage-iter_desc-prf_params.pkl

    So to solve this, I am doing this...
    >> adding roi_fit='all' to the default...    
    '''
    fit_stage = kwargs.get('fit_stage', 'iter')
    include = kwargs.get("include", [])
    exclude = kwargs.get("exclude", [])
    
    # Caught exceptions - now lets continue...
    # the folder specified in "dir_to_search" will contain several files
    # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
    # Now we need to find the relevant file, by filtering for key terms (see included)
    include += [sub, model, task, roi_fit, fit_stage] # Make sure we get the correct model and task (& subject)    
    exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

    data_path = utils.get_file_from_substring(filt=include, path=dir_to_search, exclude=exclude)

    # If the exclude is used, it returns a list of length one
    if isinstance(data_path, list) and (len(data_path)==1): # check whether this happened 
        data_path = data_path[0] # (we only want to do this if the list is length 1 - we want errors if there are more than 1 possible matches...)         

    if '.npy' in data_path:
        # Load numpy data
        data = np.load(data_path)
    elif '.pkl' in data_path:
        pkl_file = open(data_path,'rb')
        data = pickle.load(pkl_file)
        pkl_file.close()     

    return data

def get_fit_settings(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
    '''
    This will get the fitting settings stored in the pickle file associated with this model & task
    
    '''
    if isinstance(task_list, str):
        task_list = [task_list]
    if isinstance(model_list, str):
        model_list = [model_list]

    fit_settings  = {}
    for task in task_list:
        if "AS" in task:
            ses='ses-1'
        elif "2R" in task:
            ses='ses-2'            
        this_dir = opj(prf_dir, sub, ses)
        fit_settings[task] = {}
        for model in model_list:   
            this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
            fit_settings[task][model] = this_pkl['settings']
    return fit_settings

def get_model_params(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard', fit_stage='iter'):
    # Turn any strings into lists
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(model_list, list):
        model_list = [model_list]

    model_params  = {}
    for task in task_list:
        if "AS" in task:
            ses='ses-1'
        elif "2R" in task:
            ses='ses-2'            
        this_dir = opj(prf_dir, sub, ses)

        model_params[task] = {}
        for model in model_list:
            this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit, fit_stage=fit_stage)
            model_params[task][model] = this_pkl['pars']            
    return model_params

def get_number_of_vx(sub):
    # Do this by loading the ROI mask for v1
    try:
        roi_idx = get_roi(sub, label='V1')
        num_vx = roi_idx.shape[0]
    
    except:
        real_tc = get_real_tc(sub, 'task-AS0')['task-AS0']
        num_vx = real_tc.shape[0]
    
    return num_vx

def get_roi(sub, label):    
    if label=='all':
        num_vx = get_number_of_vx(sub)
        roi_idx = np.ones(num_vx, dtype=bool)
        return roi_idx
    
    if not isinstance(label, list):
        label = [label]
    
    roi_idx = []
    for this_label in label:
        if "not" in this_label:
            this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label.split("-")[-1]}.npy')        
            this_roi_idx = np.load(this_roi_file)
            this_roi_idx = this_roi_idx==0
        else:
            this_roi_file = opj(default_numpy_roi_idx_dir, sub, f'{this_label}.npy')
            this_roi_idx = np.load(this_roi_file)
        
        roi_idx.append(this_roi_idx)
    roi_idx = np.vstack(roi_idx)
    roi_idx = roi_idx.any(0)
    return roi_idx

def get_real_tc(sub, task_list, prf_dir=default_prf_dir):
    if isinstance(task_list, str):
        task_list = [task_list]

    real_tc  = {}
    for task in task_list:'            
        this_dir = opj(prf_dir, sub, ses)
        real_tc_path = utils.get_file_from_substring([sub, ses, task, 'hemi-LR', 'desc-avg_bold'], this_dir, exclude='roi')
        if isinstance(real_tc_path, list) and (len(real_tc_path)==1):
            real_tc_path = real_tc_path[0]
        real_tc[task] = np.load(real_tc_path).T

    return real_tc

def get_pred_tc(sub, task_list, model_list, prf_dir=default_prf_dir, roi_fit='all', dm_fit='standard'):
    # Turn any strings into lists
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(model_list, list):
        model_list = [model_list]

    pred_tc  = {}
    for task in task_list:
        if "AS" in task:
            ses='ses-1'
        elif "2R" in task:
            ses='ses-2'            
        this_dir = opj(prf_dir, sub, ses)

        pred_tc[task] = {}
        for model in model_list:
            this_pkl = load_prf_data(sub, ses, model, task, this_dir, roi_fit=roi_fit, dm_fit=dm_fit)
            pred_tc[task][model] = this_pkl['predictions']            

    return pred_tc

def get_design_matrix_npy(task_list, prf_dir=default_prf_dir):
    if not isinstance(task_list, list):
        task_list = [task_list]
    this_dir = opj(prf_dir)
    dm_npy  = {}    
    for task in task_list:
        dm_path = utils.get_file_from_substring(['design', task], this_dir)        
        dm_npy[task] = scipy.io.loadmat(dm_path)['stim']

    return dm_npy

def get_prfpy_stim(sub, task_list, prf_dir=default_prf_dir):
    if not isinstance(task_list, list):
        task_list = [task_list]
    dm_npy = get_design_matrix_npy(task_list, prf_dir=prf_dir)
    model_list = ['gauss']     # stimulus settings are the same for both norm & gauss models  (so only use gauss) 
    stim_settings = get_fit_settings(sub,task_list, model_list=model_list, prf_dir=prf_dir)
    prfpy_stim = {}
    for task in task_list:
        prfpy_stim[task] = PRFStimulus2D(
            screen_size_cm=stim_settings[task][model_list[0]]['screen_size_cm'],
            screen_distance_cm=stim_settings[task][model_list[0]]['screen_distance_cm'],
            design_matrix=dm_npy[task], 
            axis=0,
            TR=stim_settings[task][model_list[0]]['TR']
            )    
    return prfpy_stim



# # ********************
# def load_params_generic(params_file, load_all=False, load_var=[]):
#     """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

#     if isinstance(params_file, str):
#         if params_file.endswith('npy'):
#             params = np.load(params_file)
#         elif params_file.endswith('pkl'):
#             with open(params_file, 'rb') as input:
#                 data = pickle.load(input)
            
#             if len(load_var)==1:
#                 params = data[load_var[0]]
#             elif len(load_var)>1:
#                 params = {}
#                 # Load the specified variables
#                 for this_var in load_var:
#                     params[this_var] = data[this_var]
#             elif load_all:
#                 params = {}
#                 for this_var in data.keys():
#                     params[this_var] = data[this_var]
#             else:
#                 params = data['pars']

#     elif isinstance(params_file, np.ndarray):
#         params = params_file.copy()
#     elif isinstance(params_file, pd.DataFrame):
#         dict_keys = list(params_file.keys())
#         if not "hemi" in dict_keys:
#             # got normalization parameter file
#             params = np.array((params_file['x'][0],
#                                 params_file['y'][0],
#                                 params_file['prf_size'][0],
#                                 params_file['A'][0],
#                                 params_file['bold_bsl'][0],
#                                 params_file['B'][0],
#                                 params_file['C'][0],
#                                 params_file['surr_size'][0],
#                                 params_file['D'][0],
#                                 params_file['r2'][0]))
#         else:
#             raise NotImplementedError()
#     else:
#         raise ValueError(f"Unrecognized input type for '{params_file}'")

#     return params


