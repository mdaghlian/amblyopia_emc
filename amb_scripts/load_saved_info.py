import numpy as np
import scipy.io
import os
import sys
import yaml
import pickle
opj = os.path.join

import nibabel as nb
from prfpy_csenf.stimulus import PRFStimulus2D, CSenFStimulus
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

def amb_load_fit_settings(sub, task_list, model_list, ses, **kwargs):
    fit_settings = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, ses=ses, key='settings', **kwargs)
    return fit_settings

def amb_load_pred_tc(sub, task_list, model_list, ses, **kwargs):
    pred_tc = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, ses=ses, key='preds', **kwargs)
    return pred_tc

def amb_load_prf_params(sub, task_list, model_list, ses, **kwargs):
    prf_params = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, ses=ses, key='pars', **kwargs)
    return prf_params

def amb_load_bayes_prf(sub, task_list, model_list, ses, **kwargs):    
    kwargs['key'] = kwargs.get('key', 'samples')
    prf_params = amb_load_pkl_key(
        sub=sub, task_list=task_list, model_list=model_list, 
        ses=ses, do_bayes=True, **kwargs)
    return prf_params


def amb_load_pkl_key(sub, task_list, model_list, ses, key, **kwargs):
    if not isinstance(task_list, list):
        task_list = [task_list]
    if not isinstance(model_list, list):
        model_list = [model_list]        
    
    prf_dict = {}
    for task in task_list:
        prf_dict[task] = {}
        for model in model_list:
            this_pkl = amb_load_pkl(sub=sub, task=task, model=model, ses=ses, **kwargs)
            prf_dict[task][model] = this_pkl[key]
    return prf_dict

def amb_load_pkl(sub, task, model, ses, **kwargs):
    '''
    linescanning toolbox nicely saves everything into a pickle
    this will load the correct pickle associated with the correct, sub, ses, model and task
    roi_fit specifies which fitting run was used.  
    '''
    ow_deriv_dir  = kwargs.get('ow_deriv_dir', derivatives_dir)
    do_bayes = kwargs.get('do_bayes', False)    
    bkey = ''    
    if do_bayes:
        bkey='bayes_'

    if 'pRF' in task:
        # amb_prf_dir = opj(derivatives_dir, 'amb-prf')
        amb_prf_dir = opj(ow_deriv_dir, f'{bkey}prf')
    else:
        # amb_prf_dir = opj(derivatives_dir, 'amb-csf')
        amb_prf_dir = opj(ow_deriv_dir, f'{bkey}csf')

    dir_to_search = opj(amb_prf_dir, sub, ses)
    include = kwargs.get("include", []) # any extra details to search for in file name
    exclude = kwargs.get("exclude", []) # any extra details to search for in file name
    roi_fit = kwargs.get('roi_fit', 'all') 
    if roi_fit=='all':
        exclude += ['_x']   
    fit_stage = kwargs.get('fit_stage', 'iter')
    fit_type = kwargs.get('fit_type', 'bgfs')
    if do_bayes:
        fit_stage = ''
        fit_type = ''
    if fit_stage == 'grid':
        fit_type = ''
    # the folder specified in "dir_to_search" will contain several files
    # -> different fit types (grid vs iter), model (gauss vs norm) task (As0,AS1,AS2) and 
    # Now we need to find the relevant file, by filtering for key terms (see included)
    include += [sub, model, task, roi_fit, fit_stage, fit_type] # Make sure we get the correct model and task (& subject)    
    exclude += ['avg_bold', '.txt'] # exclude grid fits and bold time series

    data_path = dag_find_file_in_folder(filt=include, path=dir_to_search, exclude=exclude)
    if isinstance(data_path, list):
        print(include)
        print(f'Error, more than 1 match ({len(data_path)} files)')
        print(data_path)
        sys.exit()

    pkl_file = open(data_path,'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()     

    return data    

def amb_load_real_tc(sub, task_list, ses, clip_start=0):
    if not isinstance(task_list, list):
        task_list = [task_list]
    this_dir = opj(psc_tc_dir, sub, ses)
    real_tc = {}
    for task in task_list:
        real_tc_file = dag_find_file_in_folder([task, 'hemi-LR_desc-avg_bold'], this_dir)
        if isinstance(real_tc_file, list):
            print(f'Error, more than 1 match ({len(real_tc_file)} files)')
            sys.exit()
        unclipped = np.load(real_tc_file).T        
        real_tc[task] = np.copy(unclipped[:,clip_start::])

    return real_tc


def amb_load_run_corr(sub, task_list, ses):
    if not isinstance(task_list, list):
        task_list = [task_list]

    this_dir = opj(psc_tc_dir, sub, ses)
    run_corr = {}
    for task in task_list:
        run_corr_file = dag_find_file_in_folder([task, 'hemi-LR_desc-run_cor'], this_dir)
        if isinstance(run_corr_file, list):
            print(f'Error, more than 1 match ({len(run_corr_file)} files)')
            sys.exit()
        run_corr[task] = np.load(run_corr_file)
    return run_corr    

def amb_load_real_tc_run(sub, task_list, ses):
    if not isinstance(task_list, list):
        task_list = [task_list]
    # if not isinstance(run_list, list):
    #     run_list=[run_list]
    unz_dir = opj(derivatives_dir, 'pybest', sub, ses, 'unzscored')
    real_tc = {}
    for task in task_list:
        real_tc[task] = []
        run_listL = dag_find_file_in_folder([task, f'run-', 'fsnative', f'hemi-L_desc-denoised_bold'], unz_dir)
        run_listL.sort()
        run_listR = dag_find_file_in_folder([task, f'run-', 'fsnative', f'hemi-R_desc-denoised_bold'], unz_dir)
        run_listR.sort()
        for run in range(len(run_listL)):
            LH_tc = np.load(run_listL[run])
            RH_tc = np.load(run_listR[run])
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
    dm_type = dm_type.lower()
    if dm_type=='prf':
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
    elif dm_type=='csf':
        csf_dm = amb_load_dm(['sf_vect', 'c_vect'])
        sf_vect = csf_dm['sf_vect'][clip_start::]
        c_vect = csf_dm['c_vect'][clip_start::]

        # Number of stimulus types:
        prfpy_stim = CSenFStimulus(
            SF_seq=sf_vect,
            CON_seq = c_vect,
            TR=1.5,
        )


    return prfpy_stim    

def amb_load_qcsf(sub, eye_list, ses):
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

def amb_load_cmf(sub, task_list, ses, **kwargs):    
    if not isinstance(task_list, list):
        task_list = [task_list]
    this_dir = opj(derivatives_dir, 'cmf_est', sub, ses)
    cmf_data = {}
    for task in task_list:        
        cmf_file = dag_find_file_in_folder([task, 'cmf', '.pkl'], this_dir)
        if isinstance(cmf_file, list):
            print(f'Error, more than 1 match ({len(cmf_file)} files)')
            print(cmf_file)
            sys.exit()

        pkl_file = open(cmf_file,'rb')
        cmf_data[task] = pickle.load(pkl_file)['cmf']
        # replace any nans in the cmf with -1
        cmf_data[task][np.isnan(cmf_data[task])] = -1
        pkl_file.close()              
    return cmf_data     

def mat_struct_to_python_dict(mat_struct):
    # mat_struct = scipy.io.loadmat(file/path).get('entry_of_interest')
    recordarr = np.rec.array(mat_struct)
    py_dict = {}
    myList = []
    for field in recordarr.dtype.names: #iterates through field names of numpy array
        for array in recordarr[field]: #iterates through the array of each numpy array                                    
            for value in array:
                
                myList.append(np.squeeze(value))
                # myList.append(np.squeeze(value.flatten()))
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


# ********************
def load_params_generic(params_file, load_all=False, load_var=[]):
    """Load in a numpy array into the class; allows for quick plotting of voxel timecourses"""

    if isinstance(params_file, str):
        if params_file.endswith('npy'):
            params = np.load(params_file)
        elif params_file.endswith('pkl'):
            with open(params_file, 'rb') as input:
                data = pickle.load(input)
            
            if len(load_var)==1:
                params = data[load_var[0]]
            elif len(load_var)>1:
                params = {}
                # Load the specified variables
                for this_var in load_var:
                    params[this_var] = data[this_var]
            elif load_all:
                params = {}
                for this_var in data.keys():
                    params[this_var] = data[this_var]
            else:
                params = data['pars']

    elif isinstance(params_file, np.ndarray):
        params = params_file.copy()
    elif isinstance(params_file, pd.DataFrame):
        dict_keys = list(params_file.keys())
        if not "hemi" in dict_keys:
            # got normalization parameter file
            params = np.array((params_file['x'][0],
                                params_file['y'][0],
                                params_file['prf_size'][0],
                                params_file['A'][0],
                                params_file['bold_bsl'][0],
                                params_file['B'][0],
                                params_file['C'][0],
                                params_file['surr_size'][0],
                                params_file['D'][0],
                                params_file['r2'][0]))
        else:
            raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized input type for '{params_file}'")

    return params


