#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V

import numpy as np  
import nibabel as nb
import os
from datetime import datetime, timedelta
import getopt
import linescanning.utils as lsutils
opj = os.path.join

from amb_scripts.load_saved_info import *
from nibabel.freesurfer.io import read_morph_data, write_morph_data
import matplotlib as mpl
import matplotlib.pyplot as plt
derivatives_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives'

surf_scr_shot_dir = opj(derivatives_dir, 'surf_shots')
if not os.path.exists(surf_scr_shot_dir):
    os.mkdir(surf_scr_shot_dir)

def fs_auto_scrn_shot(sub, data, fs_dir, out_dir=None, **kwargs):
    '''
    fs_auto_scrn_shot:
        Create surface files for a subject, and a specific parameter.
        
    Arguments:
        sub             str             e.g. 'sub-01': Name of subject in freesurfer file
        data            np.ndarray      1D array, same length as the number of vertices in subject surface
        fs_dir          str             Location of the subjects Freesurfer folder
        out_dir         str             Where to put the screenshots. 
                                        If not specified - makes a folder in subjects freesurfer folder? dodgy?   
    **kwargs:
        data_mask       bool array      Mask to hide certain values (e.g., where rsquared is not a good fit)
        data_alpha      np.ndarray      Alpha values for plotting
        surf_name       str             Name of your surface e.g., 'polar', 'rsq'
                                        *subject name is added to front of surf_name
        do_scrn_shot    bool            Take screenshots?   Default: True
        under_surf      str             What kind of surface are we plotting on? e.g., pial, inflated...
                                                            Default: inflated
        *** COLOR
        do_col_bar      bool            Show color bar?     Default: True                                        
        cmap            str             Which colormap to use https://matplotlib.org/stable/gallery/color/colormap_reference.html
                                                            Default: viridis
        vmin            float           Minimum value for colormap
                                                            Default: minimum value in data
        vmax            float           Max value for colormap
                                                            Default: maximum value in data
        cmap_nsteps     omt             Number of steps in color map (e.g., 10 different colors)                                                            
        *** CAMERA
        azimuth         float           camera angle(0-360) Default: 0
        zoom            float           camera zoom         Default: 1.00
        elevation       float           camera angle(0-360) Default: 0
        roll            float           camera angle(0-360) Default: 0
        ***
        ow              bool            Overwrite? If surface with same name already exists, do you want to overwrite it?
                                        Default True

    TODO:
        Make option for alpha masking 
    '''
    path_to_sub_surf = opj(fs_dir, sub, 'surf')
    if out_dir==None:
        out_dir = opj(path_to_sub_surf, 'fs_scrn_shots' )
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # Check number of vertices
    n_vx = data.shape[0]
    # Load optonal arguments
    surf_name = kwargs.get('surf_name', None)
    if surf_name==None:
        print('surf_name not specified, using sub+date')
        surf_name = sub + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M')
    else:
        surf_name = sub + '_' + surf_name
    overwrite = kwargs.get('ow', True)
    print(f'File to be named: {surf_name}')        
    if (os.path.exists(opj(path_to_sub_surf, f'lh.{surf_name}'))) & (not overwrite) :
        print(f'{surf_name} already exists for {sub}, not overwriting surf files...')
    else: 
        if (os.path.exists(opj(path_to_sub_surf, f'lh.{surf_name}'))): 
            print(f'Overwriting: {surf_name} for {sub}')
        else:
            print(f'Writing: {surf_name} for {sub}')
        # **** **** SAVE DATA AS CURVE FILE **** ****    
        # Load mask for data to be plotted on surface
        data_mask = kwargs.get('data_mask', None)
        if not isinstance(data_mask, np.ndarray):
            print('Mask for data not specified, showing all points on surface')
            data_mask = np.ones(n_vx)

        # Load colormap properties: (cmap, vmin, vmax)
        cmap = kwargs.get('cmap', None)    
        if cmap==None:
            cmap = 'viridis'

        vmin = kwargs.get('vmin', None)
        if vmin==None:
            # vmin = np.min(data[data_mask])
            vmin = np.percentile(data[data_mask], 10)
        vmax = kwargs.get('vmax', None)
        if vmax==None:
            # vmax = np.max(data[data_mask])    
            vmax = np.percentile(data[data_mask],90)    
        cmap_nsteps = kwargs.get("cmap_nsteps", 20)
        hide_masked_pt_val = vmin-1 # Set the points to be masked to be lower than vmin    
        masked_data = np.ones_like(data)
        masked_data[data_mask] = data[data_mask]
        masked_data[~data_mask] = hide_masked_pt_val    

        # SAVE DATA AS A CURVE FILE
        lh_c = read_morph_data(opj(path_to_sub_surf,'lh.curv'))
        lh_masked_data = masked_data[:lh_c.shape[0]]
        rh_masked_data = masked_data[lh_c.shape[0]:]

        # now save results as a curve file, in subject folder
        print(f'saving lh & rh {surf_name} in {sub} freesurfer folder')
        write_morph_data(opj(path_to_sub_surf, f'lh.{surf_name}'),lh_masked_data)
        write_morph_data(opj(path_to_sub_surf, f'rh.{surf_name}'),rh_masked_data)        
        # **** **** DONE **** ****    
    
    # *** Now make custom overlay: ***
    # [1] rgb triple...
    fv_param_steps = np.linspace(vmin, vmax, cmap_nsteps)
    fv_color_steps = np.linspace(0,1, cmap_nsteps)
    fv_cmap = mpl.cm.__dict__[cmap]
    # Create the string that is used in the freeview command:
    # -> specifically using the overlay_custom= option
    overlay_custom_str = 'overlay_custom='
    for i, fv_param in enumerate(fv_param_steps):
        this_col_triple = fv_cmap(fv_color_steps[i])
        # For each level in the data there is a rgb triple associated with it
        # -> cmap_nsteps determines the number of levels
        # -> the command consists of:
        # [1] data value, [2] associated rgb trips
        this_str = f'{float(fv_param):.2f},{int(this_col_triple[0]*255)},{int(this_col_triple[1]*255)},{int(this_col_triple[2]*255)},'

        overlay_custom_str += this_str    
    
    # Move to the subject's surface directory
    os.chdir(path_to_sub_surf) # move to freeview dir
    # Load the relevant info for the freeview commands:
    under_surf = kwargs.get('under_surf', 'inflated') # plot on which surface?
    cam_azimuth     = kwargs.get('azimuth', 0)
    cam_zoom        = kwargs.get('zoom', 1)
    cam_elevation   = kwargs.get('elevation', 0)
    cam_roll        = kwargs.get('roll', 0)
    do_col_bar = kwargs.get('do_col_bar', True) # Turn on the colorbar?
    if do_col_bar:
        col_bar_flag = "--colorscale"
    else:
        col_bar_flag = ""
    do_scrn_shot = kwargs.get('do_scrn_shot', True) # Do the screenshots?
    if do_scrn_shot:
        scrn_shot_flag = f"--ss {out_dir}/{surf_name}"
    else:
        scrn_shot_flag = ""        
    fview_cmd = f'''freeview -f lh.{under_surf}:overlay=lh.{surf_name}:{overlay_custom_str} rh.{under_surf}:overlay=rh.{surf_name}:{overlay_custom_str} --camera Azimuth {cam_azimuth} Zoom {cam_zoom} Elevation {cam_elevation} Roll {cam_roll} {col_bar_flag} {scrn_shot_flag}'''

    print(fview_cmd)
    os.system(fview_cmd)

    # END FUNCTION


def main(argv):

    """
---------------------------------------------------------------------------------------------------
WRITE TO FREESURFER
AS PER https://docs.google.com/document/d/104Q8RV0QI0aZlYgs9sDo8eWVtEwebNsgl89sqim6LPU/edit
Drawing ROIs with freeview tutorial...
Edited by MD 

Arguments:
    -s|--sub    <sub number>        number of subject's FreeSurfer directory from which you can 
                                    omit "sub-" (e.g.,for "sub-001", enter "001").
    --model     model name
    -t|--task   <task name>         name of the experiment performed (e.g., "LE", "RE")
    -p|--param  <parameter to plot> e.g., polar angle
    --roi_fit   sometimes we fit only a subset of voxels, ("e.g. V1_exvivo")
    --rsq_th    rsq threshold
    --ecc_th    ecc threshold
Options:                                  
    -v|--verbose    print some stuff to a log-file
    --overwrite     If specified, we'll overwrite existing Gaussian parameters. If not, we'll look
                    for a file with ['model-gauss', 'stage-iter', 'params.npy'] in *outputdir* and,
                    if it exists, inject it in the normalization model (if `model=norm`)  

Example:


---------------------------------------------------------------------------------------------------
"""

    sub         = None
    eye         = None
    param       = 'pol'
    model       = None
    roi_fit     = 'all'
    rsq_th      = 0.1
    ecc_th      = 5
    scr_shot    = False
    # overwrite   = True
    # cmap        = 'viridis'
    under_surf  = 'inflated'
    # verbose     = True

    try:
        opts = getopt.getopt(argv,"h:s:n:t:m:v:e:",["help", "sub=", "eye=", "param=", "model=", "rsq_th=", "roi_fit=", "ecc_th=", "under_surf=", "cmap=", "scr_shot", "verbose", "overwrite"])[0]
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
        elif opt in ("-e", "--eye"):
            eye = arg# hyphen_parse('task', arg)
        elif opt in ("-p", "--param"):
            param = arg
        elif opt in ("--rsq_th"):
            rsq_th = float(arg)
        elif opt in ("--ecc_th"):
            ecc_th = float(arg)            
        elif opt in ("--roi_fit"):
            roi_fit = arg
        elif opt in ("--model"):
            model = arg
        elif opt in ("--under_surf"):
            under_surf = arg         
        elif opt in ("--cmap"):
            cmap = arg                        
        elif opt in ("--scr_shot"):
            scr_shot = True            
        elif opt in ("--overwrite"):
            overwrite = True
        elif opt in ("-v", "--verbose"):
            verbose = True
        else:
            print(opt)

            sys.exit()

    if len(argv) < 2:
        print(main.__doc__)
        sys.exit()
    
    # Check subject path for matching surface:    
    prf_obj = Prf1T1M(sub=sub, eye=eye, model=model)
    if model in ['gauss', 'norm']:
        total_mask = prf_obj.return_vx_mask(th={
            'min-rsq':rsq_th,
            'max-ecc':ecc_th,
        })
    else:
        total_mask = prf_obj.return_vx_mask(th={
            'min-rsq':rsq_th,
        })

    if param=='pol':
        fv_vmin = -3.14
        fv_vmax = 3.14
        fv_cmap_name = 'hsv'        
    elif param=='ecc':
        fv_vmin = 0
        fv_vmax = ecc_th
        fv_cmap_name = 'magma'
    elif param=='width_r':
        fv_vmin=0
        fv_vmax=3
        fv_cmap_name = 'Reds_r'
    elif param=='sf0':
        fv_vmin=0
        fv_vmax=6      
        fv_cmap_name = 'Blues'  
    elif param=='sfmax':
        fv_vmin=0
        fv_vmax=20      
        fv_cmap_name = 'Blues'          
    elif param=='maxC':
        fv_vmin=0
        fv_vmax=200
        fv_cmap_name = 'Greens'  
    elif param=='rsq':
        fv_vmin=0
        fv_vmax=1
        fv_cmap_name = 'viridis'
    elif param=='a_sigma':
        fv_vmin=0
        fv_vmax=5
        fv_cmap_name = 'spring'      
    # elif param=='log10_sfmax':
    #     fv_vmin=-2
    #     fv_vmax=10
    #     fv_cmap_name = 'spring'      
    else:        
        fv_vmin = None
        fv_vmax = None
        fv_cmap_name = None

    fv_steps = 10
    if sub=='sub-01':
        elevation_val = -35
    elif sub=='sub-02':
        elevation_val = -20
    else:
        elevation_val = -20
    # fview_cmd = f'''freeview -f lh.{under_surf}:overlay=lh.{surf_name}:{overlay_custom_str} rh.{under_surf}:overlay=rh.{surf_name}:{overlay_custom_str} --camera Azimuth 90 Elevation {elevation_val} --colorscale '''

    fs_auto_scrn_shot(
        sub=sub, 
        data=prf_obj.pd_params[param], 
        fs_dir=opj(derivatives_dir,'freesurfer'), 
        out_dir=surf_scr_shot_dir, 
        surf_name = f'{param}_{eye}_{model}',
        data_mask = total_mask,
        do_scrn_shot = scr_shot,
        azimuth = 90,
        elevation = elevation_val,
        zoom = 1.5,
        vmin=fv_vmin,vmax=fv_vmax,cmap=fv_cmap_name,cmap_nsteps = fv_steps,
        under_surf=under_surf,

        )
    

'''
[-3.14,255,  0,  0,
 -2.65,255,255,  0,
 -2.09,  0,128,  0,
 -1.75,  0,255,255,
 -1.05,  0,  0,255,
 - 0.5,238,130,238,
     0,255,0,0,
   0.5,255,255,0,1.05,0,128,0,1.57,0,255,255,2.09,0,0,255,2.65,238,130,238,3.14,255,0,0]
'''

# *************
if __name__ == "__main__":
    main(sys.argv[1:])


