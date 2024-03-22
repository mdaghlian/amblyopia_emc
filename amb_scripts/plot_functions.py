from ctypes import RTLD_GLOBAL
from ctypes.wintypes import RGB
from matplotlib.colors import rgb2hex
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
import linescanning.plotting as lsplt
from scipy.stats import binned_statistic
# from highlight_text import HighlightText, ax_text, fig_text
# from .utils import coord_convert, prfpy_params_dict
from dag_prf_utils.utils import dag_coord_convert as coord_convert
from dag_prf_utils.prfpy_functions import prfpy_params_dict
# import rgba
# import os 
# import nibabel as nb
# from collections import defaultdict as dd
# from pathlib import Path
# import yaml

# import cortex
# import seaborn as sns
# import pickle
# from datetime import datetime
# opj = os.path.join

def rgba(r,g,b,a):
    return [r/255,g/255,b/255,a]


def show_plot_cols():
    plot_cols = get_plot_cols()
    # fig, axs = plt.subplot()    
    plt.figure(figsize=(2,5))
    for i,key in enumerate(plot_cols.keys()):
        plt.scatter(0,i, s=500, color=plot_cols[key], label = key)
        plt.text(0, i+.1, key)
    
def get_plot_cols():
    plot_cols = {
        "LE"            : 'b',# rgba(252,141, 89, .8),#'#fd8d3c',
        "RE"            : 'r',# rgba( 67,162,202, .8),#'#43a2ca',
        "Em"            : rgba(159,150,100, .8),
        "Ed"            : rgba(159,150,100, .8),
        #
        "gauss"          : rgba( 27, 158, 119, 0.9),
        "norm"           : rgba(217,  95,   2, 0.5),
        "CSF"            : rgba(217, 222,   2, 0.5),
        "real"           : '#cccccc',
        }
    return plot_cols
    
def time_series_plot(params, prfpy_stim, real_tc=[], pred_tc=[], model='gauss', scotoma_info=[], show_stim_frame_x=[], **kwargs):
    # GET PARAMETERS....
    # axs=kwargs.get("axs", [])    
    # col=kwargs.get("col", "k")    
    # line_label=kwargs.get("line_label", " ")
    # y_label=kwargs.get("y_label", "")
    # axs_title=kwargs.get("axs_title", "")
    # y_lim=kwargs.get("y_lim", [])
    title=kwargs.get("title", None)
    fmri_TR =kwargs.get("fmri_TR", 1.5)

    if model=='gauss':
        model_idx = prfpy_params_dict()['gauss']
    elif model=='norm':
        model_idx = prfpy_params_dict()['norm']

    # ************* PRF (+stimulus) PLOT *************
    fig = plt.figure(constrained_layout=True, figsize=(15,5))
    gs00 = fig.add_gridspec(1,2, width_ratios=[10,20])
    ax1 = fig.add_subplot(gs00[0])
    # Set vf extent
    aperture_rad = prfpy_stim.screen_size_degrees/2    
    ax1.set_xlim(-aperture_rad, aperture_rad)
    ax1.set_ylim(-aperture_rad, aperture_rad)

    if show_stim_frame_x !=[]:
        this_dm = prfpy_stim.design_matrix[:,:,show_stim_frame_x]
        this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
        this_dm_rgba[:,:,0] = 1-this_dm
        this_dm_rgba[:,:,1] = 1-this_dm
        this_dm_rgba[:,:,2] = 1-this_dm
        this_dm_rgba[:,:,3] = .5
        ax1.imshow(this_dm_rgba, extent=[-aperture_rad, aperture_rad, -aperture_rad, aperture_rad])
    
    # Add prfs
    prf_x = params[0]
    prf_y = params[1]
    # Add normalizing PRF (FWHM)
    if model=='norm':
        prf_2_fwhm = 2*np.sqrt(2*np.log(2))*params[model_idx['n_sigma']] # 
        if prf_2_fwhm>aperture_rad:
            ax1.set_xlabel('*Norm PRF is too larget to show - covers whole screen')
            norm_prf_label = '*Norm PRF'
        else:
            norm_prf_label = 'Norm PRF'
        prf_2 = patches.Circle(
            (prf_x, prf_y), prf_2_fwhm, edgecolor="r", 
            facecolor=[1,0,0,.5], 
            linewidth=8, fill=False,
            label=norm_prf_label,)    
        ax1.add_patch(prf_2)
    # Add activating PRF (fwhm)
    prf_fwhm = 2*np.sqrt(2*np.log(2))*params[model_idx['a_sigma']] # 
    prf_1 = patches.Circle(
        (prf_x, prf_y), prf_fwhm, edgecolor="b", facecolor=[1,1,1,0], 
        linewidth=8, fill=False, label='PRF')
    ax1.add_patch(prf_1)

    # add scotoma
    if scotoma_info['scotoma_centre'] != []:
        scot = patches.Circle(
            scotoma_info["scotoma_centre"], scotoma_info["scotoma_radius"], 
            edgecolor="k", facecolor="w", linewidth=8, fill=True, alpha=1,
            label='scotoma')
        ax1.add_patch(scot)
    
    # Add 0 lines...
    ax1.plot((0,0), ax1.get_ylim(), 'k')
    ax1.plot(ax1.get_xlim(), (0,0), 'k')
    ax1.legend()
    ax1.set_title(model)
    # ************* TIME COURSE PLOT *************
    # Check title - if not present use parameters...
    if title == None:
        param_count = 0
        set_title = ''
        for param_key in model_idx.keys():
            set_title += f'{param_key}={round(params[model_idx[param_key]],2)}; '

            if param_count > 3:
                set_title += '\n'
                param_count = 0
            param_count += 1
    else:
        set_title = title
    x_label = "time (s)"
    x_axis = np.array(list(np.arange(0,real_tc.shape[0])*fmri_TR)) 
    ax2 = fig.add_subplot(gs00[1])
    lsplt.LazyPlot(
        [real_tc, pred_tc],
        xx=x_axis,
        color=['#cccccc', 'r'], 
        labels=['real', 'pred'], 
        add_hline='default',
        x_label=x_label,
        y_label="amplitude",
        axs=ax2,
        title=set_title,
        # xkcd=True,
        # font_size=font_size,
        line_width=[0.5, 3],
        markers=['.', None],
        # **kwargs,
        )
    # If showing stim - show corresponding time point in timeseries
    if show_stim_frame_x !=[]:
        this_time_pt = show_stim_frame_x * fmri_TR
        current_ylim = ax2.get_ylim()
        ax2.plot((this_time_pt,this_time_pt), current_ylim, 'k')

    return fig, ax1, ax2    


# ********************************
import numpy as np
from prfpy_csenf.rf import *
from dag_prf_utils.utils import *
from dag_prf_utils.cmap_functions import *
from dag_prf_utils.plot_functions import *
# from .csf_stimuli import * # load default parameters

from .utils import *

import matplotlib.pyplot as plt

# sf colors for each sf... (used in plot)
sf_cols = {
    '0.5'   :'#199E77',
    '1'     :'#D76127',
    '3'     :'#7670B3',
    '6'     :'#E62A8A',
    '12'    :'#65A744',
    '18'    :'#E5AC23',    
}


# 
from .load_saved_info import amb_load_prfpy_stim
from prfpy_csenf.model import CSenFModel
csenf_stim = amb_load_prfpy_stim('csf')

def ncsfplt_set_csf_ax(ax, **kwargs):
    """
    Create a log-log plot with customizable parameters.

    Parameters:
    - ax (matplotlib.axes._subplots.AxesSubplot): The subplot on which to create the plot.
    - box_aspect (float, optional): The aspect ratio for non-equal axes. Default is 1.
    - equal_ax (bool, optional): If True, make the axes equal in aspect ratio. Default is False.
    - xticklabels (list, optional): Labels for the x-axis ticks. Default is ['0.5', '1', '10', '50'].
    - xlim (list, optional): Limits for the x-axis. Default is [xticks[0], xticks[-1]].
    - yticklabels (list, optional): Labels for the y-axis ticks. Default is ['0.1', '1', '10', '100'].
    - ylim (list, optional): Limits for the y-axis. Default is [0.1, 500].

    Returns:
    None
    """    
    # Sort out kwargs
    box_aspect = kwargs.get('box_aspect', 1)
    equal_ax = kwargs.get('equal_ax', False)
    
    xticklabels = kwargs.get('xticklabels', ['0.5', '1', '10', '50']) 
    xticks = [float(i) for i in xticklabels]
    xlim = kwargs.get('xlim', None)
    if xlim is None:
        xlim = [xticks[0], xticks[-1]]

    # yticklabels = kwargs.get('yticklabels', ['0.1', '1', '10', '100']) 
    # yticks = [float(i) for i in yticklabels]
    # ylim = kwargs.get('ylim', [0.1, 500])
    yticklabels = kwargs.get('yticklabels', ['1', '10', '100']) 
    yticks = [float(i) for i in yticklabels]
    ylim = kwargs.get('ylim', [1, 500])




    # Add in later? As total plot?
    ax.set_xlabel('SF (c/deg)')
    ax.set_ylabel('contrast sensitivity')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Scatter the points sampled
    # ax.scatter(
    #     csenf_stim.SF_seq, 100/csenf_stim.CON_seq, color='k', alpha=0.1
    # )

    # Put a grid on the axis (only the major ones)
    # ax.grid(which='both', axis='both', linestyle='--', alpha=0.5)
    # Make the axis square
    if equal_ax:
        ax.set_aspect('equal', 'box') 
    else:
        ax.set_box_aspect(box_aspect)

    ax.set_xticks(xticks) 
    ax.set_xticklabels(xticklabels) 
    ax.set_xlim(xlim) 
    #
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(ylim)
    #
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def ncsfplt_csf_curve(params, ax,  **kwargs):
    """
    Plot the mean CSF curves with optional error shading or error bars.

    Parameters:
    - params (dict): Dictionary containing parameters for CSF calculation.
    - ax (matplotlib.axes._subplots.AxesSubplot): The subplot on which to create the plot.
    - min_SF: min sf value for line
    - max_SF: max ...
    - step_SF: number of steps in SFs 
    Returns:
    None
    """ 
    params = ncsf_plt_params_check(params)
    # Sort out kwargs
    min_SF          = kwargs.get('min_SF', csenf_stim.SFs[0])
    max_SF          = kwargs.get('max_SF', 50)
    step_Sf         = kwargs.get('step_SF', 50)
    add_sampled_SF  = kwargs.get('add_sampled_SF', False)
    add_sampled_SF_cols = kwargs.get('add_sampled_SF_cols', True)
    add_extrapolated  = kwargs.get('add_extrapolated', True)    
    add_SFp         = kwargs.get('add_SFp', False)

    # Calculate CSF curves + matrix 
    log_sf_grid  = np.linspace(
        np.log10(min_SF),
        np.log10(max_SF), 
        step_Sf)
    sfs_for_plot = 10**log_sf_grid
    con_s_grid = np.logspace(np.log10(csenf_stim.CON_Ss[-1]),np.log10(csenf_stim.CON_Ss[0]), 2)
    log_sf_grid, con_s_grid = np.meshgrid(log_sf_grid,con_s_grid)    
    
    _, csf_curves = csenf_exponential(
        log_SF_grid = log_sf_grid, #csenf_stim.log_SF_grid, 
        CON_S_grid  = con_s_grid, #csenf_stim.CON_S_grid,
        width_r     = params['width_r'], 
        SFp         = params['SFp'], 
        CSp         = params['CSp'], 
        width_l     = params['width_l'], 
        crf_exp     = params['crf_exp'],
        return_curve=True,
        )       
    dag_shaded_line(
        line_data=csf_curves, 
        xdata=sfs_for_plot, 
        ax=ax,        
        lw=1,
        **kwargs)
    if add_sampled_SF:
        _, Scsf_curves = csenf_exponential(
            log_SF_grid = csenf_stim.log_SF_grid, 
            CON_S_grid  = csenf_stim.CON_S_grid,
            width_r     = params['width_r'], 
            SFp         = params['SFp'], 
            CSp         = params['CSp'], 
            width_l     = params['width_l'], 
            crf_exp     = params['crf_exp'],
            return_curve=True,
            )       
        Sm_csf_curve = np.median(Scsf_curves, axis=1)
        if add_sampled_SF_cols:
            sampled_sf_cols = [sf_cols[key] for key in list(sf_cols.keys())]
        else:
            sampled_sf_cols = kwargs.get('line_col', None)

        ax.scatter(
            csenf_stim.SFs, 
            Sm_csf_curve, 
            s=50,
            alpha=kwargs.get('line_alpha',1),
            color=sampled_sf_cols,# kwargs.get('line_col', None), 
            marker='o',        
            label='_'
            )
    if add_extrapolated:
        log_sf_grid  = np.linspace(
            np.log10(18),
            np.log10(50), 
            step_Sf)
        sfs_for_plot = 10**log_sf_grid
        con_s_grid = np.logspace(np.log10(csenf_stim.CON_Ss[-1]),np.log10(csenf_stim.CON_Ss[0]), 2)
        log_sf_grid, con_s_grid = np.meshgrid(log_sf_grid,con_s_grid)    
        _, csf_curves = csenf_exponential(
            log_SF_grid = log_sf_grid, #csenf_stim.log_SF_grid, 
            CON_S_grid  = con_s_grid, #csenf_stim.CON_S_grid,
            width_r     = params['width_r'], 
            SFp         = params['SFp'], 
            CSp         = params['CSp'], 
            width_l     = params['width_l'], 
            crf_exp     = params['crf_exp'],
            return_curve=True,
            )       

        dag_shaded_line(
            line_data=csf_curves, 
            xdata=sfs_for_plot, 
            ax=ax,        
            line_col='w',
            lw=1,
            line_kwargs={'linestyle':'--'},
            # shade_kwargs={'step':'pre'},

            )
        # m_csf_curve = np.median(csf_curves, axis=1)        
        # ax.plot(
        #     sfs_for_plot, 
        #     m_csf_curve, 
        #     linestyle='--',
        #     alpha=1,
        #     color='w', 
        #     lw=kwargs.get('lw', 2.5))        
    if add_SFp:
        plt.axvline(x=params['SFp'])      

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    ncsfplt_set_csf_ax(ax, **kwargs)


def ncsfplt_crf_curve(params, **kwargs):
    """
    Plot the mean CSF curves with optional error shading or error bars.

    Parameters:
    - params (Dataframe): Dictionary containing parameters for CSF calculation.
    - ax (matplotlib.axes._subplots.AxesSubplot): The subplot on which to create the plot.
    - crf_col (str, optional): Color of the CSF curve. Default is 'g'.
    - lw (float): width of csf plot
    - line_alpha: alpha of line
    - shade_alpha: alpha of shade
    - error_version (str, optional): Type of error to be used ('pc-5', 'iqr', 'bound', 'std', 'ste').
    - error_bar (str, optional): Type of error representation ('shade', 'none'). Default is 'shade'.
    Returns:
    None
    """ 
    ncsf_params = ncsf_plt_params_check(params)
    # Kwargs
    sf_for_crf=kwargs.get('sf_for_crf', 0.5)
    ax = kwargs.get('ax', plt.gca())
    ow_Qs = kwargs.get('ow_Qs', True)
    Qs_at_CSp = kwargs.get('Qs_at_CSp', False)
    do_log = kwargs.get('do_log', True)
    kwargs['ax'] = ax

    # *********** Plot CRF curves ********************* 
    # *************************************************
    if ow_Qs:
        # if ncsf_params['crf_exp'].shape[0]==1:
        # return
        if not hasattr(ncsf_params['crf_exp'], 'shape'):
            # print('if not hasattr(ncsf_params[crf_exp], shape):')
            Qs = 20 * np.array([1])[...,np.newaxis]
        elif ncsf_params['crf_exp'].shape ==():
            # print('elif ncsf_params[crf_exp].shape ==():')
            Qs = 20 * np.array([1])#[...,np.newaxis]
        else:
            # print('else')
            Qs = 20 * np.ones_like(ncsf_params['crf_exp'])

    elif Qs_at_CSp:
        # Setelct
        Qs = ncsf_params['CSp'] 
    else:
        sf_grid = [0.5, sf_for_crf] # doesn't matter
        con_grid = csenf_stim.CON_Ss # [0,1] # doesn't matter
        # print(100/con_grid)
        sf_grid, con_grid = np.meshgrid(sf_grid,con_grid)
        _, csf_curve = csenf_exponential(
            log_SF_grid     = np.log10(sf_grid), 
            CON_S_grid      = con_grid, 
            width_r         = ncsf_params['width_r'], 
            SFp             = ncsf_params['SFp'], 
            CSp             = ncsf_params['CSp'], 
            width_l         = ncsf_params['width_l'], 
            crf_exp         = ncsf_params['crf_exp'],
            return_curve    = True)    
        Qs = 100/csf_curve[1,:]

    

    Cs = np.linspace(0, 100, 100)    # csenf_stim.CONs#
    Cs = np.logspace(np.log10(.25), np.log10(100), 100)
    q = ncsf_params['crf_exp']
    Q_to_q = Qs**q
    crf_curves = (Cs**q[...,np.newaxis]) / ( (Cs**q[...,np.newaxis]) + Q_to_q[...,np.newaxis])
    dag_shaded_line(
        line_data=crf_curves.T,
        xdata=Cs,   
        **kwargs,   
    )

    # ************
    # Put a grid on the axis (only the major ones)
    ax.grid(which='both', axis='both', linestyle='--', alpha=0.5)
    # ax.set_xscale('log')
    # Make the axis square
    ax.set_box_aspect(1) 
    # ax.set_title('CRF')
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim([0, 1])
    if not do_log:
        ax.set_xticks([0, 50,100])
        ax.set_xlim([0, 100]) # ax.set_xlim([0, 100])
    else:
        ax.set_xlim([0.1, 100]) # ax.set_xlim([0, 100])
        ax.set_xscale('log')
        # ax.set_xticks([0.1, 50,100])
        # ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel('contrast (%)')
    ax.set_ylabel('fMRI response (a.u.)')
    # 
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def ncsf_plt_params_check(params_in):
    params_out = {}
    eg_key = list(params_in.keys())[0]    
    # if not hasattr(params_in['width_r'], 'to_numpy'):    
    # if not hasattr(params_in[eg_key], 'to_numpy'):
    #     return params_in
    # for p in params_in.keys():
    #     params_out[p] = params_in[p].to_numpy()


    # ***
    if isinstance(params_in[eg_key], np.ndarray):
        return params_in
    for p in params_in.keys():
        params_out[p] = np.array(params_in[p])

    return params_out

# def ncsf_plt_params_check(params_in):
#     params_out = {}
#     # if not hasattr(params_in['width_r'], 'to_numpy'):
#     eg_key = list(params_in.keys())[0]    
#     if isinstance(params_in[eg_key], float) or isinstance(params_in[eg_key], int):
#         # bloop
#         for p in params_in.keys():
#             params_out[p] = np.array(params_in[p])        
#         return params_out
    
#     if hasattr(params_in[eg_key], 'to_numpy'):
#         for p in params_in.keys():
#             params_out[p] = params_in[p].to_numpy()
#     else:
#         params_out = params_in
        
#     return params_out    

def ncsf_plt_ts_plot(params, real_ts, **kwargs):
    ncsf_params = ncsf_plt_params_check(params)
    # Kwargs
    line_col = kwargs.get('line_col', 'g')
    ax = kwargs.get('ax', plt.gca())
    csenf_model = CSenFModel(stimulus = csenf_stim)
    pred_ts = csenf_model.return_prediction(
        width_r     = ncsf_params['width_r'],
        SFp         = ncsf_params['SFp'],
        CSp        = ncsf_params['CSp'],
        width_l     = ncsf_params['width_l'],
        crf_exp     = ncsf_params['crf_exp'],
        beta        = ncsf_params['amp_1'],
        baseline    = ncsf_params['bold_baseline'],
        # hrf_1       = hrf_1,
        # hrf_2       = hrf_2,
    ).squeeze()
    # max_ts = np.
    TR_in_s = 1.5
    ts_x = np.arange(0, real_ts.shape[0]) * TR_in_s
    max_ts = np.max([pred_ts, real_ts], axis=0).max()
    min_ts = np.min([pred_ts, real_ts], axis=0).min()
    max_ts = kwargs.get('max_ts', max_ts)
    min_ts = kwargs.get('min_ts', min_ts)

    # *********** ax 0,3: Time series ***********
    ax.plot(ts_x, pred_ts,  color=line_col, lw=1, alpha=0.8, label='Prediction')        
    ax.plot(ts_x, real_ts,  color='k', linestyle=':', marker='o', markersize=1, lw=1, alpha=0.8, label='Data')
    ax.set_xlim(0, ts_x.max())
    # ax.set_title('Time series')

    # Find the time for 0 stimulation, add grey patches
    id_no_stim = csenf_stim.SF_seq==0.0
    # x = np.arange(len(id_no_stim)) 
    y1 = np.ones_like(ts_x)*min_ts
    y2 = np.ones_like(ts_x)*max_ts
    ax.fill_between(ts_x, y1, y2, where=id_no_stim, facecolor='grey', alpha=0.5)    
    ax.set_ylim([min_ts, max_ts])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('BOLD signal change (%)')
    #

    # Add the SFs  
    # Find indices where the values change ( & are not to 0)
    change_indices = np.where((np.diff(csenf_stim.SF_seq) != 0) & (csenf_stim.SF_seq[1:] != 0))[0]

    # Create a list of labels corresponding to the changed values
    # labels = [f'{" " if value < 10 else ""}{value:0.1f}' for value in csenf_stim.SF_seq[change_indices + 1]]
    labels = [f'{value:0.1f}' for value in csenf_stim.SF_seq[change_indices+1]]
    labels = [value.split('.0')[0] for value in labels]
    # Add text labels at the change points on the plot
    for idx, label in zip(change_indices + 1, labels):
        ax.text(idx*TR_in_s+3*TR_in_s, min_ts*.9, label, ha='center', va='bottom', color=sf_cols[label]) #rotation=45)
    plt.legend()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def ncsf_plt_fix_ecc_ax(ax, p, **kwargs):
    do_x_axis = kwargs.get('do_x_axis', True)
    if do_x_axis:
        ax.set_xlabel('eccentricity (deg)')
        ax.set_ylabel(p)
        ax.set_xlim([0, 5])   
        ax.set_xticks(np.linspace(0,5,6))   
    # ax.set_box_aspect(2)
    if p=='SFp':          
        SFp_ylim = kwargs.get('SFp_ylim', [0,6])
        ax.set_ylim(SFp_ylim)
        ax.set_yticks([0,2,4,6])
        ax.set_ylabel('SFp (c/deg)')
    elif p=='aulcsf':
        # aulcsf_ylim = kwargs.get('aulcsf_ylim', [0,2])
        # ax.set_ylim(aulcsf_ylim)        
        # ax.set_yticks([0,1,2])
        # ax.set_ylabel('AUC (a.u.)')
        ax.set_ylabel('normalized AUC (%)')
        aulcsf_ylim = kwargs.get('aulcsf_ylim', [0,150])
        ax.set_ylim(aulcsf_ylim)        
        ax.set_yticks([0,50,100, 150])
    elif p=='crf_exp':
        ax.set_ylim([0.4,10])
        ax.set_yscale('log')
        ax.set_yticks([0.4,1,10])
        ax.set_yticklabels([0.4,1,10])
        ax.set_ylabel('slope crf (a.u.)') 
    elif p=='sfmax':
        ax.set_ylim([0,40])
        # ax.set_yticks([0,1,2])
        ax.set_ylabel('SF max (c/deg)')    
    elif p=='CSp':
        ax.set_ylim([20,200])
        ax.set_yscale('log')
        ax.set_ylabel('CSp (a.u.)')
    elif p=='width_r':
        ax.set_ylim([0,1.5])
        ax.set_ylabel('width r (a.u.)')
    
    elif p=='rsq':
        ax.set_ylim([0,1])
        ax.set_ylabel('variance explained')



# ******
def ncsf_plt_qCSF(qCSF_info, ax, **kwargs):
    line_col        = kwargs.get('line_col', None)
    line_label      = kwargs.get('line_label', None)
    min_SF          = kwargs.get('min_SF', csenf_stim.SFs[0])
    max_SF          = kwargs.get('max_SF', 50)
    step_Sf         = kwargs.get('step_SF', 50)
    do_response     = kwargs.get('do_response', False)

    # Calculate CSF curves + matrix 
    log_sf_grid  = np.linspace(
        np.log10(min_SF),
        np.log10(max_SF), 
        step_Sf)
    sfs_for_plot = 10**log_sf_grid
    b_log_csf = 10**qcsf_curve(
        sfs_for_plot,
        qCSF_info['peakCS'],
        qCSF_info['peakSF'],
        qCSF_info['bdwth'],
        qCSF_info['lowSFtrunc'],
    )

    ax.plot(
        sfs_for_plot, 
        b_log_csf, 
        color=line_col, 
        linewidth=5,
        alpha=0.5,
        label=line_label
        )
    if do_response:
        correct = qCSF_info['CORRECT_history'] == 1
        ax.scatter(
            qCSF_info['SF_history'][correct],
            1/qCSF_info['CON_history'][correct],
            s=50,
            color = line_col,
            marker='+',
            alpha=.5,
            )
        ax.scatter(
            qCSF_info['SF_history'][~correct],
            1/qCSF_info['CON_history'][~correct],
            s=50,
            color = line_col,
            marker='o',
            alpha=.5,
            )        
        
        # ax.scatter(-1, -1, s=50, color='k', marker='+', alpha=0.5, label='hit')
        # ax.scatter(-1, -1, s=50, color='k', marker='o', alpha=0.5, label='miss')
    # Also plot max
    ncsfplt_set_csf_ax(ax, **kwargs)