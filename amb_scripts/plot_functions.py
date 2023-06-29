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
# from .utils import coord_convert, print_p
from dag_prf_utils.utils import dag_coord_convert as coord_convert
from dag_prf_utils.prfpy_functions import print_p
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
        "LE"            : rgba(252,141, 89, .8),#'#fd8d3c',
        "RE"            : rgba( 67,162,202, .8),#'#43a2ca',
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
        model_idx = print_p()['gauss']
    elif model=='norm':
        model_idx = print_p()['norm']

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


