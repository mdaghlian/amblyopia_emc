import numpy as np
import sys
import matplotlib.pyplot as plt

# import matplotlib as mpl
# from matplotlib import patches
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import linescanning.plotting as lsplt
# import pandas as pd
# from scipy.stats import binned_statistic

from prfpy.rf import csenf_exponential
from prfpy.model import CSenFModel

from dag_prf_utils.plot_functions import *
from dag_prf_utils.prfpy_functions import *

from .utils import *
from .load_saved_info import *
from .plot_functions import *

def get_csf_curves(log_SFs, width_r, sf0, maxC, width_l):    
    do_1_rf = False
    if isinstance(width_r, float):
        width_r  = np.array(width_r)
        sf0  = np.array(sf0)
        maxC  = np.array(maxC)
        width_l = np.array(width_l)
        do_1_rf = True
    log_sf0 = np.log10(sf0)
    log_maxC = np.log10(maxC)
    
    # Reshape for multiple RFs
    if not do_1_rf:
        log_SFs = np.tile(log_SFs, (width_l.shape[0],1))
    else:
        log_SFs = log_SFs[np.newaxis,...]

    #
    width_r     = width_r[...,np.newaxis]
    log_sf0     = log_sf0[...,np.newaxis]
    log_maxC    = log_maxC[...,np.newaxis]
    width_l     = width_l[...,np.newaxis]

    id_L = log_SFs < log_sf0
    id_R = log_SFs >= log_sf0

    # L_curve 
    L_curve = 10**(log_maxC - ((log_SFs - log_sf0)**2) * (width_l**2))
    R_curve = 10**(log_maxC - ((log_SFs - log_sf0)**2) * (width_r**2))

    csf_curve = np.zeros_like(L_curve)
    csf_curve[id_L] = L_curve[id_L]
    csf_curve[id_R] = R_curve[id_R]

    return csf_curve

class AmbCSFPlotter(Prf1T1M):

    def __init__(self, sub, real_tc, csf_params, **kwargs):
        super().__init__(csf_params, model='csf', **kwargs)
        self.sub = sub
        self.real_tc = real_tc
        self.csf_stim = amb_load_prfpy_stim(dm_type='csf')
        self.csf_model = CSenFModel(stimulus=self.csf_stim)
        self.prf_stim  = amb_load_prfpy_stim(dm_type='prf')
        # self.csf_curves = get_csf_curves(
        #     np.log10(self.csf_stim.SFs),
        #     self.pd_params['width_r']            ,
        #     self.pd_params['sf0'],
        #     self.pd_params['maxC'],
        #     self.pd_params['width_l'])
        self.sf_x_lim = (.25,20) # sf
        self.con_y_lim = (1, 500) # con

    def return_csf_rf_curve(self, idx):
        this_csf_rf, this_csf_curve = csenf_exponential(
            log_SF_grid = self.csf_stim.log_SF_grid, 
            CON_S_grid = self.csf_stim.CON_S_grid, 
            width_r = self.pd_params['width_r'][idx], 
            sf0 = self.pd_params['sf0'][idx], 
            maxC = self.pd_params['maxC'][idx], 
            width_l = self.pd_params['width_l'][idx], 
            return_curve = True)
        return this_csf_rf, this_csf_curve        

    def csf_tc_plot(self, idx, time_pt=None, return_fig=False):
        '''
        
        '''

        # [1] Create csf_curve, pred_tc & rf
        # this_csf_rf, this_csf_curve = csenf_exponential(
        #     log_SF_grid = self.csf_stim.log_SF_grid, 
        #     CON_S_grid = self.csf_stim.CON_S_grid, 
        #     width_r = self.pd_params['width_r'][idx], 
        #     sf0 = self.pd_params['sf0'][idx], 
        #     maxC = self.pd_params['maxC'][idx], 
        #     width_l = self.pd_params['width_l'][idx], 
        #     return_curve = True)
        this_csf_rf, this_csf_curve = self.return_csf_rf_curve(idx)
        this_csf_rf = np.squeeze(this_csf_rf)
        this_pred_tc = np.squeeze(self.csf_model.return_prediction(*list(self.prf_params_np[idx,:-1])))
        this_real_tc = self.real_tc[idx,:]
        # Plotting stimuli?
        do_current_stim = True
        if time_pt==None:
            do_current_stim = False
            time_pt = 213
                
        #
        if do_current_stim:
            fig, ax = plt.subplots(
                2, 4,
                gridspec_kw={'width_ratios': [2,1,1,6], 'height_ratios':[3,1]})
            extra_stim_ax = ax[1][-1]
            for i_ax in ax[1][:-1]:
                i_ax.axis('off')
            ax = ax[0]

        else:
            fig, ax = plt.subplots(
                2, 3,
                gridspec_kw={'width_ratios': [2,1,6], 'height_ratios':[3,1]},                
                )      
            extra_stim_ax = ax[1][-1]
            for i_ax in ax[1][:-1]:
                i_ax.axis('off')            
            new_ax = [None] * 4
            new_ax[0] = ax[0][0]
            new_ax[1] = ax[0][1]
            new_ax[3] = ax[0][2]
            ax = new_ax

        fig.set_size_inches(15,8)

        sf_vect = self.csf_stim.SF_seq
        inv_c_vect = 100/self.csf_stim.CON_seq

        # Setup ax 0
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('SF')
        ax[0].set_ylabel('100/Contrast')
        ax[0].set_title(f'{self.sub}: CSF, vx={idx}')
        ax[0].plot(self.csf_stim.SFs, this_csf_curve, lw=5, color='b') # Plot csf curve
        # Plot stimuli from 0:time_pt [Different color for in vs outside rf]
        id_to_plot = np.arange(time_pt)
        ax[0].scatter(sf_vect[id_to_plot],inv_c_vect[id_to_plot], c='k', s=100)
        self.make_full_rf(idx, ax=ax[0])
        if do_current_stim:
            if sf_vect[time_pt]==0:
                ax[0].text(.5, .5, 'BASELINE',
                        horizontalalignment='center',
                        verticalalignment='top',
                        backgroundcolor='1',
                        transform=ax[0].transAxes)            
            else:
                ax[0].scatter(sf_vect[time_pt],inv_c_vect[time_pt], c='g', marker='*', s=500)

        ax[0].set_xlim(self.sf_x_lim)
        ax[0].set_ylim(self.con_y_lim)

        param_text = ''
        param_ct = 0        
        for p in self.model_labels.keys():
            if self.fixed_hrf and ('hrf' in p):
                continue
            param_text += f'{p}={self.pd_params[p][idx]:.2f}; '
            param_ct += 1
            if param_ct>3:
                param_text += '\n'
                param_ct = 0

        # RF - in DM space:
        ax[1].imshow(this_csf_rf, vmin=0, vmax=1, cmap='magma')#, alpha=.5)        
        ax[1].grid('both')
        ax[1].axis('off')
        ax[1].set_title('CSF-DM space')
        for i in range(6):
            ax[1].plot((i-.5,i-.5), (-.5,13.5), 'k')
        for i in range(14):
            ax[1].plot((-0.5,5.5), (i-.5,i-.5), 'k')

        if do_current_stim:
            ax[2].imshow(
                self.csf_stim.design_matrix[:,:,time_pt], vmin=0, vmax=1, cmap='magma')
            ax[2].axis('off')
            ax[2].set_title('DM space')
            for i in range(6):
                ax[2].plot((i-.5,i-.5), (-.5,13.5), 'k')
            for i in range(14):
                ax[2].plot((-0.5,5.5), (i-.5,i-.5), 'k')


        # TC
        tc_ymin = np.min([this_pred_tc.min(), this_real_tc.min()])
        tc_ymax = np.max([this_pred_tc.max(), this_real_tc.max()])
        tc_x = np.arange(this_pred_tc.shape[-1]) * 1.5
        ax[-1].set_ylim(tc_ymin, tc_ymax)
        ax[-1].plot(tc_x[0:time_pt],this_pred_tc[0:time_pt], '-', markersize=10, lw=5, alpha=.5) # color=self.plot_cols[eye]        
        ax[-1].plot(tc_x[0:time_pt],this_real_tc[0:time_pt], ':^', color='k', markersize=5, lw=2, alpha=.5)
        ax[-1].plot((0,tc_x[-1]), (0,0), 'k')   
        ax[-1].set_title(param_text)

        # ** Add additional axis... 
        extra_stim_ax.plot(tc_x, sf_vect, 'k')                
        con_ax = extra_stim_ax.twinx()
        con_ax.plot(tc_x, inv_c_vect, 'r')
        extra_stim_ax.set_xlabel('time (s)')
        extra_stim_ax.set_ylabel('SF', color='black')
        con_ax.set_ylabel('100/Contrast', color='red')
        if do_current_stim:
            con_ax.scatter(tc_x[time_pt], inv_c_vect[time_pt], color='r', marker='*')
            extra_stim_ax.scatter(tc_x[time_pt], sf_vect[time_pt],color='k', marker='*')
        # sf_tick_locations = tc_x[::]
        # def tick_function(X):
        #     V = 1/(1+X)
        #     return ["%.3f" % z for z in V]
        # sf_ax.xaxis.set_ticks_position("bottom")
        # sf_ax.xaxis.set_label_position("bottom")

        # # Offset the twin axis below the host
        # sf_ax.spines["bottom"].set_position(("axes", -0.15))

        # # Turn on the frame for the twin axis, but then hide all 
        # # but the bottom spine
        # sf_ax.set_frame_on(True)
        # sf_ax.patch.set_visible(False)

        # sf_ax.set_xticks(sf_tick_locations)
        # sf_ax.set_xticklabels(tick_function(sf_tick_locations))
        # sf_ax.set_xlabel("SF")        
        # # **
        dag_update_fig_fontsize(fig, 15)        
        if return_fig:
            return fig
        return
    
    def make_full_rf(self, idx, ax):
        sf_grid = np.logspace(
            np.log10(self.sf_x_lim[0]),
            np.log10(self.sf_x_lim[1]), 50)
        con_grid = np.logspace(
            np.log10(self.con_y_lim[0]),
            np.log10(self.con_y_lim[1]), 50)
        sf_grid, con_grid = np.meshgrid(sf_grid,con_grid)
        this_csf_rf = csenf_exponential(
            log_SF_grid = np.log10(sf_grid), 
            CON_S_grid = con_grid, 
            width_r = self.pd_params['width_r'][idx], 
            sf0 = self.pd_params['sf0'][idx], 
            maxC = self.pd_params['maxC'][idx], 
            width_l = self.pd_params['width_l'][idx], 
            return_curve = False)
        scat_col = ax.scatter(
            sf_grid.ravel(),
            con_grid.ravel(),
            c=this_csf_rf.ravel(),
            alpha=.5,
            cmap='magma'
        )
        
        # cb = plt.gcf().colorbar(scat_col, ax=ax)