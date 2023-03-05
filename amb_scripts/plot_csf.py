import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import linescanning.plotting as lsplt
import pandas as pd
from scipy.stats import binned_statistic
import cortex
from .load_saved_info import *
from .utils import coord_convert, print_p, rescale_bw, hyphen_parse
from .plot_functions import *
from .pyctx import *
from prfpy.rf import csf_exponential

def grate_texture(sf=1, con=1, n_pix=200):
    n_deg = 5
    x_grid,y_grid = np.meshgrid(np.linspace(-n_deg,n_deg,n_pix),np.linspace(-n_deg,n_deg,n_pix))
    ecc = np.sqrt(x_grid**2 + y_grid**2)
    grate = np.cos(x_grid*sf) * con
    grate[ecc>n_deg] = 0
    # plt.imshow(grate, vmin=-1, vmax=1)
    return grate

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


def csf_tc_plotter(real_tc, pred_tc, params, idx):
    CSF_stim = amb_load_prfpy_stim('CSF')
    rfs = csf_exponential(
        CSF_stim.log_SF_grid, 
        CSF_stim.CON_S_grid, 
        *list(params[:,0:4].T))
    csf_curves = get_csf_curves(CSF_stim.log_SFs, *list(params[:,0:4].T))
    #
    fig, ax = plt.subplots(idx.shape[0], 3,gridspec_kw={'width_ratios': [2,2, 5]})    
    fig.set_size_inches(15,idx.shape[0]*2.5)
    xmin  = np.min(CSF_stim.log_SFs)
    xmax  = np.max(CSF_stim.log_SFs)
    ymin  = np.min(CSF_stim.CON_Ss)
    ymax  = np.max(CSF_stim.CON_Ss)
    plt_i = 0
    for i in idx:

        # ax[plt_i,0].loglog(CSF_stim.SFs,csf_curves[i,:].T)
        # ax[plt_i,0].set_xlim(1,2**10)
        # ax[plt_i,0].set_ylim(1,2**10)
        ax[plt_i,0].plot(CSF_stim.SFs,csf_curves[i,:].T)        
        # ax[plt_i,0].set_ylim([0,80])
        ax[plt_i,0].set_xlabel('SF')
        ax[plt_i,0].set_ylabel('Contrast')
        ax[plt_i,1].imshow(
            rfs[i,:,:],
            extent=[xmin,xmax,ymin,ymax], alpha=0.1, vmin=0, vmax=1)
        param_text = ''
        p_label = print_p()['CSF']
        for p in p_label.keys():
            param_text += f'{p}={params[i,p_label[p]]:.2f}; '
        # Find nearest SF
        
        stim_on = CSF_stim.SF_seq != 0
        ax[plt_i,2].plot(pred_tc[i,:])
        ax[plt_i,2].plot(real_tc[i,:])
        ax[plt_i,2].plot(stim_on)        
        ax[plt_i,2].set_title(param_text)

        plt_i += 1
    update_fig_fontsize(fig, 10)






# Object to get Prf Info and parameters...
class Csf2EGetter(object):
    '''
    Used to hold parameters from LE & RE
    & To return user specified masks 

    __init__ will set up the useful information into 3 pandas data frames
    >> including: all the parameters in the numpy arrays input model specific
        gauss: "width_r", "sf0", "", "a_val", "bold_baseline", "rsq"
    >> & eccentricit, polar angle, 
        "ecc", "pol",
    Split between: 'LE'; 'RE'; 'Ed'
    
    Functions:
    return_vx_mask: returns a mask for voxels, specified by the user
    return_th_param: returns the specified parameters, masked
    '''
    def __init__(self,sub, **kwargs):
        '''
        params_LE/X        np array, of all the parameters in the LE/X condition
        model               str, model: e.g., gauss or norm
        '''
        self.sub = sub
        self.nverts = amb_load_nverts(sub)
        self.total_nverts = np.sum(self.nverts)
        self.roi_fit = kwargs.get('roi_fit', 'all')
        self.fit_stage = kwargs.get('fit_stage', 'iter')
        self.model_list = kwargs.get('model_list', ['gauss', 'norm', 'CSF'])
        # Model information (could be norm, or gauss, or CSF)
        self.p_labels = print_p()
        # Possible parameters        
        # self.possible_params = {}
        # self.possible_params['gauss'] = ['a_val', 'bold_baseline', 'rsq', 'x', 'y', 'ecc', 'pol', 'a_sigma']
        # self.possible_params['norm']  = ['a_val', 'bold_baseline', 'rsq', 'x', 'y', 'ecc', 'pol', 'a_sigma', "c_val", "n_sigma", "b_val", "d_val"]
        # self.possible_params['CSF'] = ['a_val', 'bold_baseline', 'rsq', "width_r", "sf0", "log10_sf0", "maxC", "log10_maxC", "width_l"]

        # LOAD STIMULI
        self.prfpy_stim = {}
        self.prfpy_stim['CSF'] = amb_load_prfpy_stim('CSF')
        self.prfpy_stim['pRF'] = amb_load_prfpy_stim('pRF')        
        
        # Load params
        csf_params = kwargs.get('csf_params', None)
        if csf_params==None:
            csf_params = amb_load_prf_params(
                sub=self.sub, task_list=['CSFLE', 'CSFRE'],
                model_list='CSF',
                roi_fit = self.roi_fit, fit_stage=self.fit_stage
            )

        prf_params = kwargs.get('prf_params', None)
        if prf_params==None:
            prf_params = amb_load_prf_params(
                sub=self.sub, task_list=['pRFLE', 'pRFRE'],
                model_list=['gauss', 'norm'],
                roi_fit = self.roi_fit, fit_stage=self.fit_stage
            )
        
        # -> store the model parameters 
        self.params = {}
        self.params['CSF'] = {}
        self.params['gauss'] = {}
        self.params['norm'] = {}

        for eye in ['LE', 'RE']:
            self.params['CSF'][eye] = csf_params[f'CSF{eye}']['CSF']
            self.params['gauss'][eye] = prf_params[f'pRF{eye}']['gauss']
            self.params['norm'][eye] = prf_params[f'pRF{eye}']['norm']


        # Create dictionaries to turn into PD dataframes...
        self.pd_params = {}
        self.possible_params = {} # List of possible parameters for each model
        for model in self.model_list:                        
            LE_RE_Ed_dict = {'LE' : {},'RE' : {}, 'Ed' : {}, 'Em': {}} # L,R, difference, mean
            for eye in ['LE', 'RE']:
                # First add all the parameters from the numpy arrays (x,y, etc.)
                for i_label in self.p_labels[model].keys():
                    LE_RE_Ed_dict[eye][i_label] = self.params[model][eye][:,self.p_labels[model][i_label]]

                # Now add other useful things 
                if model!='CSF': # (pol and ecc): 
                    LE_RE_Ed_dict[eye]['ecc'], LE_RE_Ed_dict[eye]['pol'] = coord_convert(
                        LE_RE_Ed_dict[eye]["x"], LE_RE_Ed_dict[eye]["y"], 'cart2pol')                    
                else:
                    # -> log10_sf0, log10_maxC,
                    LE_RE_Ed_dict[eye]['log10_sf0'] = np.log10(LE_RE_Ed_dict[eye]['sf0'])
                    LE_RE_Ed_dict[eye]['log10_maxC'] = np.log10(LE_RE_Ed_dict[eye]['maxC'])
                    LE_RE_Ed_dict[eye]['sfmax'] = 10**(
                        np.sqrt(LE_RE_Ed_dict[eye]['log10_maxC']/(LE_RE_Ed_dict[eye]['width_r']**2)) + \
                                LE_RE_Ed_dict[eye]['log10_sf0'])
                    LE_RE_Ed_dict[eye]['sfmax'][LE_RE_Ed_dict[eye]['sfmax']>100] = 100 # MAX 
                    LE_RE_Ed_dict[eye]['log10_sfmax'] = np.log10(LE_RE_Ed_dict[eye]['sfmax'])
                if model=='norm':
                    # -> size ratio:
                    LE_RE_Ed_dict[eye]['size_ratio'] = LE_RE_Ed_dict[eye]['n_sigma'] / LE_RE_Ed_dict[eye]['a_sigma']
                    LE_RE_Ed_dict[eye]['amp_ratio'] = LE_RE_Ed_dict[eye]['a_val'] / LE_RE_Ed_dict[eye]['c_val']
                    LE_RE_Ed_dict[eye]['bd_ratio'] = LE_RE_Ed_dict[eye]['b_val'] / LE_RE_Ed_dict[eye]['d_val']
            # Get differnce & mean of params:
            for i_label in LE_RE_Ed_dict["LE"].keys():
                # Difference
                LE_RE_Ed_dict['Ed'][i_label] = LE_RE_Ed_dict['RE'][i_label] - LE_RE_Ed_dict['LE'][i_label]
                LE_RE_Ed_dict['Em'][i_label] = (LE_RE_Ed_dict['RE'][i_label] + LE_RE_Ed_dict['LE'][i_label])/2

            self.pd_params[model] = {}
            # Convert to PD & save into object
            for i_E in LE_RE_Ed_dict.keys():
                self.pd_params[model][i_E] = pd.DataFrame(LE_RE_Ed_dict[i_E])

        # Save list of possible parameters for each model:
        self.possible_params = {}
        for model in self.model_list:
            self.possible_params[model] = list(self.pd_params[model]['LE'].keys())                        

        

    def return_vx_mask(self, th_dict={}):
        '''
        return_vx_mask: returns a mask for voxels, specified by the user

        Dictionary key specifies which parameter a threshold is applied to
        The value associated with that key is determines the threshold
        
        Key setup        
        th_dict = {"model-eye-thresh-param": value}
        
        E.g.1: Return mask for voxels with rsq > 0.1 for the gauss fits on the left eye
            self.return_vx_mask(th_dict={'gauss-LE-min-rsq' : 0.1})

        E.g.2: Return mask for voxels with rsq > 0.1 for gauss fits on LE
            And where the maximum eccentricity in the gauss fits on RE is less than 5
            self.return_vx_mask(th_dict={
                'gauss-LE-min-rsq' : 0.1,
                'gauss-RE-max-ecc' : 5,
                })
        Options for model:
            'CSF', 'gauss', 'norm', 'ALL' (all applies the threshold to all *possible* models)
        Option for eye:
            'LE', 'RE', 'Ed' (difference in eye),'Em' (mean of both eyes), 'ALL' (all applies to both R & L)
        Option for thresh:
            'min', 'max', 'bound' (bound requires to values, min and max)

        Options for param:
            ALL models: "a_val", "bold_baseline", "rsq" 
                        (a_val is the term I use for beta, because it alines with a_val in norm model)
            PRF (gauss & norm): "x", "y", "ecc", "pol", "a_sigma"
            norm : "c_val", "n_sigma", "b_val", "d_val"            
            CSF: "width_r", "sf0", "log10_sf0", "maxC", "log10_maxC", "width_l"
        ***** ROI *****
        You can also include an roi:
            Either by inputting a np.ndarray of your own (defined outside)
            Or by specifying the ROI by a string ('V1_exvivo')
        ***************
                
        '''        

        # Start with EVRYTHING included         
        vx_mask = np.ones(self.total_nverts, dtype=bool)
        for i_key in th_dict.keys():
            i_key_str = str(i_key)
            if 'roi' in i_key_str:
                if isinstance(th_dict[i_key], np.ndarray): # Specified outside
                    vx_mask &= th_dict[i_key] # assume it is a boolean array (can be used to add roi)
                elif isinstance(th_dict[i_key], str): # Load specified roi                    
                    roi_mask = amb_load_roi(self.sub, th_dict[i_key])
                    vx_mask &= roi_mask
                continue # MOVE TO NEXT KEY...

            # Split the key into components
            model,eye,thresh,param = i_key_str.split('-')
            # ** APPLY TO ALL MODELS **
            if model=='ALL':
                # Do all models have this parameter?
                for imodel in self.model_list:
                    if param not in self.possible_params[imodel]:
                        print(f'Not applying {param} thresh to {imodel}')
                    else:
                        vx_mask &= self.return_vx_mask(
                            {f'{imodel}-{eye}-{thresh}-{param}': th_dict[i_key]})
                continue
            # ** APPLY TO LE & RE **

            if (eye=='ALL') | (eye=='Ed'):
                for ieye in ['LE', 'RE']:
                    vx_mask &= self.return_vx_mask(
                        {f'{model}-{ieye}-{thresh}-{param}': th_dict[i_key]})
                continue

            # ** Check does this model have this parameter?
            if param not in self.possible_params[model]:
                print(f'{param} not a possible parameter for {model}')
                print('skipping threshold')
                continue

            if thresh=='min':
                vx_mask &= self.pd_params[model][eye][param].gt(th_dict[i_key]) # Greater than
            elif thresh=='max':
                vx_mask &= self.pd_params[model][eye][param].lt(th_dict[i_key]) # less than
            elif thresh=='bound':
                vx_mask &= self.pd_params[model][eye][param].gt(th_dict[i_key][0]) # Greater than
                vx_mask &= self.pd_params[model][eye][param].lt(th_dict[i_key][1]) # less than
            else:
                sys.exit()

        return vx_mask
    
    def return_th_param(self, model, eye, param, vx_mask=None):
        '''        
        Return the parameters for *model*, *eye*, and *param*
            model: gauss, norm, csf
            eye: LE,RE,Ed
            param: ... (see vx_mask entry)
        Masked by vx_mask
        '''
        if vx_mask is None:
            vx_mask = np.ones(self.total_nverts, dtype=bool)
        if not isinstance(param, list):
            param = [param]
        params_out = []
        for i_param in param:
            self.pd_params[model][eye][i_param][vx_mask]
            params_out.append(self.pd_params[model][eye][i_param][vx_mask])
        if len(params_out)==1:
            params_out = params_out[0]
        
        return params_out

    def return_w_mean(self, model, eye, param, vx_mask):
        p_masked = self.return_th_param(model, eye, param, vx_mask)
        rsq_masked = self.return_th_param(model, eye, 'rsq', vx_mask)
        p_wmean = (p_masked*rsq_masked).sum()/rsq_masked.sum()
        return p_wmean
# ************************************************************************************************
# ************************************************************************************************
# ************************************************************************************************
# Plotting object which can generate different types of shift plots
class AmbPlotter(Csf2EGetter):
    def __init__(self,sub, **kwargs):
        super().__init__(sub, **kwargs)
        #
        self.aperture_rad = kwargs.get("aperture_rad",5)
        self.ecc_bounds = kwargs.get("ecc_bounds",np.linspace(0, 5, 7))
        self.pol_bounds = kwargs.get("pol_bounds",np.linspace(0,2*np.pi,13))
        self.plot_cols = get_plot_cols()
        #
        # Load TS
        real_tc = amb_load_real_tc(sub=self.sub,task_list=['pRFLE', 'pRFRE', 'CSFLE', 'CSFRE'] )
        self.real_tc ={}
        self.real_tc['CSF'] = {}
        self.real_tc['pRF'] = {}
        for eye in ['LE', 'RE']:
            self.real_tc['CSF'][eye] = real_tc[f'CSF{eye}']                    
            self.real_tc['pRF'][eye] = real_tc[f'pRF{eye}'] 
        
        csf_pred_tc = amb_load_pred_tc(
            self.sub, task_list=['CSFLE', 'CSFRE'], model_list='CSF', 
            roi_fit=self.roi_fit, fit_stage=self.fit_stage)                        
        prf_pred_tc = amb_load_pred_tc(
            self.sub, task_list=['pRFLE', 'pRFRE'], model_list=['gauss', 'norm'], 
            roi_fit=self.roi_fit, fit_stage=self.fit_stage)                        
        self.pred_tc = {}
        self.pred_tc['CSF'] = {}
        self.pred_tc['gauss'] = {}
        self.pred_tc['norm'] = {}
        for eye in ['LE', 'RE']:
            self.pred_tc['CSF'][eye] = csf_pred_tc[f'CSF{eye}']['CSF']                    
            self.pred_tc['gauss'][eye] = prf_pred_tc[f'pRF{eye}']['gauss']
            self.pred_tc['norm'][eye] = prf_pred_tc[f'pRF{eye}']['norm']

        # Load CSF curves
        self.csf_curves = {}
        self.csf_curves['LE'] = get_csf_curves(
            self.prfpy_stim['CSF'].log_SFs, 
            *list(self.params['CSF']['LE'][:,0:4].T))
        self.csf_curves['RE'] = get_csf_curves(
            self.prfpy_stim['CSF'].log_SFs, 
            *list(self.params['CSF']['RE'][:,0:4].T))
        # Also by time
        self.csf_curve_seqs = {}
        self.csf_curve_seqs['LE'] = get_csf_curves(
            np.log10(self.prfpy_stim['CSF'].SF_seq), 
            *list(self.params['CSF']['LE'][:,0:4].T))
        self.csf_curve_seqs['RE'] = get_csf_curves(
            np.log10(self.prfpy_stim['CSF'].SF_seq), 
            *list(self.params['CSF']['RE'][:,0:4].T))
        
        # Load RFs
        self.csf_rfs = {}
        self.csf_rfs['LE'] = csf_exponential(
            self.prfpy_stim['CSF'].log_SF_grid,
            self.prfpy_stim['CSF'].CON_S_grid,
            *list(self.params['CSF']['LE'][:,0:4].T))
        self.csf_rfs['RE'] = csf_exponential(
            self.prfpy_stim['CSF'].log_SF_grid,
            self.prfpy_stim['CSF'].CON_S_grid,
            *list(self.params['CSF']['RE'][:,0:4].T))
        
    def csf_tc_plot(self, eye, idx):
        
        this_params = self.params['CSF'][eye][idx,:]
        rfs = csf_exponential(
            self.prfpy_stim['CSF'].log_SF_grid, 
            self.prfpy_stim['CSF'].CON_S_grid, 
            *list(this_params[:,0:4].T))

        csf_curves = get_csf_curves(
            self.prfpy_stim['CSF'].log_SFs, 
            *list(this_params[:,0:4].T))
        #
        fig, ax = plt.subplots(idx.shape[0], 3,gridspec_kw={'width_ratios': [2,2,5]})    
        fig.set_size_inches(15,idx.shape[0]*2.5)
        xmin  = np.min(self.prfpy_stim['CSF'].SFs)
        xmax  = np.max(self.prfpy_stim['CSF'].SFs)
        ymin  = np.min(self.prfpy_stim['CSF'].CONs)
        ymax  = np.max(self.prfpy_stim['CSF'].CONs)
        plt_i = 0
        for i in range(len(idx)):

            # ax[plt_i,0].plot(
            #     self.prfpy_stim['CSF'].SFs,
            #     csf_curves[i,:].T)        
            ax[plt_i,0].loglog(self.prfpy_stim['CSF'].SFs,csf_curves[i,:].T)
            lt_curve = (self.prfpy_stim['CSF'].CON_grid>=csf_curves[i,:].T).ravel()
            ax[plt_i,0].scatter(
                self.prfpy_stim['CSF'].SF_grid.ravel()[lt_curve],
                self.prfpy_stim['CSF'].CON_grid.ravel()[lt_curve],
                c='y'
            )
            ax[plt_i,0].scatter(
                self.prfpy_stim['CSF'].SF_grid.ravel()[~lt_curve],
                self.prfpy_stim['CSF'].CON_grid.ravel()[~lt_curve],
                c='r'
            )            
            # ax[plt_i,0].set_xlim(1,2**10)
            ax[plt_i,0].set_ylim(10**-1,ymax)
            # ax[plt_i,0].set_ylim(0,100)
            ax[plt_i,0].set_xlabel('SF')
            ax[plt_i,0].set_ylabel('Contrast')
            ax[plt_i,1].imshow(
                rfs[i,:,:], alpha=0.1, vmin=0, vmax=1)
                # extent=[xmin,xmax,ymin,ymax], alpha=0.1, vmin=0, vmax=1)
            param_text = ''
            
            for p in self.p_labels['CSF'].keys():
                param_text += f'{p}={this_params[i,self.p_labels["CSF"][p]]:.2f}; '
            # Find nearest SF
            
            ax[plt_i,2].plot(self.pred_tc['CSF'][eye][idx[i],:])
            ax[plt_i,2].plot(self.real_tc['CSF'][eye][idx[i],:])
            # stim_on = CSF_stim.SF_seq != 0
            # ax[plt_i,2].plot(stim_on)        
            ax[plt_i,2].set_title(param_text)

            plt_i += 1
        update_fig_fontsize(fig, 10)        

    def csf_tc_plotV2(self, eye, idx, time_pt=None):
        
        do_current_stim = True
        if time_pt==None:
            do_current_stim = False
            time_pt = 213

        #
        fig, ax = plt.subplots(1, 2,gridspec_kw={'width_ratios': [2,5]})    
        fig.set_size_inches(15,5)

        sf_vect = self.prfpy_stim['CSF'].SF_seq
        inv_c_vect = 100/self.prfpy_stim['CSF'].CON_seq
        # Setup ax 0
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_xlabel('SF')
        ax[0].set_ylabel('100/Contrast')
        ax[0].set_title(f'{self.sub}: CSF - {eye}, vx={idx}')
        ax[0].plot(self.prfpy_stim['CSF'].SFs, self.csf_curves[eye][idx,:].T, lw=5, color=self.plot_cols[eye]) # Plot csf curve
        
        # Plot stimuli from 0:time_pt [Different color for in vs outside rf]
        bool_lt = inv_c_vect < self.csf_curve_seqs[eye][idx,:]
        id_to_plot = np.arange(time_pt)
        id_lt = id_to_plot[bool_lt[0:time_pt]]
        id_gt = id_to_plot[~bool_lt[0:time_pt]]
        ax[0].scatter(sf_vect[id_lt],inv_c_vect[id_lt], c='r', s=100)
        ax[0].scatter(sf_vect[id_gt],inv_c_vect[id_gt], c='k', s=100)
        if do_current_stim:
            if sf_vect[time_pt]==0:
                ax[0].text(.5, .5, 'BASELINE',
                        horizontalalignment='center',
                        verticalalignment='top',
                        backgroundcolor='1',
                        transform=ax[0].transAxes)            
            else:
                ax[0].scatter(sf_vect[time_pt],inv_c_vect[time_pt], c='g', marker='*', s=500)

        x_lim = (.25,20)
        y_lim = (1, 500)
        ax[0].set_xlim(x_lim)
        ax[0].set_ylim(y_lim)
        ax[0].set_aspect('equal')
        param_text = ''
        param_ct = 0   
        plabels_to_show = ['width_r', 'log10_sf0', 'log10_maxC', 'a_val', 'bold_baseline', 'rsq', 'sfmax']     
        for p in plabels_to_show:
            param_text += f'{p}={self.pd_params["CSF"][eye][p][idx]:.2f}; '
            param_ct += 1
            if param_ct>3:
                param_text += '\n'
                param_ct = 0

        this_pred_tc = self.pred_tc['CSF'][eye][idx,:]
        this_real_tc = self.real_tc['CSF'][eye][idx,:]
        tc_ymin = np.min([this_pred_tc.min(), this_real_tc.min()])
        tc_ymax = np.max([this_pred_tc.max(), this_real_tc.max()])
        ax[1].set_ylim(tc_ymin, tc_ymax)
        tc_x = np.arange(this_pred_tc.shape[0]) * 1.5
        ax[1].plot(
            tc_x[0:time_pt],
            this_pred_tc[0:time_pt], '-', color=self.plot_cols[eye], markersize=10, lw=5, alpha=.5)
        ax[1].plot(            
            tc_x[0:time_pt],
            this_real_tc[0:time_pt], '^', color='k', markersize=5, lw=5, alpha=.5)
        ax[1].plot((0,tc_x[-1]), (0,0), 'k')   
        ax[1].set_title(param_text)
        fig.set_tight_layout('tight')
        update_fig_fontsize(fig, 20)        
        return fig
        # *** END ***

    def csf_tc_plotV3(self, eye, idx, time_pt=None):
        do_current_stim = True
        if time_pt==None:
            do_current_stim = False
            time_pt = 213
        
        this_params = self.params['CSF'][eye][idx,:]

        #
        fig, ax = plt.subplots(1, 4,gridspec_kw={'width_ratios': [2,1,1,6]})    
        fig.set_size_inches(15,5)

        sf_vect = self.prfpy_stim['CSF'].SF_seq
        inv_c_vect = 100/self.prfpy_stim['CSF'].CON_seq
        # Setup ax 0
        ax[0].set_yscale('log')
        ax[0].set_xscale('log')
        ax[0].set_aspect('equal')
        ax[0].set_xlabel('SF')
        ax[0].set_ylabel('100/Contrast')
        ax[0].set_title(f'{self.sub}: CSF - {eye}, vx={idx}')
        ax[0].plot(self.prfpy_stim['CSF'].SFs, self.csf_curves[eye][idx,:].T, lw=5, color=self.plot_cols[eye]) # Plot csf curve
        
        # Plot stimuli from 0:time_pt [Different color for in vs outside rf]
        bool_lt = inv_c_vect < self.csf_curve_seqs[eye][idx,:]
        id_to_plot = np.arange(time_pt)
        id_lt = id_to_plot[bool_lt[0:time_pt]]
        id_gt = id_to_plot[~bool_lt[0:time_pt]]
        ax[0].scatter(sf_vect[id_lt],inv_c_vect[id_lt], c='r', s=100)
        ax[0].scatter(sf_vect[id_gt],inv_c_vect[id_gt], c='k', s=100)
        if do_current_stim:
            if sf_vect[time_pt]==0:
                ax[0].text(.5, .5, 'BASELINE',
                        horizontalalignment='center',
                        verticalalignment='top',
                        backgroundcolor='1',
                        transform=ax[0].transAxes)            
            else:
                ax[0].scatter(sf_vect[time_pt],inv_c_vect[time_pt], c='g', marker='*', s=500)

        x_lim = (.25,20)
        y_lim = (1, 500)
        ax[0].set_xlim(x_lim)
        ax[0].set_ylim(y_lim)
        param_text = ''
        param_ct = 0        
        for p in self.p_labels['CSF'].keys():
            param_text += f'{p}={this_params[self.p_labels["CSF"][p]]:.2f}; '
            param_ct += 1
            if param_ct>3:
                param_text += '\n'
                param_ct = 0

        # RF - in DM space:
        ax[1].imshow(self.csf_rfs[eye][idx,:,:], vmin=0, vmax=1)#, alpha=.5)        
        for i in range(6):
            ax[1].plot((i-.5,i-.5), (-.5,13.5), 'k')
            ax[2].plot((i-.5,i-.5), (-.5,13.5), 'k')
        for i in range(14):
            ax[1].plot((-0.5,5.5), (i-.5,i-.5), 'k')
            ax[2].plot((-0.5,5.5), (i-.5,i-.5), 'k')

        ax[1].grid('both')
        ax[1].axis('off')
        ax[1].set_title('RF')
        
        ax[2].imshow(self.prfpy_stim['CSF'].design_matrix[:,:,time_pt], vmin=0, vmax=1)
        ax[2].axis('off')
        ax[2].set_title('DM space')
        if not do_current_stim:
            ax[2].set_visible(False)
        # TC
        this_pred_tc = self.pred_tc['CSF'][eye][idx,:]
        this_real_tc = self.real_tc['CSF'][eye][idx,:]
        tc_ymin = np.min([this_pred_tc.min(), this_real_tc.min()])
        tc_ymax = np.max([this_pred_tc.max(), this_real_tc.max()])
        tc_x = np.arange(this_pred_tc.shape[0]) * 1.5
        ax[-1].set_ylim(tc_ymin, tc_ymax)
        ax[-1].plot(tc_x[0:time_pt],this_pred_tc[0:time_pt], '-', color=self.plot_cols[eye], markersize=10, lw=5, alpha=.5)
        ax[-1].plot(tc_x[0:time_pt],this_real_tc[0:time_pt], '^', color='k', markersize=5, lw=5, alpha=.5)
        ax[-1].plot((0,tc_x[-1]), (0,0), 'k')   
        ax[-1].set_title(param_text)
        fig.set_tight_layout('tight')
        update_fig_fontsize(fig, 20)        
        return fig
        # *** END ***

    def csf_tc_plotV4(self, eye, idx, time_pt=213):
        
        this_params = self.params['CSF'][eye][idx,:]
        sf_vect = self.prfpy_stim['CSF'].SF_seq
        c_vect = self.prfpy_stim['CSF'].CON_seq
        inv_c_vect = 100/c_vect

        #
        fig, ax = plt.subplots(1, 5,gridspec_kw={'width_ratios': [2,2,1,1,5]})    
        fig.subplots_adjust(wspace=None)
        fig.set_size_inches(25,5)

        # Stimulus:
        grate = grate_texture(
            sf=sf_vect[time_pt],
            con=c_vect[time_pt])
        ax[0].imshow(grate, vmin=-1, vmax=1, cmap='Greys')
        ax[0].axis('off')
        ax[0].set_title(f'SF={sf_vect[time_pt]:.3f}, C={c_vect[time_pt]:.2f}')
        # Setup ax 1
        ax[1].set_yscale('log')
        ax[1].set_xscale('log')
        ax[1].set_xlabel('SF')
        ax[1].set_ylabel('100/Contrast')
        ax[1].set_title(f'{self.sub}: CSF - {eye}, vx={idx}')
        ax[1].plot(self.prfpy_stim['CSF'].SFs, self.csf_curves[eye][idx,:].T, lw=5, color=self.plot_cols[eye]) # Plot csf curve
        
        # Plot stimuli from 0:time_pt [Different color for in vs outside rf]
        bool_lt = inv_c_vect < self.csf_curve_seqs[eye][idx,:]
        id_to_plot = np.arange(time_pt)
        id_lt = id_to_plot[bool_lt[0:time_pt]]
        id_gt = id_to_plot[~bool_lt[0:time_pt]]
        ax[1].scatter(sf_vect[id_lt],inv_c_vect[id_lt], c='r', s=100)
        ax[1].scatter(sf_vect[id_gt],inv_c_vect[id_gt], c='k', s=100)
        if sf_vect[time_pt]==0:
            ax[1].text(.5, .5, 'BASELINE',
                    horizontalalignment='center',
                    verticalalignment='top',
                    backgroundcolor='1',
                    transform=ax[1].transAxes)            
        else:
            ax[1].scatter(sf_vect[time_pt],inv_c_vect[time_pt], c='g', marker='*', s=500)

        x_lim = (.25,20)
        y_lim = (1, 500)
        ax[1].set_xlim(x_lim)
        ax[1].set_ylim(y_lim)
        # NEXT AXES
        param_text = ''
        param_ct = 0        
        for p in self.p_labels['CSF'].keys():
            param_text += f'{p}={this_params[self.p_labels["CSF"][p]]:.2f}; '
            param_ct += 1
            if param_ct>3:
                param_text += '\n'
                param_ct = 0

        # RF - in DM space:
        for i in range(6):
            ax[2].plot((i-.5,i-.5), (-.5,13.5), 'k')
            ax[3].plot((i-.5,i-.5), (-.5,13.5), 'k')
        for i in range(14):
            ax[2].plot((-0.5,5.5), (i-.5,i-.5), 'k')
            ax[3].plot((-0.5,5.5), (i-.5,i-.5), 'k')

        ax[2].imshow(self.csf_rfs[eye][idx,:,:], vmin=0, vmax=1)#, alpha=.5)        
        ax[2].grid('both')
        ax[2].axis('off')
        ax[2].set_title('RF')
        
        ax[3].imshow(self.prfpy_stim['CSF'].design_matrix[:,:,time_pt], vmin=0, vmax=1)
        ax[3].axis('off')
        ax[3].set_title('DM space')

        # TC
        this_pred_tc = self.pred_tc['CSF'][eye][idx,:]
        this_real_tc = self.real_tc['CSF'][eye][idx,:]
        tc_ymin = np.min([this_pred_tc.min(), this_real_tc.min()])
        tc_ymax = np.max([this_pred_tc.max(), this_real_tc.max()])
        tc_x = np.arange(this_pred_tc.shape[0]) * 1.5
        ax[-1].set_ylim(tc_ymin, tc_ymax)
        ax[-1].plot(tc_x[0:time_pt],this_pred_tc[0:time_pt], '-', color=self.plot_cols[eye], markersize=10, lw=5, alpha=.5)
        ax[-1].plot(tc_x[0:time_pt],this_real_tc[0:time_pt], '^', color='k', markersize=5, lw=5, alpha=.5)
        ax[-1].plot((0,tc_x[-1]), (0,0), 'k')   
        ax[-1].set_title(param_text)
        fig.set_tight_layout('tight')
        update_fig_fontsize(fig, 20)        
        return fig
        # *** END ***


    def arrows_drop(self, axs, th_dict=None,model='gauss', **kwargs):
        '''
        Like arrow_plot (see below)
        >> but also include 'drop out' vx with a PRF in one condition but not another
        '''
        vx_mask = self.return_vx_mask(th_dict)
        # -> override some stuff
        drop_rsq = kwargs.get('drop_rsq', 0.1) # Voxels to be dropped based on this rsq 
        drop_ecc = kwargs.get('drop_ecc', 5)   # exclude all voxels outside this range
        kwargs['do_binning'] = False        
        kwargs['do_scatter'] = False

        dot_alpha = self._return_dot_alpha(**kwargs)
        if not isinstance(dot_alpha, np.ndarray)            :
            dot_alpha = np.ones_like(vx_mask) * dot_alpha
        # Different dot sizes for different eyes...
        dot_size = self._return_dot_size(**kwargs)

        if isinstance(dot_size, np.ndarray):
            dot_size = dot_size[vx_mask]        
        # -> specify which vox to drop...
        if 'vx_drop_in' in kwargs.keys():
            # SPECIFIED...
            vx_drop_in = kwargs['vx_drop_in']
            vx_drop_out = kwargs['vx_drop_out']

        else:
            old_vx_mask = np.copy(vx_mask) # Whatever we want to be applied to everything...

            # For arrows - apply threshold to everything
            vx_mask     = self.return_vx_mask(th_dict = {
                f'{model}-ALL-max-ecc' : drop_ecc,
                f'{model}-ALL-min-rsq' : drop_rsq,
                })            
            # - For drop points - only pts inside the ecc range  
            vx_drop_out     = self.return_vx_mask(th_dict = {
                f'{model}-ALL-max-ecc': drop_ecc,
                f'{model}-LE-min-rsq' : drop_rsq,
                f'{model}-RE-max-rsq' : drop_rsq,
                })
            vx_drop_in      = self.return_vx_mask(th_dict = {
                f'{model}-ALL-max-ecc': drop_ecc,
                f'{model}-LE-max-rsq' : drop_rsq,
                f'{model}-RE-min-rsq' : drop_rsq,
                })
            
            vx_mask     &= old_vx_mask
            vx_drop_out &= old_vx_mask
            vx_drop_in  &= old_vx_mask

        if vx_drop_out.sum() != 0:
            # Drop out points - where there is a good prf in LE, but not RE
            axs.scatter(
                self.pd_params[model]['LE']['x'][vx_drop_out], 
                self.pd_params[model]['LE']['y'][vx_drop_out], 
                alpha=dot_alpha[vx_drop_out],
                color='k', s=dot_size, marker='.')
        if vx_drop_in.sum() != 0:
            # Drop in points - where there is a good prf in *RE* but not LE
            axs.scatter(
                self.pd_params[model]['RE']['x'][vx_drop_in], 
                self.pd_params[model]['RE']['y'][vx_drop_in], 
                alpha=dot_alpha[vx_drop_in],
                color='g', s=dot_size, marker='.')
        
        # Now do the arrows
        self.arrow_plot(axs=axs, vx_mask=vx_mask, model=model, **kwargs)

    def arrow_plot(self, axs, vx_mask=None, model='gauss', **kwargs):
        ''' 
        PLOT FUNCTION: 
        Takes voxel position in LE and end coords (new_x, new_y) produces a plot, with arrows from old to new points 
        Will also show the aperture of stimuli
        Parameters
        ---------------
        axs :           matplotlib axes         where to plot
        vx_mask :       bool array              which voxels to include
        do_binning :    bool                    Bin the position (or not)
        do_scatter :    bool                    Include scatters of the voxel positions (ALL, LE, RE)
        /_LE ""
        /_RE ""                   
        do_arrows :     bool                    Include arrows
        ecc_bounds      np.ndarays              If binning, how split the visual field
        pol_/                               
        LE_col         any value for color     Gives color for points associated w/ LE/X
        RE_col        
        patch_col       any value for color     Color for the patch 
        dot_alpha       ... see function        Alpha for the points
        dot_size        ... see function        Size for the points
        '''
        
        th_dict = kwargs.get('th_dict', False)
        if th_dict:
            vx_mask = self.return_vx_mask(th_dict)

        # Get arguments related to plotting:
        do_binning = kwargs.get("do_binning", False)
        do_scatter = kwargs.get("do_scatter", False)
        do_scatter_LE = kwargs.get("do_scatter_old", True)
        do_scatter_RE = kwargs.get("do_scatter_new", True)
        if not do_scatter:
            do_scatter_LE = False
            do_scatter_RE = False
        do_arrows = kwargs.get("do_arrows", True)
        ecc_bounds = kwargs.get("ecc_bounds", self.ecc_bounds)
        pol_bounds = kwargs.get("pol_bounds", self.pol_bounds)
        LE_col = kwargs.get("LE_col", self.plot_cols['LE'])
        RE_col = kwargs.get("RE_col", self.plot_cols["RE"])
        patch_col = kwargs.get("patch_col", self.plot_cols["RE"])
        arrow_col = kwargs.get("arrow_col", 'b')
        arrow_kwargs = {
            'scale'     : 1,                                    # ALWAYS 1 -> exact pt to exact pt 
            'width'     : kwargs.get('arrow_width', .01),       # of shaft (relative to plot )
            'headwidth' : kwargs.get('arrow_headwidth', .5),    # relative to width

        }    
        # *** Get values for dot alpha & dot_size ***   (*****dodgy*****)(*****dodgy*****)              
        # if do_scatter:
        #     dot_alpha = self._return_dot_alpha(**kwargs)
        #     if isinstance(dot_alpha, np.ndarray):
        #         dot_alpha = dot_alpha[vx_mask]
        #     # -> Dot size (*****dodgy*****)
        #     dot_size = self._return_dot_size(**kwargs)
        #     if isinstance(dot_size, np.ndarray):
        #         dot_size = dot_size[vx_mask]        
        #     LE_dot_size = dot_size
        #     RE_dot_size = dot_size
        #     # -> Dot col (*****dodgy*****)
        #     dot_col,dot_cmap = self._return_dot_col(**kwargs)
        #     if isinstance(dot_col, np.ndarray):
        #         dot_col = dot_col[vx_mask]
        #     dot_vmin = kwargs.get("dot_vmin", None)
        #     dot_vmax = kwargs.get("dot_vmax", None)
        # ************* GETS COMPLEX w/ 2 eyes - sticking to the simple stuff for now... 
        dot_alpha = 0.5
        dot_size = 500
        LE_dot_size = dot_size
        RE_dot_size = dot_size
        dot_col = 'b'
        # *** *** *** *** *** *** *** *** *** *** *** 
        ALL_LE_ecc, ALL_LE_pol, ALL_LE_x, ALL_LE_y = self.return_th_param(
            model=model,eye='LE', param=['ecc', 'pol', 'x', 'y'], vx_mask=vx_mask)
        
        ALL_RE_x, ALL_RE_y = self.return_th_param(
            model=model,eye='RE', param=['x', 'y'], vx_mask=vx_mask)
        
        if do_binning:
            # print("DOING BINNING") 
            LE_x2plot, LE_y2plot = self._return_ecc_pol_bin(
                params2bin=[ALL_LE_x, ALL_LE_y],
                ecc4bin=ALL_LE_ecc, pol4bin=ALL_LE_pol,
                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,
                bin_weight=None)
            RE_x2plot, RE_y2plot = self._return_ecc_pol_bin(
                params2bin=[ALL_RE_x, ALL_RE_y],
                ecc4bin=ALL_LE_ecc, pol4bin=ALL_LE_pol,
                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,
                bin_weight=None)            
        else:
            # ID ANY GOING TO LT 0.1 rsq 
            LE_x2plot = ALL_LE_x
            LE_y2plot = ALL_LE_y
            RE_x2plot = ALL_RE_x
            RE_y2plot = ALL_RE_y

        # CHECK - IS THERE ANYTHING TO PLOT?
        if LE_x2plot.shape[0]==0:
            self._add_bin_lines(axs, ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)        
            self._add_patches(axs, patch_col=patch_col)
            self._add_axs_basics(axs, **kwargs)                
            return
        dx = RE_x2plot - LE_x2plot
        dy = RE_y2plot - LE_y2plot

        # Plot old pts and new pts (different colors)
        if do_scatter_LE:
            axs.scatter(LE_x2plot, LE_y2plot, color=LE_col, s=LE_dot_size, alpha=dot_alpha, )#c=dot_col, cmap=dot_cmap)
        if do_scatter_RE:
            axs.scatter(RE_x2plot, RE_y2plot, color=RE_col, s=RE_dot_size, alpha=dot_alpha, )#c=dot_col, cmap=dot_cmap)
        

        # Add the arrows 
        if do_arrows: # Arrows all the same color
            if arrow_col=='angle':
                # Get the angles for the arrows
                _, angle = coord_convert(dx, dy, 'cart2pol')
                q_cmap = mpl.cm.__dict__['hsv']
                q_norm = mpl.colors.Normalize()
                q_norm.autoscale(angle)
                q_col = q_cmap(q_norm(angle))                
            elif isinstance(dot_col, np.ndarray):
                q_cmap = mpl.cm.__dict__[dot_cmap]
                q_norm = mpl.colors.Normalize()
                q_norm.autoscale(dot_col)
                q_col = q_cmap(q_norm(dot_col))
            else:
                q_col = arrow_col

            axs.quiver(LE_x2plot, LE_y2plot, dx, dy, scale_units='xy', 
                       angles='xy', alpha=dot_alpha,color=q_col,  **arrow_kwargs)
            
            # # For the colorbar
            # if isinstance(dot_col, np.ndarray):
            #     scat_col = axs.scatter(
            #         np.zeros_like(LE_x2plot), np.zeros_like(LE_x2plot), s=np.zeros_like(LE_x2plot), 
            #         c=dot_col, vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
            #     fig = plt.gcf()
            #     cb = fig.colorbar(scat_col, ax=axs)        
            #     cb.set_label(kwargs['dot_col'])

        self._add_bin_lines(axs, ecc_bounds=ecc_bounds, pol_bounds=pol_bounds)        
        self._add_patches(axs, patch_col=patch_col)
        self._add_axs_basics(axs, **kwargs)    
        # END FUNCTION 

    def scatter_param(self, axs, eye, model='gauss', vx_mask=None, **kwargs):
        '''
        PLOT FUNCTION: 
        Plot a parameter around the visual field
        Can use x,y position of voxels in the LE or RE task condition ("use_task")
        Will also show the aperture of stimuli,
        Parameters
        ---------------
        axs :           matplotlib axes         where to plot
        vx_mask :       bool array              which voxels to include
        dot_col         str                     SAME AS OTHER... specify the parameter name
        xy_task         str                     X,Y positions taken from either LE, or RE
        do_binning :    bool                    Bin the position (or not)
        ecc_bounds      np.ndarays              If binning, how split the visual field
        pol_/                               
        patch_col       any value for color     Color for the patch 
        do_patch        bool 
        dot_alpha       ... see function        Alpha for the points
        dot_size        ... see function        Size for the points                
        '''
        th_dict = kwargs.get('th_dict', False)
        if th_dict:
            vx_mask = self.return_vx_mask(th_dict)

        do_binning = kwargs.get("do_binning", False)
        patch_col = kwargs.get("patch_col", self.plot_cols[eye])
        ecc_bounds = kwargs.get("ecc_bounds", self.ecc_bounds)
        pol_bounds = kwargs.get("pol_bounds", self.pol_bounds)
        do_patch = kwargs.get("do_patch", True)
        # *** Get dot alpha, dot_size  & dot_col***         
        dot_alpha = self._return_dot_alpha(**kwargs)
        if isinstance(dot_alpha, np.ndarray):
            dot_alpha = dot_alpha[vx_mask]
        dot_size = self._return_dot_size(**kwargs)
        if isinstance(dot_size, np.ndarray):
            dot_size = dot_size[vx_mask]
        dot_col,dot_cmap = self._return_dot_col(**kwargs)
        if isinstance(dot_col, np.ndarray):
            dot_col = dot_col[vx_mask]
        dot_vmin = kwargs.get("dot_vmin", None)
        dot_vmax = kwargs.get("dot_vmax", None)
        # *** *** *** *** *** *** *** *** *** *** ***         
        # X,Y positions from specified task (ub=unbinned)
        ub_X, ub_Y, ub_ecc, ub_pol  = self.return_th_param(
            model=model, eye=eye, param=['x', 'y', 'ecc', 'pol'], vx_mask=vx_mask)
        if not do_binning: # Assign plotting values
            X2plot,Y2plot = ub_X, ub_Y
            C2plot = dot_col
            S2plot = dot_size
            alpha2plot = dot_alpha
        else:
            X2plot, Y2plot, C2plot = self._return_ecc_pol_bin(
                params2bin=[ub_X, ub_Y, dot_col],
                ecc4bin=ub_ecc, pol4bin=ub_pol, 
                ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,bin_weight=None)
            if isinstance(dot_size, np.ndarray):
                S2plot = self._return_ecc_pol_bin(
                    params2bin=[dot_size],ecc4bin=ub_ecc, pol4bin=ub_pol, 
                    ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,bin_weight=None)[0]
            else:
                S2plot = dot_size
            if isinstance(dot_alpha, np.ndarray):
                alpha2plot = self._return_ecc_pol_bin(
                    params2bin=[dot_alpha],ecc4bin=ub_ecc, pol4bin=ub_pol, 
                    ecc_bounds=ecc_bounds, pol_bounds=pol_bounds,bin_weight=None)[0]
            else:
                alpha2plot = dot_alpha

        scat_col = axs.scatter(
            X2plot, Y2plot, 
            c=C2plot, s=S2plot, alpha=alpha2plot, 
            vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
        fig = plt.gcf()
        cb = fig.colorbar(scat_col, ax=axs)        
        if not isinstance(kwargs['dot_col'], np.ndarray): 
            cb.set_label(kwargs['dot_col'])
        self._add_bin_lines(axs, **kwargs)
        if do_patch:        
            self._add_patches(axs, patch_col=patch_col)
        self._add_axs_basics(axs, **kwargs)    

    def scatter_generic(self, axs, vx_mask, x_param, y_param, **kwargs):
        '''
        PLOT FUNCTION: 
        Plot any parameter vs another...
        Can use x,y position of voxels in the LE or RE task condition ("use_task")
        Will also show the aperture of stimuli,
        Parameters
        ---------------
        axs :           matplotlib axes         where to plot
        vx_mask :       bool array              which voxels to include
        dot_col         str                     SAME AS OTHER... specify the parameter name
        xy_task         str                     X,Y positions taken from either LE, or RE
        dot_alpha       ... see function        Alpha for the points
        dot_size        ... see function        Size for the points    
        do_line         Add a bin line            
        '''
        do_line = kwargs.get('do_line', False)
        do_equal = kwargs.get('do_equal', True)
        # *** Get dot alpha, dot_size  & dot_col***         
        dot_alpha = self._return_dot_alpha(**kwargs)
        if isinstance(dot_alpha, np.ndarray):
            dot_alpha = dot_alpha[vx_mask]
        dot_size = self._return_dot_size(**kwargs)
        if isinstance(dot_size, np.ndarray):
            dot_size = dot_size[vx_mask]
        dot_col,dot_cmap = self._return_dot_col(**kwargs)
        if isinstance(dot_col, np.ndarray):
            dot_col = dot_col[vx_mask]
        dot_vmin = kwargs.get("dot_vmin", None)
        dot_vmax = kwargs.get("dot_vmax", None)
        # *** *** *** *** *** *** *** *** *** *** ***         
        # X,Y positions from specified task (ub=unbinned)
        x_mod, x_eye, x_id = x_param.split('-')
        y_mod, y_eye, y_id = y_param.split('-')

        X2plot = self.return_th_param(
            model=x_mod, eye=x_eye, param=x_id, vx_mask=vx_mask)
        Y2plot = self.return_th_param(
            model=y_mod, eye=y_eye, param=y_id, vx_mask=vx_mask)
        XY_corr = np.corrcoef(X2plot,Y2plot)[0,1]
        print(f'XY correlation: {XY_corr:.3f}')
        C2plot = dot_col
        S2plot = dot_size
        alpha2plot = dot_alpha
        scat_col = axs.scatter(
            X2plot, Y2plot, 
            c=C2plot, s=S2plot, alpha=alpha2plot, 
            vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
        if isinstance(C2plot, np.ndarray):
            fig = plt.gcf()
            cb = fig.colorbar(scat_col, ax=axs)        
            if not isinstance(kwargs['dot_col'], np.ndarray): 
                cb.set_label(kwargs['dot_col'])
        if do_line:
            self._plot_bin_line(X2plot, Y2plot, X2plot, axs=axs, **kwargs)
            self._plot_bin_line(X2plot, Y2plot, Y2plot, axs=axs, **kwargs)
        if do_equal:
            axmin = np.min([X2plot.min(),Y2plot.min()])
            axmax = np.max([X2plot.max(),Y2plot.max()])
            axs.set_xlim([axmin, axmax])
            axs.set_ylim([axmin, axmax])
            axs.plot((axmin, axmax), (axmin, axmax), 'k')
            # xmax = X2plot.max()
            # ymin = 
            # ymax = Y2plot.max()

        axs.set_xlabel(x_param)
        axs.set_ylabel(y_param)
        self._add_axs_basics(axs,xlabel=x_param, ylabel=y_param, title=f'{self.sub}, corr: {XY_corr:.3f}', **kwargs)    

    def hist_generic(self, axs, vx_mask, param, **kwargs):
        '''
        PLOT FUNCTION: 
        Plot any parameter vs another...
        Can use x,y position of voxels in the LE or RE task condition ("use_task")
        Will also show the aperture of stimuli,
        Parameters
        ---------------
        axs :           matplotlib axes         where to plot
        vx_mask :       bool array              which voxels to include
        '''
        alpha = kwargs.get('alpha', 0.5)
        n_bins = kwargs.get('n_bins', 20)
        bins = kwargs.get('bins', [])
        if bins==[]:
            bins = n_bins
        p_mod, p_eye, p_id = param.split('-')

        param2plot = self.return_th_param(
            model=p_mod, eye=p_eye, param=p_id, vx_mask=vx_mask)
        
        axs.hist(param2plot, bins=bins, color=self.plot_cols[p_eye],alpha=alpha,label=param)
        axs.legend()
        self._add_axs_basics(axs, **kwargs)    

    def ecc_2eye(self, axs, vx_mask, param, **kwargs):
        '''
        PLOT FUNCTION: 
        Same as ecc_1eye, but with both eyes, so that we can do arrows between them
        Parameters
        ---------------
        '''
        do_arrow = kwargs.get('do_arrow', True)
        arrow_col = kwargs.get("arrow_col", 'b')
        arrow_alpha = kwargs.get("arrow_alpha", .5)
        arrow_kwargs = {
            'scale'     : 1,                                    # ALWAYS 1 -> exact pt to exact pt 
            'width'     : kwargs.get('arrow_width', .001),       # of shaft (relative to plot )
            'headwidth' : kwargs.get('arrow_headwidth', .5),    # relative to width

        }
        dot_col = kwargs.get('dot_col', False)
        if not dot_col==False:
            ow_dot_col_LE = self.plot_cols['LE']    
            ow_dot_col_RE = self.plot_cols['RE']
        else:
            ow_dot_col_LE = False
            ow_dot_col_RE = False            

        # [1] LE 
        Lx, Ly = self.ecc_1eye(axs=axs, vx_mask=vx_mask, param='LE-'+param, 
                               ecc_task='LE', ow_dot_task='LE', ow_dot_col=ow_dot_col_LE, return_vals=True, **kwargs)
        # [2] RE
        Rx, Ry = self.ecc_1eye(axs=axs, vx_mask=vx_mask, param='RE-'+param, 
                               ecc_task='RE', ow_dot_task='RE', ow_dot_col=ow_dot_col_RE, return_vals=True, **kwargs)

        dx = Rx - Lx
        dy = Ry - Ly
        # Add the arrows 
        if arrow_col=='angle':
            # Get the angles for the arrows
            _, angle = coord_convert(dx, dy, 'cart2pol')
            q_cmap = mpl.cm.__dict__['hsv']
            q_norm = mpl.colors.Normalize()
            q_norm.autoscale(angle)
            q_col = q_cmap(q_norm(angle))
        else:
            q_col = arrow_col
        if do_arrow:
            axs.quiver(Lx, Ly, dx, dy, scale_units='xy', 
                        angles='xy', alpha=arrow_alpha,color=q_col,  **arrow_kwargs)

    def ecc_1eye(self, axs, vx_mask, param, **kwargs):
        '''
        PLOT FUNCTION: 
        Plot a parameter by eccentricity, and by 
        -> with the option to include arrows to a second pt...
        Parameters
        ---------------
        axs :           matplotlib axes         where to plot
        vx_mask :       bool array              which voxels to include
        dot_col         str                     SAME AS OTHER... specify the parameter name
        ecc_task         str                    Ecc positions taken from either LE, or RE (default is the same as specified parameter)
        dot_alpha       ... see function        Alpha for the points
        dot_size        ... see function        Size for the points   
        dot_col         ""
        do_lines        do the binned lines?             
        '''
        ecc_info = kwargs.get('ecc_info', None)
        do_line = kwargs.get('do_line', False)
        return_vals = kwargs.get('return_vals', False) # return the ecc and param values (use this for arrows)
        do_legend = kwargs.get('do_legend', False)
        do_scatter = kwargs.get('do_scatter', True)
        # *** Get dot alpha, dot_size  & dot_col***         
        dot_alpha = self._return_dot_alpha(**kwargs)
        if isinstance(dot_alpha, np.ndarray):
            dot_alpha = dot_alpha[vx_mask]
        dot_size = self._return_dot_size(**kwargs)
        if isinstance(dot_size, np.ndarray):
            dot_size = dot_size[vx_mask]
        dot_col,dot_cmap = self._return_dot_col(**kwargs)
        if isinstance(dot_col, np.ndarray):
            dot_col = dot_col[vx_mask]
        dot_vmin = kwargs.get("dot_vmin", None)
        dot_vmax = kwargs.get("dot_vmax", None)
        # *** *** *** *** *** *** *** *** *** *** ***         
        # ID the relevant parameters
        p_mod, p_eye, p_id = param.split('-')
        if ecc_info == None:
            ecc_mod = p_mod # default to same task        
            ecc_eye = p_eye
        else:
            ecc_mod, ecc_eye = ecc_info.split('-')
        # Get the eccentricity for relevant points        
        ecc_val = self.return_th_param(model=ecc_mod, eye=ecc_eye, param='ecc', vx_mask=vx_mask) 
        param_val = self.return_th_param(
            model=p_mod, eye=p_eye, param=p_id, vx_mask=vx_mask)

        X2plot,Y2plot = ecc_val, param_val
        C2plot = dot_col
        S2plot = dot_size
        alpha2plot = dot_alpha

        if do_scatter:
            scat_col = axs.scatter(
                X2plot, Y2plot, 
                color=C2plot, s=S2plot, alpha=alpha2plot, 
                vmin=dot_vmin, vmax=dot_vmax, cmap=dot_cmap)
            if isinstance(dot_col, np.ndarray) & do_legend:
                fig = plt.gcf()
                cb = fig.colorbar(scat_col, ax=axs)        
                if not isinstance(kwargs['dot_col'], np.ndarray): 
                    cb.set_label(kwargs['dot_col'])
        if do_line:
            self._plot_bin_line(X2plot, Y2plot, X2plot, axs=axs, line_col='k',line_label=p_id, **kwargs)                
        self._add_axs_basics(axs, **kwargs)    
        if return_vals:
            return X2plot, Y2plot

    # def pyctx_plot(self, param, param_w='LE-rsq', **kwargs):
    #     '''
    #     COPIED FROM PYCORTEX ALPHA PLOTTING
    #     Function to make plotting in pycortex (using web gui) easier
    #     I found that using the "cortex.Vertex2D" didn't work (IDK why)
    #     Based on Sumiya's code -> extracts the curvature as a grey map
    #     -> puts your data on top of it...

    #     sub             subject to plot (pycortex id)
    #     data            data to plot (np.array size of the surface)
    #     data_weight     Used to mask your data. Can be boolean or a range (should be between 0 and 1.)
    #                     See other options
                        

    #     *** Optional ***
    #     Value           Default             Meaning
    #     --------------------------------------------------------------------------------------------------
    #     data_w_thresh   None                1 or 2 values (gives the threshold of values to include (lower & upper bound))                    
    #     vmin            None                Minimal value for data cmap
    #     vmax            None                Maximum value for data cmap
    #     cmap            Retinotopy_RYBCR    Color map to use for data
    #     bool_mask       True                Mask the data with absolute mask or a gradient
    #     scale_data_w    False               Scale the data weight between 0 and 1         
    #     '''
    #     return_dict = kwargs.get("return_dict", True)

    #     if not isinstance(param, list):
    #         param = [param]
        
    #     # Make dictionary for webgl...
    #     ctx_dict = {}
    #     for i_p,this_param in enumerate(param):
    #         if isinstance(this_param, np.ndarray):
    #             data = this_param
    #         else:
    #             this_p_task,this_p_param = this_param.split('-')
    #             if not isinstance(param_w, list):
    #                 this_w_task,this_w_param = param_w.split('-')
    #             else:
    #                 this_w_task,this_w_param = param_w[i_p].split('-')
                
    #             data = self.return_th_param(task=this_p_task, param=this_p_param)[0]
    #         data_weight = self.return_th_param(task=this_w_task, param=this_w_param)[0]
    #         # ctx_dict[f'{this_param}-1'],ctx_dict[f'{this_param}-2'] = pycortex_alpha_plotting(
    #         ctx_dict[f'{this_param}-1'],_ = pycortex_alpha_plotting(
    #             sub=self.sub, 
    #             data=data,
    #             data_weight=data_weight, **kwargs)
    #     if return_dict:
    #         return ctx_dict
    #     else:        
    #         cortex.webgl.show(ctx_dict)            
    # **************************************************************************************************************** 
    # ****************************************************************************************************************
    # ****************************************************************************************************************
    # ASSISTING FUNCTIONS 
    def _show_ctx(self,ctx_dict):
        cortex.webgl.show(ctx_dict)
    def _add_bin_lines(self, axs, **kwargs):
        ecc_bounds = kwargs.get("ecc_bounds", self.ecc_bounds)
        pol_bounds = kwargs.get("pol_bounds", self.pol_bounds)        
        incl_ticks = kwargs.get("incl_ticks", False)
        # **** ADD THE LINES ****
        if not incl_ticks:
            axs.set_xticks([])    
            axs.set_yticks([])
        else:
            axs.set_xticks(ecc_bounds, rotation = 90)
            axs.set_xticklabels([f"{ecc_bounds[i]:.2f}\N{DEGREE SIGN}" for i in range(len(ecc_bounds))], rotation=90)
            axs.set_yticks([])
        axs.spines['right'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.spines['top'].set_visible(False)
        axs.spines['bottom'].set_visible(False)        

        n_polar_lines = len(pol_bounds)

        for i_pol in range(n_polar_lines):
            i_pol_val = pol_bounds[i_pol]
            outer_x = np.cos(i_pol_val)*ecc_bounds[-1]
            outer_y = np.sin(i_pol_val)*ecc_bounds[-1]
            outer_x_txt = outer_x*1.1
            outer_y_txt = outer_y*1.1        
            outer_txt = f"{180*i_pol_val/np.pi:.0f}\N{DEGREE SIGN}"
            # Don't show 360, as this goes over the top of 0 degrees and is ugly...
            if not '360' in outer_txt:
                axs.plot((0, outer_x), (0, outer_y), color="k", alpha=0.3)
                if incl_ticks:
                    axs.text(outer_x_txt, outer_y_txt, outer_txt, ha='center', va='center')

        for i_ecc, i_ecc_val in enumerate(ecc_bounds):
            grid_line = patches.Circle((0, 0), i_ecc_val, color="k", alpha=0.3, fill=0)    
            axs.add_patch(grid_line)                    
        ratio = 1.0
        x_left, x_right = axs.get_xlim()
        y_low, y_high = axs.get_ylim()
        axs.set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)    

    def _add_patches(self, axs, **kwargs):
        patch_col = kwargs.get("patch_col", 'k')
        aperture_line = patches.Circle((0, 0), self.aperture_rad, color=patch_col, linewidth=8, alpha=0.5, fill=0)    
        axs.add_patch(aperture_line)        
    
    def _plot_bin_line(self,X,Y,bin_using,axs, **kwargs):
        # GET PARAMETERS....
        line_col = kwargs.get("line_col", "k")    
        line_label = kwargs.get("line_label", "_")
        lw= kwargs.get("lw", 5)
        n_bins = kwargs.get("n_bins", 10)    
        xerr = kwargs.get("xerr", False)
        # Do the binning
        X_mean = binned_statistic(bin_using, X, bins=n_bins, statistic='mean')[0]
        X_std = binned_statistic(bin_using, X, bins=n_bins, statistic='std')[0]
        count = binned_statistic(bin_using, X, bins=n_bins, statistic='count')[0]
        Y_mean = binned_statistic(bin_using, Y, bins=n_bins, statistic='mean')[0]                
        Y_std = binned_statistic(bin_using, Y, bins=n_bins, statistic='std')[0]  #/ np.sqrt(bin_data['bin_X']['count'])              
        if xerr:
            axs.errorbar(
                X_mean,
                Y_mean,
                yerr=Y_std,
                xerr=X_std,
                color=line_col,
                label=line_label, 
                lw=lw,
                )
        else:
            axs.errorbar(
                X_mean,
                Y_mean,
                yerr=Y_std,
                xerr=X_std,
                color=line_col,
                label=line_label,
                lw=lw,
                )        
        axs.legend()

    def _add_axs_basics(self, axs, **kwargs):        
        xlabel = kwargs.get("xlabel", "")
        ylabel = kwargs.get("ylabel", "")
        title = kwargs.get("title", "")
        x_lim = kwargs.get("x_lim", [])
        y_lim = kwargs.get("y_lim", [])
        axs.set_xlabel(xlabel)
        axs.set_ylabel(ylabel)
        axs.set_title(title)
        if x_lim!=[]:
            axs.set_xlim(x_lim)
        if y_lim!=[]:
            axs.set_ylim(y_lim)

    # ************************************
    # RETURN DOT FUNCTIONS
    '''
    Set of functions to return stuff for dot plotting
    > col, alpha, size
    Looks at kwargs, and at ow_* arguments (overwrite)
    Kwargs are set at when the function is called by the user
    > but some plotting functions may want to overwrite this
    > perhaps because they need to specify different dot properties for different eyes...

    dot_col
    dot_cmap
    dot_size
    dot_alpha

    
    '''
    # ************************************
    def _return_dot_property(self, dot_lbl, **kwargs):                
        ow_dot_model = kwargs.get('ow_dot_model', False)
        ow_dot_eye = kwargs.get('ow_dot_eye', False)
        ow_dot_param = kwargs.get('ow_dot_param', False)

        dmodel,deye,dparam = dot_lbl.split('-')
        if ow_dot_model: dmodel=ow_dot_model
        if ow_dot_eye: deye=ow_dot_eye
        if ow_dot_param: dparam=ow_dot_param

        dot_prop = self.return_th_param(model=dmodel,eye=deye,param=dparam).to_numpy()
                

        return dot_prop, dot_lbl
        
    def _return_dot_col(self, **kwargs):        
        # Is there a string?
        # -> check for function specific overwrite...
        dot_col = kwargs.get('dot_col', 'k')
        ow_dot_col = kwargs.get('ow_dot_col', False)
        if ow_dot_col: dot_col = ow_dot_col
        dot_cmap = kwargs.get('dot_cmap', 'viridis')
        ow_dot_cmap = kwargs.get('ow_dot_cmap', False)
        if ow_dot_cmap: dot_cmap = ow_dot_cmap
        # Check - is dot col a string used to return parameter values?
        # e.g., 'gauss-LE-rsq'
        if not '-' in dot_col:
            # Do not need a cmap
            dot_cmap = None
            return dot_col, dot_cmap
        #
            
        dot_prop, dot_lbl = self._return_dot_property(dot_lbl=dot_col, **kwargs)
        # dot_prop = rescale_bw(dot_prop, old_min=min_dot_col, old_max=max_dot_col)
        # cNorm = mpl.colors.Normalize(vmin=0, vmax=1)
        # scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=dot_cmap)
        # dot_col = scalarMap.to_rgba(dot_prop)
        # this_cmap = plt.cm.__dict__[dot_cmap]
        # dot_col = this_cmap(dot_prop)
        return dot_prop,dot_cmap


    def _return_dot_alpha(self, **kwargs):
        '''
        Function to give alpha values for plotting:
        e.g:
        dot_alpha=0.1           (float b/w 0-1) Single alpha value for all dots
        dot_alpha=
        - to set a single value for all dots alpha=0.1 (float b/w 0-1)
        - to 
        '''
        ow_dot_alpha = kwargs.get('ow_dot_alpha', False)
        if not ow_dot_alpha:
            # Is there a string?
            dot_alpha = kwargs.get("dot_alpha", 0.5)
            dot_alpha_max = kwargs.get("dot_alpha_max", 1)        
            dot_alpha_min = kwargs.get("dot_alpha_min", 0)
            if not isinstance(dot_alpha, str):
                return dot_alpha
            
            dot_prop, dot_lbl = self._return_dot_property(dot_lbl=dot_alpha, **kwargs)
            if dot_lbl != 'rsq':
                dot_alpha = rescale_bw(dot_prop)
            else: 
                dot_alpha = rescale_bw(dot_prop, old_min=0, old_max=1, new_min=dot_alpha_min, new_max=dot_alpha_max)
        else:
            dot_alpha = ow_dot_alpha
        return dot_alpha
        
    def _return_dot_size(self, **kwargs):
        ow_dot_size = kwargs.get('ow_dot_size', False)
        if not ow_dot_size:        
            dot_size = kwargs.get('dot_size', 100)
            max_dot_size = kwargs.get('max_dot_size', 500)
            min_dot_size = kwargs.get('min_dot_size', 5)
            # dot_X = kwargs.get('dot_X', None)
            # dot_Y = kwargs.get('dot_Y', None)
            if not isinstance(dot_size, str):
                return dot_size
            dot_prop, dot_lbl = self._return_dot_property(dot_lbl=dot_size, **kwargs)
            dot_size = rescale_bw(dot_prop, new_min=min_dot_size, new_max=max_dot_size)        
        else:
            dot_size = ow_dot_size
        return dot_size


    
    def _return_ecc_pol_bin(self, params2bin, ecc4bin, pol4bin, ecc_bounds, pol_bounds, bin_weight=None):
        '''
        params2bin      list of np.ndarrays, to bin
        ecc4bin         eccentricity & polar angle used in binning
        pol4bin
        ecc_bounds      how to split into bins 
        pol_bounds
        bin_weight      Something used to weight the binning (e.g., rsq)
        Return the parameters binned by the specified ecc, and pol bounds 

        '''
        if not isinstance(params2bin, list):
            params2bin = [params2bin]
        
        total_n_bins = (len(ecc_bounds)-1) * (len(pol_bounds)-1)
        params_binned = []
        for i_param in range(len(params2bin)):

            bin_mean = np.zeros((len(ecc_bounds)-1, len(pol_bounds)-1))
            bin_idx = []
            for i_ecc in range(len(ecc_bounds)-1):
                for i_pol in range(len(pol_bounds)-1):
                    ecc_lower = ecc_bounds[i_ecc]
                    ecc_upper = ecc_bounds[i_ecc + 1]

                    pol_lower = pol_bounds[i_pol]
                    pol_upper = pol_bounds[i_pol + 1]            

                    ecc_idx =(ecc4bin >= ecc_lower) & (ecc4bin <=ecc_upper)
                    pol_idx =(pol4bin >= pol_lower) & (pol4bin <=pol_upper)        
                    bin_idx = pol_idx & ecc_idx

                    if bin_weight is not None:
                        bin_mean[i_ecc, i_pol] = (params2bin[i_param][bin_idx] * bin_weight[bin_idx]).sum() / bin_weight[bin_idx].sum()
                    else:
                        bin_mean[i_ecc, i_pol] = np.mean(params2bin[i_param][bin_idx])

            bin_mean = np.reshape(bin_mean, total_n_bins)
            # REMOVE ANY NANS
            bin_mean = bin_mean[~np.isnan(bin_mean)]
            params_binned.append(bin_mean)

        return params_binned