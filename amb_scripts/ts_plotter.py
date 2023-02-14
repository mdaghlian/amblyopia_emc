# import figure_finder as ff
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from .load_saved_info import *
from .utils import *
from .plot_functions import *
from prfpy.rf import gauss2D_iso_cart
from prfpy.model import Iso2DGaussianModel, Norm_Iso2DGaussianModel
from .plot_shifts import *


class Prf1TaskInfo(object):
    '''
    Hold params & ts for a single condition

    '''
    def __init__(self, params, real_tc, pred_tc,task,model, **kwargs):
        '''
        ...
        '''
        self.sub = kwargs.get('sub', None)        
        # Model information (could be norm, or gauss)
        self.model = model
        self.task = task
        self.p_labels = print_p()[f'{self.model}_inv']  # parameters of model (used for dictionary later )  e.g. x,y,a_sigma,a_val,....
        # -> store the model parameters 
        self.params = params
        self.real_tc = real_tc
        self.pred_tc = pred_tc

        # Create dictionaries to turn into PD dataframes...
        data_dict = {} # No scotoma, scotoma, difference
        # First add all the parameters from the numpy arrays (x,y, etc.)
        for i_label in self.p_labels.keys():
            data_dict[i_label] = self.params[:,self.p_labels[i_label]]
            # Now add other useful things: 
        # -> eccentricity and polar angle 
        data_dict['ecc'], data_dict['pol'] = coord_convert(
            data_dict["x"], data_dict["y"], 'cart2pol')
        # Convert to PD
        self.pd_params = pd.DataFrame(data_dict)

    def return_vx_mask(self, ALL_th={}):
        '''
        return_vx_mask: returns a mask for voxels, specified by the user
        4 optional dictionaries:
        ALL_th      :       Applies to both AS0 and ASX condition
        Each dictionary can contain any key/s which applies to the pd_params
            gauss: "x", "y", "a_sigma", "a_val", "bold_baseline", "rsq"
            norm : "x", "y", "a_sigma", "a_val", "bold_baseline", "c_val", "n_sigma", "b_val", "d_val", "rsq"            
            ALL: "ecc", "pol", "d2s_centre"
        >> .return_vx_mask(AS0_th={'minrsq' : 0.1})
            returns a boolean array, excluding all vx where rsq < 0.1 in AS0 condition
        
        You can also specify how the threshold is applied by attaching min- or max- to the front of the key
        >> .return_vx_mask(ASX_th={max-a_sigma : 5}) # max size in ASX condition is 5

        Finally, can provide a lower and upper bound by using a list
        >> .return_vx_mask(ASX_th={a_sigma : [3,5]})

        '''        

        # Start with EVRYTHING        
        vx_mask = np.ones(self.n_vox, dtype=bool)
        for i_th in ALL_th.keys():
            if isinstance(ALL_th[i_th], np.ndarray):
                vx_mask &= ALL_th[i_th] # assume it is a boolean array (can be used to add roi)
            elif isinstance(ALL_th[i_th], list): # upper and lower bound
                vx_mask &= self.pd_params[i_th].gt(ALL_th[i_th][0]) # Greater than
                vx_mask &= self.pd_params[i_th].lt(ALL_th[i_th][1]) # Less than
            elif 'max' in i_th:
                i_th_lbl = i_th.split('-')[1]
                vx_mask &= self.pd_params[i_th_lbl].lt(ALL_th[i_th]) # Less than
            elif 'min' in i_th:
                i_th_lbl = i_th.split('-')[1]
                vx_mask &= self.pd_params[i_th_lbl].gt(ALL_th[i_th]) # Greater than
            else:
                # Default to min
                sys.exit()
                # vx_mask &= self.pd_params[task_key][i_th].gt(th_dict[task_key][i_th]) # Less than
        return vx_mask
    
    def return_th_param(self, param, vx_mask=None):
        '''
        For a specified task (AS0, ASX, ASd)
        return all the parameters listed, masked by vx_mask        
        '''
        if vx_mask is None:
            vx_mask = np.ones(self.n_vox, dtype=bool)
        if not isinstance(param, list):
            param = [param]        
        param_out = []
        for i_param in param:
            param_out.append(self.pd_params[i_param][vx_mask].to_numpy())

        return param_out
    
 

class PrfTsPlotter(Prf1TaskInfo):
    def __init__(self, params, real_tc, pred_tc,task,model, **kwargs):
        super().__init__(params, real_tc, pred_tc,task,model, **kwargs)
        # Other useful stuff...
        self.scotoma_info = get_scotoma_info(self.sub)[self.task]
        self.prfpy_stim = get_prfpy_stim(self.sub, self.task)[self.task]
        self.prfpy_stim_all = get_prfpy_stim(self.sub, ['task-AS0', 'task-AS1', 'task-AS2'])
        self.plot_cols = get_plot_cols()
        self.n_vox = self.real_tc.shape[0]
        self.n_pix = kwargs.get('n_pix', 50)
        self.normalize_RFs = kwargs.get('normalize_RFs', False)        
        self.aperture = self.scotoma_info['aperture_rad'] 

    def get_prfs(self):
        self.prf = np.zeros((self.n_vox, self.n_pix, self.n_pix))                
        i_incl = self.pd_params['a_val']!=0
        self.prf[i_incl,:] = self.pd_params['a_val'][i_incl, np.newaxis,np.newaxis]*np.rot90(
            gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.pd_params['x'][i_incl], self.pd_params['y'][i_incl]),
                sigma=self.pd_params['a_sigma'][i_incl],
                normalize_RFs=self.normalize_RFs).T,axes=(1,2))
        self.prf[:,self.aperture] = 0
    
        if self.model is 'norm':
            self.srf = np.zeros((self.n_vox, self.n_pix, self.n_pix))                
            self.srf[i_incl,:] = self.params_dict['c_val'][i_incl, np.newaxis,np.newaxis]*np.rot90(
                gauss2D_iso_cart(
                    x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                    y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                    mu=(self.params_dict['x'][i_incl], self.params_dict['y'][i_incl]),
                    sigma=self.params_dict['n_sigma'][i_incl],
                    normalize_RFs=self.normalize_RFs).T,axes=(1,2))
            
            self.srf[:,self.aperture] = 0       
    
    def ts_plot_specify(self, i_vx, ts_list, show_stim_frame_x=[]):
        # Check that everything is loaded...
        # self.check_not_missing_x(ts_list=ts_list)
        fmri_TR = 1.5
        if not hasattr(self, 'marker_dict'):
            self.marker_dict = {
                'real' : 'None',
                'pred' : '^',
            }
        if not hasattr(self, 'lw_dict'):
            self.lw_dict = {
                'real' : 3,
                'pred' : 5,
            }

        LP_data = []
        LP_lw = []
        LP_marker = []
        LP_col = []
        prfs_to_plot_id = []
        for i,this_ts_label in enumerate(ts_list):
                        
            if 'real' in this_ts_label:
                this_ts_data = self.real_tc[i_vx,:]
                LP_col.append(self.plot_cols[f'real'])
                
                LP_lw.append(self.lw_dict['real'])
                LP_marker.append(self.marker_dict['real'])
            elif 'xpred' in this_ts_label:
                if 'AS0' in this_ts_label:
                    this_task = 'task-AS0'
                elif 'AS1' in this_ts_label:
                    this_task = 'task-AS1'
                elif 'AS2' in this_ts_label:
                    this_task = 'task-AS2'

                this_ts_data = np.squeeze(self.create_pred(i_vx, prfpy_stim=self.prfpy_stim_all[this_task]))
                LP_col.append(self.plot_cols[self.model])
                LP_lw.append(self.lw_dict['pred'])
                LP_marker.append(self.marker_dict['pred'])
                prfs_to_plot_id.append(i)

            elif 'pred' in this_ts_label:
                this_ts_data = self.pred_tc[i_vx,:]
                LP_col.append(self.plot_cols[self.model])
                LP_lw.append(self.lw_dict['pred'])
                LP_marker.append(self.marker_dict['pred'])
                prfs_to_plot_id.append(i)

            print(this_ts_data.shape)

            LP_data.append(this_ts_data)
        
        # ************* PLOT *************        
        fig = plt.figure(constrained_layout=True, figsize=(20,5))
        subfigs = fig.subfigures(1, 3, width_ratios=[10,20,10])
        # ************* TIME COURSE PLOT *************
        
        x_label = "time (s)"
        x_axis = np.array(list(np.arange(0,LP_data[0].shape[0])*fmri_TR)) 
        ax2 = subfigs[1].add_subplot()
        lsplt.LazyPlot(
            LP_data,
            xx=x_axis,
            color=LP_col, 
            labels=ts_list, 
            add_hline='default',
            x_label=x_label,
            y_label="amplitude",
            axs=ax2,
            # title=set_title,
            # xkcd=True,
            # font_size=font_size,
            line_width=LP_lw,
            markers=LP_marker,
            # **kwargs,
            )
        if show_stim_frame_x!=[]:
            ylim = ax2.get_ylim()        
            ax2.plot(
                (show_stim_frame_x*fmri_TR, show_stim_frame_x*fmri_TR),
                 ylim, 'k')

        # ************* PRFs  PLOT *************
        if not prfs_to_plot_id==[]:
            n_prfs = len(prfs_to_plot_id)
            if n_prfs==1:
                gspec_prfs = subfigs[0].add_gridspec(1,1)
            elif n_prfs==2:
                gspec_prfs = subfigs[0].add_gridspec(1,2)
            elif n_prfs<=4:
                gspec_prfs = subfigs[0].add_gridspec(2,2)
            elif n_prfs<=6:
                gspec_prfs = subfigs[0].add_gridspec(3,2)
            else: 
                gspec_prfs = subfigs[0].add_gridspec(3,3)

            for plot_i, prf_id in enumerate(prfs_to_plot_id):
                this_ax = subfigs[0].add_subplot(gspec_prfs[plot_i])                                
                self.add_prf_plot(i_vx=i_vx, ax=this_ax, show_stim_frame_x=show_stim_frame_x)
            
            # PRF param text
            txt_ax = subfigs[2].add_subplot(111)
            prf_str = f'vx id = {i_vx:>3.0f}\n'
            param_count = 0
            if self.model is 'norm':
                param_keys = ['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']
            else:
                param_keys = ['x', 'y', 'a_sigma', 'a_val', 'rsq']
            for param_key in param_keys:
                param_count += 1
                prf_str += f'{param_key:>7}= {self.pd_params[param_key][i_vx]:> 7.2f}; '
                if param_count > 3:
                    prf_str += '\n'
                    param_count = 0
            txt_ax.text(0,0,prf_str, font={'size':20})
            txt_ax.axis('off')        
        else:
            if show_stim_frame_x !=[]:
                gspec_prfs = subfigs[0].add_gridspec(1,1)
                this_ax = subfigs[0].add_subplot(gspec_prfs[0])                                
                this_dm = self.prfpy_stim.design_matrix[:,:,show_stim_frame_x]
                this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
                this_dm_rgba[:,:,0] = 1-this_dm
                this_dm_rgba[:,:,1] = 1-this_dm
                this_dm_rgba[:,:,2] = 1-this_dm
                this_dm_rgba[:,:,3] = .5
                this_ax.imshow(this_dm_rgba, extent=[-self.aperture, self.aperture, -self.aperture, self.aperture])


    def add_prf_plot(self, i_vx, ax, show_stim_frame_x=[]):
        
        model_idx = self.p_labels

        this_params = self.params[i_vx,:]
        # ************* PRF (+stimulus) PLOT *************
        # Set vf extent
        aperture_rad = self.prfpy_stim.screen_size_degrees/2    
        ax.set_xlim(-aperture_rad, aperture_rad)
        ax.set_ylim(-aperture_rad, aperture_rad)

        if show_stim_frame_x !=[]:
            this_dm = self.prfpy_stim.design_matrix[:,:,show_stim_frame_x]
            this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
            this_dm_rgba[:,:,0] = 1-this_dm
            this_dm_rgba[:,:,1] = 1-this_dm
            this_dm_rgba[:,:,2] = 1-this_dm
            this_dm_rgba[:,:,3] = .5
            ax.imshow(this_dm_rgba, extent=[-aperture_rad, aperture_rad, -aperture_rad, aperture_rad])

        # Add prfs
        prf_x = this_params[0]
        prf_y = this_params[1]
        plt.scatter(prf_x, prf_y, s=50, color='k')
        # Add normalizing PRF (FWHM)
        if self.model=='norm':
            prf_2_fwhm = 2*np.sqrt(2*np.log(2))*this_params[model_idx['n_sigma']] # 
            if prf_2_fwhm>aperture_rad:
                ax.set_xlabel('*too big')
                norm_prf_label = '*Norm PRF'
            else:
                norm_prf_label = 'Norm PRF'
            prf_2 = patches.Circle(
                (prf_x, prf_y), prf_2_fwhm, edgecolor="r", 
                facecolor=[1,0,0,0.1], 
                linewidth=8, fill=True,
                label=norm_prf_label,)    
            ax.add_patch(prf_2)
        # Add activating PRF (fwhm)
        prf_fwhm = 2*np.sqrt(2*np.log(2))*this_params[model_idx['a_sigma']] # 
        prf_1 = patches.Circle(
            (prf_x, prf_y), prf_fwhm, edgecolor="b", facecolor=[0,0,1,0.1], 
            linewidth=8, fill=True, label='PRF')
        ax.add_patch(prf_1)
        
        # Add 0 lines...
        ax.plot((0,0), ax.get_ylim(), 'k')
        ax.plot(ax.get_xlim(), (0,0), 'k')
        ax.legend()


    def create_pred(self, i_vx, prfpy_stim):
        if self.model=='gauss':
            this_prfpy_model = Iso2DGaussianModel(prfpy_stim)
        elif self.model=='norm':
            this_prfpy_model = Norm_Iso2DGaussianModel(prfpy_stim)        
        
        this_params = self.params[i_vx,0:-1] # Do not include rsq            
        # AT THE MOMENT USING THE PRF    
        this_pred = this_prfpy_model.return_prediction(*list(this_params.T))
        this_prfpy_model = []
        this_params = []
        return this_pred
