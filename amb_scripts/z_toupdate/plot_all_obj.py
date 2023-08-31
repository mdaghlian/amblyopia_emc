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
from ..plot_shifts import *


class AllPrfInfo(object):
    '''
    Hold params & ts for a single condition

    '''
    def __init__(self, sub, model_list, task_list, roi_fit='all', dm_fit='standard', **kwargs):
        '''
        Function to load and hold everything...
        '''
        self.sub = sub
        self.n_vox = get_number_of_vx(sub)
        # Model information (could be norm, or gauss)
        self.model_list = model_list
        self.task_list = task_list
        self.p_labels = {}
        for model in model_list:
            self.p_labels[model] = prfpy_params_dict()[model]  # parameters of model (used for dictionary later )  e.g. x,y,a_sigma,a_val,....
        # -> store the model parameters 
        self.params_np = get_model_params(
                sub=sub, task_list=task_list, model_list=model_list,
                roi_fit=roi_fit, dm_fit=dm_fit,
                )                
        self.real_tc = get_real_tc(sub=sub, task_list=task_list)
        self.scotoma_info = get_scotoma_info(sub=sub)
        self.aperture = self.scotoma_info['task-AS0']['aperture_rad']
        self.prfpy_stim = get_prfpy_stim(sub=sub, task_list=task_list)
        # Create & save models for running time series
        self.prfpy_model = {}
        for task in task_list:
            self.prfpy_model[task] = {}
            for model in model_list:
                if model=='gauss':
                    self.prfpy_model[task][model] = Iso2DGaussianModel(self.prfpy_stim[task])
                elif model=='norm':
                    self.prfpy_model[task][model] = Norm_Iso2DGaussianModel(self.prfpy_stim[task])


        # Create dictionaries to turn into PD dataframes...
        # PUT ALL PARAMETERS INTO PD 
        self.pd_params = {}
        for task in task_list:
            self.pd_params[task] = {}
            for model in self.model_list:
                # self.pd_params[model][task]
                data_dict = {}
                for i_label in self.p_labels[model].keys(): # Go through list of model parameters
                    data_dict[i_label] = self.params_np[task][model][:,self.p_labels[model][i_label]]
                self.pd_params[task][model] = pd.DataFrame(data_dict)


class PrfTsPlotter2(AllPrfInfo):
    def __init__(self, sub, model_list, task_list, roi_fit='all', dm_fit='standard', **kwargs):
        super().__init__(sub, model_list, task_list, roi_fit=roi_fit, dm_fit=dm_fit, **kwargs)
        # Other useful stuff...
        self.fmri_TR = 1.5
        self.plot_cols = get_plot_cols()
        self.marker_dict = {
            'real' : 'None',
            'pred' : '^',
            'cross': '*',
        }

        self.lw_dict = {
            'real' : 10,
            'pred' : 5,
            'cross': 5,
        }
    
    def ts_plot_specify(self, i_vx, ts_list, show_stim_frame_x=[], **kwargs):
        # Check that everything is loaded...
        # self.check_not_missing_x(ts_list=ts_list)
        fmri_TR = 1.5
        LP_data = []
        LP_lw = []
        LP_marker = []
        LP_col = []
        # PRF ID MARKER
        prfs_to_plot_id = []    # i for the associated ts
        prfs_to_plot_task = []  # task (for parameters)
        prfs_to_plot_task2 = [] # task for stimuli
        prfs_to_plot_model = [] # model (for parameters)
        stims_to_plot = [] # id of stims to plot, if not included in the prf plots
        for i,this_ts_label in enumerate(ts_list):                        
            # ts_label: real-task-model
            if 'real' in this_ts_label:
                data_type,task = this_ts_label.split('-')
            elif 'pred' in this_ts_label:
                data_type,task,model = this_ts_label.split('-')
            elif 'cross' in this_ts_label:
                data_type,task,model,task2 = this_ts_label.split('-') # task2 = task for stimuli
                task2 = hyphen_parse('task', task2)

            task = hyphen_parse('task', task)

            LP_col.append(self.plot_cols[task])                
            LP_lw.append(self.lw_dict[data_type])
            LP_marker.append(self.marker_dict[data_type])
            if data_type=='real':
                this_ts_data = self.real_tc[task][i_vx,:]
                stims_to_plot                
            elif 'pred' in this_ts_label:
                this_ts_data = self.create_pred(i_vx=i_vx, task4param=task, task4stim=task, model=model)
                # PRF ID MARKER
                prfs_to_plot_id.append(i)
                prfs_to_plot_task.append(task)
                prfs_to_plot_task2.append(task)
                prfs_to_plot_model.append(model)
            elif 'cross' in this_ts_label:
                this_ts_data = self.create_pred(i_vx=i_vx, task4param=task, task4stim=task2, model=model)
                # PRF ID MARKER
                prfs_to_plot_id.append(i)
                prfs_to_plot_task.append(task)
                prfs_to_plot_task2.append(task2)
                prfs_to_plot_model.append(model)                

            LP_data.append(this_ts_data)
        
        # ************* PLOT *************        
        fig = plt.figure(constrained_layout=True, figsize=(40,10))
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
            **kwargs,
            )
        if show_stim_frame_x!=[]:
            ylim = ax2.get_ylim()        
            ax2.plot(
                (show_stim_frame_x*fmri_TR, show_stim_frame_x*fmri_TR),
                 ylim, 'k')

        # ************* PRFs  PLOT *************
        # -> create a string of important stuff
        
        if not prfs_to_plot_id==[]:
            prfs_id_str = [f'0/{prfs_to_plot_task[i]}/{prfs_to_plot_model[i]}/{prfs_to_plot_task2[i]}' for i in range(len(prfs_to_plot_id))]
            print(prfs_id_str)
            # sys.exit()
            prfs_id_str = list(set(prfs_id_str)) # remove duplicates
            print(prfs_id_str)
            n_prfs = len(prfs_id_str)
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

            for plot_i, prf_id_str in enumerate(prfs_id_str):
                this_ax = subfigs[0].add_subplot(gspec_prfs[plot_i])                                
                self.add_prf_plot(i_vx=i_vx, ax=this_ax, prf_id_str=prf_id_str, show_stim_frame_x=show_stim_frame_x)
            
            # PRF param text
            txt_ax = subfigs[2].add_subplot(111)
            prf_str = f'vx id = {i_vx:>3.0f}\n\n'
            for plot_i, prf_id_str in enumerate(prfs_id_str):
                this_model = prfs_to_plot_model[plot_i]
                this_task = prfs_to_plot_task[plot_i]
                prf_str += f'{this_model}-{this_task}: '
                if this_model=='norm':
                    param_keys = ['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']
                else:
                    param_keys = ['x', 'y', 'a_sigma', 'a_val', 'rsq']

                param_count = 0
                for param_key in param_keys:
                    param_count += 1
                    prf_str += f'{param_key:>7}= {self.pd_params[this_task][this_model][param_key][i_vx]:> 7.2f}; '
                    if param_count > 3:
                        prf_str += '\n'
                        param_count = 0
                prf_str += '\n'
            txt_ax.text(0,0,prf_str, font={'size':20})
            txt_ax.axis('off')        
        # else:
            # if show_stim_frame_x !=[]:
            #     gspec_prfs = subfigs[0].add_gridspec(1,1)
            #     this_ax = subfigs[0].add_subplot(gspec_prfs[0])                                
            #     this_dm = self.prfpy_stim[task].design_matrix[:,:,show_stim_frame_x]
            #     this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
            #     this_dm_rgba[:,:,0] = 1-this_dm
            #     this_dm_rgba[:,:,1] = 1-this_dm
            #     this_dm_rgba[:,:,2] = 1-this_dm
            #     this_dm_rgba[:,:,3] = .5
            #     this_ax.imshow(this_dm_rgba, extent=[-self.aperture, self.aperture, -self.aperture, self.aperture])
        
        update_fig_dotsize(fig, 15)
        update_fig_fontsize(fig,30)
        # return fig

    def add_prf_plot(self, i_vx, ax, prf_id_str, show_stim_frame_x=[]):
        # Split into:
        # -> ts number; 
        ts_id, task, model, task2 = prf_id_str.split('/')

        model_idx = self.p_labels[model]

        this_params = self.params_np[task][model][i_vx,:]
        # ************* PRF (+stimulus) PLOT *************
        # Set vf extent
        aperture_rad = self.aperture
        ax.set_xlim(-aperture_rad, aperture_rad)
        ax.set_ylim(-aperture_rad, aperture_rad)

        if show_stim_frame_x !=[]:
            this_dm = self.prfpy_stim[task2].design_matrix[:,:,show_stim_frame_x]
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
        if model=='norm':
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


    def create_pred(self, i_vx, task4param, task4stim, model):
        
        this_params = self.params_np[task4param][model][i_vx,0:-1] # Do not include rsq            
        # AT THE MOMENT USING THE PRF    
        this_pred = np.squeeze(self.prfpy_model[task4stim][model].return_prediction(*list(this_params.T)))
        return this_pred

