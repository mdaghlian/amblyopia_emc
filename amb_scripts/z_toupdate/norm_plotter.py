# import figure_finder as ff
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from .load_saved_info import *
from .utils import *
from .plot_functions import *
from prfpy.rf import gauss2D_iso_cart


def cartesian_to_polar(a, b, c, d):
    r = np.sqrt(a**2 + b**2 + c**2 + d**2)
    theta_1 = np.arccos( (a/(np.sqrt(a**2 + b**2 + c**2 + d**2))))
    theta_2 = np.arccos( (b/(np.sqrt(b**2 + c**2 + d**2))))
    theta_3 = np.arccos( (c/(np.sqrt(c**2 + d**2))))


    return r, theta_1, theta_2, theta_3

def polar_to_cartesian(r, theta_1, theta_2, theta_3):
    a = r * np.cos(theta_1)
    b = r * np.sin(theta_1) * np.cos(theta_2)
    c = r * np.sin(theta_1) * np.sin(theta_2) * np.cos(theta_3)
    d = r * np.sin(theta_1) * np.sin(theta_2) * np.sin(theta_3)
    if ((a<0).sum()>0) | ((b<0).sum()>0) | ((c<0).sum()>0) | ((d<0).sum()>0):
        print('error negative values')
    return a, b, c, d


class NormModelInfo(object):
    def __init__(self, params, vx_mask, prfpy_stim, real_tc=[], pred_tc=[], normalize_RFs=False):        
        self.params_np = params        
        self.vx_mask = vx_mask
        self.n_vox = self.vx_mask.shape[0]
        self.prfpy_stim = prfpy_stim
        self.n_pix = self.prfpy_stim.design_matrix.shape[0]
        self.normalize_RFs = normalize_RFs
        self.real_tc = real_tc
        self.pred_tc = pred_tc        
        # add the aperture
        self.aperture = self.prfpy_stim.ecc_coordinates>(self.prfpy_stim.screen_size_degrees/2)
        # parameters associated with each idx 
        self.p_labels = print_p()['norm_inv']
        self.params_dict = {}
        for this_label in self.p_labels.keys():
            this_id = self.p_labels[this_label]
            self.params_dict[this_label] = self.params_np[:,this_id]
        self.plot_cols = get_plot_cols()
        # Extra info:
        # self.get_rfs()
        self.get_norm_abcd()
        # self.get_fwhmax_fwatmin()
        # self.get_suppression_index()
        # self.n_params_dict['supp_idx'] = self.params_dict['supp_idx']


    def get_rfs(self):
        
        self.prf = np.zeros((self.n_vox, self.n_pix, self.n_pix))                
        self.prf[self.vx_mask,:] = self.params_dict['a_val'][self.vx_mask, np.newaxis,np.newaxis]*np.rot90(
            gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.params_dict['x'][self.vx_mask], self.params_dict['y'][self.vx_mask]),
                sigma=self.params_dict['a_sigma'][self.vx_mask],
                normalize_RFs=self.normalize_RFs).T,axes=(1,2))

        self.srf = np.zeros((self.n_vox, self.n_pix, self.n_pix))                
        self.srf[self.vx_mask,:] = self.params_dict['c_val'][self.vx_mask, np.newaxis,np.newaxis]*np.rot90(
            gauss2D_iso_cart(
                x=self.prfpy_stim.x_coordinates[...,np.newaxis],
                y=self.prfpy_stim.y_coordinates[...,np.newaxis],
                mu=(self.params_dict['x'][self.vx_mask], self.params_dict['y'][self.vx_mask]),
                sigma=self.params_dict['n_sigma'][self.vx_mask],
                normalize_RFs=self.normalize_RFs).T,axes=(1,2))
    
        self.prf[:,self.aperture] = 0
        self.srf[:,self.aperture] = 0

    def get_suppression_index(self):
        self.params_dict['supp_idx'] = self.srf.sum(axis=(1,2)) / self.prf.sum(axis=(1,2))   

    def get_norm_abcd(self):
        self.vector_sqrt = np.sqrt(
            self.params_dict['a_val']**2 + \
                self.params_dict['b_val']**2 + \
                    self.params_dict['c_val']**2 + \
                        self.params_dict['d_val']**2
            )
        new_ps = self.params_np.copy()

        mask = self.vector_sqrt==0
        for i in ['a_val', 'b_val', 'c_val', 'd_val']:
            new_ps[:,self.p_labels[i]] = new_ps[:,self.p_labels[i]]/ self.vector_sqrt
                
        new_ps[mask,:] = 0
        self.n_params_np = new_ps.copy()

        self.n_params_dict = {}
        for this_label in self.p_labels.keys():
            this_id = self.p_labels[this_label]
            self.n_params_dict[this_label] = self.n_params_np[:,this_id]        

        # ADD the polar stuff
        r, theta_1, theta_2, theta_3 = cartesian_to_polar(
            self.params_dict['a_val'], 
            self.params_dict['b_val'], 
            self.params_dict['c_val'], 
            self.params_dict['d_val'], 
            )
        self.n_params_dict['r'] = r
        self.n_params_dict['theta_1'] = theta_1
        self.n_params_dict['theta_2'] = theta_2
        self.n_params_dict['theta_3'] = theta_3

        self.params_dict['r'] = r
        self.params_dict['theta_1'] = theta_1
        self.params_dict['theta_2'] = theta_2
        self.params_dict['theta_3'] = theta_3

        # Also the size ratio:
        self.params_dict['size_ratio'] = self.params_dict['n_sigma'] / self.params_dict['a_sigma']
        self.n_params_dict['size_ratio'] = self.params_dict['n_sigma'] / self.params_dict['a_sigma']

    def get_fwhmax_fwatmin(self):
        x=np.linspace(-50,50,500).astype('float32')

        self.prf1d = self.params_dict['a_val'] * np.exp(-0.5*x[...,np.newaxis]**2 / self.params_dict['a_sigma']**2)
        self.vol_prf =  2*np.pi*self.params_dict['a_sigma']**2

        self.srf1d = self.params_dict['c_val'] * np.exp(-0.5*x[...,np.newaxis]**2 / self.params_dict['n_sigma']**2)
        self.vol_srf = 2*np.pi*self.params_dict['n_sigma']**2

        if self.normalize_RFs==True:
            self.profile = (self.prf1d / self.vol_prf + self.params_dict['b_val']) /\
                    (self.srf1d / self.vol_srf + self.params_dict['d_val']) - self.params_dict['b_val']/self.params_dict['d_val']
        else:
            self.profile = (self.prf1d + self.params_dict['b_val'])/(self.srf1d + self.params_dict['d_val']) - self.params_dict['b_val']/self.params_dict['d_val']


        self.params_dict['half_max']    = np.max(self.profile, axis=0)/2
        self.params_dict['fwhmax']      = np.abs(2*x[np.argmin(np.abs(self.profile-self.params_dict['half_max']), axis=0)])        
        self.params_dict['min_profile'] = np.min(self.profile, axis=0)
        self.params_dict['fwatmin']     = np.abs(2*x[np.argmin(np.abs(self.profile-self.params_dict['min_profile']), axis=0)])

    def plot_abcd(self, use_vector_norm=False, abcd_list=[]):
        if use_vector_norm:
            ps_dict = self.n_params_dict
        else:
            ps_dict = self.params_dict
        if abcd_list==[]:
            abcd_list = ['a_val', 'b_val', 'c_val', 'd_val', 'supp_idx', 'rsq']
        fig = plt.figure()
        rows = len(abcd_list)
        cols = len(abcd_list)
        fig.set_size_inches(22,22)
        plot_i = 1
        for i1,y_param in enumerate(abcd_list):
            for i2,x_param in enumerate(abcd_list):                
                ax = fig.add_subplot(rows,cols,plot_i)
                if i1>i2:
                    ax.axis('off')
                else:
                    ax.set_xlabel(x_param)
                    if i1==i2:
                        ax.hist(ps_dict[x_param][self.vx_mask])
                    else:
                        ax.set_ylabel(y_param)
                        ax.scatter(
                            ps_dict[x_param][self.vx_mask],
                            ps_dict[y_param][self.vx_mask],
                        )        
                        ax.set_title(
                            f'corr={np.corrcoef(ps_dict[x_param][self.vx_mask],ps_dict[y_param][self.vx_mask])[0,1]:.3f}')
                plot_i += 1

        fig.set_tight_layout('tight')
    
    def plot_vx_profile(self, i_vx):
        if not isinstance(i_vx, list):
            i_vx = [i_vx]

        n_vox = len(i_vx)
        cm_subsection = np.linspace(0, 1, n_vox) 
        LP_col = [ mpl.cm.jet(x) for x in cm_subsection ]
        LP_label = []
        LP_data = []
        for this_i_vx in i_vx:
            LP_data.append(self.profile[:,this_i_vx].T)
            # Make the label...            
            prf_str = f'vx id = {this_i_vx:>3.0f}\n'
            param_count = 0
            for param_key in ['x', 'y', 'a_sigma', 'a_val', 'c_val', 'n_sigma', 'b_val', 'd_val']:
                prf_str += f'{param_key:>7}= {self.params_dict[param_key][this_i_vx]:> 7.2f}; '
                if param_count > 3:
                    prf_str += '\n'
                    param_count = 0
                param_count += 1
            LP_label.append(prf_str)
        
        # ************* PLOT *************        
        fig = plt.figure(constrained_layout=True, figsize=(15,5))
        ax = fig.add_subplot()
        # ************* TIME COURSE PLOT *************
        
        x_label = "Profile"
        lsplt.LazyPlot(
            LP_data,
            color=LP_col, 
            labels=LP_label, 
            add_hline='default',
            x_label=x_label,
            axs=ax,
            # title=set_title,
            # xkcd=True,
            # font_size=font_size,
            # line_width=LP_lw,
            # markers=LP_marker,
            # **kwargs,
            )
        ax.legend(bbox_to_anchor=(1.5, 1.05), prop={'size':15,'family': 'monospace'})


    def plot_rfs(self, i_vx):
        if not isinstance(i_vx, list):
            i_vx = [i_vx]
        
        n_vox = len(i_vx)
        fig = plt.figure(constrained_layout=True, figsize=(15,5*n_vox))
        ncol = 3
        nrow = n_vox
        plot_i = 1
        for this_i_vx in i_vx:
            this_vmin = 0
            this_vmax = np.array([
                self.prf[this_i_vx,:,:].max(),
                self.srf[this_i_vx,:,:].max()
                ]).max()
            # [1] ****** PRF ************  
            ax = fig.add_subplot(nrow,ncol, plot_i)
            ax.imshow(self.prf[this_i_vx,:,:], vmin=this_vmin, vmax=this_vmax)
            plot_i += 1
            # [2] ****** SRF ************  
            ax = fig.add_subplot(nrow,ncol, plot_i)
            ax.imshow(self.srf[this_i_vx,:,:], vmin=this_vmin, vmax=this_vmax)
            plot_i += 1
            # [3] ****** TXT ************  
            prf_str = f'vx id = {this_i_vx:>3.0f}\n'
            param_count = 0
            for param_key in ['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']:
                param_count += 1
                prf_str += f'{param_key:>7}= {self.params_dict[param_key][this_i_vx]:> 7.2f}; '
                if param_count > 1:
                    prf_str += '\n'
                    param_count = 0
            ax = fig.add_subplot(nrow,ncol, plot_i)
            ax.axis('off')
            ax.text(0,0.5,prf_str, font={'size':15, 'family':'monospace'})

            plot_i += 1            

    def plot_rfsv2(self, i_vx):
        if not isinstance(i_vx, list):
            i_vx = [i_vx]
        
        n_vox = len(i_vx)
        fig = plt.figure(constrained_layout=True, figsize=(15,5*n_vox))
        ncol = 3
        nrow = n_vox
        plot_i = 1
        for this_i_vx in i_vx:
            this_vmin = 0
            this_vmax = np.array([
                self.prf[this_i_vx,:,:].max(),
                self.srf[this_i_vx,:,:].max()
                ]).max()
            # [1] ****** PRF ************  
            ax = fig.add_subplot(nrow,ncol, plot_i)
            rgb_prf = np.zeros((self.prf.shape[-1], self.prf.shape[-1], 3))
            rgb_prf[:,:,0] = self.prf[this_i_vx,:,:]#/this_vmax
            rgb_prf[:,:,1] = self.srf[this_i_vx,:,:]#/this_vmax
            ax.imshow(rgb_prf)
            prf_str = f'vx id = {this_i_vx:>3.0f}\n'
            param_count = 0
            for param_key in ['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']:
                param_count += 1
                prf_str += f'{param_key:>7}= {self.params_dict[param_key][this_i_vx]:> 7.2f}; '
                if param_count > 1:
                    prf_str += '\n'
                    param_count = 0
            ax.set_title(prf_str, loc='left')
            # ax = fig.add_subplot(nrow,ncol, plot_i)
            ax.axis('off')
            # ax.text(0,0.5,prf_str, font={'size':15, 'family':'monospace'})

            plot_i += 1            


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
            elif 'pred' in this_ts_label:
                this_ts_data = self.pred_tc[i_vx,:]
                LP_col.append(self.plot_cols['norm'])
                LP_lw.append(self.lw_dict['pred'])
                LP_marker.append(self.marker_dict['pred'])
                prfs_to_plot_id.append(i)

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
            this_time_pt = show_stim_frame_x * fmri_TR
            current_ylim = ax2.get_ylim()
            ax2.plot((this_time_pt,this_time_pt), current_ylim, 'k')
            
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
                self.add_prf_plot_v2(i_vx=i_vx, ax=this_ax, show_stim_frame_x=show_stim_frame_x)
            
            # PRF param text
            txt_ax = subfigs[2].add_subplot(111)
            prf_str = f'vx id = {i_vx:>3.0f}\n'
            param_count = 0
            for param_key in ['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']:
                param_count += 1
                prf_str += f'{param_key:>7}= {self.params_dict[param_key][i_vx]:> 7.2f}; '
                if param_count > 3:
                    prf_str += '\n'
                    param_count = 0
            txt_ax.text(0,0,prf_str, font={'size':20})
            txt_ax.axis('off')        


    def add_prf_plot_v2(self, i_vx, ax, show_stim_frame_x=[]):
        if show_stim_frame_x !=[]:
            this_dm = self.prfpy_stim.design_matrix[:,:,show_stim_frame_x]
            this_dm_rgba = np.zeros((this_dm.shape[0], this_dm.shape[1], 4))
            this_dm_rgba[:,:,0] = 1-this_dm
            this_dm_rgba[:,:,1] = 1-this_dm
            this_dm_rgba[:,:,2] = 1-this_dm
            this_dm_rgba[:,:,3] = .5
            ax.imshow(this_dm_rgba, extent=[-aperture_rad, aperture_rad, -aperture_rad, aperture_rad])        

        this_vmin = 0
        this_vmax = np.array([
            self.prf[i_vx,:,:].max(),
            self.srf[i_vx,:,:].max()]).max()
        # [1] ****** PRF ************  
        rgb_prf = np.zeros((self.prf.shape[-1], self.prf.shape[-1], 3))
        rgb_prf[:,:,0] = self.prf[i_vx,:,:]#/this_vmax
        rgb_prf[:,:,1] = self.srf[i_vx,:,:]#/this_vmax
        ax.imshow(rgb_prf)
        ax.axis('off')
    

    def add_prf_plot(self, task, model, i_vx, ax, show_stim_frame_x=[], this_prfpy_stim=[]):
        
        model_idx = self.p_labels[f'{model}_inv']

        this_params = self.model_params[task][model][i_vx,:]
        # ************* PRF (+stimulus) PLOT *************
        # Set vf extent
        aperture_rad = self.prfpy_stim[task].screen_size_degrees/2    
        ax.set_xlim(-aperture_rad, aperture_rad)
        ax.set_ylim(-aperture_rad, aperture_rad)

        if show_stim_frame_x !=[]:
            this_dm = self.prfpy_stim[task].design_matrix[:,:,show_stim_frame_x]
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

    def get_prf_str(self, i_vx, param_keys=['x', 'y', 'a_sigma', 'n_sigma', 'a_val', 'c_val', 'b_val', 'd_val', 'rsq']):
        prf_str = f'vx id = {i_vx:>3.0f}\n'
        param_count = 0
        for param_key in param_keys:
            param_count += 1
            prf_str += f'{param_key:>7}= {self.params_dict[param_key][i_vx]:> 7.2f}; '
            if param_count > 1:
                prf_str += '\n'
                param_count = 0
        return prf_str    