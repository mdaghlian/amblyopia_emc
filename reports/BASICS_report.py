#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V


import os
opj = os.path.join
from datetime import datetime
import numpy as np
import sys
import copy

# MATPLOTLIB STUFF
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import linescanning.plotting as lsplt

from amb_scripts.load_saved_info import *
from amb_scripts.plot_functions import *
from amb_scripts.utils import *
import pandas as pd
#
import figure_finder as ff

import ast
import getopt
import warnings
import json
warnings.filterwarnings('ignore')


report_folder = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/code/amb_code/reports/html_files'

def main(argv):

    """
---------------------------------------------------------------------------------------------------
sub_report_generator

Usage:
    sub_report_generator [arguments] [options]
    Creates a report for a specific subject:

Arguments:
    -s|--sub            subject        
    --roi               roi, restricts voxels to be included    
    --roi_fit           roi subset which are fit            
    --ecc_th            ecc_th, puts a threshold on voxels to include            
    --rsq_th            rsq_th, puts a threshold on voxels to include        
    --dont_open         dont open the html at the end...
Example:

---------------------------------------------------------------------------------------------------
"""    
    sub             = None
    roi             = 'all'
    roi_fit         = 'all'
    model           = 'norm'
    ecc_th          = 5
    rsq_th          = 0.1
    dm_fit          = 'standard'
    open_html       = True # open html after running
    try:
        opts = getopt.getopt(argv,"qf:s:f:d:",["help", "sub=", "roi=", "roi_fit=", "ecc_th=", "rsq_th=", "model=", "dm_fit=", "dont_open"])[0]
    except getopt.GetoptError:
        print(main.__doc__)
        sys.exit(2)
    
    for opt, arg in opts:

        if opt in ('-h', '--help'):
            print(main.__doc__)
            sys.exit()
        elif opt in ("-s", "--sub"):
            sub = hyphen_parse('sub', arg)
        elif opt in ("--roi"):
            roi = arg
        elif opt in ("--roi_fit"):
            roi_fit = arg            
        elif opt in ("--ecc_th"):
            ecc_th = float(arg)
        elif opt in ("--rsq_th"):
            rsq_th = float(arg)
        elif opt in ("--model"):
            model = arg        
        elif opt in ("--dont_open"):
            open_html = False


    ecc_str = str(ecc_th).replace('.', 'pt')
    rsq_str = str(rsq_th).replace('.', 'pt')
    # ************************************************************************************************************************
    # **** BEGIN REPORT MAKER ************************************************************************************************
    # ************************************************************************************************************************
    ff_Report = ff.ReportMaker(f'AMB-BASIC_{sub}_roi-{roi}_e-{ecc_str}_r-{rsq_str}_rf-{roi_fit}', report_folder, open_html=open_html)
    with ff_Report:
        date_now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        ff_Report.add_title(f'Basics report: {sub}, generated at {date_now}')
        ff_Report.add_title(
            f'sub={sub}, model=gauss+{model}, roi={roi}, ecc_th={ecc_th}, rsq_th={rsq_th}, dm_fit={dm_fit}, roi_fit={roi_fit}')
        # LOAD USEFUL INFORMATION
        task_list = ['LE', 'RE']#'task-2R',
        model_list = ['gauss', model]
        param_labels = print_p()
        plot_cols = get_plot_cols()

        # Get parameters & masks
        model_params = amb_load_prf_params(
            sub=sub, 
            task_list=task_list, 
            model_list=model_list, 
            roi_fit=roi_fit,
            )
        vx_mask = get_vx_mask(
            sub=sub,
            task_list=task_list,
            model_list=model_list, 
            roi=roi, rsq_th=rsq_th, ecc_th=ecc_th,
            roi_fit=roi_fit,
            dm_fit=dm_fit,
            )    
        real_tc = get_real_tc(sub=sub, task_list=task_list)
        pred_tc = get_pred_tc(
            sub=sub, 
            task_list=task_list, 
            model_list=model_list, 
            roi_fit=roi_fit,
            )
        xpred_tc = get_cross_pred_tc(
            sub=sub, 
            task_list=task_list, 
            model_list=model_list, 
            roi_fit=roi_fit,
            )

        # ************************************************************
        # ************************************************************
        # RSQUARED VALUES
        # -Create 3 sets of violin plots
        # [1] Rsquared of the fits of the models [ALL_fits] 
        # [2] Rsq of tc, generated using AS0 fits, and simulated with the scotoma data [AS0_cross_sim] 
        # [3] Rsq of tc, generated using AS0 fits, and simulated with the *AS0 STIMULI*
        
        # Create dictionary to hold the data - makes for easy violin plot with SNS 
        data_dict = {}
        data_dict['rsq_values'] = []
        data_dict['model_labels'] = []
        data_dict['task_labels'] = []

        ALL_fits_dict = copy.deepcopy(data_dict)
        AS0_cross_sim_dict = copy.deepcopy(data_dict)
        AS0_cross_no_sim_dict = copy.deepcopy(data_dict)

        for iT,task in enumerate(task_list):    
            for iM,model in enumerate(model_list):
                # this_mask = model_params[task][model][:,-1]>0.1
                this_mask = vx_mask[task][model]
                print(iM)
                # Rsq for *ALL fits*
                # ALL_fits_rsq = model_params[task][model][this_mask,-1]
                ALL_fits_rsq = get_rsq(
                    real_tc[task][this_mask,:], 
                    pred_tc[task][model][this_mask,:]
                    )
                ALL_fits_dict['rsq_values'].append(ALL_fits_rsq)
                ALL_fits_dict['model_labels'].append([model]*len(ALL_fits_rsq))
                ALL_fits_dict['task_labels'].append([task]*len(ALL_fits_rsq))

                if not task=="task-AS0":
                    # Rsq across task, using AS0
                    # -> therefore use AS0 mask. & Remove predicted tcs which are flat 
                    AS0_mask = vx_mask['task-AS0'][model]

                    # first check all tc's with zero std (these will give weird rsquared...)
                    real_tc_std_not_0_idx = np.std(real_tc[task], axis=1)!=0
                    xpred_tc_std_not_0_idx = np.std(xpred_tc[task][model], axis=1)!=0
                    pred_tc_std_not_0_idx = np.std(pred_tc['task-AS0'][model], axis=1)!=0
                    
                    #[1] Using AS0 parameters, but simulating with scotoma stimulus
                    AS0_cross_sim_mask = AS0_mask & real_tc_std_not_0_idx & xpred_tc_std_not_0_idx
                    # ff_Report.add_text(f'X sim {task} - {model} pct {AS0_cross_sim_mask.sum()} out of {AS0_mask.sum()}')
                    AS0_cross_sim_rsq = get_rsq(
                        real_tc[task][AS0_cross_sim_mask,:], 
                        xpred_tc[task][model][AS0_cross_sim_mask,:]
                        )            

                    AS0_cross_sim_dict['rsq_values'].append(AS0_cross_sim_rsq)
                    AS0_cross_sim_dict['model_labels'].append([model]*len(AS0_cross_sim_rsq))
                    AS0_cross_sim_dict['task_labels'].append([task]*len(AS0_cross_sim_rsq))

                    #[2] Using AS0 parameters, AND simulating with **AS0** stimulus
                    AS0_cross_no_sim_mask = AS0_mask & real_tc_std_not_0_idx & pred_tc_std_not_0_idx
                    # ff_Report.add_text(f'X NO sim {task} - {model} pct {AS0_cross_no_sim_mask.sum()} out of {AS0_mask.sum()}')
                    AS0_cross_no_sim_rsq = get_rsq(
                        real_tc[task][AS0_cross_no_sim_mask,:], 
                        pred_tc['task-AS0'][model][AS0_cross_no_sim_mask,:]
                        )                
                    AS0_cross_no_sim_dict['rsq_values'].append(AS0_cross_no_sim_rsq)
                    AS0_cross_no_sim_dict['model_labels'].append([model]*len(AS0_cross_no_sim_rsq))
                    AS0_cross_no_sim_dict['task_labels'].append([task]*len(AS0_cross_no_sim_rsq))

        for key in data_dict.keys():
            ALL_fits_dict[key]          = np.concatenate(ALL_fits_dict[key])
            AS0_cross_sim_dict[key]     = np.concatenate(AS0_cross_sim_dict[key])
            AS0_cross_no_sim_dict[key]  = np.concatenate(AS0_cross_no_sim_dict[key])        

        ALL_fits_PD = pd.DataFrame(ALL_fits_dict)
        AS0_cross_sim_PD = pd.DataFrame(AS0_cross_sim_dict)
        AS0_cross_no_sim_PD = pd.DataFrame(AS0_cross_no_sim_dict)

        pd_dict_for_loop = {
            'ALL_fits_PD': ALL_fits_PD,
            'AS0_cross_sim_PD': AS0_cross_sim_PD,
            'AS0_cross_no_sim_PD': AS0_cross_no_sim_PD,        
        }
        for i_rsq_type in pd_dict_for_loop.keys():
            # Add the title for the plots
            if i_rsq_type=='ALL_fits_PD':
                ff_Report.add_title('Rsq, fit on each task separately, (not cross validated)', level=2)
            
            elif i_rsq_type=='AS0_cross_sim_PD':
                ff_Report.add_title('Rsq, fit on AS0: tc simulated using AS1/AS2 stimuli (cross-task validated)', level=2)

            elif i_rsq_type=='AS0_cross_no_sim_PD':
                ff_Report.add_title('Rsq, fit on AS0: tc simulated using *AS0* stimuli (why do this? - think about filling in...)', level=2)
            
            ff_Report.add_text(f'{i_rsq_type}')
            ff_Report.add_text(f'Excluding values less than -2 for ease of viewing')
            this_pd = pd_dict_for_loop[i_rsq_type][pd_dict_for_loop[i_rsq_type].rsq_values >= -2]
            sns.violinplot(
                x="task_labels", y="rsq_values", hue="model_labels",
                data=this_pd, palette=[plot_cols["gauss"], plot_cols["norm"]], 
                width=1, linewidth=0, split=True, inner=None,)
            plt.gca().grid(True, which='both', axis="y")
            plt.gca().set_ylim(-2,1)
            plt.gca().set_title(f'Rsq values for {sub} ({i_rsq_type})')    
            fig = plt.gcf()
            ff_Report.add_img(fig)
            plt.close()        

        # ff_Report.add_title('X-task RSQ - fit on AS0, simulated with [AS1, AS2]', level=2)
        # # xtask_rsq_dict = {}
        # for iT,task in enumerate(['task-AS1', 'task-AS2']):
        #     # xtask_rsq_dict[task] = {}   
        #     for iM,model in enumerate(model_list):
        #         # this_mask = model_params[task][model][:,-1]>0.1
        #         this_mask = vx_mask[task][model]
        #         xtask_rsq = get_rsq(real_tc[task][this_mask,:], xpred_tc[task][model][this_mask,:])
        #         no_sim_task_rsq = get_rsq(real_tc[task][this_mask,:], pred_tc[task][model][this_mask,:])
        #         # xtask_rsq_dict[task][model] = xtask_rsq
        #         ff_Report.add_text(f'{task}, {model}: simulate: mean rsq {xtask_rsq.mean()}')
        #         ff_Report.add_text(f'{task}, {model}: AS0  {no_sim_task_rsq.mean()}\n')

        # ************************************************************
        # ************************************************************
        # ************************************************************

        # [2] Histograms of parameter values
        for model in model_list:
            fig = plt.figure(1)
            if model=='gauss':
                rows = 2
                cols = 3
                fig.set_size_inches(22,11)
            if model=='norm':
                rows = 4
                cols = 3
                fig.set_size_inches(22,22)
            if model=='dog':
                rows = 3
                cols = 3
                fig.set_size_inches(22,22)
            if model=='css':
                rows = 3
                cols = 3
                fig.set_size_inches(22,22)

            full_title = f'MODEL={model}\n FIT ON DATA\n Including vx with: roi={vx_mask["roi"]}, rsq_th={vx_mask["rsq_th"]}, ecc_th={vx_mask["ecc_th"]}, size_th={vx_mask["size_th"]}'
            fig.suptitle(full_title)

            for i in param_labels[model].keys():
                ax = fig.add_subplot(rows,cols,i+1)
                ax.set_title(param_labels[model][i])
                for task in task_list:
                    this_mask = vx_mask[task][model]
                    this_param = model_params[task][model][this_mask, i]
                    plt.hist(this_param,color=plot_cols[task], label=task)
                plt.legend()
            ff_Report.add_title(f'{model} - parameter histograms', level=2)
            ff_Report.add_img(fig)
            plt.close()


    # ************************************************************************************************************************
    # **** END REPORT MAKER **************************************************************************************************
    # ************************************************************************************************************************    

if __name__ == "__main__":
    main(sys.argv[1:])
