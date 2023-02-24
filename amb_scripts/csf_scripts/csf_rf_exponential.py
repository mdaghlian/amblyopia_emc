import numpy as np

def csf_rf_exp(SFs, CONs, width_r, sf0, maxC, width_l):
    '''
Translated from matlab: /data1/projects/dumoulinlab/dlDevelopment/rmDevel/csf/rfExponential

Create a 2D exponential surface for CSF fitting
   SFs        : Sample spatial frequencies          e.g., np.linspace(0,0.1,20)
   CONs       : Sample contrasts                    e.g., np.linspace(0,100)
   width_r    : width of CSF function, curvature of the parabolic
                function (larger values mean narrower function)
                width is the right side of the curve (width_right)                
   sf0        : spatial frequency with peak sensitivity  
   maxC       : maximale contrast at sf0
   width_r    : width of the left side of the CSF curve,
                
                For matlab function:...
                default:0.5308.*widht_right
                    note: other option for width_left is fixed default
                    parameter (0.4480)
    '''
    # Create grid 
    SFs_grid, CONs_grid = np.meshgrid(SFs, CONs)
    