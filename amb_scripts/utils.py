import numpy as np

def ncsf_curve(SFs, width_r, sf0, maxC, width_l, apply_0_th = True):
    log_SFs = np.log10(SFs)
    log_sf0 = np.log10(sf0)
    log_maxC = np.log10(maxC)

    id_SF_left  = log_SFs <  log_sf0
    id_SF_right = log_SFs >= log_sf0    

    L_curve = 10**(log_maxC - ((log_SFs-log_sf0)**2) * (width_l**2))
    R_curve = 10**(log_maxC - ((log_SFs-log_sf0)**2) * (width_r**2))

    csf_curve = np.zeros_like(L_curve)
    csf_curve[id_SF_left] = L_curve[id_SF_left]
    csf_curve[id_SF_right] = R_curve[id_SF_right]

    logCSF = np.log10(csf_curve)
    
    if apply_0_th:
        logCSF[logCSF<0] = 0
    return logCSF



def qcsf_curve(SFs, peakCS, peakSF, bdwth, lowSFtrunc):
    '''
    function logCSF = findQCSF(FREQ,logGain,logCenter,octaveWidth,logTrunc)
    % logCSF = findQCSF(FREQ,logGain,logCenter,octaveWidth,logTrunc)
    %

    linTrunc=10.^logTrunc;

    tauDecay = .5;
    K = log10(tauDecay);
    logWidth = [(10.^octaveWidth).*log10(2)]./2;

    logP = logGain + K.*[(1./logWidth).*(FREQ - logCenter)].^2;

    truncHalf = logGain - linTrunc;

    leftCSF = [(logP < truncHalf) & (FREQ < logCenter)].*truncHalf;
    rightCSF = [(logP >= truncHalf) | (FREQ > logCenter)].*logP;

    logCSF = (leftCSF + rightCSF);
    logCSF(find(logCSF<0))=0;
    end    
    '''
    log10_SFs = np.log10(SFs)
    linTrunc = 10 ** lowSFtrunc

    tauDecay = 0.5
    K = np.log10(tauDecay)
    logWidth = (10 ** bdwth) * np.log10(2) / 2

    logP = peakCS + K * ((1 / logWidth) * (log10_SFs - peakSF)) ** 2

    truncHalf = peakCS - linTrunc

    leftCSF = np.logical_and(logP < truncHalf, log10_SFs < peakSF) * truncHalf
    rightCSF = np.logical_or(logP >= truncHalf, log10_SFs > peakSF) * logP

    logCSF = leftCSF + rightCSF
    logCSF[np.where(logCSF < 0)] = 0

    return logCSF

from prfpy.rf import csenf_exponential
def add_aulcsf_to_obj(prf_obj, prfpy_stim):
    vx_mask = prf_obj.return_vx_mask({'min-rsq':.1})
    this_AULCSF = np.zeros(prf_obj.n_vox)
    this_rfs = csenf_exponential(
        log_SF_grid = prfpy_stim.log_SF_grid, 
        CON_S_grid = prfpy_stim.CON_S_grid, 
        width_r = prf_obj.pd_params['width_r'][vx_mask].to_numpy(), 
        sf0 = prf_obj.pd_params['sf0'][vx_mask].to_numpy(), 
        maxC = prf_obj.pd_params['maxC'][vx_mask].to_numpy(), 
        width_l = prf_obj.pd_params['width_l'][vx_mask].to_numpy(),)
    this_AULCSF[vx_mask] = this_rfs.sum(axis=(1,2)) / (this_rfs.shape[1] * this_rfs.shape[2])
    return this_AULCSF
import matplotlib.pyplot as plt
def add_aulcsf_to_objV2(prf_obj):
    vx_mask = prf_obj.return_vx_mask({'min-rsq':.1})
    SFs = np.linspace(0.5,20,100)
    
    logCSF = ncsf_curve_multi(
        SFs=SFs, 
        width_r = prf_obj.pd_params['width_r'].to_numpy(), 
        sf0 = prf_obj.pd_params['sf0'].to_numpy(), 
        maxC = prf_obj.pd_params['maxC'].to_numpy(), 
        width_l = prf_obj.pd_params['width_l'].to_numpy(), 
        apply_0_th = True)
    
    # this_AULCSF = np.trapz(10**logCSF, x=SFs, axis=0) # old version    
    this_AULCSF = np.trapz(logCSF, x=np.log10(SFs), axis=0) # Should be this? I think it should be log log...

    # Find out max possible area
    max_possible_logCSF = np.ones_like(SFs) * np.log10(200)
    max_possible_AULCSF = np.trapz(max_possible_logCSF, x=np.log10(SFs), axis=0) 
    
    # Normalize?
    this_AULCSF = this_AULCSF / max_possible_AULCSF

    # Mask 
    this_AULCSF[~vx_mask] = 0

    return this_AULCSF




def ncsf_curve_multi(SFs, width_r, sf0, maxC, width_l, apply_0_th = True):
    log_SFs = np.log10(SFs)
    log_sf0 = np.log10(sf0)
    log_maxC = np.log10(maxC)
    id_SF_left  = log_SFs[..., np.newaxis] <  log_sf0
    id_SF_right = log_SFs[..., np.newaxis] >= log_sf0    

    L_curve = 10**(log_maxC - ((log_SFs[..., np.newaxis]-log_sf0)**2) * (width_l**2))
    R_curve = 10**(log_maxC - ((log_SFs[..., np.newaxis]-log_sf0)**2) * (width_r**2))

    csf_curve = np.zeros_like(L_curve)
    csf_curve[id_SF_left] = L_curve[id_SF_left]
    csf_curve[id_SF_right] = R_curve[id_SF_right]

    logCSF = np.log10(csf_curve)
    if apply_0_th:
        logCSF[logCSF<0] = 0
    return logCSF    