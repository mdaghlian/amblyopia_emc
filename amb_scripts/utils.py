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
