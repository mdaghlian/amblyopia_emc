import numpy as np

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
    # logCSF[np.where(logCSF < 0)] = 0

    return logCSF

