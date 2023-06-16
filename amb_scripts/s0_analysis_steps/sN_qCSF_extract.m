% Extract information from qcsf .mat files. Save in cleaner format...
% qCSF: https://doi.org/10.1167/10.3.17
% Reminder: qCSF is an adaptive bayesian way of calculating the CSF in a
% more efficient way. Optimizing over 4 parameters
% (1) Peak gain (sensitivity, gamma-max) ** log
% (2) Peak spatial frequency (f-max)
% (3) Bandwidth (fwhm, beta)
% (4) Truncation at low SF (delta)

clc 

amb_code_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/code/amb_code/amb_scripts/dm_files/';
qcsf_dir = '/data1/projects/dumoulinlab/Lab_members/Marcus/projects/amblyopia_emc/derivatives/qCSF/';

sub_list = {'sub-01', 'sub-02'};
ses_list = {'ses-1'};

for i_sub=1:numel(sub_list)
for i_ses=1:numel(ses_list)
sub = char(sub_list(i_sub));
ses = char(ses_list(i_ses));
sub_qcsf_dir = fullfile(qcsf_dir, sub, ses); 
file_list = dir(sub_qcsf_dir);

saved_basic_stim = false;
for i_file = 1:length(file_list)
    this_eye = 0;
    if contains(file_list(i_file).name, 'L_qCSF')
        this_eye = 'eye-L';
    elseif contains(file_list(i_file).name, 'R_qCSF')
        this_eye = 'eye-R';
    else
        disp('...')
        continue
    end
    this_file = fullfile(sub_qcsf_dir, file_list(i_file).name);
    QCSF_data = load(this_file);
    
    % Save basic stimuli info, to the dm folder in amb_code...
    if saved_basic_stim==false
        qCSF_SF_list = QCSF_data.qcsf.stimuli.frequency; % The 12 frequencies (x values for CSF)
        save(fullfile(amb_code_dir, 'qCSF_SF_list'), 'qCSF_SF_list');
        saved_basic_stim = true;
    end
    
    % Save the data of interest as a struct:
    qCSF_struct = struct( ...
        'SF_list', [],...       % 12 SFs,
        'history', [], ...      % trial num, SF, contrast, correct
        'params', [],...        % 4 x CSF params 
        'peakCS', [],...        % 4 x CSF params, 1
        'peakSF', [],...        % 4 x CSF params, 2
        'bdwth', [],...         % 4 x CSF params, 3
        'lowSFtrunc', [],...    % 4 x CSF params, 4
        'sensitivity',[],...    % CSF curve 
        'AULCSF',[]);           % Area under log CSF...
        
    % 
    qCSF_struct.SF_list = qCSF_SF_list;
    qCSF_struct.history = QCSF_data.qcsf.data.history;
    qCSF_struct.params = QCSF_data.qcsf.data.estCSF(end,:);
    qCSF_struct.peakCS = QCSF_data.qcsf.data.estCSF(end,1);
    qCSF_struct.peakSF = QCSF_data.qcsf.data.estCSF(end,2);
    qCSF_struct.bdwth = QCSF_data.qcsf.data.estCSF(end,3);
    qCSF_struct.lowSFtrunc = QCSF_data.qcsf.data.estCSF(end,4);
    qCSF_struct.sensitivity = QCSF_data.qcsf.data.estSensitivity(end,:);
    qCSF_struct.AULCSF = QCSF_data.qcsf.data.estAULCSF(end);
    % 
    save_name = strjoin({sub, ses, this_eye, 'struct'}, '_');    
    save(fullfile(sub_qcsf_dir,save_name), 'qCSF_struct');        

end % file loop
end % ses loop
end % sub loop

% Note to estimate sensitivyt from parameters...Estimated CSF curve    
%     est_curve = findQCSF( ...
%         log10(qCSF_SF_list), ...
%         qCSF_params(1), ...
%         qCSF_params(2), ...
%         qCSF_params(3), ...
%         qCSF_params(4));


%%
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
%%
