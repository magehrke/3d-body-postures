% ------------------------------------ % 
% Define the parts you would like to run
calc_poseprior_possibility = true;

% Go to the parentfolder (called "Check-Possibility" atm)
% We add the paths to the scripts here
addpath("./src/");
addpath("./poseprior/");

mat = load('data/all_params_enc_mod_12_runs.mat');
stim = mat.stim;

% Fill an array with the kp3d data
% Fill an array with the uparam and viewpoint data
kp3d_arr={}; param_view = [];
for i=1:length(stim)-6
    info_struc = stim(i).run(1).info;
    for j=1:length(info_struc)
        kp3d_arr(end+1) = {info_struc(j).kp3d};
    end 
    param_view = [param_view;[info_struc.uparam;info_struc.viewpoint;info_struc.scale].'];
end
kp3d_arr = kp3d_arr.';


% Calculate which pose is possible/impossible by using PosePrior
if calc_poseprior_possibility
    results=[];
    res_bool=[];
    impossible_stimuli=[];
    for i=1:length(param_view)
        if mod(i, 10) == 0
            disp(i);
        end
        kp3d = kp3d_arr{i,1};
        % Rearrange joints
        % 1: belly, 2: Neck, 
        % 3: L-shldr, 4: L-elbow, 5: L-wrist, 
        % 6: R-shldr, 7: R-elbow, 8: R-wrist, 
        % 9: face, 
        % 10: L-hip, 11: L-knee, 12: L-ankle, 13: L-foot
        % 14: R-hip, 15: R-knee, 16: R-ankle, 17: R-foot, 
        joints = kp3d([ ...
            4,13, ...
            17,19,21, ...
            18,20,22, ...
            25, ...
            2,5,8,11, ...
            3,6,9,12],:);
        joints(:, 1) = joints(:, 1) - joints(1, 1);
        joints(:, 2) = joints(:, 2) - joints(1, 2);
        joints(:, 3) = joints(:, 3) - joints(1, 3); 
        joints(13,:) = NaN; % trying to avoid feet
        joints(17,:) = NaN; % trying to avoid feet
        joints = joints.';
    
        rslt = isvalid(joints);
        rslt(12) = 1; % setting feet as true
        rslt(16) = 1; % setting feet as true
        results = [results; rslt];
        res_bool = [res_bool; all(rslt)];
        if ~ all(rslt)
            impossible_stimuli = [impossible_stimuli; param_view(i, 1)];
        end
    end
    impossible_stimuli = sort(impossible_stimuli);
    disp(sum(res_bool(:) == 1) + " stimuli are possible.");

    % Visualisation: move possible/impossible stimuli to new folders
    mkdir data poseprior;
    mkdir data/poseprior possible;
    delete("data/poseprior/possible/*");
    mkdir data/poseprior impossible;
    delete("data/poseprior/impossible/*");
    for i=1:length(param_view)
        % Create stimuli file name
        uparam = int2str(param_view(i, 1));
        viewpoint = param_view(i, 2);
        scale = int2str(param_view(i, 3));  % fix() discards fraction
        viewpoint_str = "1";
        if viewpoint == 0
            viewpoint_str = "2";
        elseif viewpoint == 45
            viewpoint_str = "3";
        end
        stim_file_name = "Stim_uparam_" + uparam + "_Viewpoint_" + ...
            viewpoint_str + "_scale_" + scale + ".png";
        % Move stimuli to possible/impossible folder
        dst_folder = "data/poseprior/possible";
        if res_bool(i, 1) == 0
            dst_folder = "data/poseprior/impossible";
        end
        copyfile("data/Stim_images/" + stim_file_name, dst_folder);
    end
end

% ------------------------------------------------------------------- %
% After looking at the images and the poss/imposs division we got
% from the pose prior, the following uparams are categorized
% as imposs but look poss
poss_but_shown_impossible = [9, 17, 30, 33, 37, 42, 44, 48, 49, 52, ...
    53, 56, 68, 69, 72, 74, 79, 88, 99, 100, 111, 114, 115, 123, ...
    130, 137, 140, 146, 154, 159, 187];
disp(length(poss_but_shown_impossible));
stimuli_diff_rslts = [];
stimuli_diff_names = [];
for i=1:length(param_view)
    viewpoint = param_view(i, 2);  % we only want each uparam once
    uparam = param_view(i, 1);
    if viewpoint == 0 && ismember(uparam, poss_but_shown_impossible)
        if ~all(results(i,:)) % it is ineed categorized as impossible 
            stimuli_diff_rslts = [stimuli_diff_rslts; results(i,:)];
            stimuli_diff_names = [stimuli_diff_names; param_view(i, 1)];
        end
    end
end

stimuli_diff_joints = sum(stimuli_diff_rslts, 1);
disp(stimuli_diff_joints);

stimuli_diff_rslts = [stimuli_diff_names, stimuli_diff_rslts];
[~,idx] = sort(stimuli_diff_rslts(:,1)); % sort just the first column
stimuli_diff_rslts = stimuli_diff_rslts(idx,:);   % sort the whole matrix using the sort indices

