% ------------------------------------ % 
% Define the parts you would like to run
calc_poseprior_possibility = true;
calc_imposs_should_poss = false;
calc_expert_matching = false;
calc_ba2_matching = true;
calc_ba_third_matching = true;


% Go to the parentfolder (called "Check-Possibility" atm)
cd('/home/magehrke/Github/3D Body Postures/Check-Possiblity/');
% We add the paths to the scripts here
addpath("src");
addpath("poseprior");

mat = load('data/all_params_enc_mod_12_runs.mat');
stim = mat.stim;
% ------------------------------------------------------------------- %
% Viewpoint conversion structure
vp_map = containers.Map([-45, 0, 45], ["1", "2", "3"]);
% Edges strings to print on images
edges_str_lst = [
   "1: Belly - Neck", ...
   "2: Neck - L.Shoulder", ...
   "3: L.Shoulder - L.Elbow", ...
   "4: L.Elbow - L.Wrist", ...
   "5: Neck - R.Shoulder", ...
   "6: R.Shoulder - R.Elbow", ...
   "7: R.Elbow - R.Wrist", ...
   "8: Neck - Face", ...
   "9: Belly - L.Hip", ...
   "10: L.Hip - L.Knee", ...
   "11: L.Knee - L.Ankle", ...
   "12: L.Ankle - L.Foot", ...
   "13: Belly - R.Hip", ...
   "14: R.Hip - R.Knee", ...
   "15: R.Knee - R.Ankle", ...
   "16: R.Ankle - R.Foot"
];

% ------------------------------------------------------------------- %
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

% Delete all -45 and +45 viewpoints (3D coords are same for all views)
idx = param_view(:, 2) == 0;
param_view = param_view(idx, :);
kp3d_arr = kp3d_arr(idx, :);
fprintf("%i stimuli in total.\n", length(kp3d_arr));

% ------------------------------------------------------------------- %
% Calculate which pose is possible/impossible by using PosePrior
if calc_poseprior_possibility
    poss_array = zeros(108, 16);
    poss_bool = zeros(108, 1);
    impossible_stimuli_uparams=[];
    nearest_pose_poss_ctr = 0;
    imposs_not_adjustable = [];
    joints_lst=cell(1, length(param_view));
    for i=1:length(param_view)
        for_loop_percent(i, length(param_view));
        kp3d = kp3d_arr{i,1};
        % Rearrange joints
        % 1: belly, 2: Neck, 
        % 3: L-shldr, 4: L-elbow, 5: L-wrist, 
        % 6: R-shldr, 7: R-elbow, 8: R-wrist, 
        % 9: face, 
        % 10: L-hip, 11: L-knee, 12: L-ankle, 13: L-foot
        % 14: R-hip, 15: R-knee, 16: R-ankle, 17: R-foot, 
        joints = kp3d([ ...
            7,13, ...
            17,19,21, ...
            18,20,22, ...
            25, ...
            2,5,8,11, ...
            3,6,9,12 ...
            ],:);
        % Swap columns because the method has z, y, x
        joints_first_col = joints(:, 1);
        %joints(:, 1) = joints(:, 3); % NEGATIVE!
        %joints(:, 3) = joints_first_col; % NEGATIVE!
        % Scaling (does not matter, I hope!, except for plotting)
        factor = 20 / (joints(2, 2) - joints(16, 2));
        joints = joints * factor;
        % Center belly at (0,0,0)
        joints(:, 1) = joints(:, 1) - joints(1, 1);
        joints(:, 2) = joints(:, 2) - joints(1, 2);
        joints(:, 3) = joints(:, 3) - joints(1, 3); 

        joints_lst{i} = joints;
        joints = joints.';
    
        [poss_arr_i, pose_adj] = isvalid(joints);
        poss_arr_i(12) = 1; % setting feet as true
        poss_arr_i(16) = 1; % setting feet as true
        poss_array(i, :) = poss_arr_i;
        poss_bool(i) = all(poss_arr_i);

        if ~ all(poss_arr_i)
            
            impossible_stimuli_uparams = [impossible_stimuli_uparams; param_view(i, 1)];
            % If the pose is not possible, is the adjusted pose poss?
            rslt_adj = isvalid(pose_adj);
            rslt_adj(12) = 1; % setting feet as true
            rslt_adj(16) = 1; % setting feet as true       
            if all(rslt_adj)
                nearest_pose_poss_ctr = nearest_pose_poss_ctr + 1;
            else
                imposs_not_adjustable = [imposs_not_adjustable; param_view(i, 1), rslt_adj];
                if false % plot not adjustable impossible poses
                    poseprior_plot_joints('3d', joints_lst{i}, "Uparam " + param_view(i, 1));
                    plotbrowser;
                end
            end
        end
    end
    disp(' ');
    fprintf('\n');
    impossible_stimuli_uparams = sort(impossible_stimuli_uparams);
    fprintf("%i stimuli are possible.\n", sum(poss_bool));
    fprintf("%i imposs poses can be adjusted.\n", nearest_pose_poss_ctr);
    fprintf("%i imposs poses cannot be adjusted.\n\n", size(imposs_not_adjustable, 1));
    % -------- -------- -------- -------- -------- -------- -------- %
    % Visualisation: move possible/impossible stimuli to new folders
    if ~exist("data/poseprior/", 'dir')
        mkdir data poseprior;
    end
    if ~exist("data/poseprior/possible/", 'dir')
        mkdir data/poseprior possible;
    end
    delete("data/poseprior/possible/*");
    if ~exist("data/poseprior/impossible/", 'dir')
        mkdir data/poseprior impossible;
    end
    delete("data/poseprior/impossible/*");
    
    for i=1:length(param_view)
        % Create stimuli file name
        uparam = int2str(param_view(i, 1));
        viewpoint = param_view(i, 2);
        scale = int2str(param_view(i, 3));
        stim_file_name = "Stim_uparam_" + uparam + "_Viewpoint_" + ...
            vp_map(viewpoint) + "_scale_" + scale + ".png";
        % Move stimuli to possible/impossible folder
        if poss_bool(i, 1) == 1
            dst_folder = "data/poseprior/possible/";
            copyfile("data/Stim_images/" + stim_file_name, dst_folder);
        else % must be impossible then
            dst_folder = "data/poseprior/impossible/";
            im = imread('data/Stim_images/' + stim_file_name);
            inv_bool = ~poss_array(i,:);
            txt = edges_str_lst(inv_bool);
            txt = strjoin(txt, '\n');
            im_w_txt = insertText(im, [0 0], txt);
            imwrite(im_w_txt, dst_folder + stim_file_name, 'png');
        end
    end
end

% ------------------------------------------------------------------- %
% After looking at the images and the poss/imposs division we got
% from the pose prior, the following uparams are categorized
% as imposs but look poss
if calc_imposs_should_poss
    poss_but_shown_impossible = [9, 17, 30, 33, 37, 42, 44, 48, 49, 52, ...
        53, 56, 68, 69, 72, 74, 79, 88, 99, 100, 111, 114, 115, 123, ...
        130, 137, 140, 146, 154, 159, 187]; % 31 poses originally
    stimuli_diff_rslts = [];
    stimuli_diff_names = [];
    for i=1:length(param_view)
        viewpoint = param_view(i, 2);  % we only want each uparam once
        uparam = param_view(i, 1);
        if viewpoint == 0 && ismember(uparam, poss_but_shown_impossible)
            if ~all(poss_array(i,:)) % it is indeed categorized as impossible 
                stimuli_diff_rslts = [stimuli_diff_rslts; poss_array(i,:)];
                stimuli_diff_names = [stimuli_diff_names; param_view(i, 1)];
            end
        end
    end
    
    disp("Num of imposs stim that should be poss: " + size(stimuli_diff_rslts, 1))
    stimuli_diff_joints = sum(stimuli_diff_rslts, 1);
    disp("Edge count of these stimuli:")
    disp(stimuli_diff_joints);
    
    stimuli_diff_rslts = [stimuli_diff_names, stimuli_diff_rslts];
    [~,idx] = sort(stimuli_diff_rslts(:,1)); % sort just the first column
    stimuli_diff_rslts = stimuli_diff_rslts(idx,:);   % sort the whole matrix using the sort indices
end

% ------------------------------------------------------------------- %
% Plot the joints of kp3d against the static pose
% To judge if we got the axis right
rs = randsample(1:108, 1);
poseprior_plot_joints('3d', joints_lst{rs}, ...
    "Uparam " + param_view(rs, 1) + " VP " + param_view(rs, 2) + char(176), true);

% ------------------------------------------------------------------- %
% Compare what we have found with expert choices of possibility
if calc_expert_matching
    expert_tbl = readtable("data/possibility_by_expert.csv"); 
    
    divergence_diff_ctr = 0;
    matching_poses_ctr = 0;
    for i=1:length(param_view)
        viewpoint = param_view(i, 2); 
        if viewpoint == 0 % Because 3D, we only want the viewpoint from front
            uparam = param_view(i, 1);
            poss = poss_bool(i);
        
            % Get expert data
            expert_poss = [];
            expert_diff = []; % Was it hard for the expert to judge the pose?
            for j=1:height(expert_tbl)
                if expert_tbl{j, 'uparam'} == uparam && ...
                        expert_tbl{j, 'viewpoint'} == str2num(vp_map(viewpoint))
                    expert_poss = expert_tbl{j, 'possible_3d'};
                    expert_diff = expert_tbl{j, 'difficult'};
                    continue;
                end
            end
        
            % Compare possibility estimates of poseprior and expert
            if poss == expert_poss
                matching_poses_ctr = matching_poses_ctr + 1;
            else
                divergence_diff_ctr = divergence_diff_ctr + expert_diff;
            end
        end
    end
    fprintf("%i poses are matching with expert opinion.\n", matching_poses_ctr);
    fprintf("From the diverging poses, %i were hard to judge.\n", divergence_diff_ctr);
end

% ------------------------------------------------------------------- %
% Compare what we have found with behavioral analysis results


if calc_ba2_matching
    ba2_tbl = readtable("data/grouping_poss_real_2categories.csv", 'Delimiter', ',');
    idx = ba2_tbl.group_index < 2;
    ba2_poss_uparams = ba2_tbl{idx, "uparam"};
    idx = ba2_tbl.group_index > 1;
    ba2_imposs_uparams = ba2_tbl{idx, "uparam"};

    
    ba2_matching_tbl = table([0;0], [0;0], ...
        'RowNames', {'BA Imposs';'BA Poss'}, ...
        'VariableNames', {'PosePrior Imposs', 'PosePrior Poss'});
    for i=1:length(param_view)
        viewpoint = param_view(i, 2); 
        if viewpoint == 0 % Because 3D, we only want the viewpoint from front
            uparam = param_view(i, 1);
            scale = param_view(i, 3);
            poss = poss_bool(i);
        
            % Compare with BA
            stim_name = "Stim_uparam_" + uparam + "_Viewpoint_" + ...
                vp_map(viewpoint) + "_scale_" + scale;
            ba2_poss = 0; % works because we have all 108 stimuli
            if any(strcmp(ba2_poss_uparams, stim_name))
                ba2_poss = 1;
            end

            % Create contingency table
            ba2_matching_tbl{ba2_poss+1, poss+1} = ...
                ba2_matching_tbl{ba2_poss+1, poss+1} + 1;
        end
    end
    disp("Matches PosePrior & BA (hard cutoff at 3.0):");
    disp(ba2_matching_tbl);
end

if calc_ba_third_matching
    ba3_tbl = readtable("data/grouping_poss_real_3categories.csv", 'Delimiter', ',');
    idx = ba3_tbl.group_index == 0;
    ba3_poss_uparams = ba3_tbl{idx, "uparam"};
    idx = ba3_tbl.group_index == 3;
    ba3_imposs_uparams = ba3_tbl{idx, "uparam"};

    ba3_pose_imposs_ba_poss = {};

    % Create contingency table    
    ba3_matching_tbl = table([0;0], [0;0], ...
        'RowNames', {'BA Imposs';'BA Poss'}, ...
        'VariableNames', {'PosePrior Imposs', 'PosePrior Poss'});
    for i=1:length(param_view)
        uparam = param_view(i, 1);
        viewpoint = param_view(i, 2); % should be all 0
        scale = param_view(i, 3);
        poss = poss_bool(i);
    
        % Compare with BA
        stim_name = "Stim_uparam_" + uparam + "_Viewpoint_" + ...
            vp_map(viewpoint) + "_scale_" + scale;
        ba3_poss = nan;
        if any(strcmp(ba3_poss_uparams, stim_name))
            ba3_poss = 1;
        elseif any(strcmp(ba3_imposs_uparams, stim_name))
            ba3_poss = 0;
        end

        % Fill contingency table 
        if ~isnan(ba3_poss)
            ba3_matching_tbl{ba3_poss+1, poss+1} = ...
                ba3_matching_tbl{ba3_poss+1, poss+1} + 1;
        end

        if poss == 0 && ba3_poss == 1
            ba3_pose_imposs_ba_poss{end+1} = {uparam};
        end

    end
    disp("Matches PosePrior & BA (only upper/lower third):");
    disp(ba3_matching_tbl);
end


