mat = load('all_params_enc_mod_12_runs.mat');
stim = mat.stim;

kp3d_arr={}; param_view = [];
for i=1:length(stim)-6
    info_struc = stim(i).run(1).info;
    for j=1:length(info_struc)
        kp3d_arr(end+1) = {info_struc(j).kp3d};
    end 
    param_view = [param_view;[info_struc.uparam;info_struc.viewpoint].'];
end
kp3d_arr = kp3d_arr.';

results=[];
res_bool=[];
for i=1:length(param_view)
    if mod(i, 10) == 0
        disp(i);
    end
    kp3d = kp3d_arr{i,1};
    % Rearrange joints
    % 1: belly, 2: Neck, 2: R-shldr, 3: R-elbow, 4: R-wrist, 
    % 5: L-shldr, 6: L-elbow, 7: L-wrist, 9: face, 
    % 10: R-hip, 11: R-knee, 12: R-ankle, 13: R-foot, 
    % 14: L-hip, 15: L-knee, 16: L-ankle, 17: L-foot
    joints = kp3d([ ...
        4,13,17,19,21, ...
        18,20,22, ...
        25, ...
        2,5,8,11, ...
        3,6,9,12],:);
    joints(:, 1) = joints(:, 1) - joints(1, 1);
    joints(:, 2) = joints(:, 2) - joints(1, 2);
    joints(:, 3) = joints(:, 3) - joints(1, 3);
    joints = joints.';

    rslt = isvalid(joints);
    results = [results; rslt];
    res_bool = [res_bool; all(rslt)];
end

disp(sum(res_bool(:) == 1))

