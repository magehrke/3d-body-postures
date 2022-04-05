% This script shows how to use the isvalid function, to see if a 3D pose
% satisfies the joint-angle limits or not. The input to this function is a
% 3-by-17 matrix consisting of 3D coordiantes of the joints and output is
% 16 dimensional binary vector telling whether the corresponding bone is 
% valid or not. Please see readme to know how the bones and joints are 
% defined in the 3D pose
%
% copyright: Ijaz Akhter, MPI Tuebingen
% May 20, 2015
%

prtName = {'back-bone', 'R-shldr', 'R-Uarm', 'R-Larm', 'L-shldr', 'L-Uarm', 'L-Larm', 'head', ...
    'R-hip', 'R-Uleg', 'R-Lleg', 'R-feet', 'L-hip', 'L-Uleg', 'L-Lleg', 'L-feet'}; % the bones in the 3D pose
N = length(prtName);          % # of bones  (= 16)

%% Testing the isvalid function with a valid test pose
load('testPose.mat')        % load a testPose, S1 (3-by-17 matrix) and the connectivity, edges (16-by-2 matrix)

flags = isvalid(S1)         % flags is a 16 dimensional binary vector telling which of the bones are valid
%all the falgs should be one

%% Testing isvalid with random depth of the bones. Run this multiple times
% consider S1 get projected by an orthographic camera with viewing
% direction as X-axis

trsoBones = [1,2,5,9,13];                           % the torso bones, isvalid doesn't evaluate them
ntrso = setdiff(1:N, trsoBones);
randSigns = sign(rand(1,N)-0.5);                    % get random signs for the rest of the bones

d2 = S1(:, edges(:,1)) - S1(:, edges(:,2));         % find relative depths coordinates by taking the parent as origin
d2(1,ntrso) = randSigns(ntrso).*abs(d2(1,ntrso));   % apply random signs to the depths of non-torso bones
Sd = estimateZ(d2, edges', [0,0,0]');

% [flags2] = isvalid(Sd);
[flags2, Sd2] = isvalid(Sd);                        % Sd2 is the closest valid pose to Sd

figure(1)
viewWnS2(S1([3,2], :), S1, Sd, edges)               % will show the 2D image and the actual and the corrupted 3D pose
figure(2)
viewStruct(Sd, Sd2, edges)
title('Black: Input invalid Pose, Blue: closest valid pose')

disp('invalid bones: ')
prtName{~flags2}                                    % the invalid bones

%% Testing isvalid if feet are missing

% S1(:,13) = NaN;         % right foot
% S1(:,17) = NaN;         % left foot
% 
% disp('Testing for the pose w/o feet....');
% flags = isvalid(S1)
% 
% trsoBones = [1,2,5,9,13];                           % the torso bones, isvalid doesn't evaluate them
% ntrso = setdiff(1:N, trsoBones);
% randSigns = sign(rand(1,N)-0.5);                    % get random signs for the rest of the bones
% 
% d2 = S1(:, edges(:,1)) - S1(:, edges(:,2));         % find relative depths coordinates by taking the parent as origin
% d2(1,ntrso) = randSigns(ntrso).*abs(d2(1,ntrso));   % apply random signs to the depths of non-torso bones
% Sd = estimateZ(d2, edges', [0,0,0]');
% [flags2, Sd2] = isvalid(Sd);                        % Sd2 is the closest valid pose to Sd
% 
% figure(3)
% viewWnS2(S1([3,2], :), S1, Sd, edges)               % will show the 2D image and the actual and the corrupted 3D pose
% % figure(4)
% % viewStruct(Sd, Sd2, edges)
% % title('Black: Input invalid Pose, Blue: closest valid pose')
% 
% disp('invalid bones: ')
% prtName{~flags2}                                    % the invalid bones