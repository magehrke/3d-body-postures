%% HDF5
% Write models from mat into h5 files split by its crossvalidation runs
% Keep the Xtest (4161 param) / Xtrain (2080 params) structure
clear all;
modeldir = '/home/magehrke/data/Models';
resdir = '/home/magehrke/data/himalaya';
models = dir(fullfile(modeldir,'*transform_0.mat'));

for i =1:length(models)
    % Get Xtrain & Xtest of the model
    load(fullfile(models(1).folder,models(i).name))
    [~,name,ext] = fileparts(models(i).name);
    for f=1:3
        h5create(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_train',size(Xtrain{f}))
        h5create(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_test',size(Xtest{f}))
        h5write(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_train',Xtrain{f})
        h5write(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_test',Xtest{f})
    end
end


%% Create synthetic data
clear all;
modeldir = '/home/magehrke/data/Models';
load(fullfile(modeldir, 'kp2d_data_python_transform_0.mat'));
load(fullfile(modeldir,'hrf.mat')); % Load hrf into workspace
models = dir(fullfile(modeldir,'*transform_0.mat'));

nvox = 10;
avg_var = 24.16;

% Load onset
% TODO: HOW TO USE?? Isn't it obsolete?
load(fullfile(modeldir, 'training_runs_onsets.mat'))


for k=1:length(models)
    load(fullfile(models(k).folder,models(k).name))
    [~,name,ext] = fileparts(models(k).name);



    % Take first (of 3) crossval runs
    Xtr= cell2mat(Xtrain(1));
    indtr = find(sum(Xtr,2));

    % Convolve Xtrain with hrf function
    for i=1:size(Xtr,2)
        temp = conv(Xtr(:,i),hrf);
        Xtr(:,i) = temp(1:size(Xtr,1));
    end

    % Center and scale
    Xtr_ind = Xtr(indtr,:);
    mean_tr1 = mean(Xtr_ind, 1);
    std_tr1 = std(Xtr_ind, [], 1);
    Xtr = (Xtr - mean_tr1) ./ std_tr1;
    vari(k).mod = (var(Xtr, [], "all"));

    % Saves Xtrain*hdf of each model in one variable
    Xtr_hdf(k).mod = Xtr;

    % Take first (of 3) crossval runs
    Xte = cell2mat(Xtest(1));
    indte = find(sum(Xte,2));

    % Convolve Xtest with hrf function
    for i=1:size(Xte,2)
        temp = conv(Xte(:,i),hrf);
        Xte(:,i) = temp(1:size(Xte,1));
    end

    % Center and scale
    % TODO: Use scaling from training data?
    %mean_te1 = mean(Xte, 1);
    %std_te1 = std(Xte, [], 1);
    Xte(indte, :) = (Xte(indte, :) - mean_tr1) ./ std_tr1;

    % Saves Xtrain*hdf of each model in one variable   
    Xte_hdf(k).mod = Xte;

    % Saves *nvox* random values of model-parameter size in one variable
    bt(k).mod = 10*rand(size(Xtr,2),nvox);
end

a1 = 0;
a2 = 1;
a3 = 1; 
a4 = 0;
a5 = 0;
a6 = 0; 
s = 150;

% Multiplies the random numbers of brain voxels by constant for each model
% Convert to 1d list
btrain = [];
for i=1:length(models)
    btrain = [btrain;eval(sprintf("a%d",i))*bt(i).mod];
end

% Convert to 1d list
Xtr_hdf_lst= [];
Xte_hdf_lst = [];
for i=1:length(models)
    Xtr_hdf_lst = [Xtr_hdf_lst,(Xtr_hdf(i).mod)];
    Xte_hdf_lst = [Xte_hdf_lst,(Xte_hdf(i).mod)];
end

% Create Ytrain
Ytr = Xtr_hdf_lst*btrain;
Ytrn = Ytr +s*randn(size(Ytr)); % add noise
Ytrain{1} = Ytrn;
% Signal to Noise Ratio
SNR = var(Ytr(:,1))/var(Ytrn(:,1));
%plot(zscore(Ytr(:,1))); hold on; plot(zscore(Ytrn(:,1)));

% Create Ytest
Yte = Xte_hdf_lst*btrain;
Yten = Yte +s*randn(size(Yte));
Ytest{1} = Yten;
SNR = var(Yte(:,1))/var(Yten(:,1));

save('/home/magehrke/data/Main_effect/himalaya_fake_data_python.mat',"Ytrain","Ytest","btrain",'-v7.3');

% Plot histogram of variance inside the features
% Variance is counted for every timepoint in the training data
% The variance of the models is extremely different
calc_feat_variance = true;
if calc_feat_variance
    feat = {'VAEdec','VAEenc','VAEparam','gabor','kp2d','kp3d'};
    figure
    for i = 1:size(Xtr_hdf, 2)
        v(:,i) = var(Xtr_hdf(i).mod,[], 2);
        totalvar(:, i) = var(v(:, i));
        fprintf('Variance %s: %f (%f) (n=%f)\n', feat{i}, var(Xtr_hdf(i).mod,[], "all"), var(Xtr_hdf(i).mod, [], "all"), ...
            size(Xtr_hdf(i).mod, 2));
        m(:,i) = mean(Xtr_hdf(i).mod,2);
        histogram(v(:,i));
        hold on;
    end
    title('Variances of features per model per timepoint');
    ylabel('Number of timepoints');
    xlabel('Variance');
    legend(feat);
end


Y1 = a1*Xtr_hdf(1).mod*bt(1).mod;
Y2 = a5*Xtr_hdf(5).mod*bt(5).mod;

    
disp('Finished!')