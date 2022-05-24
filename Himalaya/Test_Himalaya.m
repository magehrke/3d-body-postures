%% HDF5 
clear all
maindir = '/home/magehrke/data/Main_effect';
resdir = fullfile(maindir,'Models');
models = dir(fullfile(maindir,'*transform_0.mat'));

for i =1:length(models)
    load(fullfile(models(1).folder,models(i).name))
    [~,name,ext] = fileparts(models(i).name);
    for f=1:3
        h5create(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_train',size(Xtrain{f}))
        h5create(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_test',size(Xtest{f}))
        h5write(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_train',Xtrain{f})
        h5write(fullfile(resdir,[name,'_cv',num2str(f),'.h5']),'/X_test',Xtest{f})
    end
end


% split = squeeze(mean(cv_scores_split));
% for i =1:size(split,1)
%     histogram(split(i,split(i,:)>0));
%     hold on
% end


%% Fake data
clear all
maindir = '/home/magehrke/data/Main_effect';
load(fullfile(maindir,'hrf.mat'));
models = dir(fullfile(maindir,'*transform_0.mat'));
% models([2,3,4,6,7,13,14]) = [];
nvox = 10;
avg_var = 24.16;
% a1 = 10*2.16e4;
a1 = 6.47;
% a1=avg_var*10;
a2 = 0;
a3= 0; 
a4=0;
a5 = 1;
a6=0;
  s=0;

for k =1:length(models)
    load(fullfile(models(1).folder,models(k).name))
    [~,name,ext] = fileparts(models(k).name);
    Xtr= cell2mat(Xtrain(1));
%     Xtr = zscore(Xtr,[],2);

    for i=1:size(Xtr,2)
        temp = conv(Xtr(:,i),hrf);
        Xtr(:,i) = temp(1:size(Xtr,1));
    end
    bt(k).mod = 10*rand(size(Xtr,2),nvox);
    Xtrr(k).mod = Xtr;
    

    Xte= cell2mat(Xtest(1));
%     Xte = zscore(Xte,[],2);

    for i=1:size(Xte,2)
        temp = conv(Xte(:,i),hrf);
        Xte(:,i) = temp(1:size(Xte,1));
    end
    Xtee(k).mod = Xte;
end
btrain = [];
for i=1:length(models)
btrain = [btrain;eval(sprintf("a%d",i))*bt(i).mod];
end
Xtra= [];
Xte = [];
for i=1:length(models)
    Xtra = [Xtra,(Xtrr(i).mod)];
    Xte = [Xte,(Xtee(i).mod)];
end

    Ytr = Xtra*btrain;
    Ytrn = Ytr +s*randn(size(Ytr));

    SNR = var(Ytr(:,1))/var(Ytrn(:,1));
    plot(zscore(Ytr(:,1))); hold on; plot(zscore(Ytrn(:,1)));
    Yte = Xte*btrain;
    Yten = Yte +s*randn(size(Yte));

    SNR = var(Yte(:,1))/var(Yten(:,1));
Ytrain{1} = Ytrn;
Ytest{1} = Yten;
% save('C:\Users\g.marrazzo\Desktop\Encoding_Python_Gallant\Data\Double\Fake_Sit_gabor_kp2d_kp3d_data_python.mat',"Ytrain","Ytest","btrain",'-v7.3');
        feat = {'VAEdec','VAEenc','VAEparam','gabor','kp2d','kp3d'};
figure
for i =1:size(Xtrr,2)
    v(:,i)= var(Xtrr(i).mod,[],2);
    m(:,i) = mean(Xtrr(i).mod,2);
    histogram(v(:,i));
    hold on;
end
    legend(feat)

   Y1 = a1*Xtrr(1).mod*bt(1).mod;

    Y2 = a5*Xtrr(5).mod*bt(5).mod;
%%
load('C:\Users\g.marrazzo\Desktop\Encoding_Python_Gallant\Data\Double\Himalaya\Fake_gabor_kp2d_kp3d_results_S_it_output_VAE_dec_VAE_enc_VAEparam_gabor_kp2d_kp3d.mat')

        feat = {'Sit','VAEdec','VAEenc','VAEparam','gabor','kp2d','kp3d'};
%                 feat = {'Sit','VAEdec','VAEenc','VAEparam','gabor'};


split = squeeze(corr_scores_split);
for i =1:size(split,1)
    histogram(split(i,:));
    hold on
end
legend(feat)
figure
splitr2 = squeeze(r2_scores_split);
splitr2 = splitr2./r2_scores;
for i =1:size(splitr2,1)
    histogram(splitr2(i,:));
    hold on
end
legend(feat)

figure
plot((Xtrr(1).mod)*bt(1).mod(:,1)); hold on; plot((Xtrr(5).mod)*bt(5).mod(:,1)); plot((Xtrr(6).mod)*bt(6).mod(:,1));