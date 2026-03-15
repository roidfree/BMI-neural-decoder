%% runPCA_LDA_GNB_twoInOne.m
clear; close all; clc;


load('monkeydata_training.mat');    
trainingData = trial;


T_class = 320;
[nTrials, nDirs] = size(trainingData);
D0  = size(trainingData(1,1).spikes,1);
M   = nTrials * nDirs;
X0  = zeros(M, D0);
y   = zeros(M,1);
idx = 0;
for i = 1:nTrials
    for k = 1:nDirs
        idx = idx + 1;
        X0(idx,:) = sum(trainingData(i,k).spikes(:,1:T_class),2)';
        y(idx)    = k;
    end
end

mu_class    = mean(X0,1);
sigma_class = std(X0,0,1) + eps;
X_norm      = (X0 - mu_class) ./ sigma_class;


C         = (X_norm' * X_norm) / (M - 1);
[V, Dmat] = eig(C);
[eigvals, order] = sort(diag(Dmat),'descend');
V_sorted  = V(:,order);
explained = eigvals / sum(eigvals);
cumVar    = cumsum(explained);
dPCA      = find(cumVar>=0.92,1,'first');
coeff     = V_sorted(:,1:dPCA);
mu_pca    = mean(X_norm,1);
X_centered= X_norm - mu_pca;
X_pca     = X_centered * coeff;


Cn    = max(y);
mu_all= mean(X_pca,1);
SW    = zeros(dPCA);
SB    = zeros(dPCA);
for c = 1:Cn
    Xc = X_pca(y==c,:);
    nc = size(Xc,1);
    mu_c = mean(Xc,1);
    SW = SW + (Xc-mu_c)'*(Xc-mu_c);
    SB = SB + nc*(mu_c-mu_all)'*(mu_c-mu_all);
end
[Vlda, Dlda] = eig(SB, SW);
[~, ordL]     = sort(diag(Dlda),'descend');
dLDA         = Cn-1;
W_lda        = Vlda(:, ordL(1:dLDA));
X_lda        = X_pca * W_lda;


markerSize = 15;             
colors     = lines(nDirs);   
colors(8,:) = [1, 0.4, 0.6];  

figure('Color','w','Position',[200 200 800 400]);
set(groot, 'DefaultAxesFontSize', 22);
set(groot, 'DefaultTextFontSize', 22);


dirLabels = arrayfun(@(c) sprintf('Direction %d', c), 1:nDirs, 'UniformOutput', false);


ax1 = axes('Position',[0.05 0.1 0.38 0.8]);
gscatter(ax1, X_pca(:,1), X_pca(:,2), y, colors, [], markerSize);
xlabel(ax1,'PC1'); ylabel(ax1,'PC2');
title(ax1,'PCA Projection');
ax1.Box = 'on';
xlim(ax1, [-8 8]);
ylim(ax1, [-7 9]);
lgd1 = legend(ax1, dirLabels, 'Location','best');
lgd1.ItemTokenSize = [markerSize markerSize];

ax2 = axes('Position',[0.6 0.1 0.38 0.8]);
gscatter(ax2, X_lda(:,1), X_lda(:,2), y, colors, [], markerSize);
xlabel(ax2,'LD1'); ylabel(ax2,'LD2');
title(ax2,'LDA Projection on PCA Subspace');
ax2.Box = 'on';
xlim(ax2, [-8 8]);
ylim(ax2, [-7 9]);
legend(ax2,'off');
