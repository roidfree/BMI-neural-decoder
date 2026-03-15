% positionEstimatorTraining_PCA_LDA_K.m

function modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData, k, pcaDim, ldaDim)
% Trains a PCA→LDA→k-NN classifier on 320 ms–summed spike counts.
%
% Usage:
%   modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData)
%   modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData, k, pcaDim, ldaDim)
%
% Inputs:
%   trainingData : (#trials×8) struct array with .spikes (98×T)
%   k            : # neighbors for k-NN (default 5)
%   pcaDim       : # PCA components (default 20)
%   ldaDim       : # LDA dimensions (≤7, default 6)
%
% Outputs modelParameters with fields:
%   .mu, .sigma        – 1×98 for z-scoring
%   .V_pca             – 98×pcaDim PCA projection
%   .W_lda             – pcaDim×ldaDim LDA projection
%   .X_proj            – N×ldaDim final features
%   .y                 – N×1 labels
%   .k                 – scalar

    if nargin<2||isempty(k),     k=5;    end
    if nargin<3||isempty(pcaDim), pcaDim=20; end
    if nargin<4||isempty(ldaDim), ldaDim=min(pcaDim,7); end

    T_class = 320;
    [numTrials, numDirs] = size(trainingData);
    N = numTrials * numDirs;
    D = 98;

    % 1) Build raw features X and labels y
    X = zeros(N,D);
    y = zeros(N,1);
    idx = 1;
    for i=1:numTrials
      for d=1:numDirs
        spk = trainingData(i,d).spikes;      % 98×T
        tEnd = min(T_class, size(spk,2));
        X(idx,:) = sum(spk(:,1:tEnd),2)';    % 1×98
        y(idx)   = d;
        idx = idx+1;
      end
    end

    % 2) Z-score normalization
    mu    = mean(X,1);
    sigma = std(X,0,1) + eps;
    Xn    = (X - mu) ./ sigma;             % N×98

    % 3) PCA via SVD: keep top pcaDim PCs
    %    Xn = U*S*V';  so columns of V are PCs
    [~,~,V] = svd(Xn,'econ');
    V_pca = V(:,1:pcaDim);                 % 98×pcaDim
    X_pca = Xn * V_pca;                    % N×pcaDim

    % 4) LDA on X_pca to get W_lda (pcaDim×ldaDim)
    classes = unique(y);
    C = numel(classes);
    meanAll = mean(X_pca,1)';              % pcaDim×1
    Sw = zeros(pcaDim,pcaDim);
    Sb = zeros(pcaDim,pcaDim);
    for c=classes'
      Xc = X_pca(y==c,:);
      mc = mean(Xc,1)';
      Sw = Sw + (Xc - mc.').'*(Xc - mc.');
      Nc = size(Xc,1);
      diff = mc - meanAll;
      Sb = Sb + Nc*(diff*diff');
    end
    [Vlda,~] = eig(Sb, Sw);
    % rank by eigenvalue ratio
    [~,ord] = sort(diag(Vlda'*Sb*Vlda)./sum((Vlda'*Sw*Vlda),2),'descend');
    W_lda = Vlda(:,ord(1:ldaDim));         % pcaDim×ldaDim
    X_lda = X_pca * W_lda;                 % N×ldaDim

    % 5) Store model
    modelParameters.mu     = mu;
    modelParameters.sigma  = sigma;
    modelParameters.V_pca  = V_pca;
    modelParameters.W_lda  = W_lda;
    modelParameters.X_proj = X_lda;
    modelParameters.y      = y;
    modelParameters.k      = k;
end
