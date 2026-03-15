% positionEstimatorTraining_LDA.m

function modelParameters = positionEstimatorTraining_LDA(trainingData, k, ldaDim)
% Trains an LDA projection followed by k-NN on spike-count features.
%
%   modelParameters = positionEstimatorTraining_LDA(trainingData, k, ldaDim)
%
% Inputs:
%   trainingData : (#trials×8) struct array with field .spikes (98×T)
%   k            : number of neighbors
%   ldaDim       : number of LDA components (≤7)
%
% Outputs modelParameters with fields:
%   .mu     (1×98), .sigma (1×98)   — for z-scoring
%   .W       (98×ldaDim)            — LDA projection
%   .X_lda   (N×ldaDim)             — projected training features
%   .y       (N×1)                  — labels 1..8
%   .k       scalar

    if nargin<2||isempty(k),     k = 5; end
    if nargin<3||isempty(ldaDim), ldaDim = 5; end

    T_class = 320;
    [numTrials, numDirs] = size(trainingData);
    N = numTrials * numDirs;
    D = 98;

    % 1) Build raw feature matrix and labels
    X = zeros(N,D);
    y = zeros(N,1);
    idx = 1;
    for i=1:numTrials
      for d=1:numDirs
        spk = trainingData(i,d).spikes;         % 98×T
        tEnd = min(T_class, size(spk,2));
        X(idx,:) = sum(spk(:,1:tEnd),2)';       % 1×98
        y(idx)   = d;
        idx = idx+1;
      end
    end

    % 2) Z-score
    mu    = mean(X,1);
    sigma = std(X,0,1) + eps;
    Xn    = (X - mu) ./ sigma;

    % 3) Compute LDA projection
    classes = unique(y);
    C = numel(classes);
    overallMean = mean(Xn,1)';       % D×1

    % Within‐class scatter
    Sw = zeros(D,D);
    Sb = zeros(D,D);
    for c=classes'
      Xc = Xn(y==c,:);
      mc = mean(Xc,1)';              % D×1
      Sw = Sw + (Xc - mc').'*(Xc - mc');
      Nc = size(Xc,1);
      diff = mc - overallMean;
      Sb = Sb + Nc * (diff * diff');
    end

    % Solve generalized eigenproblem Sb v = λ Sw v
    [V,~] = eig(Sb, Sw);
    % sort by descending eigenvalue
    [~,ord] = sort(diag(V'*Sb*V)./sum((V'*Sw*V),2),'descend');
    W = V(:,ord(1:ldaDim));         % D×ldaDim

    % 4) Project training data
    X_lda = Xn * W;                 % N×ldaDim

    % 5) Store
    modelParameters.mu    = mu;
    modelParameters.sigma = sigma;
    modelParameters.W     = W;
    modelParameters.X_lda = X_lda;
    modelParameters.y     = y;
    modelParameters.k     = k;
end
