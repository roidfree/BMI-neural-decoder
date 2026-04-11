%% —————— positionEstimatorTraining_noPCA_LDA.m ——————
function modelParameters = positionEstimatorTraining(trainingData)
% Train Gaussian Naive Bayes only (no PCA, no LDA).
% Inputs:
%   trainingData: [nTrials x nDirs] struct array with field spikes (N x T).
% Outputs:
%   modelParameters struct fields:
%     .mu_class    – global Z-score mean over raw counts (1 x D0)
%     .sigma_class – global Z-score std over raw counts (1 x D0)
%     .classifier  – NBC params: classMeans, classVars, priors, classes
%     .T_class     – spike window length (ms bins)

    %% 1. Build feature matrix X0 and labels y
    T_class = 320;
    [nTrials, nDirs] = size(trainingData);
    D0 = size(trainingData(1,1).spikes,1);
    M  = nTrials * nDirs;
    X0 = zeros(M, D0);
    y  = zeros(M,1);
    idx = 0;
    for i = 1:nTrials
        for k = 1:nDirs
            idx = idx + 1;
            % Per-neuron spike count in first T_class time bins
            X0(idx,:) = sum(trainingData(i,k).spikes(:,1:T_class),2)';
            y(idx)    = k;
        end
    end

    %% 2. Z-score normalization
    mu_class    = mean(X0,1);               
    sigma_class = std(X0,0,1) + eps;        % avoid division by zero
    X_norm = (X0 - mu_class) ./ sigma_class; 

    %% 3. Train Gaussian Naive Bayes
    Cn = max(y);
    classMeans = zeros(Cn, D0);
    classVars  = zeros(Cn, D0);
    priors     = zeros(Cn,1);
    for c = 1:Cn
        Xc = X_norm(y==c, :);
        priors(c)       = size(Xc,1) / M;
        classMeans(c,:) = mean(Xc,1);
        classVars(c,:)  = var(Xc,0,1) + eps;  % avoid zero variance
    end

    classifier.classMeans = classMeans;
    classifier.classVars  = classVars;
    classifier.priors     = priors;
    classifier.classes    = (1:Cn)';

    %% 4. Pack modelParameters
    modelParameters.mu_class    = mu_class;
    modelParameters.sigma_class = sigma_class;
    modelParameters.classifier  = classifier;
    modelParameters.T_class     = T_class;
end
