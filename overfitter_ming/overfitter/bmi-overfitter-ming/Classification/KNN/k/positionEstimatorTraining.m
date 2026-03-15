% positionEstimatorTraining.m

function modelParameters = positionEstimatorTraining(trainingData, k)
% Trains a k-NN classifier to predict movement direction from spikes.
%
% USAGE:
%   modelParameters = positionEstimatorTraining(trainingData)
%   modelParameters = positionEstimatorTraining(trainingData, k)
%
% INPUT:
%   trainingData : (#trials x 8) struct array, each with fields .spikes, .handPos
%   k (optional) : number of neighbors (default 5)
%
% OUTPUT:
%   modelParameters : struct with fields
%       .X       : N×98  matrix of normalized features
%       .y       : N×1   vector of labels in 1..8
%       .mu      : 1×98  feature mean (pre-normalization)
%       .sigma   : 1×98  feature std  (pre-normalization)
%       .k       : scalar number of neighbors

    if nargin<2 || isempty(k)
        k = 5;
    end

    T_class = 320;  % use first 320 ms for classification
    [numTrials, numDirs] = size(trainingData);

    % Build training set
    X = zeros(numTrials * numDirs, 98);
    y = zeros(numTrials * numDirs, 1);
    idx = 1;
    for i = 1:numTrials
        for d = 1:numDirs
            spk = trainingData(i,d).spikes;       % 98 x T
            T   = size(spk,2);
            tEnd = min(T_class, T);
            feat = sum(spk(:,1:tEnd), 2)';       % 1x98
            X(idx,:) = feat;
            y(idx)   = d;
            idx = idx+1;
        end
    end

    % Z-score normalization
    mu    = mean(X,1);
    sigma = std(X,0,1) + eps;
    Xn    = (X - mu) ./ sigma;

    % Store model parameters
    modelParameters.X       = Xn;
    modelParameters.y       = y;
    modelParameters.mu      = mu;
    modelParameters.sigma   = sigma;
    modelParameters.k       = k;
end
