function modelParameters = positionEstimatorTraining(trainingData, varargin)
% Train k-NN direction classifier from 320 ms spike counts.
%
% Inputs
%   trainingData  : (#trials x #dirs) struct array with field .spikes
%   varargin{1}   : optional k (default 5)
%
% Output
%   modelParameters.X      : normalized training features (N x numNeurons)
%   modelParameters.y      : direction labels (N x 1)
%   modelParameters.mu     : feature mean (1 x numNeurons)
%   modelParameters.sigma  : feature std (1 x numNeurons)
%   modelParameters.k      : neighbor count

    k = 5;
    if ~isempty(varargin) && ~isempty(varargin{1})
        k = varargin{1};
    end

    classificationHorizon = 320;
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    numSamples = numTrials * numDirs;

    features = zeros(numSamples, numNeurons);
    labels = zeros(numSamples, 1);

    sampleIdx = 1;
    for trialIdx = 1:numTrials
        for dirIdx = 1:numDirs
            spikes = trainingData(trialIdx, dirIdx).spikes;
            endIdx = min(classificationHorizon, size(spikes, 2));
            features(sampleIdx, :) = sum(spikes(:, 1:endIdx), 2)';
            labels(sampleIdx) = dirIdx;
            sampleIdx = sampleIdx + 1;
        end
    end

    mu = mean(features, 1);
    sigma = std(features, 0, 1) + eps;
    normalizedFeatures = (features - mu) ./ sigma;

    modelParameters.X = normalizedFeatures;
    modelParameters.y = labels;
    modelParameters.mu = mu;
    modelParameters.sigma = sigma;
    modelParameters.k = k;
end
