function modelParameters = positionEstimatorTraining_LDA(trainingData, varargin)
% Train LDA + k-NN direction classifier from 320 ms spike counts.
%
% Inputs
%   trainingData  : (#trials x #dirs) struct array with field .spikes
%   varargin{1}   : optional k (default 5)
%   varargin{2}   : optional ldaDim (default 5)
%
% Output
%   modelParameters.mu      : feature mean before z-score
%   modelParameters.sigma   : feature std before z-score
%   modelParameters.W       : LDA projection matrix
%   modelParameters.X_lda   : projected training features
%   modelParameters.y       : labels
%   modelParameters.k       : neighbor count

    k = 5;
    ldaDim = 5;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), k = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), ldaDim = varargin{2}; end

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

    classList = unique(labels);
    overallMean = mean(normalizedFeatures, 1)';
    sw = zeros(numNeurons, numNeurons);
    sb = zeros(numNeurons, numNeurons);
    for classIdx = classList'
        classData = normalizedFeatures(labels == classIdx, :);
        classMean = mean(classData, 1)';
        centeredClass = classData - classMean';
        sw = sw + centeredClass' * centeredClass;
        classCount = size(classData, 1);
        meanDiff = classMean - overallMean;
        sb = sb + classCount * (meanDiff * meanDiff');
    end

    [eigVec, eigVal] = eig(sb, sw);
    eigScore = real(diag(eigVal));
    [~, order] = sort(eigScore, "descend");
    ldaDim = min(ldaDim, numel(order));
    w = real(eigVec(:, order(1:ldaDim)));
    projectedFeatures = normalizedFeatures * w;

    modelParameters.mu = mu;
    modelParameters.sigma = sigma;
    modelParameters.W = w;
    modelParameters.X_lda = projectedFeatures;
    modelParameters.y = labels;
    modelParameters.k = k;
end
