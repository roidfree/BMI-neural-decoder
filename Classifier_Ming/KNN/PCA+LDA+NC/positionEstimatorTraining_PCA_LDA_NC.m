function modelParameters = positionEstimatorTraining_PCA_LDA_NC(trainingData, varargin)
% Train PCA + LDA + nearest-centroid direction classifier from 320 ms spike counts.
%
% Inputs
%   trainingData  : (#trials x #dirs) struct array with field .spikes
%   varargin{1}   : optional pcaDim (default 20)
%   varargin{2}   : optional ldaDim (default min(pcaDim,7))
%
% Output
%   modelParameters.mu        : feature mean (1 x numNeurons)
%   modelParameters.sigma     : feature std (1 x numNeurons)
%   modelParameters.V_pca     : PCA loadings
%   modelParameters.W_lda     : LDA projection
%   modelParameters.centroids : class centroids in LDA space (numClasses x ldaDim)
%   modelParameters.X_proj    : training LDA coordinates (N x ldaDim)
%   modelParameters.y         : training labels (N x 1)

    pcaDim = 20;
    ldaDim = 7;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), pcaDim = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), ldaDim = varargin{2}; end
    ldaDim = min(ldaDim, pcaDim);

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

    [~, ~, v] = svd(normalizedFeatures, "econ");
    pcaDim = min(pcaDim, size(v, 2));
    vPca = v(:, 1:pcaDim);
    pcaFeatures = normalizedFeatures * vPca;

    classList = unique(labels);
    globalMean = mean(pcaFeatures, 1)';
    sw = zeros(pcaDim, pcaDim);
    sb = zeros(pcaDim, pcaDim);
    for classIdx = classList'
        classData = pcaFeatures(labels == classIdx, :);
        classMean = mean(classData, 1)';
        centeredClass = classData - classMean';
        sw = sw + centeredClass' * centeredClass;
        classCount = size(classData, 1);
        meanDiff = classMean - globalMean;
        sb = sb + classCount * (meanDiff * meanDiff');
    end

    [eigVec, eigVal] = eig(sb, sw);
    eigScore = real(diag(eigVal));
    [~, order] = sort(eigScore, "descend");
    ldaDim = min(ldaDim, numel(order));
    wLda = real(eigVec(:, order(1:ldaDim)));
    projectedFeatures = pcaFeatures * wLda;

    numClasses = max(labels);
    centroids = zeros(numClasses, ldaDim);
    for c = 1:numClasses
        centroids(c, :) = mean(projectedFeatures(labels == c, :), 1);
    end

    modelParameters.mu = mu;
    modelParameters.sigma = sigma;
    modelParameters.V_pca = vPca;
    modelParameters.W_lda = wLda;
    modelParameters.centroids = centroids;
    modelParameters.numClasses = numClasses;
    modelParameters.X_proj = projectedFeatures;
    modelParameters.y = labels;
end
