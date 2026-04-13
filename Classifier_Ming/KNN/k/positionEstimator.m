function predictedDir = positionEstimator(testSample, modelParameters)
% Predict direction using a trained k-NN classifier.
%
% Inputs
%   testSample       : struct with field .spikes (numNeurons x T)
%   modelParameters  : output of positionEstimatorTraining
%
% Output
%   predictedDir     : scalar direction label in 1..8

    classificationHorizon = 320;
    spikes = testSample.spikes;
    endIdx = min(classificationHorizon, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';
    normalizedFeature = (feature - modelParameters.mu) ./ modelParameters.sigma;

    squaredDistances = sum((modelParameters.X - normalizedFeature) .^ 2, 2);
    [~, sortedIdx] = sort(squaredDistances, "ascend");
    nearestLabels = modelParameters.y(sortedIdx(1:modelParameters.k));
    predictedDir = mode(nearestLabels);
end
