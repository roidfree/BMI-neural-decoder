function predictedDir = positionEstimator_LDA(testSample, modelParameters)
% Predict direction using LDA features and k-NN.
%
% Inputs
%   testSample       : struct with field .spikes
%   modelParameters  : output of positionEstimatorTraining_LDA
%
% Output
%   predictedDir     : scalar direction label in 1..8

    classificationHorizon = 320;
    spikes = testSample.spikes;
    endIdx = min(classificationHorizon, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';

    normalizedFeature = (feature - modelParameters.mu) ./ modelParameters.sigma;
    ldaFeature = normalizedFeature * modelParameters.W;

    squaredDistances = sum((modelParameters.X_lda - ldaFeature) .^ 2, 2);
    [~, sortedIdx] = sort(squaredDistances, "ascend");
    nearestLabels = modelParameters.y(sortedIdx(1:modelParameters.k));
    predictedDir = mode(nearestLabels);
end
