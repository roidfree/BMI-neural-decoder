function predictedDir = positionEstimator_PCA_LDA_K(testSample, modelParameters)
% Predict direction using PCA + LDA features and k-NN.
%
% Inputs
%   testSample       : struct with field .spikes
%   modelParameters  : output of positionEstimatorTraining_PCA_LDA_K
%
% Output
%   predictedDir     : scalar direction label in 1..8

    classificationHorizon = 320;
    spikes = testSample.spikes;
    endIdx = min(classificationHorizon, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';

    normalizedFeature = (feature - modelParameters.mu) ./ modelParameters.sigma;
    pcaFeature = normalizedFeature * modelParameters.V_pca;
    ldaFeature = pcaFeature * modelParameters.W_lda;

    squaredDistances = sum((modelParameters.X_proj - ldaFeature) .^ 2, 2);
    [~, sortedIdx] = sort(squaredDistances, "ascend");
    nearestLabels = modelParameters.y(sortedIdx(1:modelParameters.k));
    predictedDir = mode(nearestLabels);
end
