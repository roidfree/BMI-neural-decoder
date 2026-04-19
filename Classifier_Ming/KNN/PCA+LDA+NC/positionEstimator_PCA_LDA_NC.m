function predictedDir = positionEstimator_PCA_LDA_NC(testSample, modelParameters)
% Predict direction using PCA + LDA features and nearest class centroid.
%
% Inputs
%   testSample       : struct with field .spikes
%   modelParameters  : output of positionEstimatorTraining_PCA_LDA_NC
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

    C = modelParameters.centroids;
    d2 = sum((C - ldaFeature) .^ 2, 2);
    [~, predictedDir] = min(d2);
end
