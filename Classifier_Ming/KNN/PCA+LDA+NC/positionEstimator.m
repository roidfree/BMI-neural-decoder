function predictedDir = positionEstimator(testSample, modelParameters)
% Standard estimator entrypoint for PCA+LDA+Nearest Centroid.
    predictedDir = positionEstimator_PCA_LDA_NC(testSample, modelParameters);
end
