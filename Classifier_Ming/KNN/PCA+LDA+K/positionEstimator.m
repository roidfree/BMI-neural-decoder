function predictedDir = positionEstimator(testSample, modelParameters)
% Standard estimator entrypoint for KNN/PCA+LDA+K.
    predictedDir = positionEstimator_PCA_LDA_K(testSample, modelParameters);
end

