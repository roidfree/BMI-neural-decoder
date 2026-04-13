function predictedDir = positionEstimator(testSample, modelParameters)
% Standard estimator entrypoint for KNN/K+LDA.
    predictedDir = positionEstimator_LDA(testSample, modelParameters);
end

