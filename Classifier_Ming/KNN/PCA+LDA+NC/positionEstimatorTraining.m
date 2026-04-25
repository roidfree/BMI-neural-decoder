function modelParameters = positionEstimatorTraining(trainingData, varargin)
% Standard training entrypoint for PCA+LDA+Nearest Centroid.
    modelParameters = positionEstimatorTraining_PCA_LDA_NC(trainingData, varargin{:});
end
