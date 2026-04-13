function modelParameters = positionEstimatorTraining(trainingData, varargin)
% Standard training entrypoint for KNN/PCA+LDA+K.
    modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData, varargin{:});
end

