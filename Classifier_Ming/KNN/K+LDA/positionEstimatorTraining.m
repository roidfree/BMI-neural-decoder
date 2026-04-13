function modelParameters = positionEstimatorTraining(trainingData, varargin)
% Standard training entrypoint for KNN/K+LDA.
    modelParameters = positionEstimatorTraining_LDA(trainingData, varargin{:});
end

