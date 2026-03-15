function predicted_direction = positionEstimator(test_data, modelParameters)
    time_window = 360; %orignial 320
    spike_data = test_data.spikes;
    firing_rates = sum(spike_data(:, 1:time_window), 2)';

    scores = zeros(1, 8);
    for k = 1:8
        model = modelParameters.models{k};
        % Instead of binary prediction, we need raw output for confidence
        if strcmp(func2str(model.kernelFunction), 'linearKernel')
            scores(k) = firing_rates * model.w + model.b;
        else
            % For RBF or other kernels: use the confidence score from svmPredict
            % (You'll need to adapt svmPredict to return raw outputs)
            scores(k) = svmPredictRaw(model, firing_rates);  % See note below
        end
    end

    [~, predicted_direction] = max(scores);  % Select class with highest score
end
