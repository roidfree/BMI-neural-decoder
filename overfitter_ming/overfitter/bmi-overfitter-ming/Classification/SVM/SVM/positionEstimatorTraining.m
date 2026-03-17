function modelParameters = positionEstimatorTraining(training_data)
    num_directions = 8;
    time_window = 320;
    features = [];
    labels = [];

    % Extract features and labels
    for trial = 1:size(training_data, 1)
        for direction = 1:num_directions
            spike_data = training_data(trial, direction).spikes;
            firing_rates = sum(spike_data(:, 1:time_window), 2)';
            features = [features; firing_rates];
            labels = [labels; direction];
        end
    end

    % One-vs-all SVM classifiers
    C = 1; tol = 1e-3; max_passes = 5;
    kernelFunction = @linearKernel;  % Change to @gaussianKernel if needed

    models = cell(num_directions, 1);
    for k = 1:num_directions
        binary_labels = double(labels == k); % 1 if current class, 0 otherwise
        models{k} = svmTrain(features, binary_labels, C, kernelFunction, tol, max_passes);
    end

    modelParameters = struct();
    modelParameters.models = models;
end
