function modelParameters = positionEstimatorTraining(training_data)
    num_directions = 8;
    time_window = 320;
    features = [];
    labels = [];

    for trial = 1:size(training_data, 1)
        for direction = 1:num_directions
            spike_data = training_data(trial, direction).spikes;
            firing_rates = sum(spike_data(:, 1:time_window), 2)';
            features = [features; firing_rates];
            labels = [labels; direction];
        end
    end

    % Train 1-vs-all SVMs
    C = 1; tol = 1e-3; max_passes = 5;
    kernelFunction = @linearKernel;
    svmModels = cell(num_directions, 1);

    for k = 1:num_directions
        binary_labels = double(labels == k); % one-vs-all
        svmModels{k} = svmTrain(features, binary_labels, C, kernelFunction, tol, max_passes);
    end

    % Train LDA
    ldaModel = fitcdiscr(features, labels);

    % Store both
    modelParameters = struct();
    modelParameters.svm_models = svmModels;
    modelParameters.lda_model = ldaModel;
end
