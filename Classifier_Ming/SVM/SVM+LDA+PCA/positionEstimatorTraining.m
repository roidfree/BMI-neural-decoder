function modelParameters = positionEstimatorTraining(training_data)
    num_directions = 8;
    time_window = 320;
    features = [];
    labels = [];

    % Extract features
    for trial = 1:size(training_data, 1)
        for direction = 1:num_directions
            spike_data = training_data(trial, direction).spikes;
            firing_rates = sum(spike_data(:, 1:time_window), 2)';
            features = [features; firing_rates];
            labels = [labels; direction];
        end
    end

    % === PCA Reduction ===
    % Center the data
    feature_mean = mean(features, 1);
    centered_features = features - feature_mean;

    % Run PCA
    [coeff, ~, ~, ~, explained] = pca(centered_features);

    % Choose number of components to retain ~95% variance
    cum_var = cumsum(explained);
    num_pcs = 12;
    reduced_features = centered_features * coeff(:, 1:num_pcs);

    % === Train SVMs ===
    C = 0.001; tol = 1e-3; max_passes = 5;
    kernelFunction = @linearKernel;
    svmModels = cell(num_directions, 1);

    for k = 1:num_directions
        binary_labels = double(labels == k);
        svmModels{k} = svmTrain(reduced_features, binary_labels, C, kernelFunction, tol, max_passes);
    end

    % === Train LDA ===
    ldaModel = fitcdiscr(reduced_features, labels);

    % Store everything
    modelParameters = struct();
    modelParameters.svm_models = svmModels;
    modelParameters.lda_model = ldaModel;
    modelParameters.pca_coeff = coeff(:, 1:num_pcs);
    modelParameters.pca_mean = feature_mean;
end
