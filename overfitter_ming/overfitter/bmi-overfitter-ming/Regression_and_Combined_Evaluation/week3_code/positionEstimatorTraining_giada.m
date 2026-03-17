function modelParameters = positionEstimatorTraining(training_data)

    % Get dataset dimensions
    [trials, movements] = size(training_data);

    % Initialise feature and label matrices
    X = [];
    Y = [];

    % Loop through all directions and trials
    for m = 1:movements
        for t = 1:trials

            % Extract spike data
            spikes = training_data(t,m).spikes;

            % Feature: spike counts in first 320 ms
            feat = sum(spikes(:,1:320),2)';

            % Store feature and corresponding label
            X = [X; feat];
            Y = [Y; m];

        end
    end

    % Get dimensions
    [n_samples, n_features] = size(X);

    % Compute class means (one per direction)
    class_means = zeros(movements, n_features);

    for k = 1:movements
        class_means(k,:) = mean(X(Y==k,:), 1);
    end

    % Compute regularised covariance matrix
    lambda = 0.01;
    Sigma = cov(X) + lambda * eye(n_features);

    % Invert covariance matrix
    Sigma_inv = inv(Sigma);

    % Store model parameters
    modelParameters.means = class_means;
    modelParameters.Sigma_inv = Sigma_inv;

end