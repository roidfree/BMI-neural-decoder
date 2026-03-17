function modelParameters = positionEstimatorTraining(training_data)

    [trials, movements] = size(training_data);
    neurons = size(training_data(1, 1).spikes, 1);

    % ---- Hyperparameters (must match positionEstimator.m) ----
    bin_width = 20;        % ms bins for regression features
    history_bins = 15;     % total bins used = history_bins+1 (e.g. 16 => 320ms)
    PCs = 500;             % # principal components for PCR

    % ---- 1) Train direction classifier ----
    X_lda = zeros(trials * movements, neurons);
    Y_lda = zeros(trials * movements, 1);
    c = 1;
    for m = 1:movements
        for t = 1:trials
            spikes = training_data(t, m).spikes;
            X_lda(c, :) = sum(spikes(:, 1:320), 2)';
            Y_lda(c) = m;
            c = c + 1;
        end
    end

    lda_means = zeros(movements, neurons);
    for k = 1:movements
        lda_means(k, :) = mean(X_lda(Y_lda == k, :), 1);
    end
    lambda = 0.01;
    lda_Sigma = cov(X_lda) + lambda * eye(neurons);
    lda_Sigma_inv = inv(lda_Sigma);

    % ---- 2) Train regression model ----
    processed = init_bin_width(training_data);
    processed = rebin_data(processed, trials, movements, neurons, bin_width);
    processed = transform_data(processed, trials, movements, "anscombe");

    max_handpos_len = min_handpos_length(processed, trials, movements);
    max_iter = floor(max_handpos_len / bin_width) - history_bins;
    if max_iter < 1
        error("Not enough data length for chosen bin_width/history_bins.");
    end

    % Build X matrices per direction and a global X for PCA basis
    X_dir = cell(movements, 1);
    Y_dir = cell(movements, 1);
    for m = 1:movements
        X_dir{m} = zeros(trials * max_iter, neurons * (history_bins + 1));
        Y_dir{m} = zeros(trials * max_iter, 2);
    end

    for m = 1:movements
        row = 1;
        for t = 1:trials
            for i = 1:max_iter
                X_dir{m}(row, :) = reshape(processed(t, m).spikes(:, i:history_bins + i), 1, []);
                t2 = (history_bins + i) * bin_width;
                t1 = (history_bins + i - 1) * bin_width;
                Y_dir{m}(row, 1) = processed(t, m).handPos(1, t2) - processed(t, m).handPos(1, t1);
                Y_dir{m}(row, 2) = processed(t, m).handPos(2, t2) - processed(t, m).handPos(2, t1);
                row = row + 1;
            end
        end
    end

    X_all = vertcat(X_dir{:});
    mu_X = mean(X_all, 1);
    centred_X = X_all - mu_X;
    [~, ~, V] = svd(centred_X, "econ");
    PCs = min(PCs, size(V, 2));
    V_reduced = V(:, 1:PCs);

    B = cell(movements, 1);
    for m = 1:movements
        Xc = X_dir{m} - mu_X;
        Xe = Xc * V_reduced;
        Xe = [Xe, ones(size(Xe, 1), 1)];
        B{m} = Xe \ Y_dir{m};
    end

    modelParameters = struct();
    modelParameters.bin_width = bin_width;
    modelParameters.history_bins = history_bins;
    modelParameters.neurons = neurons;

    modelParameters.mu_X = mu_X;
    modelParameters.V_reduced = V_reduced;
    modelParameters.B = B;

    modelParameters.lda_means = lda_means;
    modelParameters.lda_Sigma_inv = lda_Sigma_inv;
end

function training_data = init_bin_width(training_data)
    [trials, movements] = size(training_data);
    for t = 1:trials
        for m = 1:movements
            training_data(t, m).bin_width = 1;
        end
    end
end

function L = min_handpos_length(training_data, trials, movements)
    L = inf;
    for t = 1:trials
        for m = 1:movements
            L = min(L, size(training_data(t, m).handPos, 2));
        end
    end
end

function training_data = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    if new_bin_width == 0
        return;
    end
    for t = 1:trials
        for m = 1:movements
            unbinned = training_data(t, m).spikes;
            unbinned_length = size(unbinned, 2);
            binned = zeros(neurons, floor(unbinned_length / new_bin_width));
            counter = 1;
            for i = 1:new_bin_width:unbinned_length - (new_bin_width - 1)
                binned(:, counter) = sum(unbinned(:, i:i + (new_bin_width - 1)), 2);
                counter = counter + 1;
            end
            training_data(t, m).spikes = binned;
            training_data(t, m).bin_width = training_data(t, m).bin_width * new_bin_width;
        end
    end
end

function training_data = transform_data(training_data, trials, movements, transform)
    if transform == "none"
        return;
    end
    if transform == "sqrt"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = sqrt(training_data(t, m).spikes);
            end
        end
        return;
    end
    if transform == "anscombe"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = 2 * sqrt(training_data(t, m).spikes + 3 / 8);
            end
        end
        return;
    end
end

