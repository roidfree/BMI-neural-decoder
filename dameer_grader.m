function results = dameer_grader(K, pc_grid)
    if nargin < 1 || isempty(K)
        K = 5;
    end
    if nargin < 2 || isempty(pc_grid)
        pc_grid = [50, 100, 200, 300, 400, 500];
    end

    data = load('monkeydata_training.mat');
    trial = data.trial;

    rng(42);

    num_trials = size(trial, 1);
    num_directions = size(trial, 2);
    shuffled_indices = randperm(num_trials);
    fold_size = floor(num_trials / K);

    param_list = pc_grid(:);
    num_configs = numel(param_list);

    results = struct('nPC', {}, 'foldRMSE', {}, 'meanRMSE', {}, 'stdRMSE', {});

    fprintf('=============================================\n');
    fprintf('NEURAL DECODER PARAMETER SWEEP (K-FOLD CV)\n');
    fprintf('Date: %s\n', datestr(now));
    fprintf('Trials: %d, Directions: %d, Folds: %d\n', num_trials, num_directions, K);
    fprintf('=============================================\n\n');

    fprintf('Grid size: %d configs (%d PC counts)\n\n', ...
        num_configs, numel(pc_grid));

    for cfg = 1:num_configs
        nPC = param_list(cfg);
        fold_RMSE = zeros(1, K);

        fprintf('--------------------------------------------------\n');
        fprintf('Config %2d / %2d  |  nPC = %d\n', cfg, num_configs, nPC);
        fprintf('--------------------------------------------------\n');

        for fold = 1:K
            test_start = (fold - 1) * fold_size + 1;
            test_end   = fold * fold_size;

            test_idx  = shuffled_indices(test_start:test_end);
            train_idx = shuffled_indices([1:test_start-1, test_end+1:end]);

            trainData = trial(train_idx, :);
            testData  = trial(test_idx, :);
            modelParameters = trainPCRDecoder(trainData, nPC);

            meanSqError = 0;
            n_predictions = 0;

            for tr = 1:size(testData, 1)
                for direc = 1:num_directions
                    decodedHandPos = [];
                    times = 320:20:size(testData(tr, direc).spikes, 2);

                    for t = times
                        past_current_trial.trialId = testData(tr, direc).trialId;
                        past_current_trial.spikes = testData(tr, direc).spikes(:, 1:t);
                        past_current_trial.decodedHandPos = decodedHandPos;
                        past_current_trial.startHandPos = testData(tr, direc).handPos(1:2, 1);

                        [decodedPosX, decodedPosY] = predictPCRDecoder(past_current_trial, modelParameters);

                        decodedPos = [decodedPosX; decodedPosY];
                        decodedHandPos = [decodedHandPos, decodedPos];

                        actualPos = testData(tr, direc).handPos(1:2, t);
                        meanSqError = meanSqError + norm(actualPos - decodedPos)^2;
                    end

                    n_predictions = n_predictions + length(times);
                end
            end

            fold_RMSE(fold) = sqrt(meanSqError / n_predictions);
            fprintf('  Fold %d RMSE: %.4f mm\n', fold, fold_RMSE(fold));
        end

        results(cfg).nPC = nPC;
        results(cfg).foldRMSE = fold_RMSE;
        results(cfg).meanRMSE = mean(fold_RMSE);
        results(cfg).stdRMSE = std(fold_RMSE);

        fprintf('  => Mean RMSE: %.4f mm (std: %.4f)\n\n', ...
            results(cfg).meanRMSE, results(cfg).stdRMSE);
    end

    mean_rmse_all = arrayfun(@(r) r.meanRMSE, results);
    [~, rank_idx] = sort(mean_rmse_all, 'ascend');
    best = results(rank_idx(1));

    fprintf('\n=============================================\n');
    fprintf('RANKED SUMMARY (LOWER RMSE IS BETTER)\n');
    fprintf('=============================================\n');
    fprintf('Rank | nPC | Mean RMSE (mm) | Std (mm)\n');
    fprintf('---------------------------------------\n');

    for r = 1:num_configs
        id = rank_idx(r);
        fprintf('%4d | %3d | %14.4f | %8.4f\n', ...
            r, results(id).nPC, results(id).meanRMSE, results(id).stdRMSE);
    end
    fprintf('---------------------------------------\n');
    fprintf('BEST CONFIG -> nPC = %d\n', best.nPC);
    fprintf('Best Mean RMSE = %.4f mm (std: %.4f)\n', best.meanRMSE, best.stdRMSE);
    fprintf('=============================================\n');
end

function modelParameters = trainPCRDecoder(training_data, nPC)
    trials = size(training_data, 1);
    movements = size(training_data, 2);
    neurons = 98;
    bin_width = 20;
    history_bins = 15;

    processed_data = training_data;
    for t = 1:trials
        for m = 1:movements
            processed_data(t, m).bin_width = 1;
        end
    end

    processed_data = rebin_data(processed_data, trials, movements, neurons, bin_width);
    processed_data = transform_data(processed_data, trials, movements, "anscombe");

    max_iter = floor(571 / bin_width) - history_bins;
    X = zeros(trials * movements * max_iter, neurons * (history_bins + 1));
    Y = zeros(trials * movements * max_iter, 2);

    counter = 1;
    for t = 1:trials
        for m = 1:movements
            for i = 1:max_iter
                X(counter, :) = reshape( ...
                    processed_data(t, m).spikes(:, i:history_bins + i), 1, []);
                Y(counter, 1) = training_data(t, m).handPos(1, (history_bins + i) * bin_width) ...
                    - training_data(t, m).handPos(1, (history_bins + i - 1) * bin_width);
                Y(counter, 2) = training_data(t, m).handPos(2, (history_bins + i) * bin_width) ...
                    - training_data(t, m).handPos(2, (history_bins + i - 1) * bin_width);
                counter = counter + 1;
            end
        end
    end

    mu_X = mean(X, 1);
    centred_X = X - mu_X;
    [~, ~, V] = svd(centred_X, 'econ');
    nPC = min(nPC, size(V, 2));
    V_reduced = V(:, 1:nPC);
    eigen_X = centred_X * V_reduced;
    eigen_X = [eigen_X, ones(size(eigen_X, 1), 1)];
    B = eigen_X \ Y;

    modelParameters.B = B;
    modelParameters.mu_X = mu_X;
    modelParameters.V_reduced = V_reduced;
    modelParameters.bin_width = bin_width;
    modelParameters.history_bins = history_bins;
end

function [x, y] = predictPCRDecoder(test_data, modelParameters)
    B = modelParameters.B;
    mu_X = modelParameters.mu_X;
    V_reduced = modelParameters.V_reduced;
    bin_width = modelParameters.bin_width;
    history_bins = modelParameters.history_bins;
    neurons = 98;

    spikes = test_data.spikes;
    unbinned_length = size(spikes, 2);
    num_bins = floor(unbinned_length / bin_width);
    binned_data = zeros(neurons, num_bins);

    counter = 1;
    for i = 1:bin_width:unbinned_length - (bin_width - 1)
        binned_data(:, counter) = sum(spikes(:, i:i + (bin_width - 1)), 2);
        counter = counter + 1;
    end

    binned_data = 2 * sqrt(binned_data + 3 / 8);
    recent_bins = binned_data(:, end - history_bins : end);
    X_test = reshape(recent_bins, 1, []);
    X_centered = X_test - mu_X;
    X_eigen = X_centered * V_reduced;
    X_eigen = [X_eigen, 1];

    predicted_velocity = X_eigen * B;

    if isempty(test_data.decodedHandPos)
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    else
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
    end

    x = prev_x + predicted_velocity(1);
    y = prev_y + predicted_velocity(2);
end

function training_data = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    for t = 1:trials
        for m = 1:movements
            unbinned_data = training_data(t, m).spikes;
            unbinned_length = size(unbinned_data, 2);
            binned_data = zeros(neurons, floor(unbinned_length / new_bin_width));
            counter = 1;
            for i = 1:new_bin_width:unbinned_length - (new_bin_width - 1)
                binned_data(:, counter) = sum(unbinned_data(:, i:i + (new_bin_width - 1)), 2);
                counter = counter + 1;
            end
            training_data(t, m).spikes = binned_data;
            training_data(t, m).bin_width = training_data(t, m).bin_width * new_bin_width;
        end
    end
end

function training_data = transform_data(training_data, trials, movements, transform)
    if transform == "sqrt"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = sqrt(training_data(t, m).spikes);
            end
        end
    end
    if transform == "anscombe"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = 2 * sqrt(training_data(t, m).spikes + 3 / 8);
            end
        end
    end
end
