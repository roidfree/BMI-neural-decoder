clear; close all; clc;

load('monkeydata_training.mat');

% -------------------------------------------------------------
% Model parameter sweep + K-fold validation
% -------------------------------------------------------------
K = 5;
rng(42);

num_trials = size(trial, 1);
num_directions = size(trial, 2);

fprintf('=============================================\n');
fprintf('NEURAL DECODER PARAMETER SWEEP (K-FOLD CV)\n');
fprintf('Date: %s\n', datestr(now));
fprintf('Trials: %d, Directions: %d, Folds: %d\n', num_trials, num_directions, K);
fprintf('=============================================\n\n');

% Hyperparameters to sweep
alpha_grid = [0.10, 0.20, 0.35, 0.50];
pc_grid    = [6, 10, 14, 20];

[A, P] = ndgrid(alpha_grid, pc_grid);
param_list = [A(:), P(:)];
num_configs = size(param_list, 1);

fprintf('Grid size: %d configs (%d alphas x %d PC counts)\n\n', ...
    num_configs, numel(alpha_grid), numel(pc_grid));

% Keep the folds fixed across all configs so comparison is fair
shuffled_indices = randperm(num_trials);
fold_size = floor(num_trials / K);

results = struct('alpha', {}, 'nPC', {}, 'foldRMSE', {}, 'meanRMSE', {}, 'stdRMSE', {});

for cfg = 1:num_configs
    alpha = param_list(cfg, 1);
    nPC = param_list(cfg, 2);

    fprintf('--------------------------------------------------\n');
    fprintf('Config %2d / %2d  |  alpha = %.2f,  nPC = %d\n', cfg, num_configs, alpha, nPC);
    fprintf('--------------------------------------------------\n');

    fold_RMSE = zeros(1, K);

    for fold = 1:K
        test_start = (fold - 1) * fold_size + 1;
        test_end   = fold * fold_size;

        test_idx  = shuffled_indices(test_start:test_end);
        train_idx = shuffled_indices([1:test_start-1, test_end+1:end]);

        trainData = trial(train_idx, :);
        testData  = trial(test_idx, :);

        modelParameters = trainPCRDecoder(trainData, alpha, nPC);

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

    results(cfg).alpha = alpha;
    results(cfg).nPC = nPC;
    results(cfg).foldRMSE = fold_RMSE;
    results(cfg).meanRMSE = mean(fold_RMSE);
    results(cfg).stdRMSE = std(fold_RMSE);

    fprintf('  => Mean RMSE: %.4f mm (std: %.4f)\n\n', results(cfg).meanRMSE, results(cfg).stdRMSE);
end

% --------------------------
% Ranked summary
% --------------------------
mean_rmse_all = arrayfun(@(r) r.meanRMSE, results);
[~, rank_idx] = sort(mean_rmse_all, 'ascend');

fprintf('\n=============================================\n');
fprintf('RANKED SUMMARY (LOWER RMSE IS BETTER)\n');
fprintf('=============================================\n');
fprintf('Rank | alpha | nPC | Mean RMSE (mm) | Std (mm)\n');
fprintf('----------------------------------------------\n');

for r = 1:num_configs
    id = rank_idx(r);
    fprintf('%4d | %5.2f | %3d | %14.4f | %8.4f\n', ...
        r, results(id).alpha, results(id).nPC, results(id).meanRMSE, results(id).stdRMSE);
end

best = results(rank_idx(1));
fprintf('----------------------------------------------\n');
fprintf('BEST CONFIG -> alpha = %.2f, nPC = %d\n', best.alpha, best.nPC);
fprintf('Best Mean RMSE = %.4f mm (std: %.4f)\n', best.meanRMSE, best.stdRMSE);
fprintf('=============================================\n');

% Optional visual: mean RMSE for each config in ranked order
figure;
bar(mean_rmse_all(rank_idx));
hold on;
yline(best.meanRMSE, 'r--', 'LineWidth', 1.5, 'Label', 'Best');
xlabel('Config Rank');
ylabel('Mean RMSE (mm)');
title('Grid Search Results (Ranked by Mean CV RMSE)');
grid on;


% =============================================================
% Local helper functions
% =============================================================
function modelParameters = trainPCRDecoder(training_data, alpha, nPC)
    %#ok<INUSD>
    modelParameters = positionEstimatorTraining(training_data);
end

function [x, y] = predictPCRDecoder(test_data, modelParameters)
    [x, y] = positionEstimator(test_data, modelParameters);
end
