function [modelParameters] = positionEstimatorTraining(training_data)
% Updated to match Jared's latest: cosine-template direction classification
% + per-direction PCR (mu_X{dir}, V_reduced{dir}, B{dir}).

    modelParameters = struct;
    processed_data = training_data;
    trials = size(training_data, 1);
    movements = size(training_data, 2);
    neurons = size(training_data(1, 1).spikes, 1);

    % ---- CLASSIFICATION ----
    A = [];
    B = [];
    for m = 1:movements
        for t = 1:trials
            spikes = training_data(t, m).spikes;
            feat = sum(spikes(:, 1:320), 2)';
            A = [A; feat]; %#ok<AGROW>
            B = [B; m]; %#ok<AGROW>
        end
    end
    [~, n_features] = size(A);
    class_means = zeros(movements, n_features);
    for k = 1:movements
        class_means(k, :) = mean(A(B == k, :), 1);
    end
    modelParameters.means = class_means;

    % ---- MEAN TRAJECTORY (most stable baseline) ----
    % Always provide modelParameters.avgTraj{m} = [mean_x(t); mean_y(t)]
    if ~isfield(modelParameters, "avgTraj")
        avgTraj = cell(movements, 1);
        for m = 1:movements
            Tmax = 0;
            for t = 1:trials
                Tmax = max(Tmax, size(training_data(t, m).handPos, 2));
            end
    
            sumPos = zeros(2, Tmax);
            count = zeros(1, Tmax);
    
            for t = 1:trials
                pos = training_data(t, m).handPos(1:2, :);
                T_i = size(pos, 2);
                sumPos(:, 1:T_i) = sumPos(:, 1:T_i) + pos;
                count(1:T_i) = count(1:T_i) + 1;
            end
    
            count(count == 0) = 1;
            avgTraj{m} = sumPos ./ repmat(count, 2, 1);
        end
        modelParameters.avgTraj = avgTraj;
    end

    % ---- PREPROCESSING ----
    for t = 1:trials
        for m = 1:movements
            processed_data(t, m).bin_width = 1;
        end
    end
    processed_data = rebin_data(processed_data, trials, movements, neurons, 20);
    processed_data = transform_data(processed_data, trials, movements, neurons, "anscombe");

    % ---- REGRESSION ----
    history_bins = 15;
    bin_width = processed_data(1, 1).bin_width;
    max_iter = floor(571 / bin_width) - history_bins;

    X = zeros(movements, trials * max_iter, neurons * (history_bins + 1));
    Y = zeros(movements, trials * max_iter, 2);
    for m = 1:movements
        counter = 1;
        for t = 1:trials
            for i = 1:max_iter
                X(m, counter, :) = reshape(processed_data(t, m).spikes(:, i:history_bins + i), 1, []);
                Y(m, counter, 1) = processed_data(t, m).handPos(1, (history_bins + i) * bin_width) - processed_data(t, m).handPos(1, (history_bins + i - 1) * bin_width);
                Y(m, counter, 2) = processed_data(t, m).handPos(2, (history_bins + i) * bin_width) - processed_data(t, m).handPos(2, (history_bins + i - 1) * bin_width);
                counter = counter + 1;
            end
        end
    end

    modelParameters.B = cell(movements, 1);
    modelParameters.mu_X = cell(movements, 1);
    modelParameters.V_reduced = cell(movements, 1);

    for m = 1:movements
        Xmov = squeeze(X(m, :, :));
        Ymov = squeeze(Y(m, :, :));
        mu_X = mean(Xmov, 1);
        modelParameters.mu_X{m} = mu_X;
        centred_X = Xmov - mu_X;
        [~, ~, V] = svd(centred_X);
        PCs = 100;
        PCs = min(PCs, size(V, 2));
        V_reduced = V(:, 1:PCs);
        modelParameters.V_reduced{m} = V_reduced;
        eigen_X = centred_X * V_reduced;
        eigen_X = [eigen_X, ones(trials * max_iter, 1)];
        modelParameters.B{m} = eigen_X \ Ymov;
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

function training_data = transform_data(training_data, trials, movements, neurons, transform)
    %#ok<INUSD>
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

