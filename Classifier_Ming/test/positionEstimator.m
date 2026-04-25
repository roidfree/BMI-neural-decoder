function [x, y] = positionEstimator(test_data, modelParameters)
    % Match training preprocessing + PCR residual regression.
    persistent trialDir;
    spikes = test_data.spikes;
    t_now = size(spikes, 2);

    % Reset direction cache at the first decode step of each trial.
    if isempty(test_data.decodedHandPos)
        trialDir = [];
    end

    % 1) Direction classification (use PCA+LDA+kNN if available).
    if isempty(trialDir)
        if isfield(modelParameters, 'pcaKnnClassifier')
            trialDir = positionEstimator_PCA_LDA_K(test_data, modelParameters.pcaKnnClassifier);
        else
            featDir = sum(spikes(:, 1:min(320, t_now)), 2)';
            cosineSims = (modelParameters.means * featDir') ./ ...
                (vecnorm(modelParameters.means, 2, 2) * norm(featDir) + eps);
            [~, trialDir] = max(cosineSims);
        end
    end

    % 2) Preprocess exactly as in training (binning/transform/smoothing).
    neurons = size(spikes, 1);
    bin_width = 20;
    transform_name = "anscombe";
    smooth_kernel = "none";
    smooth_width = 0;
    smooth_param = 2;
    if isfield(modelParameters, 'preprocess')
        prep = modelParameters.preprocess;
        if isfield(prep, 'binWidth'), bin_width = prep.binWidth; end
        if isfield(prep, 'transform'), transform_name = string(prep.transform); end
        if isfield(prep, 'smoothKernel'), smooth_kernel = string(prep.smoothKernel); end
        if isfield(prep, 'smoothWidth'), smooth_width = prep.smoothWidth; end
        if isfield(prep, 'smoothParam'), smooth_param = prep.smoothParam; end
    end
    history_bins = 15;
    pseudo_data(1, 1).spikes = spikes; %#ok<AGROW>
    pseudo_data(1, 1).bin_width = 1;
    if smooth_kernel ~= "none" && smooth_width > 0
        pseudo_data(1, 1).spikes = smooth_spikes_matrix(pseudo_data(1, 1).spikes, smooth_kernel, smooth_param, smooth_width);
    end
    pseudo_data = rebin_data(pseudo_data, 1, 1, neurons, bin_width);
    pseudo_data = transform_data(pseudo_data, 1, 1, neurons, transform_name);
    processed_spikes = pseudo_data(1, 1).spikes;

    % Need last (history_bins+1) bins; fallback to avg trajectory if too short.
    nBinsNeed = history_bins + 1;
    if size(processed_spikes, 2) < nBinsNeed
        traj0 = modelParameters.avgTraj{trialDir};
        t0 = min(t_now, size(traj0, 2));
        x = traj0(1, t0);
        y = traj0(2, t0);
        return;
    end
    recent_bins = processed_spikes(:, end - nBinsNeed + 1:end);
    X_test = reshape(recent_bins, 1, []);

    % 3) Regression prediction according to selected method.
    regMethod = "pcr_ridge";
    if isfield(modelParameters, 'regressionMethod')
        regMethod = lower(string(modelParameters.regressionMethod));
    end
    B = modelParameters.B{trialDir};
    switch regMethod
        case "avg_only"
            predicted_vdeviation = [0, 0];
        case "ols"
            X_aug = [X_test, 1];
            predicted_vdeviation = X_aug * B;
        case "pls"
            Bpls = modelParameters.B_pls{trialDir};
            predicted_vdeviation = [1, X_test] * Bpls;
        case {"pcr", "pcr_ridge"}
            mu_X = modelParameters.mu_X{trialDir};
            V_reduced = modelParameters.V_reduced{trialDir};
            X_centered = X_test - mu_X;
            X_eigen = X_centered * V_reduced;
            X_eigen = [X_eigen, 1];
            predicted_vdeviation = X_eigen * B;
        otherwise
            error('Unknown regression method in modelParameters: %s', regMethod);
    end

    % 4) Position update (deviation mode or direct-velocity mode).
    traj = modelParameters.avgTraj{trialDir};
    t1 = min(t_now, size(traj, 2));
    t0 = max(t1 - 20, 1);
    vx_mean = traj(1, t1) - traj(1, t0);
    vy_mean = traj(2, t1) - traj(2, t0);
    targetMode = "deviation";
    if isfield(modelParameters, 'targetMode')
        targetMode = lower(string(modelParameters.targetMode));
    end

    if isempty(test_data.decodedHandPos)
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    else
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
    end

    if targetMode == "velocity"
        if regMethod == "avg_only"
            deltaX = vx_mean;
            deltaY = vy_mean;
        else
            deltaX = predicted_vdeviation(1);
            deltaY = predicted_vdeviation(2);
        end
    else
        deltaX = predicted_vdeviation(1) + vx_mean;
        deltaY = predicted_vdeviation(2) + vy_mean;
    end

    x = prev_x + deltaX;
    y = prev_y + deltaY;
end

function [training_data] = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    if new_bin_width ~= 0
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
end

function [training_data] = transform_data(training_data, trials, movements, ~, transform)
    if transform == "sqrt"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = sqrt(training_data(t, m).spikes);
            end
        end
    elseif transform == "anscombe"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = 2 * sqrt(training_data(t, m).spikes + 3 / 8);
            end
        end
    end
end

function smoothed = smooth_spikes_matrix(spikes, kernel, kernel_param, kernel_width)
    if kernel_width <= 0 || string(kernel) == "none"
        smoothed = spikes;
        return;
    end
    if string(kernel) == "MA"
        ma_kernel = (1 / kernel_width) * ones(1, kernel_width);
        smoothed = filter(ma_kernel, [1], spikes, [], 2);
    elseif string(kernel) == "CGAUSS" || string(kernel) == "gaussian"
        n_s = -kernel_width:kernel_width;
        gauss_kernel = exp(-(n_s).^2 / (2 * kernel_param.^2)) ./ (kernel_param * sqrt(2 * pi));
        if string(kernel) == "CGAUSS"
            gauss_kernel(n_s < 0) = 0;
        end
        smoothed = conv2(spikes, gauss_kernel, "same");
    else
        smoothed = spikes;
    end
end
