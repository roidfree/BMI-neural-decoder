function [x, y] = positionEstimator(test_data, modelParameters)
% Merge: use LDA to classify direction once per trial, then apply
% direction-conditional PCR regression to predict velocity and integrate.

    persistent trialDir;
    spikes = test_data.spikes;
    T = size(spikes, 2);

    % Reset at start of a new trial (first call uses 320ms window)
    if T <= 320
        trialDir = [];
    end

    % ---- 1) Direction classification (only once per trial) ----
    if isempty(trialDir)
        feat = sum(spikes(:, 1:320), 2)'; % 1 x neurons
        trialDir = lda_predict_class(feat, modelParameters.lda_means, modelParameters.lda_Sigma_inv);
    end

    % ---- 2) Regression feature extraction (match training preprocessing) ----
    bin_width = modelParameters.bin_width;
    history_bins = modelParameters.history_bins;
    neurons = modelParameters.neurons;

    pseudo = struct();
    pseudo(1, 1).spikes = spikes;
    pseudo(1, 1).bin_width = 1;

    pseudo = rebin_data(pseudo, 1, 1, neurons, bin_width);
    pseudo = transform_data(pseudo, 1, 1, "anscombe");
    proc_spikes = pseudo(1, 1).spikes;

    needed = history_bins + 1;
    if size(proc_spikes, 2) >= needed
        recent = proc_spikes(:, end - needed + 1:end);
    else
        pad = zeros(neurons, needed - size(proc_spikes, 2));
        recent = [pad, proc_spikes];
    end
    X_test = reshape(recent, 1, []);

    % PCA projection + bias
    X_centered = X_test - modelParameters.mu_X;
    X_eigen = X_centered * modelParameters.V_reduced;
    X_eigen = [X_eigen, 1];

    % ---- 3) Predict velocity and integrate to position ----
    v = X_eigen * modelParameters.B{trialDir}; % 1x2

    if isempty(test_data.decodedHandPos)
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    else
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
    end

    x = prev_x + v(1);
    y = prev_y + v(2);
end

function cls = lda_predict_class(x, means, Sigma_inv)
% x: 1xd, means: Kxd, Sigma_inv: dxd
% LDA discriminant with equal priors:
% delta_k = x*inv(Sigma)*mu_k' - 0.5*mu_k*inv(Sigma)*mu_k'
    K = size(means, 1);
    scores = zeros(K, 1);
    for k = 1:K
        mu = means(k, :);
        scores(k) = x * Sigma_inv * mu' - 0.5 * (mu * Sigma_inv * mu');
    end
    [~, cls] = max(scores);
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

