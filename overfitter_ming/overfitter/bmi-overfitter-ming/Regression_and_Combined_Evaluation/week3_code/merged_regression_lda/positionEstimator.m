function [x, y] = positionEstimator(test_data, modelParameters)
% Updated to match Jared's latest: cosine-template direction classification
% + mean trajectory decoding (most stable baseline).

    persistent trialDir;
    spikes = test_data.spikes;
    T = size(spikes, 2);

    % Reset at start of a new trial (first call uses 320ms window)
    if T <= 320
        trialDir = [];
    end

    % ---- 1) Direction classification (only once per trial) ----
    if isempty(trialDir)
        featDir = sum(spikes(:, 1:320), 2)';
        cosine_sims = (modelParameters.means * featDir') ./ (vecnorm(modelParameters.means, 2, 2) * norm(featDir));
        [~, trialDir] = max(cosine_sims);
    end

    % ---- 2) Regression feature extraction (match training preprocessing) ----
    t_now = size(test_data.spikes, 2);
    traj = modelParameters.avgTraj{trialDir};
    t_idx = min(t_now, size(traj, 2));
    x = traj(1, t_idx);
    y = traj(2, t_idx);
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


