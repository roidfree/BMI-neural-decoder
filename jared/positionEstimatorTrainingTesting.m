function [modelParameters] = positionEstimatorTrainingTesting(training_data)
    trial_rates = struct;
    trials = size(training_data, 1);
    movements = size(training_data, 2);
    neurons = 98;
    kernel_width = 10;
    window_width = kernel_width * 3;

    % Get rates via kernel smoothing
    for t = 1:trials
        for m = 1:movements
            trial_rates(t, m).rates = rates_from_spikes(training_data(t, m).spikes, kernel_width, window_width, 1);
            trial_rates(t, m).handPos = training_data(t, m).handPos;
            trial_rates(t, m).bin_width = 1;
        end
    end

    % Downsample rates if desired
    downsampled_trial_rates = downsample_rates(trial_rates, 1, trials, movements);

end


function [rate_trains] = rates_from_spikes(spike_trains, kernel_width, window_width, causal)
    % Computes the estimated firing rate of all single_units using (casual) Gaussian filtering
    % 'causal' = 0 for acausal and 'causal' = 1 for causal
    n_s = -window_width:window_width;
    gauss_kernel = exp(-(n_s).^2 / (2 * kernel_width.^2)) ./ (kernel_width * sqrt(2 * pi));
    if causal == 1
        gauss_kernel(n_s > 0) = 0;  % making it causal
    end
    rate_trains = conv2(spike_trains, gauss_kernel, "same");
end


function [downsampled_trial_rates] = downsample_rates(trial_rates, bin_width, trials, movements)
    % Takes trial_rates struct and gives new trial_rates struct with
    % downsampled rates and updated bin_width
    downsampled_trial_rates = struct;
    for t = 1:trials
        for m = 1:movements
            rate_train = trial_rates(t, m).rates;
            downsampled_rate_train = rate_train(:, 1:bin_width:end);
            downsampled_trial_rates(t, m).rates = downsampled_rate_train;
            downsampled_trial_rates(t, m).bin_width = bin_width;
            downsampled_trial_rates(t, m).handPos = trial_rates(t, m).handPos;
        end
    end
end