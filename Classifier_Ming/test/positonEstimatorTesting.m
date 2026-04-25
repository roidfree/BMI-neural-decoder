function [x, y] = positionEstimator(test_data, modelParameters)
    
end


function [rate_trains] = rates_from_spikes(spike_trains, kernel_width, window_width)
    % Computes the estimated firing rate of all single_units using casual
    % Gaussian filtering
    n_s = -window_width:window_width;
    gauss_kernel = exp(-(n_s).^2 / (2 * kernel_width.^2)) ./ (kernel_width * sqrt(2 * pi));
    gauss_kernel(n_s > 0) = 0;  % making it causal
    rate_trains = conv2(spike_trains, gauss_kernel, "same");
end