function [training_data] = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    % Take .spikes and rebin by counts
    % i.e. [1 1 1 1 1] -> [5] if bin_width 1 -> 5
    for t = 1:trials
        for m = 1:movements
            unbinned_target = training_data(t, m);
            unbinned_data = unbinned_target.spikes;
            unbinned_length = size(unbinned_data, 2);
            binned_data = zeros(neurons, floor(unbinned_length / new_bin_width));  % drop last bin of wrong size as will confuse training
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


function [training_data] = downsample_data(training_data, trials, movements, neurons, downsample_step)
    % Take .spikes and "rebin" by value at start of bin
    % i.e. [1 2 3 4 5] -> [1] if downsample_step == 5
    for t = 1:trials
        for m = 1:movements
            training_data(t, m).spikes = training_data(t, m).spikes(:, 1:downsample_step:end);
            training_data(t, m).bin_width = training_data(t, m).bin_width * downsample_step;
        end
    end
end


function [training_data] = maxpool_data(training_data, trials, movements, neurons, maxpool_bin_width)
    % Take .spikes and rebin by max value in bin (inspired by CNNs)
    % i.e. [1 2 3 2 1] -> [3] if maxpool_bin_width == 5
    for t = 1:trials
        for m = 1:movements
            unbinned_target = training_data(t, m);
            unbinned_data = unbinned_target.spikes;
            unbinned_length = size(unbinned_data, 2);
            binned_data = zeros(neurons, floor(unbinned_length / maxpool_bin_width));  % drop last bin of wrong size as will confuse training
            counter = 1;
            for i = 1:maxpool_bin_width:unbinned_length - (maxpool_bin_width - 1)
                binned_data(:, counter) = max(unbinned_data(:, i:i + (maxpool_bin_width - 1)), [], 2);
                counter = counter + 1;
            end
            training_data(t, m).spikes = binned_data;
            training_data(t, m).bin_width = training_data(t, m).bin_width * maxpool_bin_width;
        end
    end
end


function [training_data] = transform_data(training_data, trials, movements, neurons, transform)
    % Neuron firing can be modelled as a double Poisson point process
    % (Poisson but with also changing mean)
    % Poisson-ity means that the variance scales with the mean
    % Can be bad for regression / dim reduction that assumes homoscedascity
    % Apply transform (sqrt or anscombe) to make variance more independent of mean
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


function [training_data] = convolve_data(training_data, trials, movements, neurons, kernel, kernel_param, kernel_width)
    % Convolve .spikes with a convolution `kernel` (MA, EMA, CGAUSS, AGAUSS)
    % For MA `kernel_param` is irrelevant
    % For EMA `kernel_param` corresponds to alpha
    % For CGAUSS (causal half-gaussian) `kernel_param` corresponds to std
    % For AGAUSS (acausal full-gaussain) `kernel_param` corresponds to std
    if kernel == "MA"
        ma_kernel = (1 / kernel_width) * ones(1, kernel_width);
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = filter(ma_kernel, [1], training_data(t, m).spikes, [], 2);
            end
        end
    end
    if kernel == "EMA"
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = filter([kernel_param], [1, kernel_param - 1], training_data(t, m).spikes, [], 2);
            end
        end
    end
    if kernel == "CGAUSS"
        n_s = -kernel_width:kernel_width;
        gauss_kernel = exp(-(n_s).^2 / (2 * kernel_param.^2)) ./ (kernel_param * sqrt(2 * pi));
        gauss_kernel(n_s < 0) = 0;  % making it causal -- kernel filled during convolution so zero first half
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = conv2(training_data(t, m).spikes, gauss_kernel, "same");
            end
        end
    end
    if kernel == "AGAUSS"
        n_s = -kernel_width:kernel_width;
        gauss_kernel = exp(-(n_s).^2 / (2 * kernel_param.^2)) ./ (kernel_param * sqrt(2 * pi));
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = conv2(training_data(t, m).spikes, gauss_kernel, "same");
            end
        end
    end
end