function [x, y] = positionEstimator(test_data, modelParameters)
    % Extract the trained PCR parameters
    B = modelParameters.B;
    mu_X = modelParameters.mu_X;
    V_reduced = modelParameters.V_reduced;
    
    % Hyperparameters (Must exactly match the training script)
    bin_width = 20;
    history_bins = 15; % Number of lag bins. Total bins used = 16 (320ms)
    neurons = 98;
    
    % Wrap the single test_data trial into a struct to pass through helper functions
    pseudo_data = struct();
    pseudo_data(1, 1).spikes = test_data.spikes;
    pseudo_data(1, 1).bin_width = 1; 
    
    % --- 1. PREPROCESSING ---
    % Pass through the exact pipeline used in training
    pseudo_data = rebin_data(pseudo_data, 1, 1, neurons, bin_width);
    pseudo_data = transform_data(pseudo_data, 1, 1, neurons, "anscombe");
    
    % --- 2. FEATURE EXTRACTION ---
    processed_spikes = pseudo_data(1, 1).spikes;
    num_features_bins = history_bins + 1; 
    
    % Extract the last 16 columns of our processed data
    recent_bins = processed_spikes(:, end - num_features_bins + 1 : end);
    
    % Flatten into a 1D row vector (1 x 1568)
    X_test = reshape(recent_bins, 1, []);
    
    % --- 3. PCA PROJECTION ---
    % Center the data using the training mean, then project down to 500 PCs
    X_centered = X_test - mu_X;
    X_eigen = X_centered * V_reduced;
    
    % Append the bias term (1) to match the B matrix
    X_eigen = [X_eigen, 1];
    
    % --- 4. PREDICT VELOCITY ---
    % Multiply our compressed feature vector by the trained weights matrix
    predicted_velocity = X_eigen * B; 
    
    % --- 5. ESTIMATE CONTINUOUS POSITION ---
    % Position = Previous Position + Predicted Velocity
    if isempty(test_data.decodedHandPos)
        % First prediction at 320ms uses the starting coordinate.
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    else
        % Subsequent predictions grab the last X and Y from decodedHandPos
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
    end
    
    % Output the newly estimated coordinates
    x = prev_x + predicted_velocity(1);
    y = prev_y + predicted_velocity(2);
end


function [training_data] = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    if new_bin_width == 0
    else
        for t = 1:trials
            for m = 1:movements
                unbinned_target = training_data(t, m);
                unbinned_data = unbinned_target.spikes;
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


function [training_data] = transform_data(training_data, trials, movements, neurons, transform)
    if transform == "none"
    else
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
end


function [training_data] = convolve_data(training_data, trials, movements, neurons, kernel, kernel_param, kernel_width)
    if kernel_width == 0
    else
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
            gauss_kernel(n_s < 0) = 0;  
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
end