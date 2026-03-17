function [x, y] = positionEstimator(test_data, modelParameters)
    % CLASSIFICATION
    persistent trialDir;
    spikes = test_data.spikes; 
    T = size(spikes, 2);
    
    if T <= 320 
        trialDir = []; % Reset for new trial 
    end

    % 1. Direction Classification (First chunk)
    if isempty(trialDir)
        featDir = sum(spikes(:, 1:320), 2)';
        
        cosine_sims = (modelParameters.means * featDir') ./ (vecnorm(modelParameters.means, 2, 2) * norm(featDir));
        
        [~, trialDir] = max(cosine_sims);
    end

    % HOUSE KEEPING

    % Extract the trained PCR parameters
    Bs = modelParameters.B;
    mu_Xs = modelParameters.mu_X;
    V_reduceds = modelParameters.V_reduced;

    % Hyperparameters (Must exactly match the training script)
    t_now = size(test_data.spikes, 2);
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
    X_centered = X_test - mu_Xs{trialDir};
    X_eigen = X_centered * V_reduceds{trialDir};
    
    % Append the bias term (1) to match the B matrix
    X_eigen = [X_eigen, 1];
    
    % --- 4. PREDICT DEVIATION ---
    % Multiply our compressed feature vector by the trained weights matrix
    predicted_vdeviation = X_eigen * Bs{trialDir}; 
    
    % --- 5. ESTIMATE CONTINUOUS POSITION ---
    % Position = Mean_position + Previous Deviation + New Deviation
    traj = modelParameters.avgTraj{trialDir};
    vx_mean = traj(1, t_now) - traj(1, t_now - 20);
    vy_mean = traj(2, t_now) - traj(2, t_now - 20);
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
    x = prev_x + predicted_vdeviation(1) + vx_mean;
    y = prev_y + predicted_vdeviation(2) + vy_mean;
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