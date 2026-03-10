function [x, y] = positionEstimator(test_data, modelParameters)
    % Extract the trained PCR parameters
    B = modelParameters.B;
    mu_X = modelParameters.mu_X;
    V_reduced = modelParameters.V_reduced;
    
    % Hyperparameters (Must exactly match the training script!)
    bin_width = 20;
    history_bins = 15; % Number of lag bins. Total bins used = history_bins + 1 (16 bins = 320ms)
    neurons = 98;
    
    % The raw spike data from t=1 to the current time step t
    spikes = test_data.spikes;
    
    % --- 1. REBINNING ---
    % Sum spikes in 20ms windows exactly like rebin_data()
    unbinned_length = size(spikes, 2);
    num_bins = floor(unbinned_length / bin_width);
    binned_data = zeros(neurons, num_bins);
    
    counter = 1;
    for i = 1:bin_width:unbinned_length - (bin_width - 1)
        binned_data(:, counter) = sum(spikes(:, i:i + (bin_width - 1)), 2);
        counter = counter + 1;
    end
    
    % --- 2. VARIANCE STABILIZATION ---
    % Apply Anscombe transform exactly like transform_data()
    binned_data = 2 * sqrt(binned_data + 3/8);
    
    % --- 3. FEATURE EXTRACTION ---
    % We need the most recent 320ms of data (16 bins)
    num_features_bins = history_bins + 1; 
    
    % Extract the last 16 columns of our binned data
    recent_bins = binned_data(:, end - num_features_bins + 1 : end);
    
    % Flatten into a 1D row vector (1 x 1568)
    X_test = reshape(recent_bins, 1, []);
    
    % --- 4. PCA PROJECTION ---
    % Center the data using the training mean, then project
    X_centered = X_test - mu_X;
    X_eigen = X_centered * V_reduced;
    
    % Append the bias term (1) to match the B matrix
    X_eigen = [X_eigen, 1];
    
    % --- 5. PREDICT VELOCITY ---
    % Multiply our compressed feature vector by the trained weights matrix
    % predicted_velocity will be a 1x2 vector: [vel_x, vel_y]
    predicted_velocity = X_eigen * B; 
    
    % --- 6. ESTIMATE CONTINUOUS POSITION ---
    % Position = Previous Position + Predicted Velocity
    if isempty(test_data.decodedHandPos)
        % First prediction at 320ms: Hand was assumed stationary for first 300ms,
        % so the "previous" position is just the starting coordinate.
        prev_x = test_data.startHandPos(1);
        prev_y = test_data.startHandPos(2);
    else
        % Subsequent predictions: Grab the last X and Y from decodedHandPos
        prev_x = test_data.decodedHandPos(1, end);
        prev_y = test_data.decodedHandPos(2, end);
    end
    
    % Output the newly estimated coordinates
    x = prev_x + predicted_velocity(1);
    y = prev_y + predicted_velocity(2);
end