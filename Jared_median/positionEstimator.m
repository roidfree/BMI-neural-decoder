function [x, y] = positionEstimator(test_data, modelParameters)
    persistent trialDir;
    T = size(test_data.spikes, 2);
    
    % Reset on new trial
    if T <= 320 
        trialDir = []; 
    end

    % 1. Classify direction at the first possible window
    if isempty(trialDir)
        featDir = sum(test_data.spikes(:, 1:320), 2)';
        cosine_sims = (modelParameters.means * featDir') ./ (vecnorm(modelParameters.means, 2, 2) * norm(featDir));
        [~, trialDir] = max(cosine_sims);
    end

    % Map current time T to the corresponding bin in medianPos
    bin_width = 20;
    % Find which index in modelParameters.time_steps corresponds to T
    [~, bin_idx] = min(abs(modelParameters.time_steps - T));

    % 3. Output the median position for this direction and time
    x = modelParameters.medianPos(trialDir, bin_idx, 1);
    y = modelParameters.medianPos(trialDir, bin_idx, 2);
    
    
end