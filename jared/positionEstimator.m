function [x, y] = positionEstimator(test_data, modelParameters)
    W = modelParameters.W;
    P = modelParameters.P;
    mean_rates = modelParameters.mean_rates;
    alpha = modelParameters.alpha;
    
    % New test data
    spikes = test_data.spikes;
    
    % Apply same EMA filter to test data to compute rates
    filtered_chunk = filter([alpha], [1, alpha-1], spikes')';
    
    % Get CURRENT rate (as only need CURRENT position)
    current_rates = filtered_chunk(:, end);
    
    current_centered = current_rates - mean_rates;
    
    current_proj = P' * current_centered;
    
    prediction = W * current_proj;
   
    x = prediction(1);
    y = prediction(2);
end