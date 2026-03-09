function [modelParameters] = positionEstimatorTraining(training_data)
    % Rubbish version which computes rates with EMA and stacks EVERYTHING
    % to compute PCR like ASPMI
    X_train = [];
    Y_train = [];
    
    alpha = 0.2;

    num_movements = size(training_data, 2);
    num_trials = size(training_data, 1);
    
    for movement = 1:num_movements
        for trial = 1:num_trials
            spikes = training_data(trial, movement).spikes;
            pos = training_data(trial, movement).handPos(1:2, :);
          
            rates = filter([alpha], [1, alpha-1], spikes')'; 
           
            X_train = [X_train, rates];
            Y_train = [Y_train, pos];
        end
    end
   
    mean_rates = mean(X_train, 2);
    X_centered = X_train - mean_rates;
    
    % U has PCs
    [U, ~, ~] = svd(X_centered, 'econ');
    
    % Keep top 10 PCs for denoising
    P = U(:, 1:10);
    
    % Map data onto PCs
    X_proj = P' * X_centered;
    
    % Find PCR regression matrix from PC space to outputs
    W = Y_train / X_proj; 
    
    modelParameters.W = W;
    modelParameters.P = P;
    modelParameters.mean_rates = mean_rates;
    modelParameters.alpha = alpha;
end