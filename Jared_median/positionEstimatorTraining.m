function [modelParameters] = positionEstimatorTraining(training_data)
    modelParameters = struct;
    trials = size(training_data, 1);
    movements = size(training_data, 2);
    bin_width = 20;
    
    % --- 1. DIRECTION CLASSIFICATION ---
    A = []; B = [];
    for m = 1:movements
        for t = 1:trials
            feat = sum(training_data(t,m).spikes(:,1:320),2)';
            A = [A; feat];
            B = [B; m];
        end
    end
    modelParameters.means = zeros(movements, size(A, 2));
    for k = 1:movements
        modelParameters.means(k,:) = mean(A(B==k,:), 1);
    end

    % --- 2. MEDIAN POSITION TRAJECTORY ---
    % define a fixed timeline based on the shortest trial (x ms)
    
    max_time = 560; 
    time_steps = 20:bin_width:max_time;
    num_bins = length(time_steps);
    
    % Store: [Movement x TimeBin x 2 (X and Y)]
    modelParameters.medianPos = zeros(movements, num_bins, 2);

    for m = 1:movements
        for b = 1:num_bins
            currentTime = time_steps(b);
            all_x = zeros(trials, 1);
            all_y = zeros(trials, 1);
            
            for t = 1:trials
                all_x(t) = training_data(t,m).handPos(1, currentTime);
                all_y(t) = training_data(t,m).handPos(2, currentTime);
            end
            
            % Calculate median position at this specific timestamp
            modelParameters.medianPos(m, b, 1) = median(all_x);
            modelParameters.medianPos(m, b, 2) = median(all_y);
        end
    end
    modelParameters.time_steps = time_steps;
end