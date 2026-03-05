%% testDecoder.m
load('monkeydata0.mat');
training_data = trial; % [cite: 60]

% 1. Train the model
modelParams = positionEstimatorTraining(training_data);

% Variables for RMSE
squared_errors = []; 
figure(1); clf; hold on; grid on;
colors = lines(8);

for m = 1:8 % Reaching angles k=1...8 [cite: 72]
    for t = 91:100 % Use last 10 trials for testing to avoid "cheating"
        test_trial = training_data(t, m);
        decoded_pos = [];
        
        % Reset the persistent variable 'trialDir'
        clear positionEstimator 
        
        % 2. Loop through chunks (Causality Enforcement) [cite: 88]
        for T = 320:20:size(test_trial.spikes, 2)
            input_data.spikes = test_trial.spikes(:, 1:T);
            [x, y] = positionEstimator(input_data, modelParams);
            decoded_pos = [decoded_pos; x, y];
        end
        
        % 3. Extract True Trajectory (300ms start, 20ms steps) [cite: 55, 88]
        % handPos is 3xT in mm; we need 2xT in cm 
        true_pos_mm = test_trial.handPos(1:2, 320:20:end)';
        true_pos_cm = true_pos_mm / 10;
        decoded_cm = decoded_pos / 10;
        
        % 4. Calculate Distance Squared
        % Error is RMSe across both dimensions (X and Y) 
        len = min(size(true_pos_cm, 1), size(decoded_cm, 1));
        if len > 0
            % (x_pred - x_true)^2 + (y_pred - y_true)^2
            dist_sq = sum((decoded_cm(1:len, :) - true_pos_cm(1:len, :)).^2, 2);
            squared_errors = [squared_errors; dist_sq];
        end
        
        % Plot for "Star" visual
        plot(true_pos_cm(:,1), true_pos_cm(:,2), 'b', 'HandleVisibility', 'off');
        plot(decoded_cm(:,1), decoded_cm(:,2), '--', 'Color', colors(m,:));
    end
end

% 5. Final RMSE Calculation
% The mean is taken across both dimensions and all trajectories 
rmse_score = sqrt(mean(squared_errors));

fprintf('\n========================================\n');
fprintf('FINAL COMPETITION RMSE: %.4f cm\n', rmse_score);
fprintf('========================================\n');

title(['Neural Decoder Star Plot - RMSE: ', num2str(rmse_score, '%.2f'), ' cm']);
xlabel('X Position (cm)'); ylabel('Y Position (cm)');
axis equal;