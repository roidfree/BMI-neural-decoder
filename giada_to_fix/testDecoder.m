%% testDecoder.m
% Same behaviour as testFunction_for_students_MTb

load('monkeydata0.mat')

% Set random seed (same as official script)
rng(2013);

% Random permutation of trials
ix = randperm(length(trial));

% Split training / testing
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...\n')

meanSqError = 0;
n_predictions = 0;

figure
hold on
axis square
grid

%% Train model
modelParameters = positionEstimatorTraining(trainingData);

for tr = 1:size(testData,1)

    disp(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);

    for direc = randperm(8)

        decodedHandPos = [];

        times = 320:20:size(testData(tr,direc).spikes,2);

        for t = times

            past_current_trial.trialId = testData(tr,direc).trialId;

            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);

            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

            % Call your decoder
            [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);

            decodedPos = [decodedPosX; decodedPosY];

            decodedHandPos = [decodedHandPos decodedPos];

            % Compute squared error
            meanSqError = meanSqError + ...
                norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;

        end

        n_predictions = n_predictions + length(times);

        % Plot trajectories
        plot(decodedHandPos(1,:),decodedHandPos(2,:),'r')
        plot(testData(tr,direc).handPos(1,times), ...
             testData(tr,direc).handPos(2,times),'b')

    end
end

legend('Decoded Position','Actual Position')

RMSE = sqrt(meanSqError/n_predictions);

fprintf('\n========================================\n');
fprintf('FINAL COMPETITION RMSE: %.4f\n',RMSE);
fprintf('========================================\n');
