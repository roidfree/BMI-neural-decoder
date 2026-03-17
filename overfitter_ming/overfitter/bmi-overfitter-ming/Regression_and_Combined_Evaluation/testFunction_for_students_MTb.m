% Test Script to give to the students, March 2015
%% Continuous Position Estimator Test Script
% This function first calls the function "positionEstimatorTraining" to get
% the relevant modelParameters, and then calls the function
% "positionEstimator" to decode the trajectory. 

% RMSE = testFunction_for_students_MTb("overfitter")
% RMSE = testFunction_for_students_MTb("overfitter_ming")
% RMSE = testFunction_for_students_MTb("overfitter_ming_population")
% RMSE = testFunction_for_students_MTb("Test_LDA+K")
% RMSE = testFunction_for_students_MTb("week3_code")
function RMSE = testFunction_for_students_MTb(teamName)

% Use script directory so addpath/load work regardless of current folder
scriptDir = fileparts(mfilename('fullpath'));

% Load data: try script dir, then overfitter subfolder, else current dir
dataPath = fullfile(scriptDir, 'monkeydata0.mat');
if ~exist(dataPath, 'file')
    dataPath = fullfile(scriptDir, 'overfitter', 'monkeydata0.mat');
end
if exist(dataPath, 'file')
    load(dataPath);
else
    load('monkeydata0.mat');
end

% Set random number generator
rng(2013);
ix = randperm(length(trial));

% Add team folder relative to script (so "overfitter", "overfitter_ming" etc. are found)
teamPath = fullfile(scriptDir, teamName);
if ~exist(teamPath, 'dir')
    warning('Team folder not found: %s . Using current path for decoder.', teamPath);
else
    addpath(teamPath);
end

% Select training and testing data (you can choose to split your data in a different way if you wish)
trainingData = trial(ix(1:50),:);
testData = trial(ix(51:end),:);

fprintf('Testing the continuous position estimator...')

meanSqError = 0;
n_predictions = 0;  

figure
hold on
axis square
grid

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

for tr=1:size(testData,1)
    display(['Decoding block ',num2str(tr),' out of ',num2str(size(testData,1))]);
    pause(0.001)
    for direc=randperm(8) 
        decodedHandPos = [];

        times=320:20:size(testData(tr,direc).spikes,2);
        
        for t=times
            past_current_trial.trialId = testData(tr,direc).trialId;
            past_current_trial.spikes = testData(tr,direc).spikes(:,1:t); 
            past_current_trial.decodedHandPos = decodedHandPos;

            past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1); 
            
            if nargout('positionEstimator') == 3
                [decodedPosX, decodedPosY, newParameters] = positionEstimator(past_current_trial, modelParameters);
                modelParameters = newParameters;
            elseif nargout('positionEstimator') == 2
                [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            end
            
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos];
            
            meanSqError = meanSqError + norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
            
        end
        n_predictions = n_predictions+length(times);
        hold on
        plot(decodedHandPos(1,:),decodedHandPos(2,:), 'r');
        plot(testData(tr,direc).handPos(1,times),testData(tr,direc).handPos(2,times),'b')
    end
end

legend('Decoded Position', 'Actual Position')

RMSE = sqrt(meanSqError/n_predictions);

if exist(teamPath, 'dir')
    rmpath(teamPath);
end

end
