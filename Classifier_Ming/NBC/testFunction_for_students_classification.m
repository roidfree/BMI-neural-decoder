function classificationAccuracy = testFunction_for_students_classification(teamName)
% Load monkeydata_training.mat from this script's folder (not from MATLAB pwd).
if nargin < 1
    teamName = '';
end
here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['Data file not found: %s\n' ...
        'Place monkeydata_training.mat in the same folder as this script (Classification/NBC/), ' ...
        'or update the path in this file, then run run.m again.'], dataFile);
end
load(dataFile, 'trial');
addpath(here);
if nargin >= 1 && ~isempty(teamName) && exist(fullfile(here, teamName), 'dir') == 7
    addpath(fullfile(here, teamName));
end

% Randomly Split Data
rng(2013);
ix = randperm(size(trial, 1));
trainingData = trial(ix(1:50), :);
testData = trial(ix(51:end), :);

fprintf('Training movement direction classifier...\n');

% Train Model
modelParameters = positionEstimatorTraining(trainingData);

fprintf('Testing movement direction classifier...\n');

% Evaluate Accuracy
totalPredictions = 0;
correctPredictions = 0;

for tr = 1:size(testData, 1)
    for direc = 1:8
        % Create a test sample
        testSample.spikes = testData(tr, direc).spikes;
        
        % Get true label
        trueLabel = direc;

        % Predict movement direction
        pred_dir = positionEstimator(testSample, modelParameters);

        % Check accuracy
        if pred_dir == trueLabel
            correctPredictions = correctPredictions + 1;
        end
        totalPredictions = totalPredictions + 1;
    end
end

% Compute Accuracy
classificationAccuracy = (correctPredictions / totalPredictions) * 100;
fprintf('Total Predictions: %d\n', totalPredictions);
fprintf('Correct Predictions: %d\n', correctPredictions);
fprintf('Classification Accuracy: %.2f%%\n', classificationAccuracy);

end

%testFunction_for_students_classification2('overfitter');