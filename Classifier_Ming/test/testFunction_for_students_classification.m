function [RMSE, classificationAccuracy] = testFunction_for_students_classification(teamName)
% Combined test for Classifier_Ming/test:
% 1) Direction classification accuracy (PCA+LDA+kNN)
% 2) Continuous trajectory RMSE + decoded/actual trace plot
%
% Usage:
%   RMSE = testFunction_for_students_classification
%   [RMSE, acc] = testFunction_for_students_classification('test')

% Load monkeydata_training.mat from this script's folder (not MATLAB pwd).
if nargin < 1
    teamName = '';
end
here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['Data file not found: %s\n' ...
        'Place monkeydata_training.mat in the same folder as this script (Classifier_Ming/test/), ' ...
        'or update the path in this file, then run run.m again.'], dataFile);
end
load(dataFile, 'trial');
addpath(here);
if nargin >= 1 && ~isempty(teamName) && exist(fullfile(here, teamName), 'dir') == 7
    addpath(fullfile(here, teamName));
end

% Random split (kept same seed as original scripts)
rng(2013);
ix = randperm(size(trial, 1));
trainingData = trial(ix(1:50), :);
testData = trial(ix(51:end), :);

fprintf('Training model...\n');

% Train model (contains both avgTraj and PCA+LDA+kNN classifier)
modelParameters = positionEstimatorTraining(trainingData);
if ~isfield(modelParameters, 'pcaKnnClassifier')
    error('Expected field modelParameters.pcaKnnClassifier was not found after training.');
end
classifierModel = modelParameters.pcaKnnClassifier;

fprintf('Testing classification + trajectory decoding...\n');

% -------------------------------
% 1) Classification accuracy
% -------------------------------
totalPredictions = 0;
correctPredictions = 0;
for tr = 1:size(testData, 1)
    for direc = 1:8
        testSample.spikes = testData(tr, direc).spikes;
        trueLabel = direc;
        pred_dir = positionEstimator_PCA_LDA_K(testSample, classifierModel);
        if pred_dir == trueLabel
            correctPredictions = correctPredictions + 1;
        end
        totalPredictions = totalPredictions + 1;
    end
end
classificationAccuracy = (correctPredictions / totalPredictions) * 100;
fprintf('Classification Accuracy: %.2f%% (%d/%d)\n', classificationAccuracy, correctPredictions, totalPredictions);

% -------------------------------
% 2) Continuous RMSE + trace plot
% -------------------------------
meanSqError = 0;
n_predictions = 0;

figure('Color', 'w');
hold on;
axis square;
grid on;
title('Decoded vs Actual Hand Trajectories');
xlabel('X Position');
ylabel('Y Position');

for tr = 1:size(testData, 1)
    fprintf('Decoding block %d / %d\n', tr, size(testData, 1));
    for direc = randperm(8)
        decodedHandPos = [];
        times = 320:20:size(testData(tr, direc).spikes, 2);

        for t = times
            past_current_trial.trialId = testData(tr, direc).trialId;
            past_current_trial.spikes = testData(tr, direc).spikes(:, 1:t);
            past_current_trial.decodedHandPos = decodedHandPos;
            past_current_trial.startHandPos = testData(tr, direc).handPos(1:2, 1);

            [decodedPosX, decodedPosY] = positionEstimator(past_current_trial, modelParameters);
            decodedPos = [decodedPosX; decodedPosY];
            decodedHandPos = [decodedHandPos decodedPos]; %#ok<AGROW>

            err = testData(tr, direc).handPos(1:2, t) - decodedPos;
            meanSqError = meanSqError + sum(err.^2);
        end

        n_predictions = n_predictions + numel(times);
        plot(decodedHandPos(1, :), decodedHandPos(2, :), 'r');
        plot(testData(tr, direc).handPos(1, times), testData(tr, direc).handPos(2, times), 'b');
    end
end

legend('Decoded Position', 'Actual Position');
RMSE = sqrt(meanSqError / n_predictions);
fprintf('RMSE: %.4f\n', RMSE);

if nargout < 2
    clear classificationAccuracy;
end