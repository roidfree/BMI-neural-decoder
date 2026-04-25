function resultsTable = run_regression_sweep_gaussian20_with_my_classifier(teamName, targetMode, regressionMethods)
% Sweep regression methods with fixed preprocessing and classifier setup.
%
% Fixed settings:
%   - Preprocessing: Gaussian smoothing (sigma = 20), binWidth = 20, transform = none
%   - Classifier: PCA+LDA+kNN (k=7, PCA=30, LDA=5) from positionEstimatorTraining.m
%
% Output table format matches run_preprocessing_sweep_with_my_classifier.m.
%
% Usage:
%   T = run_regression_sweep_gaussian20_with_my_classifier
%   T = run_regression_sweep_gaussian20_with_my_classifier('test', 'velocity', {'pcr_ridge','pls'})

    if nargin < 1
        teamName = '';
    end
    if nargin < 2 || isempty(targetMode)
        targetMode = 'velocity';
    end
    if nargin < 3 || isempty(regressionMethods)
        regressionMethods = {'avg_only', 'ols', 'pcr', 'pcr_ridge', 'pls'};
    end
    if ischar(regressionMethods) || isstring(regressionMethods)
        regMethods = cellstr(string(regressionMethods));
    elseif iscell(regressionMethods)
        regMethods = cellfun(@char, regressionMethods, 'UniformOutput', false);
    else
        error('regressionMethods must be char/string/cellstr.');
    end

    here = fileparts(mfilename('fullpath'));
    dataFile = fullfile(here, 'monkeydata_training.mat');
    if exist(dataFile, 'file') ~= 2
        error('Data file not found: %s', dataFile);
    end
    load(dataFile, 'trial');
    addpath(here);
    if ~isempty(teamName) && exist(fullfile(here, teamName), 'dir') == 7
        addpath(fullfile(here, teamName));
    end

    rng(2013);
    ix = randperm(size(trial, 1));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

    % Fixed preprocessing requested by user.
    fixedCfg = struct( ...
        'family', "gaussian", ...
        'sampleStepMs', 20, ...
        'binWidth', 20, ...
        'useGauss', true, ...
        'gaussSigma', 20, ...
        'gaussWidth', 41, ...
        'transform', "none", ...
        'label', "Gaussian smoothing (sigma=20)");

    nRows = numel(regMethods);
    classificationMethodVals = strings(nRows, 1);
    regressionMethodVals = strings(nRows, 1);
    familyVals = strings(nRows, 1);
    sampleStepVals = zeros(nRows, 1);
    binWidthVals = zeros(nRows, 1);
    useGaussVals = false(nRows, 1);
    gaussSigmaVals = nan(nRows, 1);
    gaussWidthVals = nan(nRows, 1);
    transformVals = strings(nRows, 1);
    labels = strings(nRows, 1);
    accVals = zeros(nRows, 1);
    rmseVals = zeros(nRows, 1);
    testTimeVals = zeros(nRows, 1);

    evalStepMs = 20;
    for i = 1:nRows
        regMethod = regMethods{i};
        fprintf('\n[%d/%d] Regression=%s | Preprocessing=%s\n', i, nRows, regMethod, fixedCfg.label);

        prep = struct( ...
            'binWidth', fixedCfg.binWidth, ...
            'transform', fixedCfg.transform, ...
            'smoothKernel', "gaussian", ...
            'smoothWidth', fixedCfg.gaussWidth, ...
            'smoothParam', fixedCfg.gaussSigma, ...
            'targetMode', string(targetMode));

        modelParameters = positionEstimatorTraining(trainingData, regMethod, prep);

        classificationMethodVals(i) = "PCA+LDA+kNN (k=7, PCA=30, LDA=5)";
        regressionMethodVals(i) = string(regMethod);
        familyVals(i) = fixedCfg.family;
        sampleStepVals(i) = fixedCfg.sampleStepMs;
        binWidthVals(i) = fixedCfg.binWidth;
        useGaussVals(i) = fixedCfg.useGauss;
        gaussSigmaVals(i) = fixedCfg.gaussSigma;
        gaussWidthVals(i) = fixedCfg.gaussWidth;
        transformVals(i) = fixedCfg.transform;
        labels(i) = fixedCfg.label;

        [accVals(i), ~] = eval_classification_accuracy(testData, modelParameters.pcaKnnClassifier);
        tStart = tic;
        rmseVals(i) = eval_continuous_rmse(testData, modelParameters, evalStepMs);
        testTimeVals(i) = toc(tStart);
        fprintf('  Accuracy: %.2f%% | RMSE: %.4f | Test time: %.3fs\n', accVals(i), rmseVals(i), testTimeVals(i));
    end

    resultsTable = table( ...
        classificationMethodVals, regressionMethodVals, ...
        familyVals, sampleStepVals, binWidthVals, useGaussVals, gaussSigmaVals, gaussWidthVals, ...
        transformVals, labels, accVals, rmseVals, testTimeVals, ...
        'VariableNames', {'classificationMethod', 'regressionMethod', ...
        'family', 'sampleStepMs', 'binWidth', 'useGauss', 'gaussSigma', 'gaussWidth', ...
        'transform', 'preprocessing', 'classificationAccuracy', 'rmse', 'TestTime_s'});

    resultsTable = sortrows(resultsTable, 'rmse');
    fprintf('\n=== Regression sweep with fixed Gaussian preprocessing (sorted by RMSE) ===\n');
    disp(resultsTable);
end

function [acc, totalPred] = eval_classification_accuracy(testData, classifierModel)
    totalPred = 0;
    correctPred = 0;
    for tr = 1:size(testData, 1)
        for direc = 1:8
            sample.spikes = testData(tr, direc).spikes; %#ok<AGROW>
            predDir = positionEstimator_PCA_LDA_K(sample, classifierModel);
            correctPred = correctPred + (predDir == direc);
            totalPred = totalPred + 1;
        end
    end
    acc = 100 * correctPred / totalPred;
end

function rmse = eval_continuous_rmse(testData, modelParameters, evalStepMs)
    meanSqError = 0;
    nPred = 0;
    for tr = 1:size(testData, 1)
        for direc = randperm(8)
            decodedHandPos = [];
            times = 320:evalStepMs:size(testData(tr, direc).spikes, 2);
            for t = times
                sample.trialId = testData(tr, direc).trialId; %#ok<AGROW>
                sample.spikes = testData(tr, direc).spikes(:, 1:t);
                sample.decodedHandPos = decodedHandPos;
                sample.startHandPos = testData(tr, direc).handPos(1:2, 1);
                [x, y] = positionEstimator(sample, modelParameters);
                decodedPos = [x; y];
                decodedHandPos = [decodedHandPos decodedPos]; %#ok<AGROW>
                err = testData(tr, direc).handPos(1:2, t) - decodedPos;
                meanSqError = meanSqError + sum(err.^2);
            end
            nPred = nPred + numel(times);
        end
    end
    rmse = sqrt(meanSqError / nPred);
end
