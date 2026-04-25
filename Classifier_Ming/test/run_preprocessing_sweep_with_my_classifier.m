function resultsTable = run_preprocessing_sweep_with_my_classifier(teamName, targetMode, regressionMethod)
% Dedicated 3-method preprocessing sweep in a fixed environment:
%   1) Spike count (20 ms)
%   2) Gaussian smoothing (sigma = 20)
%   3) Spike + Anscombe transform
%
% Classifier parameters are fixed inside positionEstimatorTraining.m:
%   k = 7, PCA = 30, LDA = 5
%
% Usage:
%   T = run_preprocessing_sweep_with_my_classifier
%   T = run_preprocessing_sweep_with_my_classifier('test', 'velocity', 'pcr_ridge')
%   T = run_preprocessing_sweep_with_my_classifier('test', 'velocity', {'pcr_ridge','pls'})

    if nargin < 1
        teamName = '';
    end
    if nargin < 2 || isempty(targetMode)
        targetMode = 'velocity';
    end
    if nargin < 3 || isempty(regressionMethod)
        regressionMethod = {'pcr_ridge', 'pls'};
    end
    if ischar(regressionMethod) || isstring(regressionMethod)
        regMethods = cellstr(string(regressionMethod));
    elseif iscell(regressionMethod)
        regMethods = cellfun(@char, regressionMethod, 'UniformOutput', false);
    else
        error('regressionMethod must be char/string/cellstr.');
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

    evalStepMs = 20;

    configs = {};
    configs{end + 1} = mkcfg("spike_count", 20, 20, false, NaN, NaN, "none", "Spike count (20 ms)");
    configs{end + 1} = mkcfg("gaussian", 20, 20, true, 20, 41, "none", "Gaussian smoothing (sigma=20)");
    configs{end + 1} = mkcfg("spike_count", 20, 20, false, NaN, NaN, "anscombe", "Spike + Anscombe transform");

    nBaseCfg = numel(configs);
    nReg = numel(regMethods);
    nCfg = nBaseCfg * nReg;
    classifierMethodVals = strings(nCfg, 1);
    regressionMethodVals = strings(nCfg, 1);
    familyVals = strings(nCfg, 1);
    sampleStepVals = zeros(nCfg, 1);
    binWidthVals = zeros(nCfg, 1);
    useGaussVals = false(nCfg, 1);
    gaussSigmaVals = nan(nCfg, 1);
    gaussWidthVals = nan(nCfg, 1);
    transformVals = strings(nCfg, 1);
    labels = strings(nCfg, 1);
    accVals = zeros(nCfg, 1);
    rmseVals = zeros(nCfg, 1);
    testTimeVals = zeros(nCfg, 1);

    rowIdx = 0;
    for ri = 1:nReg
        regMethodThis = regMethods{ri};
        for ci = 1:nBaseCfg
            rowIdx = rowIdx + 1;
            cfg = configs{ci};
            classifierMethodVals(rowIdx) = "PCA+LDA+kNN (k=7, PCA=30, LDA=5)";
            regressionMethodVals(rowIdx) = string(regMethodThis);
            familyVals(rowIdx) = cfg.family;
            sampleStepVals(rowIdx) = cfg.sampleStepMs;
            binWidthVals(rowIdx) = cfg.binWidth;
            useGaussVals(rowIdx) = cfg.useGauss;
            gaussSigmaVals(rowIdx) = cfg.gaussSigma;
            gaussWidthVals(rowIdx) = cfg.gaussWidth;
            transformVals(rowIdx) = cfg.transform;
            labels(rowIdx) = string(cfg.label);
            fprintf('\n[%d/%d] Regression=%s | Preprocessing: %s\n', rowIdx, nCfg, regMethodThis, cfg.label);

            prep = struct( ...
                'binWidth', cfg.binWidth, ...
                'transform', cfg.transform, ...
                'smoothKernel', "none", ...
                'smoothWidth', 0, ...
                'smoothParam', 2, ...
                'targetMode', string(targetMode));

            if cfg.useGauss
                prep.smoothKernel = "gaussian";
                prep.smoothWidth = cfg.gaussWidth;
                prep.smoothParam = cfg.gaussSigma;
            end

            modelParameters = positionEstimatorTraining(trainingData, regMethodThis, prep);

            [accVals(rowIdx), ~] = eval_classification_accuracy(testData, modelParameters.pcaKnnClassifier);
            tStart = tic;
            rmseVals(rowIdx) = eval_continuous_rmse(testData, modelParameters, evalStepMs);
            testTimeVals(rowIdx) = toc(tStart);
            fprintf('  Accuracy: %.2f%% | RMSE: %.4f | Test time: %.3fs\n', accVals(rowIdx), rmseVals(rowIdx), testTimeVals(rowIdx));
        end
    end

    resultsTable = table( ...
        classifierMethodVals, regressionMethodVals, ...
        familyVals, sampleStepVals, binWidthVals, useGaussVals, gaussSigmaVals, gaussWidthVals, ...
        transformVals, labels, accVals, rmseVals, testTimeVals, ...
        'VariableNames', {'classificationMethod', 'regressionMethod', ...
        'family', 'sampleStepMs', 'binWidth', 'useGauss', 'gaussSigma', 'gaussWidth', ...
        'transform', 'preprocessing', 'classificationAccuracy', 'rmse', 'TestTime_s'});
    resultsTable = sortrows(resultsTable, 'rmse');

    fprintf('\n=== Preprocessing RMSE ranking (lower is better) ===\n');
    disp(resultsTable);
end

function cfg = mkcfg(family, sampleStepMs, binWidth, useGauss, gaussSigma, gaussWidth, transform, label)
    cfg = struct( ...
        'family', string(family), ...
        'sampleStepMs', sampleStepMs, ...
        'binWidth', binWidth, ...
        'useGauss', useGauss, ...
        'gaussSigma', gaussSigma, ...
        'gaussWidth', gaussWidth, ...
        'transform', string(transform), ...
        'label', string(label));
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
