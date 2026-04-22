function benchmark = dameer_grader(varargin)
%DAMEER_GRADER Unified benchmarking harness for the BMI coursework report.
%
% This function extends the original pooled-PCR cross-validation grader into
% a shared benchmarking driver for:
%   1) classifier screening
%   2) pooled regressor screening
%   3) end-to-end continuous decoder comparisons
%   4) optional exploratory Kalman comparisons
%
% Default usage:
%   benchmark = dameer_grader();
%
% Legacy-style usage (pooled PCR nPC sweep only):
%   benchmark = dameer_grader(5, [50 100 200 300 400 500]);
%
% Structured usage:
%   opts = struct('K', 5, 'includeKalman', false, 'plotFigures', true);
%   benchmark = dameer_grader(opts);

    repoRoot = fileparts(mfilename('fullpath'));
    opts = default_benchmark_options(repoRoot);
    opts = parse_benchmark_inputs(opts, varargin{:});
    benchmark = run_unified_benchmark(opts, repoRoot);
end

function opts = default_benchmark_options(repoRoot)
    opts = struct();
    opts.dataFile = fullfile(repoRoot, 'monkeydata0.mat');
    opts.seed = 42;
    opts.K = 5;
    opts.startMs = 320;
    opts.stepMs = 20;
    opts.plotFigures = true;
    opts.saveOutputs = true;
    opts.verbose = true;
    opts.timestampOutputDir = true;
    opts.outputDir = fullfile(repoRoot, 'benchmark_outputs');
    opts.metric = 'rmse';
    opts.causal = true;
    opts.taskScoreEnabled = false;

    % Screening defaults.
    opts.classifierMethods = {'cosine_jared', 'nbc_ming', 'knn_ming', 'lda_jared'};
    opts.classifierKGrid = [1, 5, 11, 21];
    opts.pooledRegressorMethods = {'pooled_pcr', 'pooled_ridge_pcr'};
    opts.pooledPCRGrid = [50, 100, 200, 300, 400, 500];
    opts.pooledRidgePCRGrid = [50, 100, 200, 300, 400, 500];
    opts.pooledRidgeLambdaGrid = [0.1, 1, 10, 100];
    opts.pooledHistoryBins = 15;
    opts.pooledBinWidth = 20;
    opts.pooledTransform = 'anscombe';
    opts.pooledPreSmoothKernel = 'none';
    opts.pooledPreSmoothWidth = 0;
    opts.pooledPreSmoothParam = NaN;
    opts.pooledPostSmoothKernel = 'none';
    opts.pooledPostSmoothWidth = 0;
    opts.pooledPostSmoothParam = NaN;

    % Final pipeline defaults.
    opts.pipelineMethods = {'jared_direct', 'jared_hybrid'};
    opts.includeKalman = false;
    opts.kalmanMethods = {'overfitter_kalman'};
end

function opts = parse_benchmark_inputs(opts, varargin)
    if nargin == 1 || isempty(varargin)
        return;
    end

    if isnumeric(varargin{1})
        opts.K = varargin{1};
        if numel(varargin) >= 2 && isnumeric(varargin{2})
            opts.pooledPCRGrid = varargin{2};
        end
        opts.classifierMethods = {};
        opts.pooledRegressorMethods = {'pooled_pcr'};
        opts.pipelineMethods = {};
        opts.includeKalman = false;
        return;
    end

    if isstruct(varargin{1})
        opts = merge_structs(opts, varargin{1});
        return;
    end

    if mod(numel(varargin), 2) ~= 0
        error('Name-value inputs must come in pairs.');
    end

    userOpts = struct();
    for idx = 1:2:numel(varargin)
        name = varargin{idx};
        value = varargin{idx + 1};
        if ~(ischar(name) || (isstring(name) && isscalar(name)))
            error('Option names must be strings.');
        end
        userOpts.(char(name)) = value;
    end

    opts = merge_structs(opts, userOpts);
end

function out = merge_structs(base, overrides)
    out = base;
    if isempty(overrides)
        return;
    end

    names = fieldnames(overrides);
    for idx = 1:numel(names)
        out.(names{idx}) = overrides.(names{idx});
    end
end

function benchmark = run_unified_benchmark(opts, repoRoot)
    dataFile = resolve_data_file(opts.dataFile, repoRoot);
    data = load(dataFile, 'trial');
    trial = data.trial;

    numTrials = size(trial, 1);
    numDirections = size(trial, 2);
    folds = make_fold_manifest(numTrials, opts.K, opts.seed);
    evalConfig = struct( ...
        'seed', opts.seed, ...
        'K', opts.K, ...
        'startMs', opts.startMs, ...
        'stepMs', opts.stepMs, ...
        'metric', opts.metric, ...
        'causal', opts.causal, ...
        'plotting', false);
    runSpecs = build_run_specs(opts, repoRoot);
    runSpecsTable = struct2table(runSpecs, 'AsArray', true);

    if isempty(runSpecs)
        error('No benchmark methods selected. Check the options passed to dameer_grader.');
    end

    outputDir = '';
    if opts.saveOutputs || opts.plotFigures
        outputDir = resolve_output_dir(opts.outputDir, opts.timestampOutputDir);
        if exist(outputDir, 'dir') ~= 7
            mkdir(outputDir);
        end
    end

    if opts.verbose
        fprintf('============================================================\n');
        fprintf('BMI BENCHMARK HARNESS\n');
        fprintf('Data file: %s\n', dataFile);
        fprintf('Trials: %d, Directions: %d, Folds: %d, Seed: %d\n', ...
            numTrials, numDirections, opts.K, opts.seed);
        fprintf('Start: %d ms, Step: %d ms, Causal: %d\n', ...
            opts.startMs, opts.stepMs, opts.causal);
        fprintf('Methods queued: %d\n', numel(runSpecs));
        fprintf('============================================================\n\n');
    end

    rawRows = repmat(empty_result_row(), 0, 1);
    timeResolvedRows = repmat(empty_timeresolved_row(), 0, 1);
    for runIdx = 1:numel(runSpecs)
        runSpec = runSpecs(runIdx);
        if opts.verbose
            fprintf('[%2d/%2d] %s | %s | config %d (%s)\n', ...
                runIdx, numel(runSpecs), runSpec.task, runSpec.method, ...
                runSpec.configId, runSpec.configLabel);
        end

        for foldIdx = 1:numel(folds)
            fold = folds(foldIdx);
            trainData = trial(fold.trainIdx, :);
            testData = trial(fold.testIdx, :);

            switch runSpec.executor
                case 'classifier_local'
                    metrics = evaluate_local_classifier_fold(runSpec, trainData, testData, evalConfig);
                case 'continuous_local'
                    metrics = evaluate_local_continuous_fold(runSpec, trainData, testData, evalConfig);
                case 'generic_position_estimator'
                    metrics = evaluate_generic_position_estimator_fold(runSpec, trainData, testData, evalConfig);
                otherwise
                    error('Unknown executor: %s', runSpec.executor);
            end

            rawRows(end + 1) = build_result_row(runSpec, foldIdx, metrics); %#ok<AGROW>

            if isfield(metrics, 'timesByT') && ~isempty(metrics.timesByT)
                for tIdx = 1:numel(metrics.timesByT)
                    timeResolvedRows(end + 1) = build_timeresolved_row( ...
                        runSpec, foldIdx, metrics.timesByT(tIdx), ...
                        metrics.accByT(tIdx), metrics.countsByT(tIdx)); %#ok<AGROW>
                end
            end

            if opts.verbose
                fprintf('    Fold %d -> %s\n', foldIdx, format_metric_string(runSpec.task, metrics));
            end
        end
    end

 
    rawTable = struct2table(rawRows, 'AsArray', true);
    if isempty(timeResolvedRows)
        timeResolvedTable = empty_timeresolved_table();
    else
       
        timeResolvedTable = struct2table(timeResolvedRows, 'AsArray', true);
    end
    [foldTable, foldStruct] = fold_manifest_table(folds);

    classifierRaw = rawTable(strcmp(rawTable.task, 'classifier'), :);
    classifierSummary = summarize_results(classifierRaw);
    classifierBest = select_best_configs(classifierSummary, 'accuracy', 'descend');

    continuousMask = ismember(rawTable.task, {'regressor_pooled', 'pipeline', 'kalman_exploratory'});
    continuousRaw = rawTable(continuousMask, :);
    continuousSummary = summarize_results(continuousRaw);
    continuousBest = select_best_configs(continuousSummary, 'taskScore', 'ascend');
    if isempty(continuousBest)
        optimizedPerformance = continuousBest;
    else
        optimizedPerformance = sortrows(continuousBest, {'meanTaskScore', 'meanRMSE', 'method'});
    end

    optimizedParameters = build_optimized_parameter_table(classifierBest, continuousBest);

    paths = struct();
    if opts.saveOutputs || opts.plotFigures
        paths = write_benchmark_outputs( ...
            outputDir, opts, evalConfig, runSpecsTable, foldTable, rawTable, timeResolvedTable, ...
            classifierSummary, classifierBest, continuousSummary, optimizedPerformance, optimizedParameters);

        if opts.plotFigures
            paths.figureFiles = generate_benchmark_figures( ...
                outputDir, classifierSummary, continuousSummary, optimizedPerformance, timeResolvedTable);
        else
            paths.figureFiles = {};
        end
    end

    benchmark = struct();
    benchmark.config = opts;
    benchmark.evalConfig = evalConfig;
    benchmark.dataFile = dataFile;
    benchmark.outputDir = outputDir;
    benchmark.folds = foldStruct;
    benchmark.tables = struct( ...
        'runSpecs', runSpecsTable, ...
        'foldManifest', foldTable, ...
        'rawResults', rawTable, ...
        'classifierRaw', classifierRaw, ...
        'classifierSummary', classifierSummary, ...
        'classifierBest', classifierBest, ...
        'classifierTimeResolved', timeResolvedTable, ...
        'continuousRaw', continuousRaw, ...
        'continuousSummary', continuousSummary, ...
        'optimizedPerformance', optimizedPerformance, ...
        'optimizedParameters', optimizedParameters);
    benchmark.paths = paths;
end

function dataFile = resolve_data_file(requestedPath, repoRoot)
    if exist(requestedPath, 'file') == 2
        dataFile = requestedPath;
        return;
    end

    candidate = fullfile(repoRoot, requestedPath);
    if exist(candidate, 'file') == 2
        dataFile = candidate;
        return;
    end

    error('Could not find data file: %s', requestedPath);
end

function outputDir = resolve_output_dir(baseDir, timestampOutputDir)
    if timestampOutputDir
        stamp = datestr(now, 'yyyymmdd_HHMMSS');
        outputDir = fullfile(baseDir, ['run_' stamp]);
    else
        outputDir = baseDir;
    end
end

function folds = make_fold_manifest(numTrials, K, seed)
    rng(seed);
    shuffled = randperm(numTrials);
    baseFoldSize = floor(numTrials / K);
    remainder = mod(numTrials, K);

    foldSizes = baseFoldSize * ones(1, K);
    if remainder > 0
        foldSizes(1:remainder) = foldSizes(1:remainder) + 1;
    end

    folds = repmat(struct('fold', 0, 'trainIdx', [], 'testIdx', []), 1, K);
    cursor = 1;
    for foldIdx = 1:K
        foldCount = foldSizes(foldIdx);
        testIdx = shuffled(cursor:cursor + foldCount - 1);
        trainMask = true(1, numTrials);
        trainMask(cursor:cursor + foldCount - 1) = false;
        trainIdx = shuffled(trainMask);
        folds(foldIdx).fold = foldIdx;
        folds(foldIdx).trainIdx = trainIdx;
        folds(foldIdx).testIdx = testIdx;
        cursor = cursor + foldCount;
    end
end

function [foldTable, foldStruct] = fold_manifest_table(folds)
    numFolds = numel(folds);
    trainCells = cell(numFolds, 1);
    testCells = cell(numFolds, 1);
    foldIds = zeros(numFolds, 1);
    for idx = 1:numFolds
        foldIds(idx) = folds(idx).fold;
        trainCells{idx} = mat2str(folds(idx).trainIdx);
        testCells{idx} = mat2str(folds(idx).testIdx);
    end

    foldTable = table(foldIds, trainCells, testCells, ...
        'VariableNames', {'fold', 'trainIndices', 'testIndices'});
    foldStruct = folds;
end

function runSpecs = build_run_specs(opts, repoRoot)
    runSpecs = repmat(empty_run_spec(), 0, 1);

    % Classifier screening.
    if ismember('cosine_jared', opts.classifierMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'classifier', ...
            'family', 'classifier_screening', ...
            'method', 'cosine_jared', ...
            'executor', 'classifier_local', ...
            'localKind', 'cosine_jared', ...
            'configId', 1, ...
            'configLabel', 'default', ...
            'classifier', 'cosine_similarity', ...
            'regressor', 'none', ...
            'preprocess', 'early_320ms_spike_counts', ...
            'notes', 'Code-grounded from jared/positionEstimatorTraining.m classification stage.');
    end

    if ismember('nbc_ming', opts.classifierMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'classifier', ...
            'family', 'classifier_screening', ...
            'method', 'nbc_ming', ...
            'executor', 'classifier_local', ...
            'localKind', 'nbc_ming', ...
            'configId', 1, ...
            'configLabel', 'default', ...
            'classifier', 'gaussian_naive_bayes', ...
            'regressor', 'none', ...
            'preprocess', 'early_320ms_spike_counts_zscore', ...
            'notes', 'Code-grounded from Classifier_Ming/NBC.');
    end

    if ismember('lda_jared', opts.classifierMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'classifier', ...
            'family', 'classifier_screening', ...
            'method', 'lda_jared', ...
            'executor', 'classifier_local', ...
            'localKind', 'lda_jared', ...
            'configId', 1, ...
            'configLabel', 'default', ...
            'classifier', 'lda_shrinkage', ...
            'regressor', 'none', ...
            'preprocess', 'early_cumulative_counts_anscombe', ...
            'notes', 'Pooled-covariance LDA on Anscombe(cumulative counts); matches the exemplar report classifier.');
    end

    if ismember('knn_ming', opts.classifierMethods)
        for idx = 1:numel(opts.classifierKGrid)
            kVal = opts.classifierKGrid(idx);
            runSpecs(end + 1) = make_run_spec( ...
                'task', 'classifier', ...
                'family', 'classifier_screening', ...
                'method', 'knn_ming', ...
                'executor', 'classifier_local', ...
                'localKind', 'knn_ming', ...
                'configId', idx, ...
                'configLabel', sprintf('k=%d', kVal), ...
                'classifier', 'knn', ...
                'regressor', 'none', ...
                'preprocess', 'early_320ms_spike_counts_zscore', ...
                'k', kVal, ...
                'notes', 'Code-grounded from Classifier_Ming/KNN/k.');
        end
    end

    % Pooled regressor screening.
    if ismember('pooled_pcr', opts.pooledRegressorMethods)
        for idx = 1:numel(opts.pooledPCRGrid)
            nPC = opts.pooledPCRGrid(idx);
            runSpecs(end + 1) = make_run_spec( ...
                'task', 'regressor_pooled', ...
                'family', 'regressor_screening', ...
                'method', 'pooled_pcr', ...
                'executor', 'continuous_local', ...
                'localKind', 'pooled_regressor', ...
                'configId', idx, ...
                'configLabel', sprintf('nPC=%d', nPC), ...
                'classifier', 'none', ...
                'regressor', 'pcr', ...
                'preprocess', '20ms_bins_anscombe_16bin_history', ...
                'nPC', nPC, ...
                'lambda', 0, ...
                'historyBins', opts.pooledHistoryBins, ...
                'binWidth', opts.pooledBinWidth, ...
                'transform', opts.pooledTransform, ...
                'preSmoothKernel', opts.pooledPreSmoothKernel, ...
                'preSmoothWidth', opts.pooledPreSmoothWidth, ...
                'preSmoothParam', opts.pooledPreSmoothParam, ...
                'postSmoothKernel', opts.pooledPostSmoothKernel, ...
                'postSmoothWidth', opts.pooledPostSmoothWidth, ...
                'postSmoothParam', opts.pooledPostSmoothParam, ...
                'notes', 'Shared pooled PCR screening baseline derived from the original dameer_grader.');
        end
    end

    if ismember('pooled_ridge_pcr', opts.pooledRegressorMethods)
        configId = 1;
        for lambdaIdx = 1:numel(opts.pooledRidgeLambdaGrid)
            lambdaVal = opts.pooledRidgeLambdaGrid(lambdaIdx);
            for pcIdx = 1:numel(opts.pooledRidgePCRGrid)
                nPC = opts.pooledRidgePCRGrid(pcIdx);
                runSpecs(end + 1) = make_run_spec( ...
                    'task', 'regressor_pooled', ...
                    'family', 'regressor_screening', ...
                    'method', 'pooled_ridge_pcr', ...
                    'executor', 'continuous_local', ...
                    'localKind', 'pooled_regressor', ...
                    'configId', configId, ...
                    'configLabel', sprintf('nPC=%d_lambda=%.3g', nPC, lambdaVal), ...
                    'classifier', 'none', ...
                    'regressor', 'ridge_pcr', ...
                    'preprocess', '20ms_bins_anscombe_16bin_history', ...
                    'nPC', nPC, ...
                    'lambda', lambdaVal, ...
                    'historyBins', opts.pooledHistoryBins, ...
                    'binWidth', opts.pooledBinWidth, ...
                    'transform', opts.pooledTransform, ...
                    'preSmoothKernel', opts.pooledPreSmoothKernel, ...
                    'preSmoothWidth', opts.pooledPreSmoothWidth, ...
                    'preSmoothParam', opts.pooledPreSmoothParam, ...
                    'postSmoothKernel', opts.pooledPostSmoothKernel, ...
                    'postSmoothWidth', opts.pooledPostSmoothWidth, ...
                    'postSmoothParam', opts.pooledPostSmoothParam, ...
                    'notes', 'Shared pooled ridge-PCR screening baseline using the dameer_grader feature pipeline.');
                configId = configId + 1;
            end
        end
    end

    % Final pipeline comparisons.
    if ismember('jared_direct', opts.pipelineMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'pipeline', ...
            'family', 'final_pipeline', ...
            'method', 'jared_direct', ...
            'executor', 'generic_position_estimator', ...
            'configId', 1, ...
            'configLabel', 'fixed', ...
            'classifier', 'none', ...
            'regressor', 'pooled_pcr', ...
            'preprocess', '20ms_bins_anscombe_16bin_history', ...
            'nPC', 500, ...
            'historyBins', 15, ...
            'methodDir', fullfile(repoRoot, 'jared', 'positionEstimator (2)'), ...
            'predictReturnsState', false, ...
            'notes', 'Direct pooled PCR baseline from jared/positionEstimator (2).');
    end

    if ismember('jared_hybrid', opts.pipelineMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'pipeline', ...
            'family', 'final_pipeline', ...
            'method', 'jared_hybrid', ...
            'executor', 'generic_position_estimator', ...
            'configId', 1, ...
            'configLabel', 'fixed', ...
            'classifier', 'cosine_similarity', ...
            'regressor', 'direction_specific_ridge_pcr', ...
            'preprocess', '20ms_bins_anscombe_16bin_history', ...
            'nPC', 500, ...
            'lambda', 1000, ...
            'historyBins', 15, ...
            'methodDir', fullfile(repoRoot, 'jared'), ...
            'predictReturnsState', false, ...
            'notes', 'Classifier-conditioned final pipeline from jared/positionEstimatorTraining.m.');
    end

    if opts.includeKalman && ismember('overfitter_kalman', opts.kalmanMethods)
        runSpecs(end + 1) = make_run_spec( ...
            'task', 'kalman_exploratory', ...
            'family', 'exploratory', ...
            'method', 'overfitter_kalman', ...
            'executor', 'generic_position_estimator', ...
            'configId', 1, ...
            'configLabel', 'fixed', ...
            'classifier', 'lda_nearest_centroid', ...
            'regressor', 'residual_ridge', ...
            'preprocess', '25ms_moving_average_multiscale_buffers', ...
            'lambda', 5, ...
            'bufferShort', 100, ...
            'bufferLong', 300, ...
            'smoothWin', 25, ...
            'methodDir', fullfile(repoRoot, 'overfitter_ming', 'overfitter', ...
                'bmi-overfitter-ming', 'Regression_and_Combined_Evaluation', 'overfitter_ming'), ...
            'predictReturnsState', true, ...
            'notes', 'Exploratory Kalman branch; uses conv2(...,''same'') smoothing and should be treated cautiously.');
    end
end

function spec = empty_run_spec()
    spec = struct( ...
        'task', '', ...
        'family', '', ...
        'method', '', ...
        'executor', '', ...
        'localKind', '', ...
        'configId', NaN, ...
        'configLabel', '', ...
        'classifier', '', ...
        'regressor', '', ...
        'preprocess', '', ...
        'k', NaN, ...
        'nPC', NaN, ...
        'lambda', NaN, ...
        'historyBins', NaN, ...
        'binWidth', NaN, ...
        'transform', '', ...
        'preSmoothKernel', 'none', ...
        'preSmoothWidth', 0, ...
        'preSmoothParam', NaN, ...
        'postSmoothKernel', 'none', ...
        'postSmoothWidth', 0, ...
        'postSmoothParam', NaN, ...
        'bufferShort', NaN, ...
        'bufferLong', NaN, ...
        'smoothWin', NaN, ...
        'methodDir', '', ...
        'predictReturnsState', false, ...
        'notes', '');
end

function spec = make_run_spec(varargin)
    spec = empty_run_spec();
    if mod(numel(varargin), 2) ~= 0
        error('Run spec arguments must come in name-value pairs.');
    end
    for idx = 1:2:numel(varargin)
        spec.(varargin{idx}) = varargin{idx + 1};
    end
end

function metrics = evaluate_local_classifier_fold(runSpec, trainData, testData, evalConfig)
    endHorizon = 560;
    timesByT = evalConfig.startMs:evalConfig.stepMs:endHorizon;
    numT = numel(timesByT);
    accByT = zeros(1, numT);
    countsByT = zeros(1, numT);
    trainTime = 0;
    testTime = 0;

    for idx = 1:numT
        horizon = timesByT(idx);
        trainTimer = tic;
        switch runSpec.localKind
            case 'cosine_jared'
                model = train_classifier_cosine_jared(trainData, horizon);
            case 'nbc_ming'
                model = train_classifier_nbc_ming(trainData, horizon);
            case 'knn_ming'
                model = train_classifier_knn_ming(trainData, runSpec.k, horizon);
            case 'lda_jared'
                model = train_classifier_lda_jared(trainData, horizon);
            otherwise
                error('Unknown local classifier kind: %s', runSpec.localKind);
        end
        trainTime = trainTime + toc(trainTimer);

        testTimer = tic;
        [accByT(idx), countsByT(idx)] = score_classifier_model(runSpec.localKind, model, testData);
        testTime = testTime + toc(testTimer);
    end

    finalAccuracy = accByT(end);
    totalPredictions = sum(countsByT);

    metrics = struct();
    metrics.rmse = NaN;
    metrics.accuracy = finalAccuracy;
    metrics.trainTimeSec = trainTime;
    metrics.testTimeSec = testTime;
    metrics.timePerPredictionMs = 1000 * testTime / max(totalPredictions, 1);
    metrics.numPredictions = totalPredictions;
    metrics.taskScore = NaN;
    metrics.timesByT = timesByT;
    metrics.accByT = accByT;
    metrics.countsByT = countsByT;
end

function metrics = evaluate_local_continuous_fold(runSpec, trainData, testData, evalConfig)
    trainTimer = tic;
    if isnan(runSpec.binWidth)
        binWidth = 20;
    else
        binWidth = runSpec.binWidth;
    end
    if isempty(runSpec.transform)
        transformName = 'anscombe';
    else
        transformName = runSpec.transform;
    end
    cfg = struct( ...
        'nPC', runSpec.nPC, ...
        'lambda', runSpec.lambda, ...
        'historyBins', runSpec.historyBins, ...
        'binWidth', binWidth, ...
        'transform', transformName, ...
        'preSmoothKernel', runSpec.preSmoothKernel, ...
        'preSmoothWidth', runSpec.preSmoothWidth, ...
        'preSmoothParam', runSpec.preSmoothParam, ...
        'postSmoothKernel', runSpec.postSmoothKernel, ...
        'postSmoothWidth', runSpec.postSmoothWidth, ...
        'postSmoothParam', runSpec.postSmoothParam);
        

    model = train_local_pooled_regressor(trainData, cfg);
    trainTime = toc(trainTimer);

    testTimer = tic;
    [rmse, numPredictions] = score_local_continuous_model(model, testData, evalConfig);
    testTime = toc(testTimer);

    metrics = struct();
    metrics.rmse = rmse;
    metrics.accuracy = NaN;
    metrics.trainTimeSec = trainTime;
    metrics.testTimeSec = testTime;
    metrics.timePerPredictionMs = 1000 * testTime / max(numPredictions, 1);
    metrics.numPredictions = numPredictions;
    metrics.taskScore = 0.9 * rmse + 0.1 * testTime;
end

function metrics = evaluate_generic_position_estimator_fold(runSpec, trainData, testData, evalConfig)
    previousPath = path();
    cleanupObj = onCleanup(@() restore_method_path(previousPath)); %#ok<NASGU>

    addpath(runSpec.methodDir, '-begin');
    clear positionEstimatorTraining positionEstimator

    trainTimer = tic;
    modelParameters = positionEstimatorTraining(trainData);
    trainTime = toc(trainTimer);

    clear positionEstimator

    testTimer = tic;
    [rmse, numPredictions] = score_generic_position_estimator( ...
        testData, modelParameters, evalConfig, runSpec.predictReturnsState);
    testTime = toc(testTimer);

    metrics = struct();
    metrics.rmse = rmse;
    metrics.accuracy = NaN;
    metrics.trainTimeSec = trainTime;
    metrics.testTimeSec = testTime;
    metrics.timePerPredictionMs = 1000 * testTime / max(numPredictions, 1);
    metrics.numPredictions = numPredictions;
    metrics.taskScore = 0.9 * rmse + 0.1 * testTime;
end

function restore_method_path(previousPath)
    path(previousPath);
    clear positionEstimatorTraining positionEstimator
end

function [accuracy, numSamples] = score_classifier_model(localKind, model, testData)
    correct = 0;
    numSamples = 0;

    for trialIdx = 1:size(testData, 1)
        for dirIdx = 1:size(testData, 2)
            sample = struct('spikes', testData(trialIdx, dirIdx).spikes);
            switch localKind
                case 'cosine_jared'
                    predDir = predict_classifier_cosine_jared(sample, model);
                case 'nbc_ming'
                    predDir = predict_classifier_nbc_ming(sample, model);
                case 'knn_ming'
                    predDir = predict_classifier_knn_ming(sample, model);
                case 'lda_jared'
                    predDir = predict_classifier_lda_jared(sample, model);
                otherwise
                    error('Unknown local classifier kind: %s', localKind);
            end
            correct = correct + double(predDir == dirIdx);
            numSamples = numSamples + 1;
        end
    end

    accuracy = 100 * correct / max(numSamples, 1);
end

function model = train_classifier_cosine_jared(trainingData, horizon)
    if nargin < 2 || isempty(horizon)
        horizon = 320;
    end
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    means = zeros(numDirs, numNeurons);

    for dirIdx = 1:numDirs
        features = zeros(numTrials, numNeurons);
        for trialIdx = 1:numTrials
            spikes = trainingData(trialIdx, dirIdx).spikes;
            endIdx = min(horizon, size(spikes, 2));
            features(trialIdx, :) = sum(spikes(:, 1:endIdx), 2)';
        end
        means(dirIdx, :) = mean(features, 1);
    end

    model = struct('means', means, 'horizon', horizon);
end

function predDir = predict_classifier_cosine_jared(testSample, model)
    spikes = testSample.spikes;
    endIdx = min(model.horizon, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';
    featureNorm = norm(feature);
    if featureNorm < eps
        featureNorm = eps;
    end
    templateNorms = sqrt(sum(model.means .^ 2, 2));
    denom = templateNorms * featureNorm;
    denom(denom < eps) = eps;
    cosineScores = (model.means * feature') ./ denom;
    [~, predDir] = max(cosineScores);
end

function model = train_classifier_nbc_ming(trainingData, horizon)
    if nargin < 2 || isempty(horizon)
        horizon = 320;
    end
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    numSamples = numTrials * numDirs;
    features = zeros(numSamples, numNeurons);
    labels = zeros(numSamples, 1);

    rowIdx = 1;
    for trialIdx = 1:numTrials
        for dirIdx = 1:numDirs
            spikes = trainingData(trialIdx, dirIdx).spikes;
            endIdx = min(horizon, size(spikes, 2));
            features(rowIdx, :) = sum(spikes(:, 1:endIdx), 2)';
            labels(rowIdx) = dirIdx;
            rowIdx = rowIdx + 1;
        end
    end

    mu = mean(features, 1);
    sigma = std(features, 0, 1) + eps;
    normalized = (features - mu) ./ sigma;
    classMeans = zeros(numDirs, numNeurons);
    classVars = zeros(numDirs, numNeurons);
    priors = zeros(numDirs, 1);

    for dirIdx = 1:numDirs
        Xc = normalized(labels == dirIdx, :);
        priors(dirIdx) = size(Xc, 1) / numSamples;
        classMeans(dirIdx, :) = mean(Xc, 1);
        classVars(dirIdx, :) = var(Xc, 0, 1) + eps;
    end

    classifier = struct( ...
        'classMeans', classMeans, ...
        'classVars', classVars, ...
        'priors', priors, ...
        'classes', (1:numDirs)');
    model = struct( ...
        'mu_class', mu, ...
        'sigma_class', sigma, ...
        'classifier', classifier, ...
        'T_class', horizon);
end

function predDir = predict_classifier_nbc_ming(testSample, model)
    endIdx = min(model.T_class, size(testSample.spikes, 2));
    feature = sum(testSample.spikes(:, 1:endIdx), 2)';
    normalizedFeature = (feature - model.mu_class) ./ model.sigma_class;
    clf = model.classifier;
    numClasses = numel(clf.classes);
    logScores = zeros(numClasses, 1);
    for classIdx = 1:numClasses
        muClass = clf.classMeans(classIdx, :);
        varClass = clf.classVars(classIdx, :);
        logGauss = -0.5 * sum(log(2 * pi * varClass)) ...
            - 0.5 * sum((normalizedFeature - muClass) .^ 2 ./ varClass);
        logScores(classIdx) = logGauss + log(clf.priors(classIdx));
    end
    [~, bestIdx] = max(logScores);
    predDir = clf.classes(bestIdx);
end

function model = train_classifier_knn_ming(trainingData, kVal, horizon)
    if nargin < 3 || isempty(horizon)
        horizon = 320;
    end
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    numSamples = numTrials * numDirs;
    features = zeros(numSamples, numNeurons);
    labels = zeros(numSamples, 1);

    rowIdx = 1;
    for trialIdx = 1:numTrials
        for dirIdx = 1:numDirs
            spikes = trainingData(trialIdx, dirIdx).spikes;
            endIdx = min(horizon, size(spikes, 2));
            features(rowIdx, :) = sum(spikes(:, 1:endIdx), 2)';
            labels(rowIdx) = dirIdx;
            rowIdx = rowIdx + 1;
        end
    end

    mu = mean(features, 1);
    sigma = std(features, 0, 1) + eps;
    normalizedFeatures = (features - mu) ./ sigma;

    model = struct( ...
        'X', normalizedFeatures, ...
        'y', labels, ...
        'mu', mu, ...
        'sigma', sigma, ...
        'k', kVal, ...
        'T_class', horizon);
end

function predDir = predict_classifier_knn_ming(testSample, model)
    spikes = testSample.spikes;
    endIdx = min(model.T_class, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';
    normalizedFeature = (feature - model.mu) ./ model.sigma;
    squaredDistances = sum((model.X - normalizedFeature) .^ 2, 2);
    [~, sortedIdx] = sort(squaredDistances, 'ascend');
    nearestLabels = model.y(sortedIdx(1:model.k));
    predDir = mode(nearestLabels);
end

function model = train_classifier_lda_jared(trainingData, horizon)
    if nargin < 2 || isempty(horizon)
        horizon = 320;
    end
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);

    features = zeros(numTrials * numDirs, numNeurons);
    labels = zeros(numTrials * numDirs, 1);
    rowIdx = 1;
    for trialIdx = 1:numTrials
        for dirIdx = 1:numDirs
            spikes = trainingData(trialIdx, dirIdx).spikes;
            endIdx = min(horizon, size(spikes, 2));
            features(rowIdx, :) = sum(spikes(:, 1:endIdx), 2)';
            labels(rowIdx) = dirIdx;
            rowIdx = rowIdx + 1;
        end
    end

    % Anscombe transform keeps variance ~constant across neurons (report §II-B).
    features = 2 * sqrt(features + 3 / 8);

    classMeans = zeros(numDirs, numNeurons);
    priors = zeros(numDirs, 1);
    pooledCov = zeros(numNeurons, numNeurons);
    total = size(features, 1);

    for dirIdx = 1:numDirs
        classMask = labels == dirIdx;
        Xc = features(classMask, :);
        classMeans(dirIdx, :) = mean(Xc, 1);
        priors(dirIdx) = size(Xc, 1) / total;
        centered = Xc - classMeans(dirIdx, :);
        pooledCov = pooledCov + centered' * centered;
    end

    pooledCov = pooledCov / max(total - numDirs, 1);
    % Shrinkage toward a diagonal target keeps the covariance invertible when
    % numNeurons is close to the per-class sample count.
    shrinkage = 1e-3;
    diagTarget = diag(diag(pooledCov));
    pooledCov = (1 - shrinkage) * pooledCov + shrinkage * diagTarget;
    regularizer = 1e-6 * mean(diag(pooledCov)) * eye(numNeurons);
    Sigma = pooledCov + regularizer;

    W = (Sigma \ classMeans')';
    b = -0.5 * sum(W .* classMeans, 2) + log(priors + eps);

    model = struct( ...
        'W', W, ...
        'b', b, ...
        'horizon', horizon);
end

function predDir = predict_classifier_lda_jared(testSample, model)
    spikes = testSample.spikes;
    endIdx = min(model.horizon, size(spikes, 2));
    feature = sum(spikes(:, 1:endIdx), 2)';
    feature = 2 * sqrt(feature + 3 / 8);
    scores = model.W * feature' + model.b;
    [~, predDir] = max(scores);
end

function model = train_local_pooled_regressor(trainingData, cfg)
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    processed = initialize_bin_width(trainingData);
    processed = smooth_dataset(processed, cfg.preSmoothKernel, cfg.preSmoothWidth, cfg.preSmoothParam);
    processed = rebin_dataset(processed, cfg.binWidth);
    processed = transform_dataset(processed, cfg.transform);
    processed = smooth_dataset(processed, cfg.postSmoothKernel, cfg.postSmoothWidth, cfg.postSmoothParam);

    minLength = minimum_trial_length(trainingData);
    maxIter = floor(minLength / cfg.binWidth) - cfg.historyBins;
    if maxIter < 1
        error('Not enough data to build the pooled regressor design matrix.');
    end

    numRows = numTrials * numDirs * maxIter;
    numFeatures = numNeurons * (cfg.historyBins + 1);
    X = zeros(numRows, numFeatures);
    Y = zeros(numRows, 2);
    rowIdx = 1;

    for trialIdx = 1:numTrials
        for dirIdx = 1:numDirs
            for timeIdx = 1:maxIter
                X(rowIdx, :) = reshape( ...
                    processed(trialIdx, dirIdx).spikes(:, timeIdx:cfg.historyBins + timeIdx), ...
                    1, []);
                currentMs = (cfg.historyBins + timeIdx) * cfg.binWidth;
                previousMs = (cfg.historyBins + timeIdx - 1) * cfg.binWidth;
                Y(rowIdx, 1) = trainingData(trialIdx, dirIdx).handPos(1, currentMs) ...
                    - trainingData(trialIdx, dirIdx).handPos(1, previousMs);
                Y(rowIdx, 2) = trainingData(trialIdx, dirIdx).handPos(2, currentMs) ...
                    - trainingData(trialIdx, dirIdx).handPos(2, previousMs);
                rowIdx = rowIdx + 1;
            end
        end
    end

    muX = mean(X, 1);
    centeredX = X - muX;
    [~, S, V] = svd(centeredX, 'econ');
    singularVals = diag(S);
    svdTol = max(size(centeredX)) * eps(max(singularVals));
    effectiveRank = sum(singularVals > svdTol);
    nPC = min(cfg.nPC, effectiveRank);
    if nPC < 1
        error('Insufficient effective rank in pooled regressor design matrix.');
    end
    Vreduced = V(:, 1:nPC);
    projectedX = centeredX * Vreduced;
    projectedX = [projectedX, ones(size(projectedX, 1), 1)];

    fitMode = 'ols';

    if cfg.lambda > 0
        penalty = cfg.lambda * eye(size(projectedX, 2));
        penalty(end, end) = 0;
        B = (projectedX' * projectedX + penalty) \ (projectedX' * Y);
        fitMode = 'ridge';
    else
        designRank = rank(projectedX);
        if designRank < size(projectedX, 2)
            B = pinv(projectedX) * Y;
            fitMode = 'pinv_fallback';
        else
            B = projectedX \ Y;
        end
    end

    model = struct();
    model.B = B;
    model.muX = muX;
    model.Vreduced = Vreduced;
    model.binWidth = cfg.binWidth;
    model.historyBins = cfg.historyBins;
    model.transform = cfg.transform;
    model.preSmoothKernel = cfg.preSmoothKernel;
    model.preSmoothWidth = cfg.preSmoothWidth;
    model.preSmoothParam = cfg.preSmoothParam;
    model.postSmoothKernel = cfg.postSmoothKernel;
    model.postSmoothWidth = cfg.postSmoothWidth;
    model.postSmoothParam = cfg.postSmoothParam;
    model.fitMode = fitMode;
end

function [rmse, numPredictions] = score_local_continuous_model(model, testData, evalConfig)
    meanSqError = 0;
    numPredictions = 0;
    numDirs = size(testData, 2);

    for trialIdx = 1:size(testData, 1)
        for dirIdx = 1:numDirs
            decodedHandPos = [];
            times = evalConfig.startMs:evalConfig.stepMs:size(testData(trialIdx, dirIdx).spikes, 2);
            fullyProcessed = preprocess_spike_matrix( ...
                testData(trialIdx, dirIdx).spikes, ...
                model.preSmoothKernel, model.preSmoothWidth, model.preSmoothParam, ...
                model.binWidth, model.transform, ...
                model.postSmoothKernel, model.postSmoothWidth, model.postSmoothParam);

            for t = times
                sample = struct();
                sample.trialId = testData(trialIdx, dirIdx).trialId;
                sample.spikes = testData(trialIdx, dirIdx).spikes(:, 1:t);
                usableBins = floor(t / model.binWidth);
                sample.preprocessedSpikes = fullyProcessed(:, 1:min(usableBins, size(fullyProcessed, 2)));
                sample.decodedHandPos = decodedHandPos;
                sample.startHandPos = testData(trialIdx, dirIdx).handPos(1:2, 1);

                [decodedX, decodedY] = predict_local_pooled_regressor(sample, model);
                decodedPos = [decodedX; decodedY];
                decodedHandPos = [decodedHandPos, decodedPos]; %#ok<AGROW>

                actualPos = testData(trialIdx, dirIdx).handPos(1:2, t);
                meanSqError = meanSqError + norm(actualPos - decodedPos) ^ 2;
            end

            numPredictions = numPredictions + numel(times);
        end
    end

    rmse = sqrt(meanSqError / max(numPredictions, 1));
end

function [decodedX, decodedY] = predict_local_pooled_regressor(testSample, model)
    rawSpikes = apply_smoothing_to_matrix( ...
        testSample.spikes, model.preSmoothKernel, model.preSmoothWidth, model.preSmoothParam);
    if isfield(testSample, 'preprocessedSpikes')
        transformedSpikes = testSample.preprocessedSpikes;
    else
        transformedSpikes = preprocess_spike_matrix( ...
            testSample.spikes, ...
            model.preSmoothKernel, model.preSmoothWidth, model.preSmoothParam, ...
            model.binWidth, model.transform, ...
            model.postSmoothKernel, model.postSmoothWidth, model.postSmoothParam);
    end
    transformedSpikes = apply_smoothing_to_matrix( ...
        transformedSpikes, model.postSmoothKernel, model.postSmoothWidth, model.postSmoothParam);
    numFeatureBins = model.historyBins + 1;
    numAvailableBins = size(transformedSpikes, 2);
    numNeurons = size(transformedSpikes, 1);

    if numAvailableBins < numFeatureBins
        padded = zeros(numNeurons, numFeatureBins);
        padded(:, end - numAvailableBins + 1:end) = transformedSpikes;
        recentBins = padded;
    else
        recentBins = transformedSpikes(:, end - numFeatureBins + 1:end);
    end

    Xtest = reshape(recentBins, 1, []);
    centered = Xtest - model.muX;
    projected = centered * model.Vreduced;
    projected = [projected, 1];
    predictedVelocity = projected * model.B;

    if isempty(testSample.decodedHandPos)
        previousX = testSample.startHandPos(1);
        previousY = testSample.startHandPos(2);
    else
        previousX = testSample.decodedHandPos(1, end);
        previousY = testSample.decodedHandPos(2, end);
    end

    decodedX = previousX + predictedVelocity(1);
    decodedY = previousY + predictedVelocity(2);
end
function transformedSpikes = preprocess_spike_matrix(rawSpikes, preKernel, preWidth, preParam, binWidth, transformName, postKernel, postWidth, postParam)
    smoothedRaw = apply_smoothing_to_matrix(rawSpikes, preKernel, preWidth, preParam);
    binnedSpikes = rebin_spike_matrix(smoothedRaw, binWidth);
    transformedSpikes = apply_transform_to_matrix(binnedSpikes, transformName);
    transformedSpikes = apply_smoothing_to_matrix(transformedSpikes, postKernel, postWidth, postParam);
end

function [rmse, numPredictions] = score_generic_position_estimator(testData, modelParameters, evalConfig, predictReturnsState)
    meanSqError = 0;
    numPredictions = 0;
    numDirs = size(testData, 2);

    for trialIdx = 1:size(testData, 1)
        for dirIdx = 1:numDirs
            decodedHandPos = [];
            times = evalConfig.startMs:evalConfig.stepMs:size(testData(trialIdx, dirIdx).spikes, 2);

            for t = times
                sample = struct();
                sample.trialId = testData(trialIdx, dirIdx).trialId;
                sample.spikes = testData(trialIdx, dirIdx).spikes(:, 1:t);
                sample.decodedHandPos = decodedHandPos;
                sample.startHandPos = testData(trialIdx, dirIdx).handPos(1:2, 1);

                if predictReturnsState
                    [decodedX, decodedY, modelParameters] = positionEstimator(sample, modelParameters);
                else
                    [decodedX, decodedY] = positionEstimator(sample, modelParameters);
                end

                decodedPos = [decodedX; decodedY];
                decodedHandPos = [decodedHandPos, decodedPos]; %#ok<AGROW>
                actualPos = testData(trialIdx, dirIdx).handPos(1:2, t);
                meanSqError = meanSqError + norm(actualPos - decodedPos) ^ 2;
            end

            numPredictions = numPredictions + numel(times);
        end
    end

    rmse = sqrt(meanSqError / max(numPredictions, 1));
end

function processed = initialize_bin_width(dataStruct)
    processed = dataStruct;
    for idx = 1:numel(processed)
        processed(idx).bin_width = 1;
    end
end

function processed = rebin_dataset(dataStruct, binWidth)
    processed = dataStruct;
    for idx = 1:numel(processed)
        processed(idx).spikes = rebin_spike_matrix(processed(idx).spikes, binWidth);
        processed(idx).bin_width = processed(idx).bin_width * binWidth;
    end
end

function binned = rebin_spike_matrix(spikes, binWidth)
    unbinnedLength = size(spikes, 2);
    numBins = floor(unbinnedLength / binWidth);
    binned = zeros(size(spikes, 1), numBins);
    for binIdx = 1:numBins
        startIdx = (binIdx - 1) * binWidth + 1;
        endIdx = startIdx + binWidth - 1;
        binned(:, binIdx) = sum(spikes(:, startIdx:endIdx), 2);
    end
end

function processed = transform_dataset(dataStruct, transformName)
    processed = dataStruct;
    for idx = 1:numel(processed)
        processed(idx).spikes = apply_transform_to_matrix(processed(idx).spikes, transformName);
    end
end

function processed = smooth_dataset(dataStruct, kernelName, kernelWidth, kernelParam)
    processed = dataStruct;
    for idx = 1:numel(processed)
        processed(idx).spikes = apply_smoothing_to_matrix( ...
            processed(idx).spikes, kernelName, kernelWidth, kernelParam);
    end
end

function smoothed = apply_smoothing_to_matrix(dataMatrix, kernelName, kernelWidth, kernelParam)
    if nargin < 4 || isempty(kernelParam) || isnan(kernelParam)
        kernelParam = max(kernelWidth / 3, 1);
    end
    if nargin < 3 || isempty(kernelWidth)
        kernelWidth = 0;
    end
    if nargin < 2 || isempty(kernelName)
        kernelName = 'none';
    end

    kName = lower(string(kernelName));
    if kernelWidth <= 1 || kName == "none"
        smoothed = dataMatrix;
        return;
    end

    switch kName
        case "rect"
            kernel = ones(1, kernelWidth) / kernelWidth;
            smoothed = filter(kernel, 1, dataMatrix, [], 2);
        case {"gauss", "gaussian"}
            support = -kernelWidth:kernelWidth;
            sigma = max(kernelParam, eps);
            kernel = exp(-(support .^ 2) / (2 * sigma ^ 2));
            kernel = kernel / sum(kernel);
            smoothed = conv2(dataMatrix, kernel, 'same');
        case "cgauss"
            support = -kernelWidth:kernelWidth;
            sigma = max(kernelParam, eps);
            kernel = exp(-(support .^ 2) / (2 * sigma ^ 2));
            kernel(support < 0) = 0;
            kernel = kernel / max(sum(kernel), eps);
            smoothed = conv2(dataMatrix, kernel, 'same');
        otherwise
            error('Unknown smoothing kernel: %s', kernelName);
    end
end


function transformed = apply_transform_to_matrix(dataMatrix, transformName)
    switch lower(transformName)
        case 'none'
            transformed = dataMatrix;
        case 'sqrt'
            transformed = sqrt(dataMatrix);
        case 'anscombe'
            transformed = 2 * sqrt(dataMatrix + 3 / 8);
        otherwise
            error('Unknown transform: %s', transformName);
    end
end

function minLength = minimum_trial_length(dataStruct)
    minLength = inf;
    for idx = 1:numel(dataStruct)
        minLength = min(minLength, size(dataStruct(idx).handPos, 2));
    end
end

function row = empty_result_row()
    row = struct( ...
        'task', '', ...
        'family', '', ...
        'method', '', ...
        'classifier', '', ...
        'regressor', '', ...
        'preprocess', '', ...
        'fold', NaN, ...
        'configId', NaN, ...
        'configLabel', '', ...
        'k', NaN, ...
        'nPC', NaN, ...
        'lambda', NaN, ...
        'historyBins', NaN, ...
        'bufferShort', NaN, ...
        'bufferLong', NaN, ...
        'smoothWin', NaN, ...
        'rmse', NaN, ...
        'accuracy', NaN, ...
        'trainTimeSec', NaN, ...
        'testTimeSec', NaN, ...
        'timePerPredictionMs', NaN, ...
        'taskScore', NaN, ...
        'notes', '');
end

function row = build_result_row(runSpec, foldIdx, metrics)
    row = empty_result_row();
    row.task = runSpec.task;
    row.family = runSpec.family;
    row.method = runSpec.method;
    row.classifier = runSpec.classifier;
    row.regressor = runSpec.regressor;
    row.preprocess = runSpec.preprocess;
    row.fold = foldIdx;
    row.configId = runSpec.configId;
    row.configLabel = runSpec.configLabel;
    row.k = runSpec.k;
    row.nPC = runSpec.nPC;
    row.lambda = runSpec.lambda;
    row.historyBins = runSpec.historyBins;
    row.bufferShort = runSpec.bufferShort;
    row.bufferLong = runSpec.bufferLong;
    row.smoothWin = runSpec.smoothWin;
    row.rmse = metrics.rmse;
    row.accuracy = metrics.accuracy;
    row.trainTimeSec = metrics.trainTimeSec;
    row.testTimeSec = metrics.testTimeSec;
    row.timePerPredictionMs = metrics.timePerPredictionMs;
    row.taskScore = metrics.taskScore;
    row.notes = runSpec.notes;
end

function row = empty_timeresolved_row()
    row = struct( ...
        'task', '', ...
        'method', '', ...
        'configId', NaN, ...
        'configLabel', '', ...
        'k', NaN, ...
        'fold', NaN, ...
        't', NaN, ...
        'accuracy', NaN, ...
        'numPredictions', NaN);
end

function row = build_timeresolved_row(runSpec, foldIdx, t, accuracy, numPredictions)
    row = empty_timeresolved_row();
    row.task = runSpec.task;
    row.method = runSpec.method;
    row.configId = runSpec.configId;
    row.configLabel = runSpec.configLabel;
    row.k = runSpec.k;
    row.fold = foldIdx;
    row.t = t;
    row.accuracy = accuracy;
    row.numPredictions = numPredictions;
end

function tbl = empty_timeresolved_table()
    tbl = struct2table(repmat(empty_timeresolved_row(), 0, 1));
end

function metricString = format_metric_string(taskName, metrics)
    switch taskName
        case 'classifier'
            metricString = sprintf('accuracy = %.2f%% | train = %.3fs | test = %.3fs', ...
                metrics.accuracy, metrics.trainTimeSec, metrics.testTimeSec);
        otherwise
            metricString = sprintf('RMSE = %.4f | train = %.3fs | test = %.3fs', ...
                metrics.rmse, metrics.trainTimeSec, metrics.testTimeSec);
    end
end

function summaryTable = summarize_results(rawTable)
    if isempty(rawTable)
        summaryTable = struct2table(repmat(empty_summary_row(), 0, 1));
        return;
    end

    keys = make_group_keys(rawTable);
    [uniqueKeys, ~, groupIdx] = unique(keys);
    numGroups = numel(uniqueKeys);
    summaryRows = repmat(empty_summary_row(), numGroups, 1);

    for idx = 1:numGroups
        mask = groupIdx == idx;
        subset = rawTable(mask, :);
        template = subset(1, :);

        summaryRows(idx).task = template.task{1};
        summaryRows(idx).family = template.family{1};
        summaryRows(idx).method = template.method{1};
        summaryRows(idx).classifier = template.classifier{1};
        summaryRows(idx).regressor = template.regressor{1};
        summaryRows(idx).preprocess = template.preprocess{1};
        summaryRows(idx).configId = template.configId(1);
        summaryRows(idx).configLabel = template.configLabel{1};
        summaryRows(idx).k = template.k(1);
        summaryRows(idx).nPC = template.nPC(1);
        summaryRows(idx).lambda = template.lambda(1);
        summaryRows(idx).historyBins = template.historyBins(1);
        summaryRows(idx).bufferShort = template.bufferShort(1);
        summaryRows(idx).bufferLong = template.bufferLong(1);
        summaryRows(idx).smoothWin = template.smoothWin(1);
        summaryRows(idx).notes = template.notes{1};
        summaryRows(idx).numFolds = height(subset);

        rmseVals = subset.rmse(~isnan(subset.rmse));
        accVals = subset.accuracy(~isnan(subset.accuracy));
        trainVals = subset.trainTimeSec(~isnan(subset.trainTimeSec));
        testVals = subset.testTimeSec(~isnan(subset.testTimeSec));
        perPredVals = subset.timePerPredictionMs(~isnan(subset.timePerPredictionMs));
        taskVals = subset.taskScore(~isnan(subset.taskScore));

        if isempty(rmseVals)
            summaryRows(idx).meanRMSE = NaN;
            summaryRows(idx).stdRMSE = NaN;
        else
            summaryRows(idx).meanRMSE = mean(rmseVals);
            summaryRows(idx).stdRMSE = std(rmseVals);
        end

        if isempty(accVals)
            summaryRows(idx).meanAccuracy = NaN;
            summaryRows(idx).stdAccuracy = NaN;
        else
            summaryRows(idx).meanAccuracy = mean(accVals);
            summaryRows(idx).stdAccuracy = std(accVals);
        end

        summaryRows(idx).meanTrainTimeSec = mean(trainVals);
        summaryRows(idx).stdTrainTimeSec = std(trainVals);
        summaryRows(idx).meanTestTimeSec = mean(testVals);
        summaryRows(idx).stdTestTimeSec = std(testVals);
        summaryRows(idx).meanTimePerPredictionMs = mean(perPredVals);
        summaryRows(idx).stdTimePerPredictionMs = std(perPredVals);
        if isempty(taskVals)
            summaryRows(idx).meanTaskScore = NaN;
            summaryRows(idx).stdTaskScore = NaN;
        else
            summaryRows(idx).meanTaskScore = mean(taskVals);
            summaryRows(idx).stdTaskScore = std(taskVals);
        end
    end

    summaryTable = struct2table(summaryRows, 'AsArray', true);
end

function keys = make_group_keys(rawTable)
    keys = cell(height(rawTable), 1);
    for idx = 1:height(rawTable)
        keys{idx} = sprintf('%s|%s|%d', ...
            rawTable.task{idx}, rawTable.method{idx}, rawTable.configId(idx));
    end
end

function row = empty_summary_row()
    row = struct( ...
        'task', '', ...
        'family', '', ...
        'method', '', ...
        'classifier', '', ...
        'regressor', '', ...
        'preprocess', '', ...
        'configId', NaN, ...
        'configLabel', '', ...
        'k', NaN, ...
        'nPC', NaN, ...
        'lambda', NaN, ...
        'historyBins', NaN, ...
        'bufferShort', NaN, ...
        'bufferLong', NaN, ...
        'smoothWin', NaN, ...
        'meanRMSE', NaN, ...
        'stdRMSE', NaN, ...
        'meanAccuracy', NaN, ...
        'stdAccuracy', NaN, ...
        'meanTrainTimeSec', NaN, ...
        'stdTrainTimeSec', NaN, ...
        'meanTestTimeSec', NaN, ...
        'stdTestTimeSec', NaN, ...
        'meanTimePerPredictionMs', NaN, ...
        'stdTimePerPredictionMs', NaN, ...
        'meanTaskScore', NaN, ...
        'stdTaskScore', NaN, ...
        'numFolds', NaN, ...
        'notes', '');
end

function bestTable = select_best_configs(summaryTable, metricName, direction)
    if isempty(summaryTable)
        bestTable = summaryTable([], :);
        return;
    end

    methods = unique(summaryTable.method);
    keepRows = false(height(summaryTable), 1);

    for idx = 1:numel(methods)
        mask = strcmp(summaryTable.method, methods{idx});
        subsetIdx = find(mask);
        subset = summaryTable(mask, :);

        switch metricName
            case 'accuracy'
                metricVals = subset.meanAccuracy;
            case 'rmse'
                metricVals = subset.meanRMSE;
            case 'taskScore'
                metricVals = subset.meanTaskScore;
            otherwise
                error('Unsupported metric for best-config selection: %s', metricName);
        end

        if strcmp(direction, 'descend')
            [~, localIdx] = max(metricVals);
        else
            [~, localIdx] = min(metricVals);
        end

        keepRows(subsetIdx(localIdx)) = true;
    end

    bestTable = summaryTable(keepRows, :);
end

function parameterTable = build_optimized_parameter_table(classifierBest, continuousBest)
    if isempty(classifierBest) && isempty(continuousBest)
        parameterTable = table();
        return;
    end

    merged = [classifierBest; continuousBest];
    keepVars = {'task', 'family', 'method', 'classifier', 'regressor', ...
        'configId', 'configLabel', 'k', 'nPC', 'lambda', 'historyBins', ...
        'bufferShort', 'bufferLong', 'smoothWin', 'notes'};
    parameterTable = merged(:, keepVars);
end

function paths = write_benchmark_outputs(outputDir, opts, evalConfig, runSpecsTable, foldTable, rawTable, timeResolvedTable, classifierSummary, classifierBest, continuousSummary, optimizedPerformance, optimizedParameters)
    paths = struct();
    paths.outputDir = outputDir;

    paths.evalConfigMat = fullfile(outputDir, 'eval_config.mat');
    save(paths.evalConfigMat, 'evalConfig', 'opts', 'runSpecsTable');

    paths.benchmarkPlanCsv = fullfile(outputDir, 'benchmark_plan.csv');
    writetable(runSpecsTable, paths.benchmarkPlanCsv);

    paths.foldManifestMat = fullfile(outputDir, 'fold_manifest.mat');
    save(paths.foldManifestMat, 'foldTable');

    paths.foldManifestCsv = fullfile(outputDir, 'fold_manifest.csv');
    writetable(foldTable, paths.foldManifestCsv);

    paths.rawResultsCsv = fullfile(outputDir, 'raw_results.csv');
    writetable(rawTable, paths.rawResultsCsv);

    paths.classifierSummaryCsv = fullfile(outputDir, 'classifier_summary.csv');
    writetable(classifierSummary, paths.classifierSummaryCsv);

    paths.classifierBestCsv = fullfile(outputDir, 'classifier_best.csv');
    writetable(classifierBest, paths.classifierBestCsv);

    paths.classifierTimeResolvedCsv = fullfile(outputDir, 'classifier_timeresolved.csv');
    writetable(timeResolvedTable, paths.classifierTimeResolvedCsv);

    paths.continuousSummaryCsv = fullfile(outputDir, 'continuous_summary.csv');
    writetable(continuousSummary, paths.continuousSummaryCsv);

    paths.optimizedPerformanceCsv = fullfile(outputDir, 'optimized_performance.csv');
    writetable(optimizedPerformance, paths.optimizedPerformanceCsv);

    paths.optimizedParametersCsv = fullfile(outputDir, 'optimized_parameters.csv');
    writetable(optimizedParameters, paths.optimizedParametersCsv);

    paths.bundleMat = fullfile(outputDir, 'benchmark_bundle.mat');
    save(paths.bundleMat, 'opts', 'runSpecsTable', 'foldTable', 'rawTable', ...
        'timeResolvedTable', 'classifierSummary', 'classifierBest', ...
        'continuousSummary', 'optimizedPerformance', 'optimizedParameters');
end

function figureFiles = generate_benchmark_figures(outputDir, classifierSummary, continuousSummary, optimizedPerformance, timeResolvedTable)
    figureFiles = {};

    classifierFigure = plot_classifier_accuracy_curve(outputDir, classifierSummary);
    if ~isempty(classifierFigure)
        figureFiles{end + 1} = classifierFigure; %#ok<AGROW>
    end

    timeFigure = plot_classifier_accuracy_vs_time(outputDir, timeResolvedTable);
    if ~isempty(timeFigure)
        figureFiles{end + 1} = timeFigure;
    end

    hyperFiles = plot_rmse_hyperparameter_curves(outputDir, continuousSummary);
    for idx = 1:numel(hyperFiles)
        figureFiles{end + 1} = hyperFiles{idx}; %#ok<AGROW>
    end

    taskHyperFiles = plot_taskscore_hyperparameter_curves(outputDir, continuousSummary);
    for idx = 1:numel(taskHyperFiles)
        figureFiles{end + 1} = taskHyperFiles{idx}; %#ok<AGROW>
    end


    comparisonFigure = plot_with_vs_without_classifier(outputDir, optimizedPerformance);
    if ~isempty(comparisonFigure)
        figureFiles{end + 1} = comparisonFigure; %#ok<AGROW>
    end

    runtimeFigure = plot_runtime_scatter(outputDir, optimizedPerformance);
    if ~isempty(runtimeFigure)
        figureFiles{end + 1} = runtimeFigure; %#ok<AGROW>
    end

    taskScoreFigure = plot_taskscore_bar(outputDir, optimizedPerformance);
    if ~isempty(taskScoreFigure)
        figureFiles{end + 1} = taskScoreFigure;
    end
end

function figureFile = plot_classifier_accuracy_curve(outputDir, classifierSummary)
    figureFile = '';
    if isempty(classifierSummary)
        return;
    end

    knnMask = strcmp(classifierSummary.method, 'knn_ming');
    knnSummary = classifierSummary(knnMask, :);
    if isempty(knnSummary) || numel(unique(knnSummary.k)) < 2
        return;
    end

    [xVals, order] = sort(knnSummary.k);
    yVals = knnSummary.meanAccuracy(order);

    fig = figure('Visible', 'off');
    plot(xVals, yVals, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('k');
    ylabel('Mean CV accuracy (%)');
    title('k-NN classifier screening');

    figureFile = fullfile(outputDir, 'classifier_accuracy_vs_k.png');
    saveas(fig, figureFile);
    close(fig);
end

function figureFiles = plot_rmse_hyperparameter_curves(outputDir, continuousSummary)
    figureFiles = {};
    if isempty(continuousSummary)
        return;
    end

    methods = {'pooled_pcr', 'pooled_ridge_pcr'};
    for methodIdx = 1:numel(methods)
        methodName = methods{methodIdx};
        mask = strcmp(continuousSummary.method, methodName);
        subset = continuousSummary(mask, :);
        if isempty(subset) || numel(unique(subset.nPC(~isnan(subset.nPC)))) < 2
            continue;
        end

        fig = figure('Visible', 'off');
        hold on;
        grid on;

        lambdaVals = unique(subset.lambda(~isnan(subset.lambda)));
        if isempty(lambdaVals)
            lambdaVals = NaN;
        end

        if numel(lambdaVals) == 1 && isnan(lambdaVals)
            [xVals, order] = sort(subset.nPC);
            yVals = subset.meanRMSE(order);
            plot(xVals, yVals, '-o', 'LineWidth', 1.5);
        elseif numel(lambdaVals) <= 1
            [xVals, order] = sort(subset.nPC);
            yVals = subset.meanRMSE(order);
            plot(xVals, yVals, '-o', 'LineWidth', 1.5, ...
                'DisplayName', sprintf('\\lambda = %.3g', lambdaVals(1)));
            legend('Location', 'best');
        else
            for lambdaIdx = 1:numel(lambdaVals)
                lambdaVal = lambdaVals(lambdaIdx);
                lineMask = abs(subset.lambda - lambdaVal) < 1e-12;
                lineSubset = subset(lineMask, :);
                [xVals, order] = sort(lineSubset.nPC);
                yVals = lineSubset.meanRMSE(order);
                plot(xVals, yVals, '-o', 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('\\lambda = %.3g', lambdaVal));
            end
            legend('Location', 'best');
        end

        xlabel('Number of principal components');
        ylabel('Mean CV RMSE');
        title(sprintf('RMSE vs nPC: %s', pretty_method_name(methodName)));

        figureFile = fullfile(outputDir, sprintf('rmse_vs_npc_%s.png', methodName));
        saveas(fig, figureFile);
        close(fig);
        figureFiles{end + 1} = figureFile; %#ok<AGROW>
    end
end

function figureFiles = plot_taskscore_hyperparameter_curves(outputDir, continuousSummary)
    figureFiles = {};
    if isempty(continuousSummary)
        return;
    end

    methods = {'pooled_pcr', 'pooled_ridge_pcr'};
    for methodIdx = 1:numel(methods)
        methodName = methods{methodIdx};
        mask = strcmp(continuousSummary.method, methodName);
        subset = continuousSummary(mask, :);
        if isempty(subset) || numel(unique(subset.nPC(~isnan(subset.nPC)))) < 2
            continue;
        end

        validRows = ~isnan(subset.meanTaskScore);
        subset = subset(validRows, :);
        if isempty(subset)
            continue;
        end

        fig = figure('Visible', 'off');
        hold on;
        grid on;

        lambdaVals = unique(subset.lambda(~isnan(subset.lambda)));
        if isempty(lambdaVals)
            lambdaVals = NaN;
        end

        if numel(lambdaVals) == 1 && isnan(lambdaVals)
            [xVals, order] = sort(subset.nPC);
            yVals = subset.meanTaskScore(order);
            plot(xVals, yVals, '-o', 'LineWidth', 1.5);
        elseif numel(lambdaVals) <= 1
            [xVals, order] = sort(subset.nPC);
            yVals = subset.meanTaskScore(order);
            plot(xVals, yVals, '-o', 'LineWidth', 1.5, ...
                'DisplayName', sprintf('\\lambda = %.3g', lambdaVals(1)));
            legend('Location', 'best');
        else
            for lambdaIdx = 1:numel(lambdaVals)
                lambdaVal = lambdaVals(lambdaIdx);
                lineMask = abs(subset.lambda - lambdaVal) < 1e-12;
                lineSubset = subset(lineMask, :);
                [xVals, order] = sort(lineSubset.nPC);
                yVals = lineSubset.meanTaskScore(order);
                plot(xVals, yVals, '-o', 'LineWidth', 1.5, ...
                    'DisplayName', sprintf('\\lambda = %.3g', lambdaVal));
            end
            legend('Location', 'best');
        end

        xlabel('Number of principal components');
        ylabel('Mean CV Task Score');
        title(sprintf('Task Score vs nPC: %s', pretty_method_name(methodName)));

        figureFile = fullfile(outputDir, sprintf('taskscore_vs_npc_%s.png', methodName));
        saveas(fig, figureFile);
        close(fig);
        figureFiles{end + 1} = figureFile; %#ok<AGROW>
    end
end

function figureFile = plot_with_vs_without_classifier(outputDir, optimizedPerformance)
    figureFile = '';
    if isempty(optimizedPerformance)
        return;
    end

    methods = {'jared_direct', 'jared_hybrid'};
    mask = ismember(optimizedPerformance.method, methods);
    subset = optimizedPerformance(mask, :);
    if height(subset) ~= 2
        return;
    end

    [~, order] = ismember(methods, subset.method);
    subset = subset(order, :);

    fig = figure('Visible', 'off');
    meanVals = subset.meanRMSE;
    stdVals = subset.stdRMSE;
    bar(meanVals);
    hold on;
    errorbar(1:numel(meanVals), meanVals, stdVals, '.k', 'LineWidth', 1.25);
    set(gca, 'XTick', 1:numel(methods), ...
        'XTickLabel', {'Direct pooled regressor', 'Classifier-first pipeline'});
    ylabel('Mean CV RMSE');
    title('With vs without classifier');
    grid on;

    figureFile = fullfile(outputDir, 'with_vs_without_classifier.png');
    saveas(fig, figureFile);
    close(fig);
end

function figureFile = plot_runtime_scatter(outputDir, optimizedPerformance)
    figureFile = '';
    if isempty(optimizedPerformance)
        return;
    end

    fig = figure('Visible', 'off');
    scatter(optimizedPerformance.meanTestTimeSec, optimizedPerformance.meanRMSE, 60, 'filled');
    grid on;
    xlabel('Mean test time per fold (s)');
    ylabel('Mean CV RMSE');
    title('RMSE vs test-time cost');
    labels = optimizedPerformance.method;
    text(optimizedPerformance.meanTestTimeSec, optimizedPerformance.meanRMSE, ...
        labels, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left');

    figureFile = fullfile(outputDir, 'rmse_vs_test_time.png');
    saveas(fig, figureFile);
    close(fig);
end

function label = pretty_method_name(methodName)
    label = strrep(methodName, '_', ' ');
end

function figureFile = plot_classifier_accuracy_vs_time(outputDir, timeResolvedTable)
    figureFile = '';
    if isempty(timeResolvedTable) || height(timeResolvedTable) == 0
        return;
    end

    methods = unique(timeResolvedTable.method);
    if isempty(methods)
        return;
    end

    fig = figure('Visible', 'off');
    hold on;
    grid on;

    for idx = 1:numel(methods)
        methodName = methods{idx};
        methodMask = strcmp(timeResolvedTable.method, methodName);
        methodRows = timeResolvedTable(methodMask, :);

        configIds = unique(methodRows.configId);
        for cIdx = 1:numel(configIds)
            configMask = methodRows.configId == configIds(cIdx);
            subset = methodRows(configMask, :);

            timesUnique = unique(subset.t);
            meanAcc = zeros(size(timesUnique));
            for tIdx = 1:numel(timesUnique)
                tMask = subset.t == timesUnique(tIdx);
                meanAcc(tIdx) = mean(subset.accuracy(tMask));
            end

            if ~isempty(subset.configLabel) && iscell(subset.configLabel)
                labelText = sprintf('%s (%s)', pretty_method_name(methodName), subset.configLabel{1});
            else
                labelText = pretty_method_name(methodName);
            end
            plot(timesUnique, meanAcc, '-o', 'LineWidth', 1.5, 'DisplayName', labelText);
        end
    end

    xlabel('Time since movement onset (ms)');
    ylabel('Mean CV accuracy (%)');
    title('Time-resolved classifier accuracy');
    legend('Location', 'southeast');

    figureFile = fullfile(outputDir, 'classifier_accuracy_vs_time.png');
    saveas(fig, figureFile);
    close(fig);
end

function figureFile = plot_taskscore_bar(outputDir, optimizedPerformance)
    figureFile = '';
    if isempty(optimizedPerformance) || height(optimizedPerformance) == 0
        return;
    end
    if ~ismember('meanTaskScore', optimizedPerformance.Properties.VariableNames)
        return;
    end

    valid = ~isnan(optimizedPerformance.meanTaskScore);
    subset = optimizedPerformance(valid, :);
    if isempty(subset)
        return;
    end

    [~, order] = sort(subset.meanTaskScore);
    subset = subset(order, :);

    fig = figure('Visible', 'off');
    bar(subset.meanTaskScore);
    hold on;
    if ismember('stdTaskScore', subset.Properties.VariableNames)
        errorbar(1:height(subset), subset.meanTaskScore, subset.stdTaskScore, ...
            '.k', 'LineWidth', 1.25);
    end
    labels = cell(height(subset), 1);
    for idx = 1:height(subset)
        labels{idx} = sprintf('%s (%s)', ...
            pretty_method_name(subset.method{idx}), subset.configLabel{idx});
    end
    set(gca, 'XTick', 1:height(subset), 'XTickLabel', labels, 'XTickLabelRotation', 30);
    ylabel('Mean Task Score (0.9 RMSE + 0.1 Time)');
    title('Optimised continuous methods ranked by Task Score');
    grid on;

    figureFile = fullfile(outputDir, 'optimized_taskscore.png');
    saveas(fig, figureFile);
    close(fig);
end
