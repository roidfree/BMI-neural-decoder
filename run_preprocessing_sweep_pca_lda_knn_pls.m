 function results = run_preprocessing_sweep_pca_lda_knn_pls(varargin)
%RUN_PREPROCESSING_SWEEP_PCA_LDA_KNN_PLS
% Preprocessing sweep for:
%   - Classifier: PCA -> LDA -> kNN
%   - Regressor: direction-specific PLS (selected by predicted direction)
%
% Swept preprocessing knobs:
%   Spike-count family:
%     - bin width in [5, 10, 15, 20, 40]
%     - transform in {'none','anscombe'}
%   Gaussian family:
%     - sigma in [3, 5, 10, 20]
%     - sampling step in [5, 10, 15, 20, 40]
%     - transform in {'none','sqrt'}
%     - Gaussian total width fixed to (2*sigma + 1)
%
% Notes:
%   - Spike-count path: rebin (sum counts) -> optional Anscombe.
%   - Gaussian path: smooth -> sample (no summing after smoothing) -> optional sqrt.
%   - CV scoring matches dameer_grader timing style (predict from 320 ms every 20 ms).
%
% Usage:
%   results = run_preprocessing_sweep_pca_lda_knn_pls();
%   results = run_preprocessing_sweep_pca_lda_knn_pls(struct('K',5,'seed',42));
%
% Returns:
%   results struct containing raw table, summary table, best config, and sweep options.

    userOpts = struct();
    if nargin >= 1 && isstruct(varargin{1})
        userOpts = varargin{1};
    end

    opts = default_opts();
    opts = merge_structs(opts, userOpts);
    opts.dataFile = resolve_data_file(opts.dataFile);

    loaded = load(opts.dataFile, 'trial');
    trial = loaded.trial;

    numTrials = size(trial, 1);
    folds = make_folds(numTrials, opts.K, opts.seed);
    sweepConfigs = build_sweep_configs();

    rawRows = repmat(empty_row(), 0, 1);

    fprintf('Running preprocessing sweep (%d configs, %d folds)\n', numel(sweepConfigs), numel(folds));
    for cfgIdx = 1:numel(sweepConfigs)
        cfg = sweepConfigs(cfgIdx);
        fprintf('[%3d/%3d] %s\n', cfgIdx, numel(sweepConfigs), cfg.label);

        for foldIdx = 1:numel(folds)
            trainData = trial(folds(foldIdx).trainIdx, :);
            testData = trial(folds(foldIdx).testIdx, :);

            metrics = evaluate_fold(trainData, testData, cfg, opts);
            row = empty_row();
            row.family = cfg.family;
            row.configLabel = cfg.label;
            row.fold = foldIdx;
            row.rmse = metrics.rmse;
            row.accuracy = metrics.accuracy;
            row.trainTimeSec = metrics.trainTimeSec;
            row.testTimeSec = metrics.testTimeSec;
            row.kNN = opts.knnK;
            row.nPLS = opts.nPLS;
            row.historyBins = opts.historyBins;
            row.binWidth = cfg.binWidth;
            row.sigma = cfg.sigma;
            row.sampleStep = cfg.sampleStep;
            row.transform = cfg.transform;
            rawRows(end + 1) = row; %#ok<AGROW>
        end
    end

    rawTable = struct2table(rawRows);
    summaryTable = groupsummary(rawTable, {'family','configLabel','binWidth','sigma','sampleStep','transform'}, ...
        {'mean','std'}, {'rmse','accuracy','trainTimeSec','testTimeSec'});
    summaryTable = sortrows(summaryTable, {'mean_rmse', 'mean_accuracy'}, {'ascend', 'descend'});

    bestConfig = summaryTable(1, :);

    results = struct();
    results.raw = rawTable;
    results.summary = summaryTable;
    results.best = bestConfig;
    results.options = opts;

    if opts.saveOutputs
        writetable(rawTable, opts.rawCsv);
        writetable(summaryTable, opts.summaryCsv);
        save(opts.matFile, 'results');
    end

    disp('=== BEST PREPROCESSING CONFIG (lowest mean RMSE) ===');
    disp(bestConfig);
end

function opts = default_opts()
    root = fileparts(fileparts(mfilename('fullpath')));
    opts = struct();
    % Default to monkeydata0.mat in the repo root so this matches the
    % coursework grader-style data location.
    opts.dataFile = fullfile(root, 'monkeydata0.mat');
    opts.K = 5;
    opts.seed = 42;
    opts.startMs = 320;
    opts.stepMs = 20;
    opts.historyBins = 15;
    opts.knnK = 11;
    opts.nPLS = 25;
    opts.pcaVarKeep = 0.99;
    opts.saveOutputs = true;
    opts.rawCsv = fullfile(root, 'sweeps', 'preprocess_sweep_raw.csv');
    opts.summaryCsv = fullfile(root, 'sweeps', 'preprocess_sweep_summary.csv');
    opts.matFile = fullfile(root, 'sweeps', 'preprocess_sweep_results.mat');
end

function out = merge_structs(base, overrides)
    out = base;
    names = fieldnames(overrides);
    for i = 1:numel(names)
        out.(names{i}) = overrides.(names{i});
    end
end

function dataFile = resolve_data_file(dataFileInput)
    dataFile = dataFileInput;
    if exist(dataFile, 'file') == 2
        return;
    end

    [scriptDir, ~, ~] = fileparts(mfilename('fullpath'));
    repoRoot = fileparts(scriptDir);
    [~, fileName, fileExt] = fileparts(dataFileInput);
    targetName = [fileName, fileExt];

    candidateList = {};
    if ~isempty(fileparts(dataFileInput))
        candidateList{end + 1} = dataFileInput; %#ok<AGROW>
    else
        % Common places users run from.
        candidateList{end + 1} = fullfile(repoRoot, dataFileInput); %#ok<AGROW>
        candidateList{end + 1} = fullfile(scriptDir, dataFileInput); %#ok<AGROW>
        candidateList{end + 1} = fullfile(pwd, dataFileInput); %#ok<AGROW>

        % Also scan one folder below cwd/scriptDir (covers layouts like:
        % Downloads/BMI-neural-decoder-main 3/monkeydata0.mat).
        candidateList = [candidateList, find_file_one_level_down(pwd, targetName)]; %#ok<AGROW>
        candidateList = [candidateList, find_file_one_level_down(scriptDir, targetName)]; %#ok<AGROW>
    end

    for i = 1:numel(candidateList)
        if exist(candidateList{i}, 'file') == 2
            dataFile = candidateList{i};
            return;
        end
    end

    % Backward-compatible fallback for repos that still store the training
    % set as monkeydata_training.mat.
    legacyCandidate = fullfile(repoRoot, 'monkeydata_training.mat');
    if exist(legacyCandidate, 'file') == 2
        dataFile = legacyCandidate;
        warning('run_preprocessing_sweep_pca_lda_knn_pls:DataFileFallback', ...
            ['Requested data file not found: %s. Falling back to %s. ', ...
             'Set opts.dataFile explicitly to silence this warning.'], ...
            dataFileInput, legacyCandidate);
        return;
    end

    error(['Data file not found: %s\n', ...
           'Also checked fallback: %s\n', ...
           'Tip: pass opts.dataFile, e.g. run_preprocessing_sweep_pca_lda_knn_pls(struct(''dataFile'',''/path/to/monkeydata0.mat''));'], ...
           dataFileInput, legacyCandidate);
end

function found = find_file_one_level_down(baseDir, targetName)
    found = {};
    if exist(baseDir, 'dir') ~= 7
        return;
    end

    entries = dir(baseDir);
    for i = 1:numel(entries)
        if ~entries(i).isdir
            continue;
        end
        dname = entries(i).name;
        if strcmp(dname, '.') || strcmp(dname, '..')
            continue;
        end
        candidate = fullfile(baseDir, dname, targetName);
        if exist(candidate, 'file') == 2
            found{end + 1} = candidate; %#ok<AGROW>
        end
    end
end

function cfgs = build_sweep_configs()
    binWidths = [5, 10, 15, 20, 40];
    spikeTransforms = {'none', 'anscombe'};

    sigmas = [3, 5, 10, 20];
    sampleSteps = [5, 10, 15, 20, 40];
    gaussTransforms = {'none', 'sqrt'};

    cfgs = repmat(struct('family','','label','','binWidth',NaN,'sigma',NaN,'sampleStep',NaN,'transform',''), 0, 1);

    for b = binWidths
        for t = 1:numel(spikeTransforms)
            cfg = struct();
            cfg.family = 'spike_count';
            cfg.binWidth = b;
            cfg.sigma = NaN;
            cfg.sampleStep = b;
            cfg.transform = spikeTransforms{t};
            cfg.label = sprintf('spike_count_bw%d_%s', b, spikeTransforms{t});
            cfgs(end + 1) = cfg; %#ok<AGROW>
        end
    end

    for s = sigmas
        for st = sampleSteps
            for t = 1:numel(gaussTransforms)
                cfg = struct();
                cfg.family = 'gaussian';
                cfg.binWidth = NaN;
                cfg.sigma = s;
                cfg.sampleStep = st;
                cfg.transform = gaussTransforms{t};
                cfg.label = sprintf('gaussian_sigma%d_step%d_%s', s, st, gaussTransforms{t});
                cfgs(end + 1) = cfg; %#ok<AGROW>
            end
        end
    end
end

function folds = make_folds(numTrials, K, seed)
    rng(seed);
    idx = randperm(numTrials);
    edges = round(linspace(0, numTrials, K + 1));
    folds = repmat(struct('trainIdx',[],'testIdx',[]), K, 1);
    for k = 1:K
        testIdx = idx(edges(k) + 1:edges(k + 1));
        trainMask = true(1, numTrials);
        trainMask(testIdx) = false;
        folds(k).testIdx = testIdx;
        folds(k).trainIdx = find(trainMask);
    end
end

function metrics = evaluate_fold(trainData, testData, cfg, opts)
    tTrain = tic;

    clsModel = train_classifier_pca_lda_knn(trainData, cfg, opts);
    regModels = train_directional_pls_regressors(trainData, cfg, opts);

    trainTimeSec = toc(tTrain);

    tTest = tic;
    [rmse, acc] = score_pipeline(testData, clsModel, regModels, cfg, opts);
    testTimeSec = toc(tTest);

    metrics = struct('rmse', rmse, 'accuracy', acc, ...
        'trainTimeSec', trainTimeSec, 'testTimeSec', testTimeSec);
end

function model = train_classifier_pca_lda_knn(trainData, cfg, opts)
    [X, y] = build_classifier_dataset(trainData, cfg, opts.startMs);

    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    sigma(sigma < eps) = 1;
    Xz = (X - mu) ./ sigma;

    [~, S, V] = svd(Xz, 'econ');
    eigVals = diag(S).^2;
    frac = cumsum(eigVals) / max(sum(eigVals), eps);
    nPC = find(frac >= opts.pcaVarKeep, 1, 'first');
    if isempty(nPC)
        nPC = min(size(V, 2), 30);
    end
    PC = V(:, 1:nPC);
    Xp = Xz * PC;

    [Wlda, ldaMeans, ldaPriors] = fit_lda_projection(Xp, y);
    Xlda = Xp * Wlda;

    model = struct();
    model.mu = mu;
    model.sigma = sigma;
    model.PC = PC;
    model.Wlda = Wlda;
    model.ldaMeans = ldaMeans;
    model.ldaPriors = ldaPriors;
    model.XldaTrain = Xlda;
    model.yTrain = y;
    model.k = opts.knnK;
end

function [X, y] = build_classifier_dataset(data, cfg, horizonMs)
    [nTrials, nDirs] = size(data);
    nNeurons = size(data(1,1).spikes, 1);
    X = zeros(nTrials * nDirs, nNeurons);
    y = zeros(nTrials * nDirs, 1);

    row = 1;
    for tr = 1:nTrials
        for d = 1:nDirs
            proc = preprocess_spikes(data(tr,d).spikes, cfg);
            featBins = max(1, floor(horizonMs / feature_dt(cfg)));
            featBins = min(featBins, size(proc, 2));
            X(row, :) = sum(proc(:, 1:featBins), 2)';
            y(row) = d;
            row = row + 1;
        end
    end
end

function [W, classMeans, priors] = fit_lda_projection(X, y)
    classes = unique(y);
    c = numel(classes);
    [n, d] = size(X);

    muGlobal = mean(X, 1);
    Sw = zeros(d, d);
    Sb = zeros(d, d);
    classMeans = zeros(c, d);
    priors = zeros(c, 1);

    for i = 1:c
        mask = (y == classes(i));
        Xi = X(mask, :);
        mui = mean(Xi, 1);
        classMeans(i, :) = mui;
        priors(i) = size(Xi, 1) / n;

        centered = Xi - mui;
        Sw = Sw + centered' * centered;

        dm = (mui - muGlobal)';
        Sb = Sb + size(Xi, 1) * (dm * dm');
    end

    reg = 1e-6 * trace(Sw) / max(d, 1);
    Sw = Sw + reg * eye(d);
    [eigVecs, eigVals] = eig(Sb, Sw);
    [~, ord] = sort(real(diag(eigVals)), 'descend');
    numComp = max(1, c - 1);
    W = real(eigVecs(:, ord(1:numComp)));
end

function regModels = train_directional_pls_regressors(trainData, cfg, opts)
    [~, nDirs] = size(trainData);
    regModels = cell(1, nDirs);

    for d = 1:nDirs
        [X, Y] = build_pls_dataset_for_direction(trainData(:, d), cfg, opts.historyBins);

        nComp = min([opts.nPLS, size(X, 2), size(X, 1) - 1]);
        if nComp < 1
            nComp = 1;
        end

        [~, ~, ~, ~, BETA] = plsregress(X, Y, nComp);

        model = struct();
        model.BETA = BETA;
        model.nComp = nComp;
        model.historyBins = opts.historyBins;
        model.dt = feature_dt(cfg);
        model.cfg = cfg;
        regModels{d} = model;
    end
end

function [X, Y] = build_pls_dataset_for_direction(dirTrials, cfg, historyBins)
    nTrials = numel(dirTrials);
    nNeurons = size(dirTrials(1).spikes, 1);

    rowsEstimate = 0;
    for i = 1:nTrials
        p = preprocess_spikes(dirTrials(i).spikes, cfg);
        rowsEstimate = rowsEstimate + max(0, size(p,2) - historyBins);
    end

    X = zeros(rowsEstimate, nNeurons * (historyBins + 1));
    Y = zeros(rowsEstimate, 2);

    row = 1;
    dt = feature_dt(cfg);
    for i = 1:nTrials
        spikesRaw = dirTrials(i).spikes;
        handPos = dirTrials(i).handPos;
        p = preprocess_spikes(spikesRaw, cfg);

        maxT = min(size(p,2), floor(size(handPos,2) / dt));
        for t = (historyBins + 1):maxT
            window = p(:, t-historyBins:t);
            X(row, :) = reshape(window, 1, []);

            curMs = t * dt;
            prevMs = (t - 1) * dt;
            Y(row, 1) = handPos(1, curMs) - handPos(1, prevMs);
            Y(row, 2) = handPos(2, curMs) - handPos(2, prevMs);
            row = row + 1;
        end
    end

    X = X(1:row-1, :);
    Y = Y(1:row-1, :);
end

function [rmse, accuracy] = score_pipeline(testData, clsModel, regModels, cfg, opts)
    sqErr = 0;
    nPred = 0;
    nCorrect = 0;
    nCls = 0;

    [nTrials, nDirs] = size(testData);

    for tr = 1:nTrials
        for trueDir = 1:nDirs
            sample = testData(tr, trueDir);

            predDir = predict_direction(sample.spikes, clsModel, cfg, opts.startMs);
            nCorrect = nCorrect + double(predDir == trueDir);
            nCls = nCls + 1;

            decoded = [];
            times = opts.startMs:opts.stepMs:size(sample.spikes, 2);
            for t = times
                s.spikes = sample.spikes(:, 1:t);
                s.decodedHandPos = decoded;
                s.startHandPos = sample.handPos(1:2, 1);

                [xhat, yhat] = predict_position_pls(s, regModels{predDir});
                decoded = [decoded, [xhat; yhat]]; %#ok<AGROW>

                actual = sample.handPos(1:2, t);
                sqErr = sqErr + sum((actual - [xhat; yhat]).^2);
                nPred = nPred + 1;
            end
        end
    end

    rmse = sqrt(sqErr / max(nPred, 1));
    accuracy = 100 * nCorrect / max(nCls, 1);
end

function predDir = predict_direction(spikesRaw, model, cfg, horizonMs)
    proc = preprocess_spikes(spikesRaw, cfg);
    featBins = max(1, floor(horizonMs / feature_dt(cfg)));
    featBins = min(featBins, size(proc, 2));
    x = sum(proc(:, 1:featBins), 2)';

    xz = (x - model.mu) ./ model.sigma;
    xp = xz * model.PC;
    xlda = xp * model.Wlda;

    dist2 = sum((model.XldaTrain - xlda).^2, 2);
    [~, ord] = sort(dist2, 'ascend');
    k = min(model.k, numel(ord));
    predDir = mode(model.yTrain(ord(1:k)));
end

function [xhat, yhat] = predict_position_pls(testSample, model)
    proc = preprocess_spikes(testSample.spikes, model.cfg);
    needBins = model.historyBins + 1;
    nBins = size(proc, 2);
    nNeurons = size(proc, 1);

    if nBins < needBins
        padded = zeros(nNeurons, needBins);
        padded(:, end-nBins+1:end) = proc;
        recent = padded;
    else
        recent = proc(:, end-needBins+1:end);
    end

    Xrow = reshape(recent, 1, []);
    v = [1, Xrow] * model.BETA;

    if isempty(testSample.decodedHandPos)
        prev = testSample.startHandPos;
    else
        prev = testSample.decodedHandPos(:, end);
    end

    xhat = prev(1) + v(1);
    yhat = prev(2) + v(2);
end

function proc = preprocess_spikes(spikes, cfg)
    switch cfg.family
        case 'spike_count'
            proc = rebin_sum(spikes, cfg.binWidth);
            proc = apply_transform(proc, cfg.transform);
        case 'gaussian'
            kernel = gaussian_kernel(cfg.sigma);
            smooth = conv2(spikes, kernel, 'same');
            proc = smooth(:, 1:cfg.sampleStep:end);
            proc = apply_transform(proc, cfg.transform);
        otherwise
            error('Unknown family: %s', cfg.family);
    end
end

function dt = feature_dt(cfg)
    switch cfg.family
        case 'spike_count'
            dt = cfg.binWidth;
        case 'gaussian'
            dt = cfg.sampleStep;
        otherwise
            error('Unknown family: %s', cfg.family);
    end
end

function out = rebin_sum(spikes, bw)
    n = size(spikes, 2);
    nBins = floor(n / bw);
    out = zeros(size(spikes, 1), nBins);
    for i = 1:nBins
        s = (i - 1) * bw + 1;
        e = i * bw;
        out(:, i) = sum(spikes(:, s:e), 2);
    end
end

function k = gaussian_kernel(sigma)
    half = sigma;
    x = -half:half;
    k = exp(-(x.^2) / (2 * sigma^2));
    k = k / sum(k);
    % Length is always exactly 2*sigma + 1.
end

function out = apply_transform(x, tname)
    switch lower(tname)
        case 'none'
            out = x;
        case 'sqrt'
            out = sqrt(x);
        case 'anscombe'
            out = 2 * sqrt(x + 3/8);
        otherwise
            error('Unknown transform: %s', tname);
    end
end

function row = empty_row()
    row = struct( ...
        'family', '', ...
        'configLabel', '', ...
        'fold', NaN, ...
        'rmse', NaN, ...
        'accuracy', NaN, ...
        'trainTimeSec', NaN, ...
        'testTimeSec', NaN, ...
        'kNN', NaN, ...
        'nPLS', NaN, ...
        'historyBins', NaN, ...
        'binWidth', NaN, ...
        'sigma', NaN, ...
        'sampleStep', NaN, ...
        'transform', '');
end