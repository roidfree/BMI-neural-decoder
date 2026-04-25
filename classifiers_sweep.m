clc; clear; close all;

%% --- Configuration & Paths ---
repoRoot = fileparts(fileparts(mfilename('fullpath')));
dataFile = fullfile(repoRoot, 'BMI-neural-decoder-main 3', 'monkeydata0.mat');
outDir = fullfile(repoRoot, 'benchmark_outputs', 'full_param_sweep_final');
if ~exist(outDir, 'dir'), mkdir(outDir); end

cfg = struct();
cfg.binWidth = 20;
cfg.transform = 'anscombe';
cfg.horizonMs = 320;
cfg.K_outer = 5;      % Outer folds for leaderboard reporting
cfg.K_inner = 3;      % Inner folds for finding best PCA 'r' and kNN 'k'
cfg.seed = 42;

% Hyperparameter Search Space
cfg.pcaRange = 10:5:60;  % Searching PCA dimensions from 10 to 60
cfg.knnRange = 1:2:15;   % Searching k from 1 to 15 (odd numbers only)

methodKeys = {'pca_lda_knn', 'pca_lda_svm', 'pca_lda_nbc'};
methodLabels = {'PCA+LDA+kNN', 'PCA+LDA+SVM', 'PCA+LDA+NBC'};

%% --- Load Data ---
data = load(dataFile, 'trial');
trial = data.trial;
nTrials = size(trial, 1);
folds = make_folds(nTrials, cfg.K_outer, cfg.seed);
rows = {};

%% --- Main Benchmarking Loop ---
for m = 1:numel(methodKeys)
    key = methodKeys{m};
    label = methodLabels{m};
    acc = nan(cfg.K_outer, 1);
    tSec = nan(cfg.K_outer, 1);

    fprintf('Running Nested Sweep for: %s\n', label);

    for f = 1:cfg.K_outer
        trainData = trial(folds(f).trainIdx, :);
        testData = trial(folds(f).testIdx, :);

        tFold = tic;
        % Perform internal Tuning (Nested Cross-Validation)
        model = train_method_with_sweep(key, trainData, cfg);

        % Test on the held-out fold
        acc(f) = test_method(key, model, testData, cfg);
        tSec(f) = toc(tFold);

        fprintf('  Fold %d/%d | Best nPC: %d | Best k: %d | Accuracy: %.2f%%\n', ...
            f, cfg.K_outer, model.bestPC, model.bestK, acc(f));
    end

    rows(end + 1, :) = {string(label), mean(acc), std(acc), mean(tSec)}; %#ok<AGROW>
end

%% --- Final Leaderboard Display ---
results = cell2table(rows, 'VariableNames', {'Method', 'MeanAccuracy', 'StdDev', 'TimeSec'});
results = sortrows(results, 'MeanAccuracy', 'descend');
disp(results);
writetable(results, fullfile(outDir, 'final_leaderboard.csv'));

%% ---------------- LOCAL FUNCTIONS ----------------

function model = train_method_with_sweep(key, trainData, cfg)
    % Extract features from training set for tuning
    [X_full, y_full] = feature_matrix(trainData, cfg);
    innerFolds = make_folds(size(X_full, 1), cfg.K_inner, cfg.seed);

    bestScore = -1;
    bestPC = 20;
    bestK = 5;

    % --- THE NESTED HYPERPARAMETER SWEEP ---
    for nPC = cfg.pcaRange
        foldAccs = nan(cfg.K_inner, 1);
        k_winners = nan(cfg.K_inner, 1);

        for iF = 1:cfg.K_inner
            % Internal Split
            X_in_trn = X_full(innerFolds(iF).trainIdx, :);
            y_in_trn = y_full(innerFolds(iF).trainIdx);
            X_in_val = X_full(innerFolds(iF).testIdx, :);
            y_in_val = y_full(innerFolds(iF).testIdx);

            % Step 1: Z-score
            [Xz, mu, sd] = zscore_fit(X_in_trn);

            % Step 2: PCA (fit on inner train, apply to inner val)
            [~, pcaMu, coeff] = pca_project_fit(Xz, nPC);
            Z_trn = pca_project_apply(Xz, pcaMu, coeff);
            Z_val = pca_project_apply(zscore_apply(X_in_val, mu, sd), pcaMu, coeff);

            % Step 3: LDA (fit on inner train, apply to inner val)
            lda = lda_fit(Z_trn, y_in_trn);
            L_trn = lda_project(Z_trn, lda);
            L_val = lda_project(Z_val, lda);

            % Step 4: Classifier Evaluation
            if strcmp(key, 'pca_lda_knn')
                k_scores = nan(numel(cfg.knnRange), 1);
                for kidx = 1:numel(cfg.knnRange)
                    kVal = cfg.knnRange(kidx);
                    p = arrayfun(@(r) knn_predict(L_trn, y_in_trn, L_val(r, :), kVal), 1:size(L_val,1))';
                    k_scores(kidx) = mean(p == y_in_val);
                end
                [foldAccs(iF), maxKIdx] = max(k_scores);
                k_winners(iF) = cfg.knnRange(maxKIdx);
            else
                % SVM, NBC, or NC (No extra hyperparams beyond PCA)
                if strcmp(key, 'pca_lda_svm')
                    if exist('fitcecoc', 'file') == 2
                        m = fitcecoc(L_trn, y_in_trn); p = predict(m, L_val);
                    else
                        p = ones(size(L_val,1), 1); % deterministic fallback
                    end
                elseif strcmp(key, 'pca_lda_nbc')
                    m = nbc_fit(L_trn, y_in_trn); p = arrayfun(@(r) nbc_predict(m, L_val(r,:)), 1:size(L_val,1))';
                else
                    m = nc_fit(L_trn, y_in_trn); p = arrayfun(@(r) nc_predict(m, L_val(r,:)), 1:size(L_val,1))';
                end
                foldAccs(iF) = mean(p == y_in_val);
            end
        end

        % Check if this PCA component count (and k) is the best so far
        if mean(foldAccs) > bestScore
            bestScore = mean(foldAccs);
            bestPC = nPC;
            if strcmp(key, 'pca_lda_knn'), bestK = round(mean(k_winners)); end
        end
    end

    % --- FINAL TRAIN: Fit the model using the WINNING parameters on all training data ---
    [Xz, mu, sd] = zscore_fit(X_full);
    [Xpca, pcaMu, coeff] = pca_project_fit(Xz, bestPC);
    lda = lda_fit(Xpca, y_full);
    L_full = lda_project(Xpca, lda);

    model = struct('key', key, 'mu', mu, 'sd', sd, 'pcaMu', pcaMu, ...
                   'coeff', coeff, 'lda', lda, 'bestPC', bestPC, 'bestK', bestK);

    if strcmp(key, 'pca_lda_knn')
        model.X = L_full; model.y = y_full;
    elseif strcmp(key, 'pca_lda_svm')
        if exist('fitcecoc', 'file') == 2
            model.svm = fitcecoc(L_full, y_full); model.hasSvm = true;
        else
            model.svm = []; model.hasSvm = false;
        end
    elseif strcmp(key, 'pca_lda_nbc')
        model.nbc = nbc_fit(L_full, y_full);
    else
        model.nc = nc_fit(L_full, y_full);
    end
end

function acc = test_method(key, model, testData, cfg)
    hit = 0; total = 0;
    for i = 1:size(testData, 1)
        for d = 1:size(testData, 2)
            % Preprocess
            x = feature_vector(testData(i, d).spikes, cfg);
            x = zscore_apply(x, model.mu, model.sd);
            xp = pca_project_apply(x, model.pcaMu, model.coeff);
            z = lda_project(xp, model.lda);

            % Classify using the "best" k or model found during tuning
            if strcmp(key, 'pca_lda_knn')
                pred = knn_predict(model.X, model.y, z, model.bestK);
            elseif strcmp(key, 'pca_lda_svm')
                if isfield(model, 'hasSvm') && model.hasSvm
                    pred = predict(model.svm, z);
                else
                    pred = 1;
                end
            elseif strcmp(key, 'pca_lda_nbc')
                pred = nbc_predict(model.nbc, z);
            else
                pred = nc_predict(model.nc, z);
            end

            hit = hit + double(pred == d);
            total = total + 1;
        end
    end
    acc = 100 * hit / max(total, 1);
end

%% --- Helper Logic Functions ---

function f = feature_vector(spikes, cfg)
    T = min(cfg.horizonMs, size(spikes, 2));
    nBins = floor(T / cfg.binWidth);
    binned = zeros(size(spikes, 1), nBins);
    for i = 1:nBins
        binned(:, i) = sum(spikes(:, (i-1)*cfg.binWidth+1 : i*cfg.binWidth), 2);
    end
    if strcmp(cfg.transform, 'anscombe'), binned = 2 * sqrt(binned + 3/8); end
    f = sum(binned, 2)'; % Consolidate into 1x98 vector for the classification window
end

function yhat = knn_predict(Xtrain, ytrain, x, k)
    % Standard Euclidean Distance for kNN
    dist = sum((Xtrain - x) .^ 2, 2);
    [~, idx] = sort(dist, 'ascend');
    yhat = mode(ytrain(idx(1:min(k, length(idx)))));
end

function [Xz, mu, sd] = zscore_fit(X)
    mu = mean(X, 1); sd = std(X, 0, 1); sd(sd < eps) = 1; Xz = (X - mu) ./ sd;
end

function z = zscore_apply(x, mu, sd)
    z = (x - mu) ./ sd;
end

function [Xp, pcaMu, coeff] = pca_project_fit(X, nPC)
    pcaMu = mean(X, 1); [~, ~, V] = svd(X - pcaMu, 'econ');
    coeff = V(:, 1:min(nPC, size(V,2))); Xp = (X - pcaMu) * coeff;
end

function xp = pca_project_apply(x, pcaMu, coeff)
    xp = (x - pcaMu) * coeff;
end

function lda = lda_fit(X, y)
    classes = unique(y); c = numel(classes); p = size(X, 2); M = zeros(c, p); S = zeros(p, p);
    for i = 1:c
        Xi = X(y == classes(i), :); M(i, :) = mean(Xi, 1);
        S = S + (Xi - M(i, :))' * (Xi - M(i, :));
    end
    S = S / max(size(X, 1) - c, 1) + 1e-6 * eye(p);
    W = (S \ M')'; b = -0.5 * sum(W .* M, 2);
    lda = struct('W', W, 'b', b);
end

function z = lda_project(x, lda)
    z = x * lda.W' + lda.b';
end

function nbc = nbc_fit(X, y)
    classes = unique(y);
    for i = 1:numel(classes)
        Xi = X(y == classes(i), :);
        nbc.mu(i, :) = mean(Xi, 1); nbc.v(i, :) = var(Xi, 0, 1) + eps;
    end
    nbc.classes = classes;
end

function yhat = nbc_predict(nbc, x)
    logP = -0.5 * sum(log(2 * pi * nbc.v), 2) - 0.5 * sum((x - nbc.mu).^2 ./ nbc.v, 2);
    [~, idx] = max(logP); yhat = nbc.classes(idx);
end

function nc = nc_fit(X, y)
    classes = unique(y);
    for i = 1:numel(classes)
        nc.centroids(i, :) = mean(X(y == classes(i), :), 1);
    end
    nc.classes = classes;
end

function yhat = nc_predict(nc, x)
    d2 = sum((nc.centroids - x) .^ 2, 2); [~, idx] = min(d2); yhat = nc.classes(idx);
end

function [X, y] = feature_matrix(trialData, cfg)
    [nT, nD] = size(trialData); nN = size(trialData(1,1).spikes, 1);
    X = zeros(nT*nD, nN); y = zeros(nT*nD, 1); row = 1;
    for t = 1:nT
        for d = 1:nD
            X(row, :) = feature_vector(trialData(t,d).spikes, cfg);
            y(row) = d;
            row = row + 1;
        end
    end
end

function folds = make_folds(n, K, seed)
    rng(seed); idx = randperm(n);
    base = floor(n / K); remn = mod(n, K);
    sizes = base * ones(1, K); sizes(1:remn) = sizes(1:remn) + 1;
    folds = repmat(struct('trainIdx', [], 'testIdx', []), 1, K);
    cursor = 1;
    for k = 1:K
        nFold = sizes(k);
        tst = idx(cursor:cursor+nFold-1);
        folds(k).testIdx = tst;
        folds(k).trainIdx = setdiff(idx, tst);
        cursor = cursor + nFold;
    end
end