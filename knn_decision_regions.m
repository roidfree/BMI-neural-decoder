function stats = knn_decision_regions(varargin)
% knn_decision_regions
% Plot side-by-side kNN decision regions:
%   Left  - PCA space (PC1 vs PC2)
%   Right - LDA space (LD1 vs LD2)
%
% Usage:
%   knn_decision_regions();
%   load('monkeydata_training.mat', 'trial');
%   knn_decision_regions(trial, 'knnK', 7, 'pcaDim', 25, 'ldaDim', 7);

    p = inputParser;
    p.addOptional('trainingData', [], @(x) isempty(x) || isstruct(x));
    p.addParameter('trainingFcn', @positionEstimatorTraining_PCA_LDA_K_local, @(f) isa(f, 'function_handle'));
    p.addParameter('knnK', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.addParameter('pcaDim', 20, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    p.addParameter('ldaDim', 7, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    p.addParameter('gridRes', 180, @(x) isnumeric(x) && isscalar(x) && x >= 20);
    p.parse(varargin{:});
    opts = p.Results;

    if isempty(opts.trainingData)
        data = loadLocalTrainingData();
        trainingData = data.trial;
    else
        trainingData = opts.trainingData;
    end

    model = opts.trainingFcn(trainingData, opts.knnK, opts.pcaDim, opts.ldaDim);
    required = {'X_proj', 'y', 'k'};
    for i = 1:numel(required)
        if ~isfield(model, required{i})
            error('Model missing required field "%s".', required{i});
        end
    end

    [features, labels] = extractClassifierFeatures(trainingData, 320);
    Xlda = model.X_proj;
    y = model.y(:);
    if numel(y) ~= size(features, 1)
        y = labels;
    end

    Xpca = getPcaFeatures(model, features);
    if size(Xpca, 2) < 2
        error('Need at least 2 PCA dimensions for left panel.');
    end
    if size(Xlda, 2) < 2
        error('Need at least 2 LDA dimensions for right panel.');
    end

    figure('Color', 'w', 'Position', [60 60 1280 560]);
    t = tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

    nexttile;
    [misPca, errPca] = drawDecisionPanel(Xpca, y, model.k, opts.gridRes, 'PCA space', 'PC1', 'PC2');

    nexttile;
    [misLda, errLda] = drawDecisionPanel(Xlda, y, model.k, opts.gridRes, 'LDA space', 'LD1', 'LD2');

    title(t, sprintf('kNN decision regions (k=%d)', model.k), 'FontWeight', 'bold');

    stats = struct;
    stats.k = model.k;
    stats.nSamples = numel(y);
    stats.pcaLooErrorRate = errPca;
    stats.ldaLooErrorRate = errLda;
    stats.nMisPca = sum(misPca);
    stats.nMisLda = sum(misLda);
    stats.modelParameters = model;

    if nargout == 0
        fprintf('PCA LOO error: %.2f%% (%d/%d)\n', 100 * errPca, sum(misPca), numel(y));
        fprintf('LDA LOO error: %.2f%% (%d/%d)\n', 100 * errLda, sum(misLda), numel(y));
        clear stats;
    end
end

function [mis, err] = drawDecisionPanel(X, y, k, gridRes, panelTitle, xlab, ylab)
    x1 = X(:, 1);
    x2 = X(:, 2);
    pad1 = 0.08 * (max(x1) - min(x1) + eps);
    pad2 = 0.08 * (max(x2) - min(x2) + eps);
    g1 = linspace(min(x1) - pad1, max(x1) + pad1, gridRes);
    g2 = linspace(min(x2) - pad2, max(x2) + pad2, gridRes);
    [G1, G2] = meshgrid(g1, g2);

    n = numel(G1);
    predGrid = zeros(n, 1);
    tail = mean(X, 1);
    for i = 1:n
        row = tail;
        row(1:2) = [G1(i), G2(i)];
        predGrid(i) = knnMode1(row, X, y, k);
    end
    Z = reshape(predGrid, gridRes, gridRes);

    imagesc(g1, g2, Z);
    set(gca, 'YDir', 'normal');
    colormap(gca, lines(max(8, max(y))));
    caxis([0.5 max(y) + 0.5]);
    hold on;
    axis equal tight;
    box on;
    grid on;

    scatter(x1, x2, 36, y, 'filled', 'MarkerEdgeColor', [0.2 0.2 0.2], 'LineWidth', 0.3);
    mis = knnLooMisclass(X, y, k);
    if any(mis)
        plot(x1(mis), x2(mis), 'rx', 'MarkerSize', 10, 'LineWidth', 1.8);
    end

    classList = unique(y)';
    for c = classList
        m = mean(X(y == c, 1:2), 1);
        plot(m(1), m(2), 'k*', 'MarkerSize', 14, 'LineWidth', 1.0);
    end

    xlabel(xlab);
    ylabel(ylab);
    title(panelTitle);

    err = mean(mis);
end

function Xpca = getPcaFeatures(model, features)
    if isfield(model, 'X_pca')
        Xpca = model.X_pca;
        return;
    end

    if isfield(model, 'V_pca') && isfield(model, 'mu') && isfield(model, 'sigma')
        Xn = (features - model.mu) ./ model.sigma;
        Xpca = Xn * model.V_pca;
        return;
    end

    mu = mean(features, 1);
    sig = std(features, 0, 1) + eps;
    Xn = (features - mu) ./ sig;
    [~, ~, v] = svd(Xn, 'econ');
    keep = min(20, size(v, 2));
    Xpca = Xn * v(:, 1:keep);
end

function [features, labels] = extractClassifierFeatures(trainingData, horizon)
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1, 1).spikes, 1);
    n = numTrials * numDirs;

    features = zeros(n, numNeurons);
    labels = zeros(n, 1);
    idx = 1;
    for t = 1:numTrials
        for d = 1:numDirs
            s = trainingData(t, d).spikes;
            stopIdx = min(horizon, size(s, 2));
            features(idx, :) = sum(s(:, 1:stopIdx), 2)';
            labels(idx) = d;
            idx = idx + 1;
        end
    end
end

function data = loadLocalTrainingData()
    roots = {pwd, fileparts(mfilename('fullpath')), fileparts(fileparts(mfilename('fullpath')))};
    candidates = {};
    for i = 1:numel(roots)
        candidates{end + 1} = fullfile(roots{i}, 'monkeydata_training.mat'); %#ok<AGROW>
        candidates{end + 1} = fullfile(roots{i}, 'jared', 'monkeydata_training.mat'); %#ok<AGROW>
        candidates{end + 1} = fullfile(roots{i}, 'Classifier_Ming', 'NBC', 'monkeydata_training.mat'); %#ok<AGROW>
    end
    candidates = unique(candidates, 'stable');

    filePath = '';
    for i = 1:numel(candidates)
        if exist(candidates{i}, 'file') == 2
            filePath = candidates{i};
            break;
        end
    end
    if isempty(filePath)
        error('Could not find monkeydata_training.mat. Pass trainingData explicitly.');
    end

    data = load(filePath, 'trial');
    if ~isfield(data, 'trial')
        error('File %s does not contain variable ''trial''.', filePath);
    end
end

function pred = knnMode1(x, Xref, yref, k)
    d2 = sum((Xref - x) .^ 2, 2);
    [~, ord] = sort(d2, 'ascend');
    pred = mode(yref(ord(1:k)));
end

function mis = knnLooMisclass(X, y, k)
    n = size(X, 1);
    pred = zeros(n, 1);
    for i = 1:n
        d2 = sum((X - X(i, :)) .^ 2, 2);
        d2(i) = inf;
        [~, ord] = sort(d2, 'ascend');
        pred(i) = mode(y(ord(1:k)));
    end
    mis = pred ~= y;
end

function modelParameters = positionEstimatorTraining_PCA_LDA_K_local(trainingData, varargin)
    k = 5;
    pcaDim = 20;
    ldaDim = 7;
    if numel(varargin) >= 1 && ~isempty(varargin{1}), k = varargin{1}; end
    if numel(varargin) >= 2 && ~isempty(varargin{2}), pcaDim = varargin{2}; end
    if numel(varargin) >= 3 && ~isempty(varargin{3}), ldaDim = varargin{3}; end
    ldaDim = min(ldaDim, pcaDim);

    [features, labels] = extractClassifierFeatures(trainingData, 320);

    mu = mean(features, 1);
    sigma = std(features, 0, 1) + eps;
    normalizedFeatures = (features - mu) ./ sigma;

    [~, ~, v] = svd(normalizedFeatures, 'econ');
    pcaDim = min(pcaDim, size(v, 2));
    vPca = v(:, 1:pcaDim);
    pcaFeatures = normalizedFeatures * vPca;

    classList = unique(labels);
    globalMean = mean(pcaFeatures, 1)';
    sw = zeros(pcaDim, pcaDim);
    sb = zeros(pcaDim, pcaDim);
    for classIdx = classList'
        classData = pcaFeatures(labels == classIdx, :);
        classMean = mean(classData, 1)';
        centeredClass = classData - classMean';
        sw = sw + centeredClass' * centeredClass;
        classCount = size(classData, 1);
        meanDiff = classMean - globalMean;
        sb = sb + classCount * (meanDiff * meanDiff');
    end

    [eigVec, eigVal] = eig(sb, sw);
    eigScore = real(diag(eigVal));
    [~, order] = sort(eigScore, 'descend');
    ldaDim = min(ldaDim, numel(order));
    wLda = real(eigVec(:, order(1:ldaDim)));
    projectedFeatures = pcaFeatures * wLda;

    modelParameters.mu = mu;
    modelParameters.sigma = sigma;
    modelParameters.V_pca = vPca;
    modelParameters.X_pca = pcaFeatures;
    modelParameters.W_lda = wLda;
    modelParameters.X_proj = projectedFeatures;
    modelParameters.y = labels;
    modelParameters.k = k;
end
run_preprocessing_sweep_pca_lda_knn_pls.m
