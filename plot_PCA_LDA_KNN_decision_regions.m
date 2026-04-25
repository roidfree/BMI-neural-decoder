function stats = plot_PCA_LDA_KNN_decision_regions(varargin)
% plot_PCA_LDA_KNN_decision_regions
% Visualize PCA->LDA->kNN classifier decision regions in LD1/LD2.
%
% This is intended for the classifier stage of a combined
% "PCA + LDA + kNN classifier, then PLS regressor" pipeline:
%   1) Train classifier and project training points into LDA space.
%   2) Draw kNN decision regions in LD1/LD2 (all remaining LD dimensions
%      fixed at the training-set mean to create a 2D slice).
%   3) Mark leave-one-out kNN misclassifications.
%
% Usage examples
%   % 1) Use local monkeydata_training.mat and default trainer:
%   plot_PCA_LDA_KNN_decision_regions();
%
%   % 2) Pass your own trial data and hyperparameters:
%   load('my_training_data.mat', 'trial');
%   plot_PCA_LDA_KNN_decision_regions(trial, 'knnK', 7, 'pcaDim', 25, 'ldaDim', 7);
%
%   % 3) Use a custom training function that returns fields:
%   %    X_proj (Nxd), y (Nx1 labels), k
%   plot_PCA_LDA_KNN_decision_regions(trial, 'trainingFcn', @myTrainingFcn);
%
% Inputs (all optional)
%   trainingData : trial struct array (#trials x #dirs). If omitted,
%                  the function attempts to load monkeydata_training.mat.
%
% Name-value options
%   'trainingFcn'                 : classifier trainer function handle
%                                   (default: @positionEstimatorTraining_PCA_LDA_K)
%   'knnK'                        : k for kNN (default 5)
%   'pcaDim'                      : PCA latent dimension (default 20)
%   'ldaDim'                      : requested LDA dimension (default 7)
%   'gridRes'                     : decision-map grid resolution (default 180)
%   'useClassifierTrainingSplit'  : if true, randomly keep 50 trial rows
%                                   (default false)
%   'splitSeed'                   : rng seed for split mode (default 2013)
%
% Output
%   stats : struct containing misclassification statistics and model fields.

    p = inputParser;
    p.addOptional('trainingData', [], @(x) isempty(x) || isstruct(x));
    p.addParameter('trainingFcn', [], @(f) isempty(f) || isa(f, 'function_handle') || ischar(f) || isstring(f));
    p.addParameter('knnK', 5, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.addParameter('pcaDim', 20, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.addParameter('ldaDim', 7, @(x) isnumeric(x) && isscalar(x) && x >= 1);
    p.addParameter('gridRes', 180, @(x) isnumeric(x) && isscalar(x) && x >= 20);
    p.addParameter('useClassifierTrainingSplit', false, @(x) islogical(x) || isnumeric(x));
    p.addParameter('splitSeed', 2013, @(x) isnumeric(x) && isscalar(x));

    p.parse(varargin{:});
    opts = p.Results;
    opts.trainingFcn = normalizeTrainingFcn(opts.trainingFcn);

    if isempty(opts.trainingData)
        data = loadLocalTrainingData();
        trainingData = data.trial;
    else
        trainingData = opts.trainingData;
    end

    if opts.useClassifierTrainingSplit
        rng(opts.splitSeed);
        ix = randperm(size(trainingData, 1));
        keep = min(50, numel(ix));
        trainingData = trainingData(ix(1:keep), :);
    end

    modelParameters = opts.trainingFcn(trainingData, opts.knnK, opts.pcaDim, opts.ldaDim);

    requiredFields = {'X_proj', 'y', 'k'};
    for i = 1:numel(requiredFields)
        if ~isfield(modelParameters, requiredFields{i})
            error('Model is missing required field "%s". Ensure your training function returns X_proj, y, and k.', requiredFields{i});
        end
    end

    Xtr = modelParameters.X_proj;
    ytr = modelParameters.y(:);
    k = modelParameters.k;
    ldaD = size(Xtr, 2);

    if ldaD < 2
        error('Need at least 2 LDA dimensions to plot LD1 vs LD2. Current projected dimension: %d.', ldaD);
    end

    sliceTail = mean(Xtr, 1);

    ld1 = Xtr(:, 1);
    ld2 = Xtr(:, 2);
    pad1 = 0.08 * (max(ld1) - min(ld1) + eps);
    pad2 = 0.08 * (max(ld2) - min(ld2) + eps);
    g1 = linspace(min(ld1) - pad1, max(ld1) + pad1, opts.gridRes);
    g2 = linspace(min(ld2) - pad2, max(ld2) + pad2, opts.gridRes);
    [LG1, LG2] = meshgrid(g1, g2);

    nGrid = numel(LG1);
    predGrid = zeros(nGrid, 1);
    for gi = 1:nGrid
        row = sliceTail;
        row(1:2) = [LG1(gi), LG2(gi)];
        predGrid(gi) = knnMode1(row, Xtr, ytr, k);
    end
    Z = reshape(predGrid, opts.gridRes, opts.gridRes);

    mis = knnLooMisclass(Xtr, ytr, k);

    maxClass = max(ytr);
    cmap = lines(max(8, maxClass));

    figure('Color', 'w', 'Position', [80 80 780 620]);
    axes;
    imagesc(g1, g2, Z);
    set(gca, 'YDir', 'normal');
    colormap(gca, cmap);
    caxis([0.5, maxClass + 0.5]);
    hold on;
    axis equal tight;
    box on;
    set(gca, 'Layer', 'top');
    set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.35);
    grid on;

    scatter(ld1, ld2, 44, ytr, 'filled', ...
        'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.4, ...
        'MarkerFaceAlpha', 0.92);

    if any(mis)
        plot(ld1(mis), ld2(mis), 'rx', 'MarkerSize', 11, 'LineWidth', 2.2);
    end

    classList = unique(ytr)';
    for d = classList
        classPts = Xtr(ytr == d, 1:2);
        m = mean(classPts, 1);
        plot(m(1), m(2), 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
        text(m(1), m(2), sprintf('  %d', d), ...
            'FontSize', 11, 'FontWeight', 'bold', ...
            'Color', [0 0 0], 'VerticalAlignment', 'middle');
    end

    xlabel('LD1 (first LDA axis on PCA subspace)');
    ylabel('LD2');
    if ldaD > 2
        sliceNote = sprintf('; LD3-LD%d fixed at training mean', ldaD);
    else
        sliceNote = '';
    end
    title(sprintf(['LDA space with kNN decision regions ' ...
        '(k=%d, pcaDim=%d, ldaDim=%d%s)'], k, opts.pcaDim, ldaD, sliceNote));

    cb = colorbar('Ticks', classList, ...
        'TickLabels', arrayfun(@num2str, classList, 'UniformOutput', false));
    cb.Label.String = 'Predicted class (kNN) / true label (markers)';

    hPatch = patch('XData', [NaN NaN NaN], 'YData', [NaN NaN NaN], ...
        'FaceColor', [0.7 0.85 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
    hPt = scatter(NaN, NaN, 44, 0.5, 'filled', 'MarkerEdgeColor', 'k');
    hX = plot(NaN, NaN, 'rx', 'LineWidth', 2.2);
    hStar = plot(NaN, NaN, 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
    legend([hPatch, hPt, hX, hStar], ...
        {'Decision regions (kNN, full LD dim)', 'Data points (true direction)', ...
        'Misclassified (LOO-kNN)', 'Class means (LD1-LD2)'}, ...
        'Location', 'northwest', 'FontSize', 9);

    hold off;

    stats = struct;
    stats.k = k;
    stats.pcaDimRequested = opts.pcaDim;
    stats.ldaDimActual = ldaD;
    stats.nSamples = size(Xtr, 1);
    stats.nMisclassifiedLOO = sum(mis);
    stats.looErrorRate = mean(mis);
    stats.modelParameters = modelParameters;

    fprintf('LOO-kNN misclassified: %d / %d (%.2f%%)\n', ...
        stats.nMisclassifiedLOO, stats.nSamples, 100 * stats.looErrorRate);
end

function trainingFcn = normalizeTrainingFcn(trainingFcn)
    if isempty(trainingFcn)
        trainerName = 'positionEstimatorTraining_PCA_LDA_K';
        thisDir = fileparts(mfilename('fullpath'));
        if exist(trainerName, 'file') ~= 2
            addpath(thisDir);
        end
        if exist(trainerName, 'file') ~= 2
            error(['Could not resolve default training function "%s". ', ...
                'Ensure %s is on MATLAB path or pass ''trainingFcn'' explicitly.'], ...
                trainerName, [trainerName '.m']);
        end
        trainingFcn = str2func(trainerName);
        return;
    end

    if ischar(trainingFcn) || isstring(trainingFcn)
        fName = char(trainingFcn);
        if exist(fName, 'file') ~= 2
            error('Training function "%s" not found on MATLAB path.', fName);
        end
        trainingFcn = str2func(fName);
    end
end

function data = loadLocalTrainingData()
    here = fileparts(mfilename('fullpath'));
    candidate = {
        fullfile(here, 'monkeydata_training.mat'), ...
        fullfile(pwd, 'monkeydata_training.mat'), ...
        fullfile(pwd, 'jared', 'monkeydata_training.mat'), ...
        fullfile(pwd, 'Classifier_Ming', 'NBC', 'monkeydata_training.mat') ...
    };

    found = '';
    for i = 1:numel(candidate)
        if exist(candidate{i}, 'file') == 2
            found = candidate{i};
            break;
        end
    end

    if isempty(found)
        error(['Could not find monkeydata_training.mat. ', ...
            'Pass trainingData explicitly, or place the MAT file in one of:\n  %s'], ...
            strjoin(candidate, '\n  '));
    end

    data = load(found, 'trial');
    if ~isfield(data, 'trial')
        error('File found (%s) does not contain variable ''trial''.', found);
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