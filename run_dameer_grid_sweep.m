 clc; clear;

% Ordered preprocessing + regressor sweeps.
% Goal:
%   - enforce exact preprocessing order per method
%   - compare three preprocessing methods on pooled regressor RMSE
%   - run Ming classifier hyperparameter sweeps

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outputDir = fullfile(repoRoot, 'benchmark_outputs', 'ordered_preprocess_and_classifier_sweeps');
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

baseOpts = struct();
baseOpts.plotFigures = false;
baseOpts.saveOutputs = false;
baseOpts.verbose = false;
baseOpts.timestampOutputDir = false;
baseOpts.K = 5;
baseOpts.pipelineMethods = {};
baseOpts.includeKalman = false;

% ============================================================
% PART A: ORDERED PREPROCESSING COMPARISON (three methods)
% ============================================================
% Method 1: gaussian smoothing on raw spikes -> binning -> sqrt
% Method 2: rectangular smoothing on raw spikes -> anscombe -> gaussian smoothing
% Method 3: baseline (binning -> anscombe)

binWidthGrid = [5, 10, 20, 25, 40];
historyGrid = [1, 4, 8, 12, 15, 20];
fixedNpc = 300;

methods = struct([]);

methods(1).name = 'gauss_then_bin_then_sqrt';
methods(1).transform = 'sqrt';
methods(1).preKernel = 'gauss';
methods(1).preWidthGrid = [3, 5, 9];
methods(1).preParamGrid = [1.0, 2.0];
methods(1).postKernel = 'none';
methods(1).postWidthGrid = 0;
methods(1).postParamGrid = 0;

methods(2).name = 'rect_then_anscombe_then_gauss';
methods(2).transform = 'anscombe';
methods(2).preKernel = 'rect';
methods(2).preWidthGrid = [3, 5, 9];
methods(2).preParamGrid = 0;
methods(2).postKernel = 'gauss';
methods(2).postWidthGrid = [3, 5, 9];
methods(2).postParamGrid = [1.0, 2.0];

methods(3).name = 'baseline_bin_then_anscombe';
methods(3).transform = 'anscombe';
methods(3).preKernel = 'none';
methods(3).preWidthGrid = 0;
methods(3).preParamGrid = 0;
methods(3).postKernel = 'none';
methods(3).postWidthGrid = 0;
methods(3).postParamGrid = 0;

rowsA = {};
expId = 0;

for mIdx = 1:numel(methods)
    method = methods(mIdx);
    for bwIdx = 1:numel(binWidthGrid)
        binWidth = binWidthGrid(bwIdx);
        for hbIdx = 1:numel(historyGrid)
            historyBins = historyGrid(hbIdx);
            for pWIdx = 1:numel(method.preWidthGrid)
                preWidth = method.preWidthGrid(pWIdx);
                for pPIdx = 1:numel(method.preParamGrid)
                    preParam = method.preParamGrid(pPIdx);
                    for qWIdx = 1:numel(method.postWidthGrid)
                        postWidth = method.postWidthGrid(qWIdx);
                        for qPIdx = 1:numel(method.postParamGrid)
                            postParam = method.postParamGrid(qPIdx);
                            preParamLabel = format_smoothing_param(method.preKernel, preParam);
                            postParamLabel = format_smoothing_param(method.postKernel, postParam);

                            expId = expId + 1;
                            fprintf('[A-%03d] %s | bin=%d | history=%d | pre=%s(w=%g,p=%s) | post=%s(w=%g,p=%s)\n', ...
                                expId, method.name, binWidth, historyBins, ...
                                method.preKernel, preWidth, preParamLabel, method.postKernel, postWidth, postParamLabel);

                            maxIterEstimate = floor(571 / binWidth) - historyBins;
                            if maxIterEstimate <= 0
                                fprintf('    -> skipped (insufficient bins)\n');
                                meanRmse = NaN;
                                stdRmse = NaN;
                            else
                                opts = baseOpts;
                                opts.classifierMethods = {};
                                opts.pooledRegressorMethods = {'pooled_pcr'};
                                opts.pooledPCRGrid = fixedNpc;
                                opts.pooledBinWidth = binWidth;
                                opts.pooledHistoryBins = historyBins;
                                opts.pooledTransform = method.transform;
                                opts.pooledPreSmoothKernel = method.preKernel;
                                opts.pooledPreSmoothWidth = preWidth;
                                opts.pooledPreSmoothParam = preParam;
                                opts.pooledPostSmoothKernel = method.postKernel;
                                opts.pooledPostSmoothWidth = postWidth;
                                opts.pooledPostSmoothParam = postParam;

                                try
                                    benchmark = dameer_grader(opts);
                                    summary = benchmark.tables.continuousSummary;
                                    mask = strcmp(summary.method, 'pooled_pcr') & summary.nPC == fixedNpc;
                                    subset = summary(mask, :);
                                    if isempty(subset)
                                        meanRmse = NaN;
                                        stdRmse = NaN;
                                    else
                                        meanRmse = subset.meanRMSE(1);
                                        stdRmse = subset.stdRMSE(1);
                                    end
                                catch runErr
                                    fprintf('    -> failed (%s)\n', runErr.message);
                                    meanRmse = NaN;
                                    stdRmse = NaN;
                                end
                            end

                            rowsA(end + 1, :) = { ...
                                expId, string(method.name), binWidth, historyBins, string(method.transform), ...
                                string(method.preKernel), preWidth, preParam, ...
                                string(method.postKernel), postWidth, postParam, ...
                                fixedNpc, meanRmse, stdRmse}; %#ok<AGROW>
                        end
                    end
                end
            end
        end
    end
end

partA = cell2table(rowsA, 'VariableNames', { ...
    'expId', 'method', 'binWidth', 'historyBins', 'transform', ...
    'preKernel', 'preWidth', 'preParam', 'postKernel', 'postWidth', 'postParam', ...
    'nPC', 'meanRMSE', 'stdRMSE'});
partA = sortrows(partA, {'meanRMSE', 'stdRMSE'}, {'ascend', 'ascend'});

partA_csv = fullfile(outputDir, 'ordered_preprocessing_method_comparison.csv');
writetable(partA, partA_csv);

methodNames = unique(partA.method);
if isempty(partA)
    bestByMethod = table();
else
    bestByMethod = repmat(partA(1, :), 0, 1);
    for i = 1:numel(methodNames)
        subset = partA(partA.method == methodNames(i), :);
        subset = subset(~isnan(subset.meanRMSE), :);
        if isempty(subset)
            continue;
        end
        bestByMethod(end + 1, :) = subset(1, :); %#ok<AGROW>
    end
    bestByMethod = sortrows(bestByMethod, {'meanRMSE', 'method'});
end
best_method_csv = fullfile(outputDir, 'ordered_preprocessing_best_by_method.csv');
writetable(bestByMethod, best_method_csv);

if ~isempty(bestByMethod)
    figA = figure('Visible', 'off');
    bar(bestByMethod.meanRMSE);
    grid on;
    set(gca, 'XTick', 1:height(bestByMethod), 'XTickLabel', cellstr(bestByMethod.method), 'XTickLabelRotation', 25);
    ylabel('Mean CV RMSE (lower is better)');
    title('Best config from each ordered preprocessing method');
    saveas(figA, fullfile(outputDir, 'ordered_preprocessing_best_method_rmse.png'));
    close(figA);
end

% ============================================================
% PART B: FIXED PREPROCESSING REGRESSOR HYPERPARAMETER SWEEP
% ============================================================
% Uses the single best ordered method from Part A and sweeps pooled regressors.

if isempty(bestByMethod)
    warning('Part B skipped because Part A produced no valid rows.');
    partB = table();
else
    winner = bestByMethod(1, :);
    optsB = baseOpts;
    optsB.classifierMethods = {};
    optsB.pooledRegressorMethods = {'pooled_pcr', 'pooled_ols_ridge', 'pooled_pls'};
    maxPcFromFeatures = 98 * (winner.historyBins + 1);
    compGrid = unique([1, 3, 5, 7, 10, 30, 50, 70, 100, ...
        min(300, maxPcFromFeatures), min(500, maxPcFromFeatures), min(700, maxPcFromFeatures)]);
    optsB.pooledPCRGrid = compGrid;
    optsB.pooledOLSRidgeLambdaGrid = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000];
    optsB.pooledPLSGrid = compGrid;

    optsB.pooledBinWidth = winner.binWidth;
    optsB.pooledHistoryBins = winner.historyBins;
    optsB.pooledTransform = char(winner.transform);
    optsB.pooledPreSmoothKernel = char(winner.preKernel);
    optsB.pooledPreSmoothWidth = winner.preWidth;
    optsB.pooledPreSmoothParam = winner.preParam;
    optsB.pooledPostSmoothKernel = char(winner.postKernel);
    optsB.pooledPostSmoothWidth = winner.postWidth;
    optsB.pooledPostSmoothParam = winner.postParam;

    fprintf('\n[PART B] winner preprocessing = %s\n', winner.method);
    fprintf('[PART B] sweeping component grid (PCR/PLS): %s\n', mat2str(compGrid));
    fprintf('[PART B] sweeping lambda grid (OLS+Ridge): %s\n', mat2str(optsB.pooledOLSRidgeLambdaGrid));
    benchmarkB = dameer_grader(optsB);
    summaryB = benchmarkB.tables.continuousSummary;
    partB = summaryB(strcmp(summaryB.task, 'regressor_pooled'), :);
    partB = sortrows(partB, {'method', 'lambda', 'nPC'});
end

partB_csv = fullfile(outputDir, 'winner_preprocessing_regressor_hyper_sweep.csv');
writetable(partB, partB_csv);

if ~isempty(partB)
    figB = figure('Visible', 'off');
    hold on; grid on;
    pcr = partB(strcmp(partB.method, 'pooled_pcr'), :);
    if ~isempty(pcr)
        [xp, ordp] = sort(pcr.nPC);
        plot(xp, pcr.meanRMSE(ordp), '-s', 'LineWidth', 1.5, 'DisplayName', 'PCR');
    end
    pls = partB(strcmp(partB.method, 'pooled_pls'), :);
    if ~isempty(pls)
        [xp, ordp] = sort(pls.nPC);
        plot(xp, pls.meanRMSE(ordp), '-^', 'LineWidth', 1.5, 'DisplayName', 'PLS');
    end
    ols = partB(strcmp(partB.method, 'pooled_ols_ridge'), :);
    if ~isempty(ols)
        [xl, ordl] = sort(ols.lambda);
        plot(xl, ols.meanRMSE(ordl), '-o', 'LineWidth', 1.5, 'DisplayName', 'OLS+Ridge');
    end
    legend('Location', 'best');
    xlabel('nComponents / \lambda'); ylabel('Mean CV RMSE');
    title('Regressor hyperparameter sweep on winning ordered preprocessing');
    saveas(figB, fullfile(outputDir, 'winner_preprocessing_regressor_hyper_sweep.png'));
    close(figB);
end

% ============================================================
% PART C: Ming classifier hyperparameter sweep
% ============================================================
% Available Ming classifiers in dameer_grader: nbc_ming and knn_ming.

optsC = baseOpts;
optsC.classifierMethods = {'nbc_ming', 'knn_ming'};
optsC.classifierKGrid = [1, 3, 5, 7, 11, 15, 21, 31];
optsC.pooledRegressorMethods = {};

benchmarkC = dameer_grader(optsC);
cls = benchmarkC.tables.classifierSummary;
cls = sortrows(cls, {'method', 'k'});

cls_csv = fullfile(outputDir, 'ming_classifier_hyper_sweep.csv');
writetable(cls, cls_csv);

if ~isempty(cls)
    figC = figure('Visible', 'off');
    hold on; grid on;
    knn = cls(strcmp(cls.method, 'knn_ming'), :);
    if ~isempty(knn)
        [xk, ordk] = sort(knn.k);
        yk = knn.meanAccuracy(ordk);
        plot(xk, yk, '-o', 'LineWidth', 1.5, 'DisplayName', 'knn_ming');
    end
    nbc = cls(strcmp(cls.method, 'nbc_ming'), :);
    if ~isempty(nbc)
        yNbc = nbc.meanAccuracy(1);
        xSpan = [min(optsC.classifierKGrid), max(optsC.classifierKGrid)];
        plot(xSpan, [yNbc, yNbc], '--', 'LineWidth', 1.3, ...
            'DisplayName', sprintf('nbc_ming = %.2f%%', yNbc));
    end
    xlabel('k (for kNN)');
    ylabel('Mean CV Accuracy (%)');
    title('Ming classifier hyperparameter sweep');
    legend('Location', 'best');
    saveas(figC, fullfile(outputDir, 'ming_classifier_hyper_sweep.png'));
    close(figC);
end

fprintf('\nSaved outputs to: %s\n', outputDir);
fprintf('Part A method comparison: %s\n', partA_csv);
fprintf('Part B regressor sweep:   %s\n', partB_csv);
fprintf('Part C classifier sweep:  %s\n', cls_csv);

function label = format_smoothing_param(kernelName, paramVal)
    kernel = lower(string(kernelName));
    if kernel == "none" || kernel == "rect"
        label = 'n/a';
    elseif isnan(paramVal)
        label = 'auto';
    else
        label = sprintf('%.3g', paramVal);
    end
end