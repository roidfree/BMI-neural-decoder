clc; clear;

% Simple pooled-regressor sweeps aligned to the project ask:
%   1) preprocessing sweep -> fixed pooled PCR regressor (general / pooled)
%   2) fixed preprocessing (20ms + Anscombe) -> sweep all pooled regressors
%
% NOTE: "pooled" here means trained on all directions (not movement-specific).

repoRoot = fileparts(fileparts(mfilename('fullpath')));
outputDir = fullfile(repoRoot, 'benchmark_outputs', 'preprocess_regressor_sweeps');
if exist(outputDir, 'dir') ~= 7
    mkdir(outputDir);
end

baseOpts = struct();
baseOpts.plotFigures = false;
baseOpts.saveOutputs = false;
baseOpts.verbose = false;
baseOpts.timestampOutputDir = false;
baseOpts.classifierMethods = {};
baseOpts.pipelineMethods = {};
baseOpts.includeKalman = false;

%% ============================================================
%  PART A: PREPROCESSING SWEEP (fixed pooled PCR regressor)
% =============================================================
%
% Regressor fixed to pooled PCR with nPC = 300.
% We only move preprocessing knobs and track RMSE movement.

preBinWidthGrid = [10, 20, 25, 40];
preHistoryGrid = [8, 12, 15, 20];
preTransformGrid = {'none', 'sqrt', 'anscombe'};
fixedNpcForPreSweep = 300;

rowsA = {};
expId = 0;

for tIdx = 1:numel(preTransformGrid)
    transformName = preTransformGrid{tIdx};
    for bIdx = 1:numel(preBinWidthGrid)
        binWidth = preBinWidthGrid(bIdx);
        for hIdx = 1:numel(preHistoryGrid)
            historyBins = preHistoryGrid(hIdx);
            expId = expId + 1;

            fprintf('[A-%03d] transform=%s | bin=%d | history=%d | nPC=%d\n', ...
                expId, transformName, binWidth, historyBins, fixedNpcForPreSweep);

            opts = baseOpts;
            opts.pooledRegressorMethods = {'pooled_pcr'};
            opts.pooledPCRGrid = fixedNpcForPreSweep;
            opts.pooledBinWidth = binWidth;
            opts.pooledHistoryBins = historyBins;
            opts.pooledTransform = transformName;

            benchmark = dameer_grader(opts);
            summary = benchmark.tables.continuousSummary;
            mask = strcmp(summary.method, 'pooled_pcr') & summary.nPC == fixedNpcForPreSweep;
            subset = summary(mask, :);

            if isempty(subset)
                meanRmse = NaN;
                stdRmse = NaN;
            else
                meanRmse = subset.meanRMSE(1);
                stdRmse = subset.stdRMSE(1);
            end

            rowsA(end + 1, :) = {expId, string(transformName), binWidth, historyBins, ... %#ok<AGROW>
                fixedNpcForPreSweep, meanRmse, stdRmse};
        end
    end
end

preprocessSweep = cell2table(rowsA, 'VariableNames', ...
    {'expId', 'transform', 'binWidth', 'historyBins', 'nPC', 'meanRMSE', 'stdRMSE'});
preprocessSweep = sortrows(preprocessSweep, {'meanRMSE', 'stdRMSE'}, {'ascend', 'ascend'});

partA_csv = fullfile(outputDir, 'preprocessing_rmse_sweep.csv');
writetable(preprocessSweep, partA_csv);

bestA = preprocessSweep(1, :);
partA_txt = fullfile(outputDir, 'best_preprocessing_for_fixed_pcr.txt');
fid = fopen(partA_txt, 'w');
fprintf(fid, 'Best preprocessing for pooled PCR (nPC=%d):\n', fixedNpcForPreSweep);
fprintf(fid, 'transform=%s\n', bestA.transform);
fprintf(fid, 'binWidth=%d\n', bestA.binWidth);
fprintf(fid, 'historyBins=%d\n', bestA.historyBins);
fprintf(fid, 'meanRMSE=%.6f\n', bestA.meanRMSE);
fprintf(fid, 'stdRMSE=%.6f\n', bestA.stdRMSE);
fclose(fid);

figA = figure('Visible', 'off');
for tIdx = 1:numel(preTransformGrid)
    transformName = string(preTransformGrid{tIdx});
    subset = preprocessSweep(preprocessSweep.transform == transformName, :);
    matrix = build_heatmap_matrix(subset, preBinWidthGrid, preHistoryGrid, 'meanRMSE');

    subplot(1, numel(preTransformGrid), tIdx);
    imagesc(preBinWidthGrid, preHistoryGrid, matrix);
    axis xy;
    colorbar;
    xlabel('Bin width (ms)');
    ylabel('History bins');
    title(sprintf('transform=%s', transformName));
end
sgtitle(sprintf('RMSE movement for pooled PCR (nPC=%d)', fixedNpcForPreSweep));
saveas(figA, fullfile(outputDir, 'preprocessing_rmse_heatmaps_fixed_pcr.png'));
close(figA);

%% ============================================================
%  PART B: FIXED PREPROCESSING (20ms + Anscombe), SWEEP REGRESSORS
% =============================================================
%
% Fixed preprocessing:
%   - bin width = 20 ms
%   - history bins = 15
%   - transform = anscombe
%
% Sweep both pooled regressors and their hyperparameters.

optsB = baseOpts;
optsB.pooledRegressorMethods = {'pooled_pcr', 'pooled_ridge_pcr'};
optsB.pooledPCRGrid = [50, 100, 200, 300, 400, 500];
optsB.pooledRidgePCRGrid = [50, 100, 200, 300, 400, 500];
optsB.pooledRidgeLambdaGrid = [0.1, 1, 10, 100, 1000];
optsB.pooledBinWidth = 20;
optsB.pooledHistoryBins = 15;
optsB.pooledTransform = 'anscombe';

fprintf('\n[PART B] fixed preprocessing sweep: bin=20, history=15, transform=anscombe\n');
benchmarkB = dameer_grader(optsB);
summaryB = benchmarkB.tables.continuousSummary;
regMask = strcmp(summaryB.task, 'regressor_pooled');
regSummary = summaryB(regMask, :);
regSummary = sortrows(regSummary, {'method', 'lambda', 'nPC'});

partB_csv = fullfile(outputDir, 'fixed_preprocessing_regressor_sweep.csv');
writetable(regSummary, partB_csv);

% Best-by-RMSE per method
methods = unique(regSummary.method);
bestRows = repmat(regSummary(1, :), 0, 1);
for mIdx = 1:numel(methods)
    methodName = methods{mIdx};
    subset = regSummary(strcmp(regSummary.method, methodName), :);
    [~, bestIdx] = min(subset.meanRMSE);
    bestRows(end + 1, :) = subset(bestIdx, :); %#ok<AGROW>
end
bestRows = sortrows(bestRows, {'meanRMSE', 'method'});
partB_best_csv = fullfile(outputDir, 'best_regressor_by_method_fixed_preprocessing.csv');
writetable(bestRows, partB_best_csv);

% Plot RMSE vs nPC for pooled_pcr
figB1 = figure('Visible', 'off');
subsetPCR = regSummary(strcmp(regSummary.method, 'pooled_pcr'), :);
[xPcr, ordPcr] = sort(subsetPCR.nPC);
yPcr = subsetPCR.meanRMSE(ordPcr);
plot(xPcr, yPcr, '-o', 'LineWidth', 1.5);
grid on;
xlabel('nPC');
ylabel('Mean CV RMSE');
title('Fixed preprocessing: pooled PCR RMSE vs nPC (20ms + Anscombe + 15 history)');
saveas(figB1, fullfile(outputDir, 'fixed_preproc_rmse_vs_npc_pooled_pcr.png'));
close(figB1);

% Plot RMSE vs nPC for pooled_ridge_pcr by lambda
figB2 = figure('Visible', 'off');
hold on;
grid on;
subsetRidge = regSummary(strcmp(regSummary.method, 'pooled_ridge_pcr'), :);
lambdaVals = unique(subsetRidge.lambda(~isnan(subsetRidge.lambda)));
for lIdx = 1:numel(lambdaVals)
    lambdaVal = lambdaVals(lIdx);
    mask = abs(subsetRidge.lambda - lambdaVal) < 1e-12;
    lineSubset = subsetRidge(mask, :);
    [xVals, ord] = sort(lineSubset.nPC);
    yVals = lineSubset.meanRMSE(ord);
    plot(xVals, yVals, '-o', 'LineWidth', 1.5, ...
        'DisplayName', sprintf('\\lambda = %.3g', lambdaVal));
end
legend('Location', 'best');
xlabel('nPC');
ylabel('Mean CV RMSE');
title('Fixed preprocessing: pooled ridge-PCR RMSE vs nPC (20ms + Anscombe + 15 history)');
saveas(figB2, fullfile(outputDir, 'fixed_preproc_rmse_vs_npc_pooled_ridge_pcr.png'));
close(figB2);

fprintf('\nSaved all sweep outputs to:\n%s\n', outputDir);
fprintf('Part A CSV: %s\n', partA_csv);
fprintf('Part B CSV: %s\n', partB_csv);
fprintf('Best-per-method CSV: %s\n', partB_best_csv);

function matrix = build_heatmap_matrix(subset, xVals, yVals, fieldName)
    matrix = nan(numel(yVals), numel(xVals));
    for yIdx = 1:numel(yVals)
        for xIdx = 1:numel(xVals)
            mask = subset.binWidth == xVals(xIdx) & subset.historyBins == yVals(yIdx);
            if any(mask)
                vals = subset.(fieldName)(mask);
                matrix(yIdx, xIdx) = mean(vals, 'omitnan');
            end
        end
    end
end