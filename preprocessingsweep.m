clc; clear;
% Preprocessing sweep using dameer_grader for consistency.
% Methods fixed to:
%   - Classifier: pca_lda_knn_ming
%   - Regressor: pooled_pls (nComp = 30)
%
% Families:
%   1) spike_count: binning + transform (none/anscombe)
%   2) gaussian: Gaussian smoothing  + transform (none/sqrt)
%


repoRoot = fileparts(fileparts(mfilename('fullpath')));
outDir = fullfile(repoRoot, 'benchmark_outputs', 'preprocessing_sweep_pls_knn');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

base = struct();
base.K = 5;
base.seed = 42;
base.startMs = 320;
base.computeProfile = 'full';
base.plotFigures = false;
base.saveOutputs = false;
base.verbose = false;
base.timestampOutputDir = false;
base.classifierMethods = {'pca_lda_knn_ming'};
base.classifierPipelineK = 11;
base.classifierPcaComponents = 20;
base.pooledRegressorMethods = {'pooled_pls'};
base.pooledPLSGrid = 30;
base.pipelineMethods = {};
base.includeKalman = false;

sampleSteps = [5, 10, 15, 20, 40];
spikeTransforms = {'none', 'anscombe'};
gaussSigmas = [3, 5, 10, 20];
gaussTransforms = {'none', 'sqrt'};

totalRuns = numel(sampleSteps) * numel(spikeTransforms) ...
    + numel(sampleSteps) * numel(gaussSigmas) * numel(gaussTransforms);
fprintf('Total runs: %d\n', totalRuns);

rows = {};
runId = 0;

for b = 1:numel(sampleSteps)
    for t = 1:numel(spikeTransforms)
        runId = runId + 1;
        local = base;
        local.family = 'spike_count';
        local.sampleStep = sampleSteps(b);
        local.binWidth = sampleSteps(b);
        local.useGauss = false;
        local.sigma = NaN;
        local.gaussWidth = NaN;
        local.transform = spikeTransforms{t};

        rows(end + 1, :) = run_single(local, runId, totalRuns, repoRoot); %#ok<AGROW>
    end
end

for ss = 1:numel(sampleSteps)
    for s = 1:numel(gaussSigmas)
        for t = 1:numel(gaussTransforms)
            runId = runId + 1;
            local = base;
            local.family = 'gaussian';
            local.sampleStep = sampleSteps(ss);
            local.binWidth = sampleSteps(ss);
            local.useGauss = true;
            local.sigma = gaussSigmas(s);
            local.gaussWidth = 2 * gaussSigmas(s) + 1;
            local.transform = gaussTransforms{t};

            rows(end + 1, :) = run_single(local, runId, totalRuns, repoRoot); %#ok<AGROW>
        end
    end
end

results = cell2table(rows, 'VariableNames', ...
    {'family', 'sampleStepMs', 'binWidth', 'useGauss', 'gaussSigma', 'gaussWidth', 'transform', ...
     'meanAccuracy', 'stdAccuracy', ...
     'meanRMSE', 'stdRMSE', ...
     'meanTrainSec', 'meanTestSec', ...
     'status', 'message'});

writetable(results, fullfile(outDir, 'preprocessing_sweep_pls_knn.csv'));

validCls = results(strcmp(results.status, "ok") & ~isnan(results.meanAccuracy), :);
if ~isempty(validCls)
    bestCls = sortrows(validCls, {'meanAccuracy', 'meanRMSE'}, {'descend', 'ascend'});
    writetable(bestCls(1, :), fullfile(outDir, 'best_for_classifier_knn.csv'));
end

validReg = results(strcmp(results.status, "ok") & ~isnan(results.meanRMSE), :);
if ~isempty(validReg)
    bestReg = sortrows(validReg, {'meanRMSE', 'meanAccuracy'}, {'ascend', 'descend'});
    writetable(bestReg(1, :), fullfile(outDir, 'best_for_regressor_pls.csv'));
end

fprintf('Saved outputs to: %s\n', outDir);

function row = run_single(local, runId, totalRuns, repoRoot)
    fprintf('[%03d/%03d] family=%s step=%d gauss=%d sigma=%g width=%g transform=%s\n', ...
        runId, totalRuns, local.family, local.sampleStep, local.useGauss, local.sigma, local.gaussWidth, local.transform);

    opts = build_dameer_opts(local, repoRoot);

    try
        bench = dameer_grader(opts);

        cls = bench.tables.classifierSummary;
        cls = cls(strcmp(cls.method, 'pca_lda_knn_ming'), :);
        if isempty(cls)
            meanAcc = NaN; stdAcc = NaN; clsTrain = NaN; clsTest = NaN;
        else
            meanAcc = cls.meanAccuracy(1);
            stdAcc = cls.stdAccuracy(1);
            clsTrain = cls.meanTrainTimeSec(1);
            clsTest = cls.meanTestTimeSec(1);
        end

        reg = bench.tables.continuousSummary;
        reg = reg(strcmp(reg.method, 'pooled_pls') & reg.nPC == 30, :);
        if isempty(reg)
            meanRmse = NaN; stdRmse = NaN; regTrain = NaN; regTest = NaN;
        else
            meanRmse = reg.meanRMSE(1);
            stdRmse = reg.stdRMSE(1);
            regTrain = reg.meanTrainTimeSec(1);
            regTest = reg.meanTestTimeSec(1);
        end

        row = {string(local.family), local.sampleStep, local.binWidth, local.useGauss, local.sigma, local.gaussWidth, string(local.transform), ...
            meanAcc, stdAcc, meanRmse, stdRmse, clsTrain + regTrain, clsTest + regTest, "ok", ""};
    catch runErr
        row = {string(local.family), local.sampleStep, local.binWidth, local.useGauss, local.sigma, local.gaussWidth, string(local.transform), ...
            NaN, NaN, NaN, NaN, NaN, NaN, "failed", string(runErr.message)};
    end
end

function opts = build_dameer_opts(local, repoRoot)
    opts = struct();
    opts.dataFile = fullfile(repoRoot, 'BMI', 'monkeydata0.mat');
    opts.seed = local.seed;
    opts.K = local.K;
    opts.startMs = local.startMs;
    opts.stepMs = local.binWidth;
    opts.computeProfile = local.computeProfile;
    opts.plotFigures = local.plotFigures;
    opts.saveOutputs = local.saveOutputs;
    opts.verbose = local.verbose;
    opts.timestampOutputDir = local.timestampOutputDir;

    opts.classifierMethods = local.classifierMethods;
    opts.classifierPipelineK = local.classifierPipelineK;
    opts.classifierPcaComponents = local.classifierPcaComponents;
    opts.classifierBinWidth = local.binWidth;
    opts.classifierTransform = local.transform;

    opts.pooledRegressorMethods = local.pooledRegressorMethods;
    opts.pooledPLSGrid = local.pooledPLSGrid;
    opts.pooledHistoryBins = 15;
    opts.pooledBinWidth = local.binWidth;
    opts.pooledTransform = local.transform;

    opts.pipelineMethods = local.pipelineMethods;
    opts.includeKalman = local.includeKalman;

    if local.useGauss
        sigma = local.sigma;
        width = floor((local.gaussWidth - 1) / 2); % map total width to dameer half-width convention

        opts.classifierPreSmoothKernel = 'gauss';
        opts.classifierPreSmoothWidth = width;
        opts.classifierPreSmoothParam = sigma;
        opts.classifierPostSmoothKernel = 'none';
        opts.classifierPostSmoothWidth = 0;
        opts.classifierPostSmoothParam = NaN;

        opts.pooledPreSmoothKernel = 'gauss';
        opts.pooledPreSmoothWidth = width;
        opts.pooledPreSmoothParam = sigma;
        opts.pooledPostSmoothKernel = 'none';
        opts.pooledPostSmoothWidth = 0;
        opts.pooledPostSmoothParam = NaN;
    else
        opts.classifierPreSmoothKernel = 'none';
        opts.classifierPreSmoothWidth = 0;
        opts.classifierPreSmoothParam = NaN;
        opts.classifierPostSmoothKernel = 'none';
        opts.classifierPostSmoothWidth = 0;
        opts.classifierPostSmoothParam = NaN;

        opts.pooledPreSmoothKernel = 'none';
        opts.pooledPreSmoothWidth = 0;
        opts.pooledPreSmoothParam = NaN;
        opts.pooledPostSmoothKernel = 'none';
        opts.pooledPostSmoothWidth = 0;
        opts.pooledPostSmoothParam = NaN;
    end
end