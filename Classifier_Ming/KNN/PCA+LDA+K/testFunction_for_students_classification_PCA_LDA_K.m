% testFunction_for_students_classification_PCA_LDA_K.m

function [bestK, bestPCA, bestLDA, sortedResults] = testFunction_for_students_classification_PCA_LDA_K(teamName)
% Hyperparameter search for PCA + LDA + k-NN (same train/test split for all runs).
%
% Usage:
%   [bestK,bestPCA,bestLDA] = testFunction_for_students_classification_PCA_LDA_K;
%   [bestK,bestPCA,bestLDA,sortedResults] = ... % sortedResults is Nx4 [k,pca,lda,acc], desc acc
%
% Edit the "User settings" block below.

    %% ========== User settings ==========
    % searchMode:
    %   'paper_table' - only the 6 rows used in the paper table (fast)
    %   'grid_wide'   - moderate k x PCA x LDA grid (recommended to explore)
    %   'grid_fine'   - denser k steps, more PCA dims (slower)
    %   'custom'      - use kList / pcaList / ldaList (nested loops, skip LDA>=PCA)
    searchMode = 'grid_wide';

    % Custom mode only: k must be <= num training samples (50 trials x 8 dirs = 400)
    kList    = [1, 27, 75];
    pcaList  = 5:5:30;
    ldaList  = 1:7;

    % Optional: cap how many configs to run (inf = all). Useful for quick smoke tests.
    maxEvaluations = inf;

    % Paper / reproducibility table (used when searchMode == 'paper_table')
    hyperConfigs = [
        1,   5,  1;
        1,  10,  4;
        27, 10,  4;
        27, 20,  4;
        75, 10,  4;
        75, 20,  6;
    ];
    %% ========== End user settings ==========

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(size(trial,1));
    trainData = trial(ix(1:50),:);
    testData  = trial(ix(51:end),:);

    nTrain = numel(trainData) * 8;
    ldaMax = 7;

    switch searchMode
        case 'paper_table'
            runs = buildRunsFromMatrix(hyperConfigs, nTrain, ldaMax);
        case 'grid_wide'
            kGrid = [1 3 5 7 9 11 15 19 23 27 31 39 47 55 65 75];
            pcaGrid = 5:5:35;
            runs = buildRunsGrid(kGrid, pcaGrid, ldaMax, nTrain);
        case 'grid_fine'
            kGrid = unique([1:2:51, 55 65 75 95 115]);
            kGrid = kGrid(kGrid <= nTrain);
            pcaGrid = 5:5:40;
            runs = buildRunsGrid(kGrid, pcaGrid, ldaMax, nTrain);
        case 'custom'
            runs = buildRunsCustom(kList, pcaList, ldaList, nTrain, ldaMax);
        otherwise
            error('Unknown searchMode: use paper_table | grid_wide | grid_fine | custom');
    end

    nRuns = size(runs, 1);
    if isfinite(maxEvaluations) && maxEvaluations < nRuns
        runs = runs(1:maxEvaluations, :);
        nRuns = size(runs, 1);
        fprintf('Limited to first %d configurations (maxEvaluations).\n', nRuns);
    end

    fprintf('PCA -> LDA -> k-NN | mode=%s | %d configurations | train N=%d\n\n', searchMode, nRuns, nTrain);

    allRes = zeros(0, 4);
    t0 = tic;
    for ri = 1:nRuns
        k = runs(ri, 1);
        p = runs(ri, 2);
        d = runs(ri, 3);
        acc = evalConfig(trainData, testData, k, p, d);
        allRes(end+1, :) = [k, p, d, acc]; %#ok<AGROW>
        if mod(ri, max(1, floor(nRuns/10))) == 1 || ri == nRuns
            fprintf('  [%4d / %4d] k=%3d  PCA=%2d  LDA=%2d  ->  %.2f%%\n', ri, nRuns, k, p, d, acc);
        end
    end
    fprintf('\nElapsed %.1f s\n\n', toc(t0));

    [~, ord] = sort(allRes(:,4), 'descend');
    sortedResults = allRes(ord, :);
    bestK = sortedResults(1, 1);
    bestPCA = sortedResults(1, 2);
    bestLDA = sortedResults(1, 3);
    bestAcc = sortedResults(1, 4);

    nTop = min(25, size(sortedResults, 1));
    fprintf('Top %d configurations (by accuracy):\n', nTop);
    fprintf('%-6s %-10s %-10s %-12s\n', 'k', 'PCA Dim', 'LDA Dim', 'Accuracy (%)');
    fprintf('%s\n', repmat('-', 1, 44));
    for i = 1:nTop
        fprintf('%-6d %-10d %-10d %.2f%%\n', sortedResults(i,1), sortedResults(i,2), sortedResults(i,3), sortedResults(i,4));
    end
    fprintf('\n=> Best: k=%d, PCA=%d, LDA=%d with %.2f%%\n', bestK, bestPCA, bestLDA, bestAcc);
end

function runs = buildRunsFromMatrix(hyperConfigs, nTrain, ldaMax)
    runs = zeros(0, 3);
    for r = 1:size(hyperConfigs, 1)
        k = hyperConfigs(r, 1);
        p = hyperConfigs(r, 2);
        d = hyperConfigs(r, 3);
        if d >= p || k > nTrain || d > ldaMax
            continue;
        end
        runs(end+1, :) = [k, p, d]; %#ok<AGROW>
    end
end

function runs = buildRunsGrid(kGrid, pcaGrid, ldaMax, nTrain)
    runs = zeros(0, 3);
    for k = kGrid(:)'
        if k > nTrain || k < 1, continue; end
        for p = pcaGrid(:)'
            dMax = min(ldaMax, p - 1);
            if dMax < 1, continue; end
            for d = 1:dMax
                runs(end+1, :) = [k, p, d]; %#ok<AGROW>
            end
        end
    end
end

function runs = buildRunsCustom(kList, pcaList, ldaList, nTrain, ldaMax)
    runs = zeros(0, 3);
    for k = kList(:)'
        if k > nTrain || k < 1, continue; end
        for p = pcaList(:)'
            for d = ldaList(:)'
                if d >= p || d > ldaMax || d < 1, continue; end
                runs(end+1, :) = [k, p, d]; %#ok<AGROW>
            end
        end
    end
end

function acc = evalConfig(trainData, testData, k, p, d)
    mdl = positionEstimatorTraining_PCA_LDA_K(trainData, k, p, d);
    correct = 0;
    total = 0;
    for tr = 1:size(testData,1)
        for dir = 1:8
            sample.spikes = testData(tr,dir).spikes;
            pred = positionEstimator_PCA_LDA_K(sample, mdl);
            correct = correct + (pred == dir);
            total = total + 1;
        end
    end
    acc = 100 * correct / total;
end
