%% plot_PCA_LDA_decision_regions — LDA decision regions in the PC1-PC2 plane + scatter
% Pipeline: 320 ms spike counts, z-score, SVD, first two PCs; train multiclass
% linear LDA (pooled covariance) on (PC1,PC2); shaded grid, true-label markers,
% red crosses for misclassified points, black stars for class means.
%
% Note: boundaries are linear (LDA), unlike kNN in PCA+LDA+kNN.
%
% Requires monkeydata_training.mat in the same folder as this script.
%   plot_PCA_LDA_decision_regions

here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['File not found: %s\nPlace monkeydata_training.mat next to this script.'], dataFile);
end
load(dataFile, 'trial');

classificationHorizon = 320;
useClassifierTrainingSplit = false;
gridRes = 250;

if useClassifierTrainingSplit
    rng(2013);
    ix = randperm(size(trial, 1));
    trainingData = trial(ix(1:50), :);
else
    trainingData = trial;
end

[numTrials, numDirs] = size(trainingData);
numNeurons = size(trainingData(1, 1).spikes, 1);
numSamples = numTrials * numDirs;
features = zeros(numSamples, numNeurons);
labels = zeros(numSamples, 1);
sampleIdx = 1;
for trialIdx = 1:numTrials
    for dirIdx = 1:numDirs
        spikes = trainingData(trialIdx, dirIdx).spikes;
        endIdx = min(classificationHorizon, size(spikes, 2));
        features(sampleIdx, :) = sum(spikes(:, 1:endIdx), 2)';
        labels(sampleIdx) = dirIdx;
        sampleIdx = sampleIdx + 1;
    end
end

muF = mean(features, 1);
sigmaF = std(features, 0, 1) + eps;
X = (features - muF) ./ sigmaF;

[~, ~, v] = svd(X, "econ");
v2 = v(:, 1:2);
PC = X * v2;

numClasses = 8;
ldaMdl = trainLdaPooled2d(PC, labels, numClasses);

predTrain = predictLda2d(PC, ldaMdl);
mis = predTrain ~= labels;

pc1 = PC(:, 1);
pc2 = PC(:, 2);
pad1 = 0.08 * (max(pc1) - min(pc1) + eps);
pad2 = 0.08 * (max(pc2) - min(pc2) + eps);
x1 = linspace(min(pc1) - pad1, max(pc1) + pad1, gridRes);
x2 = linspace(min(pc2) - pad2, max(pc2) + pad2, gridRes);
[PC1g, PC2g] = meshgrid(x1, x2);
gridPts = [PC1g(:), PC2g(:)];
predGrid = predictLda2d(gridPts, ldaMdl);
Z = reshape(predGrid, gridRes, gridRes);

% 8-class discrete colors (replace with lines(8) if preferred)
cmap8 = [
    0.85 0.15 0.15
    1.00 0.55 0.10
    0.65 0.90 0.25
    0.15 0.65 0.30
    0.20 0.80 0.90
    0.15 0.35 0.95
    0.55 0.25 0.85
    0.95 0.25 0.75
];

figure('Color', 'w', 'Position', [80 80 720 580]);
axes;
imagesc(x1, x2, Z);
set(gca, 'YDir', 'normal');
colormap(gca, cmap8);
caxis([0.5 8.5]);
hold on;
axis equal tight;
box on;
set(gca, 'Layer', 'top');
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.35);
grid on;

scatter(pc1, pc2, 46, labels, 'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.4, ...
    'MarkerFaceAlpha', 0.92);

if any(mis)
    plot(pc1(mis), pc2(mis), 'rx', 'MarkerSize', 11, 'LineWidth', 2.2, ...
        'DisplayName', 'Misclassified');
end

for d = 1:numClasses
    m = mean(PC(labels == d, :), 1);
    plot(m(1), m(2), 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
    text(m(1), m(2), sprintf('  %d', d), 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0 0 0], 'VerticalAlignment', 'middle');
end

xlabel('PC1');
ylabel('PC2');
title('PCA-Projected Feature Space with LDA Decision Regions');

cb = colorbar('Ticks', 1:8, 'TickLabels', arrayfun(@num2str, 1:8, 'UniformOutput', false));
cb.Label.String = 'Predicted class (regions) / true label (markers)';

hPatch = patch('XData', [NaN NaN NaN], 'YData', [NaN NaN NaN], ...
    'FaceColor', [0.7 0.85 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hPt = scatter(NaN, NaN, 46, 0.5, 'filled', 'MarkerEdgeColor', 'k');
hX = plot(NaN, NaN, 'rx', 'LineWidth', 2.2);
hStar = plot(NaN, NaN, 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
legend([hPatch, hPt, hX, hStar], ...
    {'Decision regions (LDA)', 'Data points (color = true direction)', ...
    'Misclassified', 'Class means'}, ...
    'Location', 'northwest', 'FontSize', 9);

hold off;

%% ---- local functions (pooled-covariance LDA in 2D) ----
function mdl = trainLdaPooled2d(X, y, C)
    n = size(X, 1);
    mu = zeros(C, 2);
    nk = zeros(C, 1);
    for k = 1:C
        idx = y == k;
        nk(k) = sum(idx);
        mu(k, :) = mean(X(idx, :), 1);
    end
    Sigma = zeros(2, 2);
    for k = 1:C
        Xk = X(y == k, :);
        Xk = Xk - mean(Xk, 1);
        Sigma = Sigma + Xk' * Xk;
    end
    df = n - C;
    if df < 1
        df = 1;
    end
    Sigma = Sigma / df;
    Sigma = Sigma + 1e-5 * eye(2);
    mdl.mu = mu;
    mdl.SigmaInv = Sigma \ eye(2);
    mdl.logPrior = log(nk / n + eps);
    mdl.C = C;
end

function pred = predictLda2d(X, mdl)
    n = size(X, 1);
    scores = zeros(n, mdl.C);
    for k = 1:mdl.C
        mk = mdl.mu(k, :)';
        scores(:, k) = mdl.logPrior(k) - 0.5 * (mk' * mdl.SigmaInv * mk) + X * mdl.SigmaInv * mk;
    end
    [~, pred] = max(scores, [], 2);
end
