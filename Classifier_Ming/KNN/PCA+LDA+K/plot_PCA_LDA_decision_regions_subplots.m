%% plot_PCA_LDA_decision_regions_subplots — Left: LDA in PCA plane; right: kNN in LDA plane
% One figure, two tiles; same content as plot_PCA_LDA_decision_regions and
% plot_PCA_LDA_KNN_decision_regions combined.
%
%   plot_PCA_LDA_decision_regions_subplots

here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['File not found: %s\nPlace monkeydata_training.mat next to this script.'], dataFile);
end
load(dataFile, 'trial');

useClassifierTrainingSplit = false;
gridRes = 200;
classificationHorizon = 320;
knnK = 5;
pcaDim = 20;
ldaDim = 7;
numClasses = 8;

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

%% ----- Left: PC1–PC2, LDA regions -----
[~, ~, v] = svd(X, "econ");
v2 = v(:, 1:2);
PC = X * v2;
ldaMdl2d = trainLdaPooled2d(PC, labels, numClasses);
predLda = predictLda2d(PC, ldaMdl2d);

pc1 = PC(:, 1);
pc2 = PC(:, 2);
pad1 = 0.08 * (max(pc1) - min(pc1) + eps);
pad2 = 0.08 * (max(pc2) - min(pc2) + eps);
x1 = linspace(min(pc1) - pad1, max(pc1) + pad1, gridRes);
x2 = linspace(min(pc2) - pad2, max(pc2) + pad2, gridRes);
[PC1g, PC2g] = meshgrid(x1, x2);
Zpc = reshape(predictLda2d([PC1g(:), PC2g(:)], ldaMdl2d), gridRes, gridRes);
predAtPcPoints = labelFromPlottedGrid(pc1, pc2, x1, x2, Zpc);
misLda = predAtPcPoints ~= labels;

%% ----- Right: LD1–LD2 slice, kNN (full LDA dim) -----
modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData, knnK, pcaDim, ldaDim);
Xtr = modelParameters.X_proj;
ytr = modelParameters.y;
kVal = modelParameters.k;
ldaD = size(Xtr, 2);
sliceTail = mean(Xtr, 1);

ld1 = Xtr(:, 1);
ld2 = Xtr(:, 2);
padL1 = 0.08 * (max(ld1) - min(ld1) + eps);
padL2 = 0.08 * (max(ld2) - min(ld2) + eps);
g1 = linspace(min(ld1) - padL1, max(ld1) + padL1, gridRes);
g2 = linspace(min(ld2) - padL2, max(ld2) + padL2, gridRes);
[LG1, LG2] = meshgrid(g1, g2);
nGrid = numel(LG1);
predGrid = zeros(nGrid, 1);
for gi = 1:nGrid
    row = sliceTail;
    row(1:2) = [LG1(gi), LG2(gi)];
    predGrid(gi) = knnMode1(row, Xtr, ytr, kVal);
end
Zld = reshape(predGrid, gridRes, gridRes);
predAtLdPoints = labelFromPlottedGrid(ld1, ld2, g1, g2, Zld);
misKnn = predAtLdPoints ~= ytr;

%% ----- Figure -----
figure('Color', 'w', 'Position', [60 80 1200 520]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
imagesc(x1, x2, Zpc);
set(gca, 'YDir', 'normal');
colormap(gca, cmap8);
caxis([0.5 8.5]);
hold on;
axis equal tight;
box on;
set(gca, 'Layer', 'top', 'GridLineStyle', ':', 'GridAlpha', 0.35);
grid on;
hPatchL = patch('XData', [NaN NaN NaN], 'YData', [NaN NaN NaN], ...
    'FaceColor', [0.7 0.85 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
scatter(pc1, pc2, 46, labels, 'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.35, 'MarkerFaceAlpha', 0.92);
hPtL = scatter(NaN, NaN, 46, 0.5, 'filled', 'MarkerEdgeColor', 'k');
if any(misLda)
    plot(pc1(misLda), pc2(misLda), 'rx', 'MarkerSize', 8, 'LineWidth', 1.8);
end
hXL = plot(NaN, NaN, 'rx', 'LineWidth', 1.8);
for d = 1:numClasses
    m = mean(PC(labels == d, :), 1);
    plot(m(1), m(2), 'k*', 'MarkerSize', 14, 'LineWidth', 1.1);
    text(m(1), m(2), sprintf('  %d', d), 'FontSize', 10, 'FontWeight', 'bold');
end
hStarL = plot(NaN, NaN, 'k*', 'MarkerSize', 14, 'LineWidth', 1.1);
xlabel('PC1');
ylabel('PC2');
title('PCA space');
legend([hPatchL, hPtL, hXL, hStarL], ...
    {'Decision regions (LDA)', 'Data points (true direction)', ...
    'Misclassified (LDA)', 'Class means (PC1-PC2)'}, ...
    'Location', 'northwest', 'FontSize', 8);
hold off;

nexttile;
imagesc(g1, g2, Zld);
set(gca, 'YDir', 'normal');
colormap(gca, cmap8);
caxis([0.5 8.5]);
hold on;
axis equal tight;
box on;
set(gca, 'Layer', 'top', 'GridLineStyle', ':', 'GridAlpha', 0.35);
grid on;
hPatchR = patch('XData', [NaN NaN NaN], 'YData', [NaN NaN NaN], ...
    'FaceColor', [0.7 0.85 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
scatter(ld1, ld2, 46, ytr, 'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.35, 'MarkerFaceAlpha', 0.92);
hPtR = scatter(NaN, NaN, 46, 0.5, 'filled', 'MarkerEdgeColor', 'k');
if any(misKnn)
    plot(ld1(misKnn), ld2(misKnn), 'rx', 'MarkerSize', 8, 'LineWidth', 1.8);
end
hXR = plot(NaN, NaN, 'rx', 'LineWidth', 1.8);
for d = 1:numClasses
    m = mean(Xtr(ytr == d, 1:2), 1);
    plot(m(1), m(2), 'k*', 'MarkerSize', 14, 'LineWidth', 1.1);
    text(m(1), m(2), sprintf('  %d', d), 'FontSize', 10, 'FontWeight', 'bold');
end
hStarR = plot(NaN, NaN, 'k*', 'MarkerSize', 14, 'LineWidth', 1.1);
xlabel('LD1');
ylabel('LD2');
if ldaD > 2
    sliceNote = sprintf('; LD3–LD%d @ mean', ldaD);
else
    sliceNote = '';
end
title(sprintf('LDA space%s', sliceNote));
legend([hPatchR, hPtR, hXR, hStarR], ...
    {'Decision regions (kNN, full LDA dim)', 'Data points (true direction)', ...
    'Mismatch with plotted 2D slice', 'Class means (LD1-LD2)'}, ...
    'Location', 'northwest', 'FontSize', 8);
hold off;

cb = colorbar;
cb.Layout.Tile = 'east';
cb.Ticks = 1:8;
cb.TickLabels = arrayfun(@num2str, 1:8, 'UniformOutput', false);
cb.Label.String = 'Class (1–8)';

sgtitle('Decision regions: PCA Space (left) vs LDA Space (right)');

%% ---- local functions ----
function mdl = trainLdaPooled2d(X, y, C)
    n = size(X, 1);
    mu = zeros(C, 2);
    nk = zeros(C, 1);
    for kk = 1:C
        idx = y == kk;
        nk(kk) = sum(idx);
        mu(kk, :) = mean(X(idx, :), 1);
    end
    Sigma = zeros(2, 2);
    for kk = 1:C
        Xk = X(y == kk, :);
        Xk = Xk - mean(Xk, 1);
        Sigma = Sigma + Xk' * Xk;
    end
    df = max(n - C, 1);
    Sigma = Sigma / df + 1e-5 * eye(2);
    mdl.mu = mu;
    mdl.SigmaInv = Sigma \ eye(2);
    mdl.logPrior = log(nk / n + eps);
    mdl.C = C;
end

function pred = predictLda2d(X, mdl)
    n = size(X, 1);
    scores = zeros(n, mdl.C);
    for kk = 1:mdl.C
        mk = mdl.mu(kk, :)';
        scores(:, kk) = mdl.logPrior(kk) - 0.5 * (mk' * mdl.SigmaInv * mk) + X * mdl.SigmaInv * mk;
    end
    [~, pred] = max(scores, [], 2);
end

function pred = knnMode1(x, Xref, yref, k)
    d2 = sum((Xref - x) .^ 2, 2);
    [~, ord] = sort(d2, 'ascend');
    pred = mode(yref(ord(1:k)));
end

function pred = labelFromPlottedGrid(xq, yq, xGrid, yGrid, Z)
    nx = numel(xGrid);
    ny = numel(yGrid);
    ix = round((xq - xGrid(1)) ./ (xGrid(end) - xGrid(1) + eps) * (nx - 1)) + 1;
    iy = round((yq - yGrid(1)) ./ (yGrid(end) - yGrid(1) + eps) * (ny - 1)) + 1;
    ix = min(max(ix, 1), nx);
    iy = min(max(iy, 1), ny);
    lin = sub2ind([ny, nx], iy, ix);
    pred = Z(lin);
end

