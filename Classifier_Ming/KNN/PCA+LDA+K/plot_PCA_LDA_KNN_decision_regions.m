%% plot_PCA_LDA_KNN_decision_regions — PCA+LDA+kNN shaded decision map
% Training matches positionEstimatorTraining_PCA_LDA_K; plot LD1 vs LD2 with
% kNN distances in full LDA space (LD3+ fixed at training-set mean; 2D slice).
%
% Red crosses mark points whose true label disagrees with the plotted 2D slice
% (LD3+ fixed at training-set mean), i.e., visual mismatches in this map.
%
%   plot_PCA_LDA_KNN_decision_regions

here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['File not found: %s\nPlace monkeydata_training.mat next to this script.'], dataFile);
end
load(dataFile, 'trial');

useClassifierTrainingSplit = false;
gridRes = 180;

% Match testFunction / training hyperparameters here
knnK = 5;
pcaDim = 20;
ldaDim = 7;

if useClassifierTrainingSplit
    rng(2013);
    ix = randperm(size(trial, 1));
    trainingData = trial(ix(1:50), :);
else
    trainingData = trial;
end

modelParameters = positionEstimatorTraining_PCA_LDA_K(trainingData, knnK, pcaDim, ldaDim);
Xtr = modelParameters.X_proj;
ytr = modelParameters.y;
k = modelParameters.k;
ldaD = size(Xtr, 2);
sliceTail = mean(Xtr, 1);

ld1 = Xtr(:, 1);
ld2 = Xtr(:, 2);
pad1 = 0.08 * (max(ld1) - min(ld1) + eps);
pad2 = 0.08 * (max(ld2) - min(ld2) + eps);
g1 = linspace(min(ld1) - pad1, max(ld1) + pad1, gridRes);
g2 = linspace(min(ld2) - pad2, max(ld2) + pad2, gridRes);
[LG1, LG2] = meshgrid(g1, g2);
nGrid = numel(LG1);
predGrid = zeros(nGrid, 1);
for gi = 1:nGrid
    row = sliceTail;
    row(1:2) = [LG1(gi), LG2(gi)];
    predGrid(gi) = knnMode1(row, Xtr, ytr, k);
end
Z = reshape(predGrid, gridRes, gridRes);

mis = false(size(ytr));
for i = 1:numel(ytr)
    row = sliceTail;
    row(1:2) = Xtr(i, 1:2);
    mis(i) = knnMode1(row, Xtr, ytr, k) ~= ytr(i);
end

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

figure('Color', 'w', 'Position', [80 80 740 600]);
axes;
imagesc(g1, g2, Z);
set(gca, 'YDir', 'normal');
colormap(gca, cmap8);
caxis([0.5 8.5]);
hold on;
axis equal tight;
box on;
set(gca, 'Layer', 'top');
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.35);
grid on;

scatter(ld1, ld2, 46, ytr, 'filled', ...
    'MarkerEdgeColor', [0.15 0.15 0.15], 'LineWidth', 0.4, ...
    'MarkerFaceAlpha', 0.92);

if any(mis)
    plot(ld1(mis), ld2(mis), 'rx', 'MarkerSize', 11, 'LineWidth', 2.2);
end

numClasses = 8;
for d = 1:numClasses
    m = mean(Xtr(ytr == d, 1:2), 1);
    plot(m(1), m(2), 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
    text(m(1), m(2), sprintf('  %d', d), 'FontSize', 11, 'FontWeight', 'bold', ...
        'Color', [0 0 0], 'VerticalAlignment', 'middle');
end

xlabel('LD1 (first LDA axis on PCA subspace)');
ylabel('LD2');
if ldaD > 2
    sliceNote = sprintf('; LD3–LD%d fixed at training mean', ldaD);
else
    sliceNote = '';
end
title(sprintf(['LDA-projected space with k-NN decision regions ' ...
    '(k=%d, pcaDim=%d, ldaDim=%d%s)'], k, pcaDim, ldaD, sliceNote));

cb = colorbar('Ticks', 1:8, 'TickLabels', arrayfun(@num2str, 1:8, 'UniformOutput', false));
cb.Label.String = 'Predicted class (kNN) / true label (markers)';

hPatch = patch('XData', [NaN NaN NaN], 'YData', [NaN NaN NaN], ...
    'FaceColor', [0.7 0.85 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.5);
hPt = scatter(NaN, NaN, 46, 0.5, 'filled', 'MarkerEdgeColor', 'k');
hX = plot(NaN, NaN, 'rx', 'LineWidth', 2.2);
hStar = plot(NaN, NaN, 'k*', 'MarkerSize', 16, 'LineWidth', 1.2);
legend([hPatch, hPt, hX, hStar], ...
    {'Decision regions (kNN, full LD dim)', 'Data points (true direction)', ...
    'Mismatch with plotted 2D slice', 'Class means (LD1–LD2)'}, ...
    'Location', 'northwest', 'FontSize', 9);

hold off;

function pred = knnMode1(x, Xref, yref, k)
    d2 = sum((Xref - x) .^ 2, 2);
    [~, ord] = sort(d2, 'ascend');
    pred = mode(yref(ord(1:k)));
end

