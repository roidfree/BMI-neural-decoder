%% plot_PCA_LDA_2D — Scatter: first two PCs vs first two LDA directions on PCA subspace
% Features, z-score, PCA (SVD), LDA (eig(SB,SW)) match positionEstimatorTraining_PCA_LDA_K.
%
% Run from a folder that contains monkeydata_training.mat (same folder as this script).
%   plot_PCA_LDA_2D
%
% Edit parameters below as needed.

here = fileparts(mfilename('fullpath'));
dataFile = fullfile(here, 'monkeydata_training.mat');
if exist(dataFile, 'file') ~= 2
    error(['File not found: %s\nPlace monkeydata_training.mat next to this script or set dataFile.'], dataFile);
end
load(dataFile, 'trial');

%% Parameters (same style as classifier)
classificationHorizon = 320;
pcaDim = 20;   % >= 2; figure shows PC1-PC2
ldaDim = 7;    % >= 2; figure shows LD1-LD2

% true: match test script (rng(2013), first 50 trial rows only)
% false: use all trials (denser scatter)
useClassifierTrainingSplit = false;

if useClassifierTrainingSplit
    rng(2013);
    ix = randperm(size(trial, 1));
    trainingData = trial(ix(1:50), :);
else
    trainingData = trial;
end

%% Build features (same as positionEstimatorTraining_PCA_LDA_K)
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

mu = mean(features, 1);
sigma = std(features, 0, 1) + eps;
X = (features - mu) ./ sigma;

%% PCA (econ SVD on X, right singular vectors)
[~, ~, v] = svd(X, "econ");
pcaDim = min(pcaDim, size(v, 2));
vPca = v(:, 1:pcaDim);
pcaFeatures = X * vPca;
ldaDim = min(ldaDim, pcaDim);

%% LDA on PCA subspace (same as training script)
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
[~, order] = sort(eigScore, "descend");
ldaDim = min(ldaDim, numel(order));
wLda = real(eigVec(:, order(1:ldaDim)));
ldaFeatures = pcaFeatures * wLda;

%% Plot
cmap = lines(8);
figure('Color', 'w', 'Position', [100 100 900 380]);
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');

nexttile;
hold on;
for d = 1:8
    m = labels == d;
    scatter(pcaFeatures(m, 1), pcaFeatures(m, 2), 12, cmap(d, :), 'filled', ...
        'MarkerFaceAlpha', 0.65, 'DisplayName', sprintf('Direction %d', d));
end
hold off;
grid on;
xlabel('PC1');
ylabel('PC2');
title('PCA projection');
legend('Location', 'bestoutside', 'NumColumns', 2);

nexttile;
hold on;
for d = 1:8
    m = labels == d;
    scatter(ldaFeatures(m, 1), ldaFeatures(m, 2), 12, cmap(d, :), 'filled', ...
        'MarkerFaceAlpha', 0.65, 'DisplayName', sprintf('Direction %d', d));
end
hold off;
grid on;
xlabel('LD1');
ylabel('LD2');
title('LDA projection on PCA subspace');

sgtitle(sprintf(['Neural firing-rate features (320 ms counts, z-scored) - ' ...
    'pcaDim=%d, ldaDim=%d'], pcaDim, ldaDim));
