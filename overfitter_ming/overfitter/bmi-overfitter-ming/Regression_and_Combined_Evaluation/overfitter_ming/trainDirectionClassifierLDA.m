function ldaModel = trainDirectionClassifierLDA(trainingData, varargin)
% Train LDA direction classifier from smoothed early spike features.
% Same style as trainDirectionClassifier: input training_data, output model struct.
%
% ldaModel = trainDirectionClassifierLDA(trainingData)
% ldaModel = trainDirectionClassifierLDA(trainingData, 'dirWindowEnd', 320, 'smoothWinLen', 25)
%
% Output: ldaModel.W_lda, ldaModel.dirTemplatesLDA, ldaModel.smoothKernel,
%         ldaModel.dirWindowEnd, ldaModel.smoothWinLen

[trials, numDirs] = size(trainingData);
numNeurons = size(trainingData(1,1).spikes, 1);

% Defaults (match positionEstimatorTraining)
dirWindowEnd = 320;
smoothWinLen = 25;
ldaReg = 1e-3;
for k = 1:2:length(varargin)
    if strcmpi(varargin{k}, 'dirWindowEnd'), dirWindowEnd = varargin{k+1}; end
    if strcmpi(varargin{k}, 'smoothWinLen'), smoothWinLen = varargin{k+1}; end
    if strcmpi(varargin{k}, 'ldaReg'), ldaReg = varargin{k+1}; end
end

smoothKernel = ones(1, smoothWinLen) / smoothWinLen;

% ---------- 1) Per-direction early features (smoothed sum over [1, dirWindowEnd] ms) ----------
dirFeatAll = cell(1, numDirs);
for d = 1:numDirs
    F = [];
    for i = 1:trials
        raw = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        Tend = min(size(cleaned, 2), dirWindowEnd);
        feat = sum(cleaned(:, 1:Tend), 2)';
        F = [F; feat];
    end
    dirFeatAll{d} = F;
end

% ---------- 2) LDA: within-class Sw, between-class Sb, generalized eig(Sb, Sw) ----------
AllFeat = [];
for d = 1:numDirs
    AllFeat = [AllFeat; dirFeatAll{d}];
end
mu_all = mean(AllFeat, 1);
Sw = zeros(numNeurons, numNeurons);
Sb = zeros(numNeurons, numNeurons);
for d = 1:numDirs
    Xd = dirFeatAll{d};
    mu_d = mean(Xd, 1);
    n_d = size(Xd, 1);
    Sw = Sw + (Xd - mu_d)' * (Xd - mu_d);
    Sb = Sb + n_d * (mu_d - mu_all)' * (mu_d - mu_all);
end
Sw = Sw + ldaReg * eye(numNeurons);
[V, D] = eig(Sb, Sw);
[D, ord] = sort(diag(D), 'descend');
V = V(:, ord);
nLDA = min(numDirs - 1, size(V, 2));
W_lda = V(:, 1:nLDA);

% Class means in LDA space (templates for nearest-centroid at test)
dirTemplatesLDA = zeros(numDirs, nLDA);
for d = 1:numDirs
    dirTemplatesLDA(d, :) = mean(dirFeatAll{d}, 1) * W_lda;
end

% Pack (same naming as used in positionEstimatorTraining / positionEstimator)
ldaModel.W_lda = W_lda;
ldaModel.dirTemplatesLDA = dirTemplatesLDA;
ldaModel.smoothKernel = smoothKernel;
ldaModel.dirWindowEnd = dirWindowEnd;
ldaModel.smoothWinLen = smoothWinLen;
end
