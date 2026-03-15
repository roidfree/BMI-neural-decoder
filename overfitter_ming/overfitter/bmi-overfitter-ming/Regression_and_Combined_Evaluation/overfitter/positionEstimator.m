function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Predict with: direction classification (early spikes) + per-direction ridge regression
% Uses same cleaned (smoothed) signal as in training.

bufferBinsLong  = modelParameters.bufferSizeLong;
bufferBinsShort = modelParameters.bufferSizeShort;
dirWindowEnd    = modelParameters.dirWindowEnd;
smoothKernel    = modelParameters.smoothKernel;
W_lda           = modelParameters.W_lda;
dirTemplatesLDA = modelParameters.dirTemplatesLDA;

% Raw spikes -> cleaner signal (smoothed rate)
raw = testData.spikes;                            % 98 x T
cleaned = conv2(1, smoothKernel, raw, 'same');

trialId = double(testData.trialId);

% --------- Decide direction once per trial (cache using trialId) ---------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    % LDA: project early feature then nearest centroid in LDA space
    Tend = min(size(cleaned,2), dirWindowEnd);
    featEarly = sum(cleaned(:,1:Tend), 2)';       % 1 x 98
    featProj = featEarly * W_lda;                 % 1 x nLDA

    diffs = dirTemplatesLDA - featProj;           % 8 x nLDA
    dist2 = sum(diffs.^2, 2);
    [~, d] = min(dist2);
    modelParameters.trialDirMap(trialId) = d;
end

% --------- Multi-scale buffer feature, z-score, decode velocity, integrate ---------
T = size(cleaned, 2);
startLong  = T - bufferBinsLong  + 1;
startShort = T - bufferBinsShort + 1;
if startLong  < 1, startLong  = 1; end
if startShort < 1, startShort = 1; end

featLong  = sum(cleaned(:, startLong:T), 2)';
featShort = sum(cleaned(:, startShort:T), 2)';
feat = [featShort, featLong];                      % 1 x 196
mu = modelParameters.featMean{d};
sig = modelParameters.featStd{d};
sig(sig < 1e-6) = 1;
featN = (feat - mu) ./ sig;
featB = [featN 1];

wVx = modelParameters.linearModelVx{d};
wVy = modelParameters.linearModelVy{d};
decodedVelX = featB * wVx;
decodedVelY = featB * wVy;

% Integrate: new position = last position + velocity (per step)
if isempty(testData.decodedHandPos)
    lastX = testData.startHandPos(1);
    lastY = testData.startHandPos(2);
else
    lastX = testData.decodedHandPos(1, end);
    lastY = testData.decodedHandPos(2, end);
end
% Regression prediction (velocity integration)
decodedPosX = lastX + decodedVelX;
decodedPosY = lastY + decodedVelY;

% BMI trick: blend with direction-specific average trajectory (stabilizes RMSE)
avgTraj = modelParameters.avgTraj{d};
wAvg = modelParameters.avgTrajWeight;
tIdx = min(T, size(avgTraj, 2));
avgX = avgTraj(1, tIdx);
avgY = avgTraj(2, tIdx);
decodedPosX = (1 - wAvg) * decodedPosX + wAvg * avgX;
decodedPosY = (1 - wAvg) * decodedPosY + wAvg * avgY;
end