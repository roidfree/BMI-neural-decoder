function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Predict with: direction classification (early spikes) + per-direction ridge regression

bufferBins   = modelParameters.bufferSize;
dirWindowEnd = modelParameters.dirWindowEnd;

trialId = double(testData.trialId);

% --------- Decide direction once per trial (cache using trialId) ---------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    % Use early spikes to classify direction (nearest centroid)
    spikes = testData.spikes;                     % 98 x T
    Tend = min(size(spikes,2), dirWindowEnd);
    featEarly = sum(spikes(:,1:Tend), 2)';        % 1 x 98

    templates = modelParameters.dirTemplates;     % 8 x 98
    diffs = templates - featEarly;                % 8 x 98
    dist2 = sum(diffs.^2, 2);                     % 8 x 1
    [~, d] = min(dist2);
    modelParameters.trialDirMap(trialId) = d;
end

% --------- Build buffered feature for regression ---------
T = size(testData.spikes, 2);
startBin = T - bufferBins + 1;
if startBin < 1
    startBin = 1;
end

feat = sum(testData.spikes(:, startBin:T), 2)';   % 1 x 98
featB = [feat 1];

wX = modelParameters.linearModelX{d};
wY = modelParameters.linearModelY{d};

decodedPosX = featB * wX;
decodedPosY = featB * wY;
end