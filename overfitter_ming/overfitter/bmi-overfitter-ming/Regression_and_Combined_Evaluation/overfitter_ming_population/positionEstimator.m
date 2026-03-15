function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Direction from POPULATION VECTOR (rate + baseline + tuning strength); then velocity regression + blend.

bufferMsLong  = modelParameters.bufferMsLong;
bufferMsShort = modelParameters.bufferMsShort;
dirWindowEnd  = modelParameters.dirWindowEnd;
smoothKernel  = modelParameters.smoothKernel;
prefVec       = modelParameters.prefVec;
tuningStrength = modelParameters.tuningStrength;
baselineRate  = modelParameters.baselineRate;
tuningPower   = 1;
if isfield(modelParameters, 'tuningPower'), tuningPower = modelParameters.tuningPower; end

raw = testData.spikes;
cleaned = conv2(1, smoothKernel, raw, 'same');

trialId = double(testData.trialId);

% --------- Direction: population vector (rate, baseline, tuning strength) ---------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    Tend = min(size(cleaned,2), dirWindowEnd);
    featEarly = sum(cleaned(:,1:Tend), 2)' / Tend;
    featCentered = featEarly - baselineRate';
    weightedFeat = featCentered .* (tuningStrength .^ tuningPower)';
    popVec = weightedFeat * prefVec;
    angle = atan2(popVec(2), popVec(1));
    d = mod(round(angle / (pi/4)), 8) + 1;
    if d < 1, d = 1; end
    if d > 8, d = 8; end
    modelParameters.trialDirMap(trialId) = d;
end

% --------- Multi-scale buffer (rate), z-score, velocity decode, integrate ---------
T = size(cleaned, 2);
startLong  = max(1, T - bufferMsLong  + 1);
startShort = max(1, T - bufferMsShort + 1);
lenLong  = T - startLong  + 1;
lenShort = T - startShort + 1;

featLong  = sum(cleaned(:, startLong:T), 2)' / lenLong;
featShort = sum(cleaned(:, startShort:T), 2)' / lenShort;
feat = [featShort, featLong];

mu = modelParameters.featMean{d};
sig = modelParameters.featStd{d};
sig(sig < 1e-6) = 1;
featN = (feat - mu) ./ sig;
featB = [featN 1];

decodedVelX = featB * modelParameters.linearModelVx{d};
decodedVelY = featB * modelParameters.linearModelVy{d};

if isempty(testData.decodedHandPos)
    lastX = testData.startHandPos(1);
    lastY = testData.startHandPos(2);
else
    lastX = testData.decodedHandPos(1, end);
    lastY = testData.decodedHandPos(2, end);
end

decodedPosX = lastX + decodedVelX;
decodedPosY = lastY + decodedVelY;

avgTraj = modelParameters.avgTraj{d};
wAvg = modelParameters.avgTrajWeight;
tIdx = min(T, size(avgTraj, 2));
decodedPosX = (1 - wAvg) * decodedPosX + wAvg * avgTraj(1, tIdx);
decodedPosY = (1 - wAvg) * decodedPosY + wAvg * avgTraj(2, tIdx);
end
