function [decodedPosX, decodedPosY, modelParameters] = positionEstimator(testData, modelParameters)
% Residual decoding: avgTraj + scale*decodedResidual, then Kalman filter.

bufferMsLong  = modelParameters.bufferMsLong;
bufferMsShort = modelParameters.bufferMsShort;
dirWindowEnd  = modelParameters.dirWindowEnd;
smoothKernel  = modelParameters.smoothKernel;
W_lda         = modelParameters.W_lda;
dirTemplatesLDA = modelParameters.dirTemplatesLDA;

raw = testData.spikes;
cleaned = conv2(1, smoothKernel, raw, 'same');

trialId = double(testData.trialId);

% --------- Direction (LDA, cached) ---------
if isKey(modelParameters.trialDirMap, trialId)
    d = modelParameters.trialDirMap(trialId);
else
    Tend = min(size(cleaned,2), dirWindowEnd);
    featEarly = sum(cleaned(:,1:Tend), 2)';
    featProj = featEarly * W_lda;
    diffs = dirTemplatesLDA - featProj;
    dist2 = sum(diffs.^2, 2);
    [~, d] = min(dist2);
    modelParameters.trialDirMap(trialId) = d;
end

% --------- Residual decode: avgTraj + scale * decodedResidual ---------
T = size(cleaned, 2);
startLong  = max(1, T - bufferMsLong  + 1);
startShort = max(1, T - bufferMsShort + 1);

lenLong  = T - startLong  + 1;
lenShort = T - startShort + 1;
featLong  = sum(cleaned(:, startLong:T), 2)'  / lenLong;
featShort = sum(cleaned(:, startShort:T), 2)' / lenShort;
startPos  = [testData.startHandPos(1), testData.startHandPos(2)];
feat = [featShort, featLong, startPos];

mu = modelParameters.featMean{d};
sig = modelParameters.featStd{d};
sig(sig < 1e-6) = 1;
featN = (feat - mu) ./ sig;
featB = [featN 1];

resX = featB * modelParameters.linearModelRx{d};
resY = featB * modelParameters.linearModelRy{d};

% Clip residual (robust to erroneous signals / outliers)
clip = modelParameters.residualClip(d);
resX = max(-clip, min(clip, resX));
resY = max(-clip, min(clip, resY));

avgTraj = modelParameters.avgTraj{d};
tIdx = min(T, size(avgTraj, 2));
scale = modelParameters.residualScale;

decodedPosX = avgTraj(1, tIdx) + scale * resX;
decodedPosY = avgTraj(2, tIdx) + scale * resY;

% --------- Kalman filter (smooth trajectory) ---------
dt = modelParameters.kf_dt * 20;   % effective dt in ms
Q = modelParameters.kf_Q;
R = modelParameters.kf_R;
F = [1 dt; 0 1];
H = [1 0];

if ~isKey(modelParameters.kfState, trialId)
    modelParameters.kfState(trialId) = struct('sx', [testData.startHandPos(1); 0], ...
        'sy', [testData.startHandPos(2); 0], 'Px', eye(2)*50, 'Py', eye(2)*50);
end
st = modelParameters.kfState(trialId);

% Predict
st.sx = F * st.sx;
st.sy = F * st.sy;
st.Px = F * st.Px * F' + Q * [dt^3/3 dt^2/2; dt^2/2 dt];
st.Py = F * st.Py * F' + Q * [dt^3/3 dt^2/2; dt^2/2 dt];

% Update with measurement (decodedPos)
Kx = st.Px * H' / (H * st.Px * H' + R);
Ky = st.Py * H' / (H * st.Py * H' + R);
st.sx = st.sx + Kx * (decodedPosX - H * st.sx);
st.sy = st.sy + Ky * (decodedPosY - H * st.sy);
st.Px = (eye(2) - Kx * H) * st.Px;
st.Py = (eye(2) - Ky * H) * st.Py;

modelParameters.kfState(trialId) = st;
decodedPosX = st.sx(1);
decodedPosY = st.sy(1);
end
