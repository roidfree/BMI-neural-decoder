function modelParameters = positionEstimatorTraining(trainingData)
% Direction from POPULATION VECTOR: rate + baseline + tuning strength + min-threshold + power weighting.
% Then per-direction velocity regression using rate features.

step = 20;
bufferMsLong  = 300;
bufferMsShort = 100;
initTime = 300;
dirWindowEnd = 320;
smoothWinLen = 25;
lambda = 15;
avgTrajWeight = 0.4;
% PV improvements: ignore weakly tuned neurons; emphasize strong modulators
minTuningThreshold = 0.15;   % zero out neurons with normalized tuning strength below this
tuningPower = 2;             % use tuningStrength.^tuningPower so well-tuned neurons dominate

numTrials  = size(trainingData,1);
numNeurons = size(trainingData(1,1).spikes,1);

smoothKernel = ones(1, smoothWinLen) / smoothWinLen;

%% ---------- 1) Per-neuron early firing rate per direction ----------
dirMeanRates = zeros(numNeurons, 8);

for d = 1:8
    F = [];
    for i = 1:numTrials
        raw = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        Tend = min(size(cleaned,2), dirWindowEnd);
        feat = sum(cleaned(:,1:Tend), 2)' / Tend;   % mean firing rate
        F = [F; feat];
    end
    dirMeanRates(:, d) = mean(F, 1)';
end

%% ---------- 2) Baseline rate per neuron ----------
baselineRate = mean(dirMeanRates, 2);   % 98 x 1

%% ---------- 3) Preferred direction per neuron + tuning strength ----------
angles = (0:7)' * (pi/4);
unitVecs = [cos(angles), sin(angles)];   % 8 x 2

prefVec = zeros(numNeurons, 2);
tuningStrength = zeros(numNeurons, 1);

for n = 1:numNeurons
    w = dirMeanRates(n,:) - baselineRate(n);   % 1 x 8, direction modulation
    p = w * unitVecs;   % 1 x 2
    mag = norm(p);
    tuningStrength(n) = mag;
    if mag > 1e-8
        prefVec(n,:) = p / mag;
    else
        prefVec(n,:) = [0 0];
    end
end

if max(tuningStrength) > 0
    tuningStrength = tuningStrength / max(tuningStrength);
end
% Zero out weakly tuned neurons (reduces noise from untuned units)
tuningStrength(tuningStrength < minTuningThreshold) = 0;

%% ---------- 4) Per-direction velocity regression using rate features ----------
linearModelVx = cell(1,8);
linearModelVy = cell(1,8);
featMean = cell(1,8);
featStd  = cell(1,8);

for d = 1:8
    X = [];
    Yvel = [];

    for i = 1:numTrials
        raw  = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        pos  = trainingData(i,d).handPos(1:2,:);
        T    = size(cleaned,2);

        for t = (initTime+step):step:T
            startLong  = max(1, t - bufferMsLong  + 1);
            startShort = max(1, t - bufferMsShort + 1);
            lenLong  = t - startLong + 1;
            lenShort = t - startShort + 1;

            featLong  = sum(cleaned(:, startLong:t), 2)' / lenLong;
            featShort = sum(cleaned(:, startShort:t), 2)' / lenShort;
            feat = [featShort, featLong];
            vel = (pos(:,t) - pos(:,t-step))';

            X = [X; feat];
            Yvel = [Yvel; vel];
        end
    end

    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig < 1e-6) = 1;
    Xn = (X - mu) ./ sig;
    Xb = [Xn ones(size(Xn,1),1)];
    featMean{d} = mu;
    featStd{d}  = sig;

    I = eye(size(Xb,2));
    I(end,end) = 0;
    wVx = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,1));
    wVy = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,2));
    linearModelVx{d} = wVx;
    linearModelVy{d} = wVy;
end

%% ---------- 5) Direction-specific average trajectory ----------
avgTraj = cell(1,8);
for d = 1:8
    Tmax = 0;
    for i = 1:numTrials
        Tmax = max(Tmax, size(trainingData(i,d).handPos, 2));
    end
    sumPos = zeros(2, Tmax);
    count  = zeros(1, Tmax);
    for i = 1:numTrials
        pos = trainingData(i,d).handPos(1:2,:);
        T_i = size(pos, 2);
        sumPos(:,1:T_i) = sumPos(:,1:T_i) + pos;
        count(1:T_i) = count(1:T_i) + 1;
    end
    count(count == 0) = 1;
    avgTraj{d} = sumPos ./ repmat(count, 2, 1);
end

%% ---------- Pack ----------
modelParameters.step = step;
modelParameters.bufferMsLong  = bufferMsLong;
modelParameters.bufferMsShort = bufferMsShort;
modelParameters.dirWindowEnd = dirWindowEnd;
modelParameters.smoothWinLen = smoothWinLen;
modelParameters.smoothKernel = smoothKernel;
modelParameters.prefVec = prefVec;
modelParameters.tuningStrength = tuningStrength;
modelParameters.baselineRate = baselineRate;
modelParameters.minTuningThreshold = minTuningThreshold;
modelParameters.tuningPower = tuningPower;
modelParameters.linearModelVx = linearModelVx;
modelParameters.linearModelVy = linearModelVy;
modelParameters.featMean = featMean;
modelParameters.featStd  = featStd;
modelParameters.avgTraj = avgTraj;
modelParameters.avgTrajWeight = avgTrajWeight;
modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
end
