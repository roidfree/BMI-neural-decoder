function modelParameters = positionEstimatorTraining(trainingData)
% Direction classification + per-direction regression (buffered)
% Uses smoothed (cleaned) firing rate instead of raw spike counts.

step = 20;                  % ms between decoded samples
% Windows in MILLISECONDS (each spikes column is 1 ms)
bufferMsLong  = 300;        % long history: 300 ms
bufferMsShort = 100;        % short history: 100 ms
initTime = 300;             % start using data after 300 ms
dirWindowEnd = 320;         % direction window [1, 320] ms
smoothWinLen = 25;          % smoothing window 25 ms
lambda = 5;                 % ridge for residual regression
residualScale = 0.6;        % decodedPos = avgTraj + 0.6 * residual
residualClipPct = 88;       % clip residual at 88th percentile (robust)
% Kalman (light smoothing on top)
kf_dt = 1;
kf_Q = 0.5;
kf_R = 60;                  % larger R => smoother trajectory

numTrials  = size(trainingData,1);
numNeurons = size(trainingData(1,1).spikes,1);

% ---------- 0) Raw spikes -> cleaner signal (smoothed rate per bin) ----------
smoothKernel = ones(1, smoothWinLen) / smoothWinLen;

% ---------- 1) Train direction templates (nearest-centroid classifier) ----------
dirFeatAll = cell(1,8);
for d = 1:8
    F = [];
    for i = 1:numTrials
        raw = trainingData(i,d).spikes;          % 98 x T
        cleaned = conv2(1, smoothKernel, raw, 'same');  % 98 x T
        Tend = min(size(cleaned,2), dirWindowEnd);
        feat = sum(cleaned(:,1:Tend), 2)';       % 1 x 98 (early cleaned sum)
        F = [F; feat];
    end
    dirFeatAll{d} = F;
end

% ---------- 1b) LDA projection for direction (better than nearest-centroid in raw space) ----------
AllFeat = []; labels = [];
for d = 1:8
    F = dirFeatAll{d};
    AllFeat = [AllFeat; F];
    labels = [labels; d * ones(size(F,1), 1)];
end
mu_all = mean(AllFeat, 1);
Sw = zeros(numNeurons, numNeurons);
Sb = zeros(numNeurons, numNeurons);
for d = 1:8
    Xd = dirFeatAll{d};
    mu_d = mean(Xd, 1);
    n_d = size(Xd, 1);
    Sw = Sw + (Xd - mu_d)' * (Xd - mu_d);
    Sb = Sb + n_d * (mu_d - mu_all)' * (mu_d - mu_all);
end
Sw = Sw + 1e-3 * eye(numNeurons);   % slightly stronger for stable LDA
[V, D] = eig(Sb, Sw);
[D, ord] = sort(diag(D), 'descend');
V = V(:, ord);
nLDA = min(7, size(V,2));
W_lda = V(:, 1:nLDA);   % 98 x nLDA
dirTemplatesLDA = zeros(8, nLDA);
for d = 1:8
    dirTemplatesLDA(d,:) = mean(dirFeatAll{d}, 1) * W_lda;
end

% ---------- 2) Direction-specific average trajectory: MEDIAN (robust to outlier trials) ----------
avgTraj = cell(1,8);
for d = 1:8
    Tmax = 0;
    for i = 1:numTrials
        Tmax = max(Tmax, size(trainingData(i,d).handPos, 2));
    end
    avgTraj{d} = zeros(2, Tmax);
    for t = 1:Tmax
        xlist = []; ylist = [];
        for i = 1:numTrials
            pos = trainingData(i,d).handPos(1:2,:);
            if size(pos, 2) >= t
                xlist = [xlist; pos(1,t)];
                ylist = [ylist; pos(2,t)];
            end
        end
        avgTraj{d}(1, t) = median(xlist);
        avgTraj{d}(2, t) = median(ylist);
    end
end

% ---------- 3) RESIDUAL regression: predict (pos - avgTraj), not velocity ----------
% Output = avgTraj + decodedResidual; residual is smaller, easier to predict
linearModelRx = cell(1,8);
linearModelRy = cell(1,8);
featMean = cell(1,8);
featStd  = cell(1,8);
residualClip = zeros(1, 8);   % per-direction clip threshold (robust)
for d = 1:8
    X = [];
    Yres = [];
    for i = 1:numTrials
        raw  = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        pos  = trainingData(i,d).handPos(1:2,:);
        T    = size(cleaned,2);
        avg  = avgTraj{d};
        Tavg = size(avg, 2);

        startPos = pos(1:2, 1)';
        for t = (initTime+step):step:T
            startLong  = t - bufferMsLong  + 1;
            startShort = t - bufferMsShort + 1;
            if startLong  < 1, startLong  = 1; end
            if startShort < 1, startShort = 1; end
            lenLong  = t - startLong  + 1;
            lenShort = t - startShort + 1;
            featLong  = sum(cleaned(:, startLong:t), 2)'  / lenLong;
            featShort = sum(cleaned(:, startShort:t), 2)' / lenShort;
            feat = [featShort, featLong, startPos];
            tIdx = min(t, Tavg);
            res = (pos(:,t) - avg(:, tIdx))';   % residual from avg trajectory
            X = [X; feat];
            Yres = [Yres; res];
        end
    end

    mu = mean(X, 1);
    sig = std(X, 0, 1);
    sig(sig < 1e-6) = 1;
    Xn = (X - mu) ./ sig;
    Xb = [Xn ones(size(Xn,1),1)];
    featMean{d} = mu;
    featStd{d}  = sig;

    I = eye(size(Xb,2)); I(end,end) = 0;
    wRx = (Xb' * Xb + lambda * I) \ (Xb' * Yres(:,1));
    wRy = (Xb' * Xb + lambda * I) \ (Xb' * Yres(:,2));
    linearModelRx{d} = wRx;
    linearModelRy{d} = wRy;
    % Clip threshold from training residuals (limit extreme corrections at test)
    residualClip(d) = max(prctile(abs(Yres(:)), residualClipPct), 1);
end

% Pack parameters
modelParameters.step = step;
modelParameters.bufferMsLong  = bufferMsLong;
modelParameters.bufferMsShort = bufferMsShort;
modelParameters.dirWindowEnd = dirWindowEnd;
modelParameters.smoothWinLen = smoothWinLen;
modelParameters.smoothKernel = smoothKernel;
modelParameters.W_lda = W_lda;
modelParameters.dirTemplatesLDA = dirTemplatesLDA;
modelParameters.linearModelRx = linearModelRx;
modelParameters.linearModelRy = linearModelRy;
modelParameters.featMean = featMean;
modelParameters.featStd  = featStd;
modelParameters.avgTraj = avgTraj;
modelParameters.residualScale = residualScale;
modelParameters.residualClip = residualClip;
modelParameters.kf_dt = kf_dt;
modelParameters.kf_Q = kf_Q;
modelParameters.kf_R = kf_R;

modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
modelParameters.kfState = containers.Map('KeyType','double','ValueType','any');  % Kalman state per trial
end
