function modelParameters = positionEstimatorTraining(trainingData)
    % Direction classification + per-direction regression (buffered)
    % Uses smoothed (cleaned) firing rate instead of raw spike counts.
    
    step = 20;              % must match test script (sample every 20 bins)
    % Buffer in BIN count: if 1 bin = 20ms, then 20 bins = 400ms, 5 bins = 100ms
    bufferBinsLong  = 20;
    bufferBinsShort = 5;
    initTime = 300;
    dirWindowEnd = 320;
    smoothWinLen = 5;
    lambda = 15;
    avgTrajWeight = 0.4;    % blend: (1-w)*regression + w*averageTrajectory (BMI trick)
    
    numTrials  = size(trainingData,1);
    numNeurons = size(trainingData(1,1).spikes,1);
    
    % ---------- 0) Raw spikes -> cleaner signal (smoothed rate per bin) ----------
    % smoothed = conv with boxcar; same length as input
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
    Sw = Sw + 1e-4 * eye(numNeurons);   % regularize for invertibility
    [V, D] = eig(Sb, Sw);
    [D, ord] = sort(diag(D), 'descend');
    V = V(:, ord);
    nLDA = min(7, size(V,2));
    W_lda = V(:, 1:nLDA);   % 98 x nLDA
    % Projected class means (8 x nLDA)
    dirTemplatesLDA = zeros(8, nLDA);
    for d = 1:8
        dirTemplatesLDA(d,:) = mean(dirFeatAll{d}, 1) * W_lda;
    end
    
    % ---------- 2) Per-direction VELOCITY regression with multi-scale buffer ----------
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
                startLong  = t - bufferBinsLong  + 1;
                startShort = t - bufferBinsShort + 1;
                if startLong  < 1, startLong  = 1; end
                if startShort < 1, startShort = 1; end
                featLong  = sum(cleaned(:, startLong:t), 2)';
                featShort = sum(cleaned(:, startShort:t), 2)';
                feat = [featShort, featLong];            % 1 x 196 (multi-scale)
                vel = (pos(:,t) - pos(:,t-step))';
                X = [X; feat];
                Yvel = [Yvel; vel];
            end
        end
    
        % Z-score features (stabilize ridge)
        mu = mean(X, 1);
        sig = std(X, 0, 1);
        sig(sig < 1e-6) = 1;
        Xn = (X - mu) ./ sig;
        Xb = [Xn ones(size(Xn,1),1)];
        featMean{d} = mu;
        featStd{d}  = sig;
    
        I = eye(size(Xb,2)); I(end,end) = 0;
        wVx = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,1));
        wVy = (Xb' * Xb + lambda * I) \ (Xb' * Yvel(:,2));
        linearModelVx{d} = wVx;
        linearModelVy{d} = wVy;
    end
    
    % ---------- 3) Direction-specific average trajectory (BMI competition trick) ----------
    % Per direction: average handPos(1:2,:) across trials, aligned by bin index.
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
            sumPos(:, 1:T_i) = sumPos(:, 1:T_i) + pos;
            count(1:T_i) = count(1:T_i) + 1;
        end
        count(count == 0) = 1;
        avgTraj{d} = sumPos ./ repmat(count, 2, 1);   % 2 x Tmax
    end
    
    % Pack parameters
    modelParameters.step = step;
    modelParameters.bufferSizeLong  = bufferBinsLong;
    modelParameters.bufferSizeShort = bufferBinsShort;
    modelParameters.dirWindowEnd = dirWindowEnd;
    modelParameters.smoothWinLen = smoothWinLen;
    modelParameters.smoothKernel = smoothKernel;
    modelParameters.W_lda = W_lda;
    modelParameters.dirTemplatesLDA = dirTemplatesLDA;
    modelParameters.linearModelVx = linearModelVx;
    modelParameters.linearModelVy = linearModelVy;
    modelParameters.featMean = featMean;
    modelParameters.featStd  = featStd;
    modelParameters.avgTraj = avgTraj;
    modelParameters.avgTrajWeight = avgTrajWeight;
    
    modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
    end