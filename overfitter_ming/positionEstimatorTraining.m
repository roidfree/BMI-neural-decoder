function modelParameters = positionEstimatorTraining(trainingData)
% Direction classification + per-direction regression (buffered)
% No toolbox needed

step = 20;              % must match test script
bufferBins = 10;        % 10 bins ~ 200ms (if each bin is 20ms)
initTime = 300;         % start regression samples after 300ms
dirWindowEnd = 320;     % use early window [1:320] for direction classification

numTrials  = size(trainingData,1);
numNeurons = size(trainingData(1,1).spikes,1);

% ---------- 1) Train direction templates (nearest-centroid classifier) ----------
dirFeatAll = cell(1,8);
for d = 1:8
    F = [];
    for i = 1:numTrials
        spikes = trainingData(i,d).spikes;       % 98 x T
        Tend = min(size(spikes,2), dirWindowEnd);
        feat = sum(spikes(:,1:Tend), 2)';        % 1 x 98 (early spike counts)
        F = [F; feat];
    end
    dirFeatAll{d} = F;
end

dirTemplates = zeros(8, numNeurons);
for d = 1:8
    dirTemplates(d,:) = mean(dirFeatAll{d}, 1);  % 8 x 98
end

% ---------- 2) Train per-direction regression models ----------
linearModelX = cell(1,8);
linearModelY = cell(1,8);

lambda = 100;  % ridge strength; try 1,10,100,1000 for tuning
for d = 1:8
    X = [];
    Y = [];
    for i = 1:numTrials
        spikes = trainingData(i,d).spikes;          % 98 x T
        pos    = trainingData(i,d).handPos(1:2,:);  % 2 x T
        T      = size(spikes,2);

        for t = (initTime+step):step:T
            startBin = t - bufferBins + 1;
            if startBin < 1
                startBin = 1;
            end

            feat = sum(spikes(:, startBin:t), 2)';  % 1 x 98
            X = [X; feat];
            Y = [Y; pos(:,t)'];
        end
    end

    Xb = [X ones(size(X,1),1)];     % bias
    % Ridge regression (do not regularize bias)
    I = eye(size(Xb,2)); I(end,end) = 0;

    wX = (Xb' * Xb + lambda * I) \ (Xb' * Y(:,1));
    wY = (Xb' * Xb + lambda * I) \ (Xb' * Y(:,2));

    linearModelX{d} = wX;
    linearModelY{d} = wY;
end

% Pack parameters
modelParameters.step = step;
modelParameters.bufferSize = bufferBins;
modelParameters.dirWindowEnd = dirWindowEnd;
modelParameters.dirTemplates = dirTemplates;
modelParameters.linearModelX = linearModelX;
modelParameters.linearModelY = linearModelY;

% We'll store trialId->direction here during decoding
modelParameters.trialDirMap = containers.Map('KeyType','double','ValueType','double');
end