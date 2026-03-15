function pvModel = trainDirectionClassifierPV(trainingData)
% Population-vector direction classifier: firing rate + baseline + tuning strength.
% Same logic as positionEstimatorTraining (improved PV).
%
% Output: pvModel.prefVec, pvModel.tuningStrength, pvModel.baselineRate,
%         pvModel.smoothKernel, pvModel.dirWindowEnd, pvModel.smoothWinLen

[trials, numDirs] = size(trainingData);
numNeurons = size(trainingData(1,1).spikes, 1);

dirWindowEnd = 320;
smoothWinLen = 25;

smoothKernel = ones(1, smoothWinLen) / smoothWinLen;

%% ---------- 1) Per-neuron early firing rate per direction ----------
dirMeanRates = zeros(numNeurons, numDirs);
for d = 1:numDirs
    F = [];
    for i = 1:trials
        raw = trainingData(i,d).spikes;
        cleaned = conv2(1, smoothKernel, raw, 'same');
        Tend = min(size(cleaned, 2), dirWindowEnd);
        feat = sum(cleaned(:, 1:Tend), 2)' / Tend;   % mean firing rate
        F = [F; feat];
    end
    dirMeanRates(:, d) = mean(F, 1)';
end

%% ---------- 2) Baseline rate per neuron ----------
baselineRate = mean(dirMeanRates, 2);   % 98 x 1

%% ---------- 3) Preferred direction per neuron + tuning strength ----------
angles = (0:(numDirs-1))' * (pi/4);
unitVecs = [cos(angles), sin(angles)];   % 8 x 2

prefVec = zeros(numNeurons, 2);
tuningStrength = zeros(numNeurons, 1);

for n = 1:numNeurons
    w = dirMeanRates(n, :) - baselineRate(n);   % 1 x 8, direction modulation
    p = w * unitVecs;
    mag = norm(p);
    tuningStrength(n) = mag;
    if mag > 1e-8
        prefVec(n, :) = p / mag;
    else
        prefVec(n, :) = [0 0];
    end
end

if max(tuningStrength) > 0
    tuningStrength = tuningStrength / max(tuningStrength);
end

pvModel.prefVec = prefVec;
pvModel.tuningStrength = tuningStrength;
pvModel.baselineRate = baselineRate;
pvModel.smoothKernel = smoothKernel;
pvModel.dirWindowEnd = dirWindowEnd;
pvModel.smoothWinLen = smoothWinLen;
end
