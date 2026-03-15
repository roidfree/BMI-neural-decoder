function direction = predictDirectionPV(spikes, pvModel)
% Predict direction: rate + baseline + tuning-strength^power weighting (weakly tuned neurons down-weighted).

cleaned = conv2(1, pvModel.smoothKernel, spikes, 'same');
Tend = min(size(cleaned, 2), pvModel.dirWindowEnd);

featEarly = sum(cleaned(:, 1:Tend), 2)' / Tend;
featCentered = featEarly - pvModel.baselineRate';
tp = 1;
if isfield(pvModel, 'tuningPower'), tp = pvModel.tuningPower; end
weightedFeat = featCentered .* (pvModel.tuningStrength .^ tp)';

popVec = weightedFeat * pvModel.prefVec;   % 1 x 2
angle = atan2(popVec(2), popVec(1));

d = mod(round(angle / (pi/4)), 8) + 1;
if d < 1, d = 1; end
if d > 8, d = 8; end
direction = d;
end
