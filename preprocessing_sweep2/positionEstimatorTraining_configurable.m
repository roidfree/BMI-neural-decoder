function modelParameters = positionEstimatorTraining_configurable(training_data, config)

trials = size(training_data,1);
movements = size(training_data,2);
neurons = 98;

processed_data = training_data;

% =========================
% INITIALISE BIN WIDTH
% =========================
for t = 1:trials
    for m = 1:movements
        processed_data(t,m).bin_width = 1;
    end
end

% =========================
% APPLY PREPROCESSING FIRST
% =========================

% Rebin
if config.bin_width ~= 0
    for t = 1:trials
        for m = 1:movements
            data = processed_data(t,m).spikes;
            L = size(data,2);

            newL = floor(L / config.bin_width);
            binned = zeros(neurons, newL);

            for i = 1:newL
                idx_start = (i-1)*config.bin_width + 1;
                idx_end   = i*config.bin_width;
                binned(:,i) = sum(data(:, idx_start:idx_end),2);
            end

            processed_data(t,m).spikes = binned;
            processed_data(t,m).bin_width = config.bin_width;
        end
    end
end

% Transform
if config.transform == "sqrt"
    for t = 1:trials
        for m = 1:movements
            processed_data(t,m).spikes = sqrt(processed_data(t,m).spikes);
        end
    end
elseif config.transform == "anscombe"
    for t = 1:trials
        for m = 1:movements
            processed_data(t,m).spikes = 2*sqrt(processed_data(t,m).spikes + 3/8);
        end
    end
end

% =========================
% FEATURE EXTRACTION (NOW CORRECT)
% =========================

A = [];
B = [];

for m = 1:movements
    for t = 1:trials

        spikes = processed_data(t,m).spikes;

        T = floor(320 / processed_data(t,m).bin_width);
        T = min(T, size(spikes,2));

        feat = sum(spikes(:,1:T),2)';

        A = [A; feat];
        B = [B; m];

    end
end

% =========================
% NEAREST MEAN CLASSIFIER
% =========================

class_means = zeros(movements, size(A,2));

for k = 1:movements
    class_means(k,:) = mean(A(B==k,:),1);
end

modelParameters.means = class_means;

end