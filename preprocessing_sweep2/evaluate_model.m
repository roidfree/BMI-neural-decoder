function accuracy = evaluate_model(testData, modelParameters, config)

correct = 0;
total = 0;

for tr = 1:size(testData,1)
for d = 1:8

    spikes = testData(tr,d).spikes;

    % === SAME PREPROCESSING ===

    % Rebin
    if config.bin_width ~= 0
        data = spikes;
        L = size(data,2);
        newL = floor(L / config.bin_width);

        binned = zeros(98, newL);

        for i = 1:newL
            idx_start = (i-1)*config.bin_width + 1;
            idx_end   = i*config.bin_width;
            binned(:,i) = sum(data(:, idx_start:idx_end),2);
        end

        spikes = binned;
    end

    % Transform
    if config.transform == "sqrt"
        spikes = sqrt(spikes);
    elseif config.transform == "anscombe"
        spikes = 2*sqrt(spikes + 3/8);
    end

    % Feature
    T = floor(320 / config.bin_width);
    T = min(T, size(spikes,2));

    feat = sum(spikes(:,1:T),2)';

    % Predict
    dists = sum((modelParameters.means - feat).^2,2);
    [~, pred] = min(dists);

    if pred == d
        correct = correct + 1;
    end

    total = total + 1;

end
end

accuracy = 100 * correct / total;

end