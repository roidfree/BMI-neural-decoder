function accuracy = evaluate_model(testData, modelParameters, config)

correct = 0;
total = 0;

for tr = 1:size(testData,1)
for d = 1:8

    % Get spikes
    spikes = testData(tr,d).spikes;

    % === APPLY SAME PREPROCESSING ===
    trials = 1;
    movements = 1;
    neurons = 98;

    temp(1,1).spikes = spikes;
    temp(1,1).bin_width = 1;

    temp = rebin_data(temp, trials, movements, neurons, config.bin_width);
    temp = transform_data(temp, trials, movements, neurons, config.transform);

    if config.kernel ~= "none"
        temp = convolve_data(temp, trials, movements, neurons, ...
            config.kernel, config.kernel_param, config.kernel_width);
    end

    % Feature extraction (same as training)
    feat = sum(temp(1,1).spikes(:,1:320),2)';

    % === CLASSIFICATION (nearest mean) ===
    dists = sum((modelParameters.means - feat).^2,2);
    [~, pred] = min(dists);

    if pred == d
        correct = correct + 1;
    end

    total = total + 1;

end
end

accuracy = (correct / total) * 100;
end