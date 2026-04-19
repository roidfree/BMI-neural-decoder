clc; clear;
load monkeydata0.mat

rng(2013);

bin_widths = [10, 20, 40];
transforms = ["none", "sqrt", "anscombe"];
kernels = ["none", "MA"];
kernel_widths = [0, 5];

results_all = [];
exp_id = 0;

for bw = bin_widths
for trf = transforms
for k = kernels

    if k == "none"
        kw_list = 0;
    else
        kw_list = kernel_widths;
    end

for kw = kw_list

    exp_id = exp_id + 1;

    % === CONFIG ===
    config.bin_width = bw;
    config.transform = trf;
    config.kernel = k;
    config.kernel_width = kw;
    config.kernel_param = 0.3;

    fprintf('\n[%d] BW=%d | TR=%s | K=%s | KW=%d\n',...
        exp_id, bw, trf, k, kw);

    % Split
    ix = randperm(size(trial,1));
    trainData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    % Train
    tic;
    model = positionEstimatorTraining_configurable(trainData, config);
    train_time = toc;

    % === SIMPLE CLASSIFIER (his means) ===
    correct = 0;
    total = 0;

    for tr = 1:size(testData,1)
    for d = 1:8

        spikes = testData(tr,d).spikes;
        feat = sum(spikes(:,1:320),2)';

        % nearest mean classifier
        dists = sum((model.means - feat).^2,2);
        [~, pred] = min(dists);

        if pred == d
            correct = correct + 1;
        end

        total = total + 1;

    end
    end

    acc = correct / total * 100;

    results_all = [results_all; {bw, string(trf), string(k), kw, acc, train_time}];

end
end
end
end

results = cell2table(results_all, ...
    'VariableNames', {'BinWidth','Transform','Kernel','KernelWidth','Accuracy','TrainTime'});

[~, idx] = max(results.Accuracy);
disp('=== BEST PREPROCESSING ===');
disp(results(idx,:));

save('jared_preprocessing_sweep.mat','results');