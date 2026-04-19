clc; clear;
load monkeydata0.mat

rng(2013);

% FIXED SPLIT (IMPORTANT)
ix = randperm(size(trial,1));
trainData = trial(ix(1:50),:);
testData  = trial(ix(51:end),:);

bin_widths = [10,20,40];
transforms = ["none","sqrt","anscombe"];

results_all = [];
exp_id = 0;

for bw = bin_widths
for trf = transforms

    exp_id = exp_id + 1;

    config.bin_width = bw;
    config.transform = trf;

    fprintf('\n[%d] BW=%d TR=%s\n', exp_id, bw, trf);

    % Train
    model = positionEstimatorTraining_configurable(trainData, config);

    % Test
    acc = evaluate_model(testData, model, config);

    results_all = [results_all; {bw, string(trf), acc}];

end
end

results = cell2table(results_all,...
    'VariableNames',{'BinWidth','Transform','Accuracy'});

disp(results)

[~,idx] = max(results.Accuracy);
disp('=== BEST PREPROCESSING ===');
disp(results(idx,:));