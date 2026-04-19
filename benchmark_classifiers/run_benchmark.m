clc; clear;

% === MODEL PATHS (FIXED) ===
models = {
    'NBC', 'NBC';
    'SVM', 'SVM/SVM';
    'SVM+LDA', 'SVM/SVM+LDA';
    'SVM+LDA+PCA', 'SVM/SVM+LDA+PCA';
};

numModels = size(models,1);

accuracies = zeros(numModels,1);
train_times = zeros(numModels,1);
test_times = zeros(numModels,1);

for i = 1:numModels
    
    name = models{i,1};
    path = models{i,2};

    fprintf('\n=============================\n');
    fprintf('Running %s\n', name);
    fprintf('=============================\n');

    % === ADD PATH (RECURSIVE FIX) ===
    addpath(genpath(path));

    % === RUN BENCHMARK ===
    results = benchmark_test_function(name);

    accuracies(i) = results.accuracy;
    train_times(i) = results.train_time;
    test_times(i) = results.test_time;

    % === REMOVE PATH ===
    rmpath(genpath(path));
end

% === ACCURACY COMPARISON ===
figure;
bar(accuracies);
set(gca,'XTickLabel',models(:,1));
ylabel('Accuracy (%)');
title('Model Accuracy Comparison');

% === TRAIN TIME ===
figure;
bar(train_times);
set(gca,'XTickLabel',models(:,1));
ylabel('Seconds');
title('Training Time Comparison');

% === TEST TIME ===
figure;
bar(test_times);
set(gca,'XTickLabel',models(:,1));
ylabel('Seconds');
title('Testing Time Comparison');