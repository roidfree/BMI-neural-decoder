function results = benchmark_test_function(modelName)

    % Load data
    load monkeydata0.mat

    % Split data
    rng(2013);
    ix = randperm(size(trial, 1));
    trainingData = trial(ix(1:50), :);
    testData = trial(ix(51:end), :);

    fprintf('Training...\n');

    % === TRAIN ===
    tic;
    modelParameters = positionEstimatorTraining(trainingData);
    train_time = toc;

    fprintf('Testing...\n');

    % === TEST ===
    totalPredictions = 0;
    correctPredictions = 0;
    confMat = zeros(8,8);

    tic;
    for tr = 1:size(testData, 1)
        for direc = 1:8
            
            testSample.spikes = testData(tr, direc).spikes;
            trueLabel = direc;

            pred_dir = positionEstimator(testSample, modelParameters);

            % Confusion matrix
            confMat(trueLabel, pred_dir) = confMat(trueLabel, pred_dir) + 1;

            if pred_dir == trueLabel
                correctPredictions = correctPredictions + 1;
            end

            totalPredictions = totalPredictions + 1;
        end
    end
    test_time = toc;

    % === METRICS ===
    accuracy = (correctPredictions / totalPredictions) * 100;
    perClassAcc = diag(confMat) ./ sum(confMat,2);

    % === STORE RESULTS ===
    results.accuracy = accuracy;
    results.train_time = train_time;
    results.test_time = test_time;
    results.confMat = confMat;
    results.perClassAcc = perClassAcc;

    % === PLOTS ===
    figure;
    imagesc(confMat);
    colorbar;
    title(['Confusion Matrix - ' modelName]);
    xlabel('Predicted'); ylabel('True');

    figure;
    bar(perClassAcc);
    title(['Per-Class Accuracy - ' modelName]);
    xlabel('Direction'); ylabel('Accuracy');

    fprintf('Accuracy: %.2f%%\n', accuracy);
    fprintf('Train Time: %.2fs\n', train_time);
    fprintf('Test Time: %.2fs\n', test_time);

end