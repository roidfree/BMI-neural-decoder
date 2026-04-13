function bestK = testFunction_for_students_classification(teamName)
% testFunction_for_students_classification  —  Evaluate k‐NN over several k’s
%
%   bestK = testFunction_for_students_classification(teamName)
%     loads the data, splits randomly, then for each k in kList:
%       • trains: modelParameters = positionEstimatorTraining(trainingData, k)
%       • tests:  classificationAccuracy(k_idx)
%     and finally prints and returns the k with highest accuracy.

    % Load & split
    load monkeydata_training.mat
    rng(2013);
    ix = randperm(size(trial,1));
    trainingData = trial(ix(1:50),:);
    testData     = trial(ix(51:end),:);

    % List of k values to try
    kList = [22];

    bestAcc = -inf;
    bestK   = NaN;

    fprintf('Tuning k for k-NN classifier...\n');
    for ki = 1:numel(kList)
        k = kList(ki);

        % Train with this k
        modelParameters = positionEstimatorTraining(trainingData, k);

        % Test
        totalPred = 0;
        correct   = 0;
        for tr = 1:size(testData,1)
            for direc = 1:8
                sample.spikes = testData(tr,direc).spikes;
                pred_dir = positionEstimator(sample, modelParameters);
                if pred_dir == direc
                    correct = correct + 1;
                end
                totalPred = totalPred + 1;
            end
        end
        acc = 100 * correct / totalPred;
        fprintf('  k = %2d  →  Accuracy = %.2f%%  (%d/%d)\n', k, acc, correct, totalPred);

        % Track best
        if acc > bestAcc
            bestAcc = acc;
            bestK   = k;
        end
    end

    fprintf('\n*** Best k = %d with %.2f%% accuracy ***\n', bestK, bestAcc);
end
