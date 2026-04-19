% testFunction_for_students_classification_PCA_LDA_NC.m

function [bestPCA, bestLDA] = testFunction_for_students_classification_PCA_LDA_NC(teamName)
% Sweep PCA and LDA dimensions; report best classification accuracy (nearest centroid).
%
% Usage:
%   [bestPCA,bestLDA] = testFunction_for_students_classification_PCA_LDA_NC('teamName');

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(size(trial,1));
    trainData = trial(ix(1:50),:);
    testData  = trial(ix(51:end),:);

    pcaList  = [7];
    ldaList  = [6];

    bestAcc = -inf;
    bestPCA = NaN;
    bestLDA = NaN;

    fprintf('Tuning PCA→LDA→Nearest Centroid...\n');
    for p = pcaList
      for d = ldaList
        if d >= p, continue; end

        mdl = positionEstimatorTraining_PCA_LDA_NC(trainData, p, d);

        correct = 0;
        total   = 0;
        for tr=1:size(testData,1)
          for dir=1:8
            sample.spikes = testData(tr,dir).spikes;
            pred = positionEstimator_PCA_LDA_NC(sample, mdl);
            correct = correct + (pred==dir);
            total   = total + 1;
          end
        end
        acc = 100*correct/total;
        fprintf('  PCA=%d, LDA=%d → %.2f%%\n', p,d,acc);

        if acc>bestAcc
          bestAcc = acc;
          bestPCA = p;
          bestLDA = d;
        end
      end
    end

    fprintf('→ Best: PCA=%d, LDA=%d with %.2f%%\n', bestPCA,bestLDA,bestAcc);
end
