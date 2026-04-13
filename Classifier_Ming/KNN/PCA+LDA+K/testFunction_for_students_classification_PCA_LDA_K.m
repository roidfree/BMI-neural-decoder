% testFunction_for_students_classification_PCA_LDA_K.m

function [bestK, bestPCA, bestLDA] = testFunction_for_students_classification_PCA_LDA_K(teamName)
% Sweeps over k, PCA dims, LDA dims to find best classification accuracy.
%
% Usage:
%   [bestK,bestPCA,bestLDA] = testFunction_for_students_classification_PCA_LDA_K('teamName');

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(size(trial,1));
    trainData = trial(ix(1:50),:);
    testData  = trial(ix(51:end),:);

    kList    = [1, 27, 75];
    pcaList  = [7];
    ldaList  = [6];   % <= min(pcaDim,7)

    bestAcc = -inf;
    bestK = NaN;
    bestPCA = NaN;
    bestLDA = NaN;

    fprintf('Tuning PCA→LDA→k-NN...\n');
    for k = kList
      for p = pcaList
        for d = ldaList
          if d >= p, continue; end

          mdl = positionEstimatorTraining_PCA_LDA_K(trainData, k, p, d);

          correct=0; total=0;
          for tr=1:size(testData,1)
            for dir=1:8
              sample.spikes = testData(tr,dir).spikes;
              pred = positionEstimator_PCA_LDA_K(sample, mdl);
              correct = correct + (pred==dir);
              total   = total + 1;
            end
          end
          acc = 100*correct/total;
          fprintf('  k=%d, PCA=%d, LDA=%d → %.2f%%\n', k,p,d,acc);

          if acc>bestAcc
            bestAcc = acc;
            bestK = k;
            bestPCA = p;
            bestLDA = d;
          end
        end
      end
    end

    fprintf('→ Best: k=%d, PCA=%d, LDA=%d with %.2f%%\n', bestK,bestPCA,bestLDA,bestAcc);
end
