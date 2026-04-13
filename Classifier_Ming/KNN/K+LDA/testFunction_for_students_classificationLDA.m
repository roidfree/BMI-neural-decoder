% testFunction_for_students_classificationLDA.m

function [bestK, bestDim] = testFunction_for_students_classificationLDA(teamName)
% Sweeps over k and LDA dimensions; reports best accuracy.
%
% Usage:
%   [bestK,bestDim] = testFunction_for_students_classificationLDA('teamName');

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(size(trial,1));
    trainData = trial(ix(1:50),:);
    testData  = trial(ix(51:end),:);

    kList    = [2];
    dimList  = 6;  % max LDA dims = #classes−1 = 7

    bestAcc = -inf;
    bestK   = NaN;
    bestDim = NaN;

    fprintf('Tuning k and LDA dimension...\n');
    for k = kList
      for d = dimList
        % train
        modelParams = positionEstimatorTraining_LDA(trainData, k, d);

        % test
        correct = 0;
        total   = 0;
        for tr=1:size(testData,1)
          for dir=1:8
            sample.spikes = testData(tr,dir).spikes;
            pred = positionEstimator_LDA(sample, modelParams);
            correct = correct + (pred==dir);
            total   = total + 1;
          end
        end
        acc = 100 * correct/total;
        fprintf('  k=%d, LDA dim=%d → acc=%.2f%%\n', k, d, acc);

        if acc>bestAcc
          bestAcc = acc;
          bestK   = k;
          bestDim = d;
        end
      end
    end

    fprintf('→ Best: k=%d, LDA dim=%d with %.2f%% accuracy\n', bestK, bestDim, bestAcc);
end
