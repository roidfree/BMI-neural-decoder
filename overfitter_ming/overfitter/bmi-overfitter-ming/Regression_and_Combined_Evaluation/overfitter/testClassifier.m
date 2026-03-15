% Test the separated LDA+NC direction classifier (trainDirectionClassifierLDA)
% for this folder (velocity regression / population-vector style decoder).
load monkeydata0.mat

trainingData = trial(1:80,:);
testData = trial(81:100,:);

ldaModel = trainDirectionClassifierLDA(trainingData);

confMat = zeros(8,8);
correct = 0;
total = 0;

for trueDir = 1:8
    for t = 1:20
        spikes = testData(t,trueDir).spikes;
        predDir = predictDirectionLDA(spikes, ldaModel);
        confMat(trueDir,predDir) = confMat(trueDir,predDir) + 1;
        if predDir == trueDir
            correct = correct + 1;
        end
        total = total + 1;
    end
end

accuracy = correct / total;
fprintf('Classification accuracy: %.2f%%\n', accuracy*100)
disp('Confusion Matrix:')
disp(confMat)

figure
imagesc(confMat)
xlabel('Predicted Direction')
ylabel('True Direction')
title('LDA+NC Direction Confusion Matrix (overfitter / population-vector folder)')
colorbar
