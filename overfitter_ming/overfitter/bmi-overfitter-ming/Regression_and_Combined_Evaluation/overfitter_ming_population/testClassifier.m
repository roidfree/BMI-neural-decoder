% Test the separated population-vector direction classifier (trainDirectionClassifierPV)
load monkeydata0.mat

trainingData = trial(1:80,:);
testData = trial(81:100,:);

pvModel = trainDirectionClassifierPV(trainingData);

confMat = zeros(8,8);
correct = 0;
total = 0;

for trueDir = 1:8
    for t = 1:20
        spikes = testData(t,trueDir).spikes;
        predDir = predictDirectionPV(spikes, pvModel);
        confMat(trueDir,predDir) = confMat(trueDir,predDir) + 1;
        if predDir == trueDir
            correct = correct + 1;
        end
        total = total + 1;
    end
end

accuracy = correct / total;
fprintf('Classification accuracy (Population Vector): %.2f%%\n', accuracy*100)
disp('Confusion Matrix:')
disp(confMat)

figure
imagesc(confMat)
xlabel('Predicted Direction')
ylabel('True Direction')
title('Population Vector Direction Confusion Matrix (overfitter_ming_population)')
colorbar
