function direction = lda_predict(spikes, ldaModel)
% LDA_PREDICT  Predict reach direction from spike data using a trained LDA+NC model.
%
%   direction = lda_predict(spikes, ldaModel)
%
%   Inputs
%     spikes   - numNeurons x T spike matrix (from a single trial/direction)
%     ldaModel - struct produced by trainDirectionClassifier, with fields:
%                  W_lda          - numNeurons x nLDA projection matrix
%                  dirTemplatesLDA - numDirs x nLDA class centroids in LDA space
%                  smoothKernel   - 1 x smoothWinLen averaging kernel
%                  dirWindowEnd   - time window length used during training
%
%   Output
%     direction - predicted direction index (integer 1–8)

smoothKernel = ldaModel.smoothKernel;
dirWindowEnd = ldaModel.dirWindowEnd;
W_lda        = ldaModel.W_lda;
templates    = ldaModel.dirTemplatesLDA;

% Apply same preprocessing as training: smooth then sum over time window
cleaned = conv2(1, smoothKernel, spikes, 'same');
Tend    = min(size(cleaned, 2), dirWindowEnd);
feat    = sum(cleaned(:, 1:Tend), 2)';   % 1 x numNeurons

% Project into LDA space and find nearest class centroid
proj      = feat * W_lda;                % 1 x nLDA
dists     = sum((templates - proj) .^ 2, 2);
[~, direction] = min(dists);

end
