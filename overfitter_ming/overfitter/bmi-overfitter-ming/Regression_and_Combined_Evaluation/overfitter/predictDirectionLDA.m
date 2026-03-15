function direction = predictDirectionLDA(spikes, ldaModel)
% Predict direction with LDA+NC model from trainDirectionClassifierLDA.
% Feature: smoothed sum over [1, dirWindowEnd]; project with W_lda; nearest centroid.

cleaned = conv2(1, ldaModel.smoothKernel, spikes, 'same');
Tend = min(size(cleaned, 2), ldaModel.dirWindowEnd);
feat = sum(cleaned(:, 1:Tend), 2)';

feat_proj = feat * ldaModel.W_lda;
dists = sum((ldaModel.dirTemplatesLDA - feat_proj).^2, 2);
[~, direction] = min(dists);
end
