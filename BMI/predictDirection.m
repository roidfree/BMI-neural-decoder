function direction = predictDirection(spikes, ldaModel)

feat = sum(spikes(:,1:320),2)';

means = ldaModel.means;
Sigma_inv = ldaModel.Sigma_inv;

num_classes = size(means,1);

scores = zeros(num_classes,1);

for k = 1:num_classes
    
    mu = means(k,:);
    
    scores(k) = feat * Sigma_inv * mu' - 0.5 * mu * Sigma_inv * mu';
    
end

[~, direction] = max(scores);

end