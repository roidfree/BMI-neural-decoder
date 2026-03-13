function ldaModel = trainDirectionClassifier(training_data)

[trials, movements] = size(training_data);

X = [];
Y = [];

for m = 1:movements
    for t = 1:trials
        
        spikes = training_data(t,m).spikes;
        
        feat = sum(spikes(:,1:320),2)';
        
        X = [X; feat];
        Y = [Y; m];
        
    end
end

[n_samples, n_features] = size(X);

class_means = zeros(movements, n_features);

for k = 1:movements
    class_means(k,:) = mean(X(Y==k,:),1);
end

lambda = 0.01;
Sigma = cov(X) + lambda * eye(n_features);

Sigma_inv = inv(Sigma);

ldaModel.means = class_means;
ldaModel.Sigma_inv = Sigma_inv;

end