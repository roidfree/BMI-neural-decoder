%% —————— positionEstimator_noPCA_LDA.m ——————
function pred_dir = positionEstimator(testSample, modelParameters)
% Predict direction with Gaussian Naive Bayes only (no PCA, no LDA).
    T_class = modelParameters.T_class;
    % 1) Feature extraction
    feat0 = sum(testSample.spikes(:,1:T_class),2)';
    % 2) Z-score normalization
    feat_norm = (feat0 - modelParameters.mu_class) ./ modelParameters.sigma_class;

    % 3) Gaussian NB prediction
    clf = modelParameters.classifier;
    Cn = numel(clf.classes);
    logScores = zeros(Cn,1);
    for c = 1:Cn
        mu_c  = clf.classMeans(c,:);
        var_c = clf.classVars(c,:);
        % Per-dimension Gaussian log-likelihood (naive independence)
        logGauss = -0.5 * sum(log(2*pi*var_c)) ...
                   -0.5 * sum((feat_norm - mu_c).^2 ./ var_c);
        logScores(c) = logGauss + log(clf.priors(c));
    end
    [~, idxMax] = max(logScores);
    pred_dir = clf.classes(idxMax);
end
