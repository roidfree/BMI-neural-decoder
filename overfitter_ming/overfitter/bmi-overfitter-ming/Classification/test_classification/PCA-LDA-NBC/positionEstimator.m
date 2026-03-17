%% —————— positionEstimator_noToolbox_GNB.m ——————
function pred_dir = positionEstimator(testSample, modelParameters)
% 用手动实现的 PCA→LDA→Gaussian NB 模型做预测
    T_class = modelParameters.T_class;
    % 1) 特征提取
    feat0 = sum(testSample.spikes(:,1:T_class),2)';  
    % 2) Z-score 归一化
    feat_norm = (feat0 - modelParameters.mu_class) ./ modelParameters.sigma_class;
    % 3) PCA 投影
    feat_centered = feat_norm - modelParameters.pca.mu;
    feat_pca      = feat_centered * modelParameters.pca.coeff;
    % 4) LDA 投影
    feat_lda = feat_pca * modelParameters.lda.W;
    % 5) 保留 NB 特征
    feat_nb  = feat_lda(modelParameters.predictorIdx);
    % 6) Gaussian NB 预测
    clf = modelParameters.classifier;
    Cn = numel(clf.classes);
    logScores = zeros(Cn,1);
    for c = 1:Cn
        mu_c   = clf.classMeans(c,:);
        var_c  = clf.classVars(c,:);
        % Gaussian log-likelihood
        logGauss = -0.5 * sum(log(2*pi*var_c)) ...
                   -0.5 * sum((feat_nb - mu_c).^2 ./ var_c);
        logScores(c) = logGauss + log(clf.priors(c));
    end
    [~, idxMax] = max(logScores);
    pred_dir = clf.classes(idxMax);
end
