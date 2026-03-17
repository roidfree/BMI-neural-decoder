%% —————— positionEstimator_noPCA_LDA.m ——————
function pred_dir = positionEstimator(testSample, modelParameters)
% 用纯 Gaussian Naïve Bayes 模型做预测（无 PCA、无 LDA）
    T_class = modelParameters.T_class;
    % 1) 特征提取
    feat0 = sum(testSample.spikes(:,1:T_class),2)';
    % 2) Z-score 归一化
    feat_norm = (feat0 - modelParameters.mu_class) ./ modelParameters.sigma_class;

    % 3) Gaussian NB 预测
    clf = modelParameters.classifier;
    Cn = numel(clf.classes);
    logScores = zeros(Cn,1);
    for c = 1:Cn
        mu_c  = clf.classMeans(c,:);
        var_c = clf.classVars(c,:);
        % 逐维高斯对数似然
        logGauss = -0.5 * sum(log(2*pi*var_c)) ...
                   -0.5 * sum((feat_norm - mu_c).^2 ./ var_c);
        logScores(c) = logGauss + log(clf.priors(c));
    end
    [~, idxMax] = max(logScores);
    pred_dir = clf.classes(idxMax);
end
