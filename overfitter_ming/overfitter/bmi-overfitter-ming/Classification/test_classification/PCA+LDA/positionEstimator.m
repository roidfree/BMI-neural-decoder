%% —————— positionEstimator_noToolbox.m ——————
function pred_dir = positionEstimator_noToolbox(testSample, modelParameters)
    % 基于手动实现的 PCA→LDA 做方向预测
    % 输入:
    %   testSample: struct，含 spikes (N×T)
    %   modelParameters: training时返回的参数
    % 输出:
    %   pred_dir: 预测方向（1~K）

    T_class = 320;
    feat = sum(testSample.spikes(:,1:T_class), 2)';  % 1×N

    % 1) Z-score 归一化
    mu_class    = modelParameters.mu_class;
    sigma_class = modelParameters.sigma_class;
    feat_norm = (feat - mu_class) ./ sigma_class;   % 1×N

    % 2) PCA 投影
    coeff = modelParameters.pca.coeff;               % N×dPCA
    % 因为训练时数据已中心化，此处只需直接投影
    feat_pca = feat_norm * coeff;                    % 1×dPCA

    % 3) LDA 判别：计算各类得分
    clf = modelParameters.classifier;
    invSigma = clf.invSigma;
    means = clf.means;       % K×dPCA
    priors = clf.priors;     % K×1
    K = numel(clf.classes);

    scores = zeros(K,1);
    for k = 1:K
        mu_k = means(k,:)';  % dPCA×1
        x = feat_pca';       % dPCA×1
        % 判别函数：δ_k(x) = x' Σ⁻¹ μ_k - 0.5 μ_k' Σ⁻¹ μ_k + log π_k
        scores(k) = x' * (invSigma * mu_k) ...
                    - 0.5 * (mu_k' * (invSigma * mu_k)) ...
                    + log(priors(k));
    end

    % 4) 取最大得分对应的类
    [~, idx] = max(scores);
    pred_dir = clf.classes(idx);
end
