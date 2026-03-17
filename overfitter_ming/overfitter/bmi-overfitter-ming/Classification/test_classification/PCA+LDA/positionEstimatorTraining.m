%% —————— positionEstimatorTraining_noToolbox.m ——————
function modelParameters = positionEstimatorTraining_noToolbox(trainingData)
    % 不依赖任何 Toolbox 的 PCA→LDA 训练
    % 输入:
    %   trainingData: [#trials × #dirs] struct，含 spikes (N×T) 和 handPos （未用）
    % 输出:
    %   modelParameters: 包含所有手动实现的参数

    %% 1. 构造 X, y
    T_class   = 320;
    [numTrials, numDirs] = size(trainingData);
    numNeurons = size(trainingData(1,1).spikes,1);
    M = numTrials * numDirs;

    X = zeros(M, numNeurons);
    y = zeros(M,1);
    idx = 0;
    for i = 1:numTrials
        for k = 1:numDirs
            idx = idx + 1;
            X(idx,:) = sum(trainingData(i,k).spikes(:,1:T_class),2)';  % N 维向量
            y(idx)   = k;
        end
    end

    %% 2. Z-score 归一化
    mu_class    = mean(X,1);                 % 1×N
    sigma_class = std(X,0,1) + eps;          % 1×N
    X_norm = (X - mu_class) ./ sigma_class;  % M×N

    %% 3. PCA 降维（累计解释 ≥95%）
    % 3.1 计算协方差矩阵
    C = (X_norm' * X_norm) / (M - 1);        % N×N
    % 3.2 特征分解
    [V, D] = eig(C);                         % V: 特征向量列，D: 对角特征值
    eigvals = diag(D);
    % 3.3 从大到小排序
    [eigvals_sorted, perm] = sort(eigvals,'descend');
    V_sorted = V(:, perm);
    % 3.4 计算累计方差贡献率
    explained = eigvals_sorted / sum(eigvals_sorted);
    cumVar = cumsum(explained);
    dPCA = find(cumVar >= 0.92, 1,'first');
    % 3.5 选取前 dPCA 主成分
    coeff = V_sorted(:,1:dPCA);             % N×dPCA
    mu_pca = mean(X_norm,1);                 % 应当接近零向量
    X_pca = X_norm * coeff;                  % M×dPCA

    %% 4. 手动 LDA 训练（线性判别分析）
    K = numDirs;
    classMeans = zeros(K, dPCA);
    for k = 1:K
        Xi = X_pca(y==k, :);                 % 属于第 k 类的所有样本
        classMeans(k,:) = mean(Xi,1);        % 1×dPCA
    end
    % 池化协方差
    Sigma = zeros(dPCA);
    for k = 1:K
        Xi = X_pca(y==k, :) - classMeans(k,:);
        Sigma = Sigma + (Xi' * Xi);
    end
    Sigma = Sigma / (M - K);
    invSigma = pinv(Sigma);                 % dPCA×dPCA

    % 假设等先验
    priors = ones(K,1) * (1/K);

    % 打包 classifier 参数
    classifier.invSigma   = invSigma;
    classifier.means      = classMeans;
    classifier.priors     = priors;
    classifier.classes    = 1:K;

    %% 5. 返回
    modelParameters.classifier   = classifier;
    modelParameters.mu_class     = mu_class;
    modelParameters.sigma_class  = sigma_class;
    modelParameters.pca.coeff    = coeff;
    modelParameters.pca.mu       = mu_pca;
    modelParameters.pca.d        = dPCA;
end
