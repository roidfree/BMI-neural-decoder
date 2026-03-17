%% —————— positionEstimatorTraining_noPCA_LDA.m ——————
function modelParameters = positionEstimatorTraining(trainingData)
% 纯 Gaussian Naïve Bayes 训练（无 PCA、无 LDA）
% 输入:
%   trainingData: [nTrials × nDirs] struct 数组，含 spikes (N×T)
% 输出:
%   modelParameters: struct，字段：
%     .mu_class    – 原始特征归一化均值 (1×D0)
%     .sigma_class – 原始特征归一化 std  (1×D0)
%     .classifier  – NBC 参数 struct，含 classMeans、classVars、priors、classes
%     .T_class     – 时间窗长度

    %% 1. 构造特征矩阵 X0 和标签 y
    T_class = 320;
    [nTrials, nDirs] = size(trainingData);
    D0 = size(trainingData(1,1).spikes,1);
    M  = nTrials * nDirs;
    X0 = zeros(M, D0);
    y  = zeros(M,1);
    idx = 0;
    for i = 1:nTrials
        for k = 1:nDirs
            idx = idx + 1;
            % 每个神经元在前 T_class ms 的 spike 总数
            X0(idx,:) = sum(trainingData(i,k).spikes(:,1:T_class),2)';
            y(idx)    = k;
        end
    end

    %% 2. Z-score 归一化
    mu_class    = mean(X0,1);               
    sigma_class = std(X0,0,1) + eps;        % 防止 0
    X_norm = (X0 - mu_class) ./ sigma_class; 

    %% 3. 训练 Gaussian Naïve Bayes
    Cn = max(y);
    classMeans = zeros(Cn, D0);
    classVars  = zeros(Cn, D0);
    priors     = zeros(Cn,1);
    for c = 1:Cn
        Xc = X_norm(y==c, :);
        priors(c)       = size(Xc,1) / M;
        classMeans(c,:) = mean(Xc,1);
        classVars(c,:)  = var(Xc,0,1) + eps;  % 防止 0 方差
    end

    classifier.classMeans = classMeans;
    classifier.classVars  = classVars;
    classifier.priors     = priors;
    classifier.classes    = (1:Cn)';

    %% 4. 打包返回
    modelParameters.mu_class    = mu_class;
    modelParameters.sigma_class = sigma_class;
    modelParameters.classifier  = classifier;
    modelParameters.T_class     = T_class;
end
