%% —————— positionEstimatorTraining_noToolbox_GNB.m ——————
function modelParameters = positionEstimatorTraining(trainingData)
% 手动实现 PCA→LDA→Gaussian Naïve Bayes 训练
% 输入:
%   trainingData: [nTrials × nDirs] struct 数组，含 spikes (N×T)
% 输出:
%   modelParameters: struct，字段参见下方注释

    %% 1. 构造原始特征 X0, 标签 y
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
            X0(idx,:) = sum(trainingData(i,k).spikes(:,1:T_class),2)';  
            y(idx)    = k;
        end
    end

    %% 2. Z-score 归一化
    mu_class    = mean(X0,1);               
    sigma_class = std(X0,0,1) + eps;        
    X_norm = (X0 - mu_class) ./ sigma_class; 

    %% 3. PCA 降维（累计方差 ≥95%）
    C = (X_norm' * X_norm) / (M - 1);        % N×N 协方差矩阵
    [V, Dmat] = eig(C);                     
    [eigvals, order] = sort(diag(Dmat),'descend');
    V_sorted = V(:, order);
    explained = eigvals / sum(eigvals);
    cumVar = cumsum(explained);
    dPCA = find(cumVar >= 0.92, 1, 'first');
    coeff = V_sorted(:, 1:dPCA);            
    mu_pca = mean(X_norm,1);                
    X_centered = X_norm - mu_pca;           
    X_pca = X_centered * coeff;             

    %% 4. LDA 监督降维（降到 C-1 维）
    Cn = max(y);
    mu_all = mean(X_pca,1);
    SW = zeros(dPCA);
    SB = zeros(dPCA);
    for c = 1:Cn
        Xc = X_pca(y==c, :);
        nc = size(Xc,1);
        mu_c = mean(Xc,1);
        SW = SW + (Xc - mu_c)'*(Xc - mu_c);
        SB = SB + nc*(mu_c - mu_all)'*(mu_c - mu_all);
    end
    [Vlda, Dlda] = eig(SB, SW);
    [~, ordL] = sort(diag(Dlda),'descend');
    dLDA = 7;                          
    W_lda = Vlda(:, ordL(1:dLDA));          
    X_lda = X_pca * W_lda;                  

    %% 5. 剔除任何类别中零方差的维度
    zeroVar = false(1,dLDA);
    for c = 1:Cn
        zeroVar = zeroVar | (var(X_lda(y==c,:),0,1)==0);
    end
    predictorIdx = find(~zeroVar);
    X_nb = X_lda(:, predictorIdx);

    %% 6. 训练 Gaussian Naïve Bayes
    classMeans = zeros(Cn, numel(predictorIdx));
    classVars  = zeros(Cn, numel(predictorIdx));
    priors     = zeros(Cn,1);
    for c = 1:Cn
        Xc = X_nb(y==c, :);
        priors(c) = size(Xc,1) / M;
        classMeans(c,:) = mean(Xc,1);
        classVars(c,:)  = var(Xc,0,1) + eps;  % 防止除零
    end
    classifier.classMeans = classMeans;  
    classifier.classVars  = classVars;
    classifier.priors     = priors;
    classifier.classes    = (1:Cn)';

    %% 7. 打包返回
    modelParameters.mu_class     = mu_class;
    modelParameters.sigma_class  = sigma_class;
    modelParameters.pca.coeff    = coeff;
    modelParameters.pca.mu       = mu_pca;
    modelParameters.pca.d        = dPCA;
    modelParameters.lda.W        = W_lda;
    modelParameters.predictorIdx = predictorIdx;
    modelParameters.classifier   = classifier;
    modelParameters.T_class      = T_class;
end
