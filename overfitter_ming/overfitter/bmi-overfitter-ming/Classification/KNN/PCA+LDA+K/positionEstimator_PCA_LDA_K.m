% positionEstimator_PCA_LDA_K.m

function pred_dir = positionEstimator_PCA_LDA_K(testSample, modelParameters)
% Predicts direction via PCA→LDA→k-NN.
%
% Usage:
%   pred_dir = positionEstimator_PCA_LDA_K(testSample, modelParameters)
%
% Inputs:
%   testSample       : struct with .spikes (98×T)
%   modelParameters  : output of training above
%
% Output:
%   pred_dir : scalar 1..8

    T_class = 320;
    spk = testSample.spikes;
    tEnd = min(T_class, size(spk,2));
    feat = sum(spk(:,1:tEnd),2)';           % 1×98

    % normalize
    Xn = (feat - modelParameters.mu) ./ modelParameters.sigma;  % 1×98
    % PCA project
    x_pca = Xn * modelParameters.V_pca;      % 1×pcaDim
    % LDA project
    x_lda = x_pca * modelParameters.W_lda;   % 1×ldaDim

    % k-NN in LDA space
    diffsq = (modelParameters.X_proj - x_lda).^2;  % N×ldaDim
    d2     = sum(diffsq,2);                        % N×1
    [~,ord] = sort(d2,'ascend');
    nn      = modelParameters.y(ord(1:modelParameters.k));
    pred_dir = mode(nn);
end
