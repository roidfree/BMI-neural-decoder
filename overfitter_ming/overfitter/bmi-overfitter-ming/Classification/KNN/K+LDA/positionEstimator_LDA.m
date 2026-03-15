% positionEstimator_LDA.m

function pred_dir = positionEstimator_LDA(testSample, modelParameters)
% Predict direction via LDA+kNN.
%
%   pred_dir = positionEstimator_LDA(testSample, modelParameters)
%
% Inputs:
%   testSample       : struct with .spikes (98×T)
%   modelParameters  : output of positionEstimatorTraining_LDA
%
% Output:
%   pred_dir : scalar in 1..8

    T_class = 320;
    spk = testSample.spikes;
    tEnd = min(T_class, size(spk,2));
    feat = sum(spk(:,1:tEnd),2)';                  % 1×98

    % normalize
    Xn = (feat - modelParameters.mu) ./ modelParameters.sigma;  % 1×98
    % project
    x_lda = Xn * modelParameters.W;                % 1×ldaDim

    % compute distances to all N training points
    diffsq = (modelParameters.X_lda - x_lda).^2;    % N×ldaDim
    d2 = sum(diffsq,2);                            % N×1

    % find k nearest
    [~,ord] = sort(d2);
    nn      = modelParameters.y(ord(1:modelParameters.k));
    % majority vote
    pred_dir = mode(nn);
end
