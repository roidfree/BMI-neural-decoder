% positionEstimator.m

function pred_dir = positionEstimator(testSample, modelParameters)
% Predicts movement direction with a trained k-NN classifier.
%
% USAGE:
%   pred_dir = positionEstimator(testSample, modelParameters)
%
% INPUT:
%   testSample      : struct with field .spikes (98 x T)
%   modelParameters : output of positionEstimatorTraining
%
% OUTPUT:
%   pred_dir : scalar in 1..8, predicted direction

    % Extract & normalize feature
    spk = testSample.spikes;               % 98 x T
    tEnd = min(320, size(spk,2));
    feat = sum(spk(:,1:tEnd), 2)';         % 1x98 raw
    Xn = (feat - modelParameters.mu) ./ modelParameters.sigma;

    % Compute distances to all training points
    D2 = sum((modelParameters.X - Xn).^2, 2);   % Euclid. distance^2

    % Find k nearest
    [~, ord] = sort(D2, 'ascend');
    nn = modelParameters.y(ord(1:modelParameters.k));

    % Majority vote
    pred_dir = mode(nn);
end
