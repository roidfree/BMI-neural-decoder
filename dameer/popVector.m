% PSEUDO CODE for Population Vector Decoder (direction)

% Step 1: Compute firing rates in first 320ms - chosen cuz initial spikes more relevant 
%for each direction d
%   for each trial t
%       for each neuron n
%           rate(t,d,n) = sum(spikes(n,1:320)) / 320; % feature vector 
%       end
%   end
%end
%----------

% Step 2: Compute preferred direction of each neuron
% for each neuron n
%   for each direction d
%       mean_rate(n,d) = average firing rate(:,d,n) across trials;
%   end 
% end

% Step 3: Find the preferred direction for each neuron
% for each neuron n
%   combine the 8 direction unit vectors
%   weight each one by mean_rate(n,d)
%   preferred_direction(n) = sum(weighted unit vectors)
% end

%---
% Testing
% Compute test firing rates from first 320ms of test trials
% for each neuron n
%  test_rate(n) = sum(test spikes(n,1:320)) / 320;
% end

% Build the population vector for each test trial
% PV = sum over neurons of test_rate * preferred_direction(n)

% Compute the angle of the population vector
% take the angle of the PV vector to get the predicted direction
% angle = atan2(PV_y, PV_x)
%Choose the closest of the 8 directions...

% -------
% ACTUAL MATLAB code implementation

load('monkeydata_training.mat');

num_trials = size(trial, 1);
num_directions = size(trial, 2);
num_neurons = size(trial(1,1).spikes, 1);
T = 320;

% 8 movement angles
angles = linspace(0, 2*pi, num_directions + 1);
angles(end) = [];

% Step 1: compute firing rates in first 320 ms
rate = zeros(num_trials, num_directions, num_neurons);

for d = 1:num_directions
    for t = 1:num_trials
        for n = 1:num_neurons
            spike_count = sum(trial(t,d).spikes(n, 1:T));
            rate(t,d,n) = spike_count / T;
        end
    end
end

% Step 2: average firing rate across trials
mean_rate = zeros(num_neurons, num_directions);

for n = 1:num_neurons
    for d = 1:num_directions
        mean_rate(n,d) = mean(rate(:,d,n));
    end
end

% Step 3: find preferred direction of each neuron
preferred_direction = zeros(num_neurons, 2);

for n = 1:num_neurons
    vx = 0;
    vy = 0;

    for d = 1:num_directions
        vx = vx + mean_rate(n,d) * cos(angles(d));
        vy = vy + mean_rate(n,d) * sin(angles(d));
    end

    preferred_direction(n,:) = [vx vy];
end

% example trial to test
test_trial = trial(50,1);

% Step 4: compute test firing rates
test_rate = zeros(num_neurons, 1);

for n = 1:num_neurons
    test_rate(n) = sum(test_trial.spikes(n, 1:T)) / T;
end

% Step 5: build population vector
PV = [0 0];

for n = 1:num_neurons
    PV = PV + test_rate(n) * preferred_direction(n,:);
end

% Step 6: convert vector angle into one of the 8 directions
predicted_angle = atan2(PV(2), PV(1));

angle_diff = zeros(num_directions, 1);
for d = 1:num_directions
    angle_diff(d) = abs(atan2(sin(predicted_angle - angles(d)), cos(predicted_angle - angles(d))));
end

[~, predicted_direction] = min(angle_diff);

fprintf('Predicted direction: %d\n', predicted_direction);
fprintf('Population vector angle: %.2f radians\n', predicted_angle);


