% PSEUDO CODE for Population Vector Decoder (direction)
%
% Step 1: Compute firing rates in first 320ms
% Step 2: Average those rates across training trials
% Step 3: Use those averages to find each neuron's preferred direction
% Step 4: For a test trial, compute firing rates in first 320ms
% Step 5: Build population vector from all neurons
% Step 6: Pick the closest of the 8 movement directions

load('monkeydata_training.mat');

num_trials = size(trial, 1);
num_directions = size(trial, 2);
num_neurons = size(trial(1,1).spikes, 1);
T = 320;

train_idx = 1:50;
test_idx = 51:100;

if num_trials < 100
    error('This script expects 100 trials per direction.');
end

angles = linspace(0, 2*pi, num_directions + 1);
angles(end) = [];

% Step 1: firing rates from the training trials
rate = zeros(length(train_idx), num_directions, num_neurons);

for d = 1:num_directions
    for t = 1:length(train_idx)
        trial_num = train_idx(t);
        for n = 1:num_neurons
            spike_count = sum(trial(trial_num, d).spikes(n, 1:T));
            rate(t,d,n) = spike_count / T;
        end
    end
end

% Step 2: average firing rate for each neuron and direction
mean_rate = zeros(num_neurons, num_directions);

for n = 1:num_neurons
    for d = 1:num_directions
        mean_rate(n,d) = mean(rate(:,d,n));
    end
end

% Step 3: preferred direction vector for each neuron
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

% Step 4-6: test on the other 50 trials
correct = 0;
total = 0;

for d_true = 1:num_directions
    for t = 1:length(test_idx)
        trial_num = test_idx(t);
        test_rate = zeros(num_neurons, 1);

        for n = 1:num_neurons
            test_rate(n) = sum(trial(trial_num, d_true).spikes(n, 1:T)) / T;
        end

        PV = [0 0];

        for n = 1:num_neurons
            PV = PV + test_rate(n) * preferred_direction(n,:);
        end

        predicted_angle = atan2(PV(2), PV(1));

        angle_diff = zeros(num_directions, 1);
        for d = 1:num_directions
            angle_diff(d) = abs(atan2(sin(predicted_angle - angles(d)), cos(predicted_angle - angles(d))));
        end

        [~, predicted_direction] = min(angle_diff);

        if predicted_direction == d_true
            correct = correct + 1;
        end

        total = total + 1;
    end
end

accuracy = correct / total;

fprintf('Training trials per direction: %d\n', length(train_idx));
fprintf('Testing trials per direction: %d\n', length(test_idx));
fprintf('Classification accuracy: %.2f%%\n', accuracy * 100);
