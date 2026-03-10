%% BMI Pre-Homework
clear
clf
load("monkeydata_training.mat");
%% 1 Raster plot for single trial
trial_num = 1; % needs to be 1 - 100
movement_num = 1; % needs to be 1 - 8
spike_matrix = trial(trial_num, movement_num).spikes(:, :);
[neuron_ids, time_bins] = find(spike_matrix);
figure_1 = figure(1); box on; grid on; hold on;
scatter(time_bins, neuron_ids, "b", "filled");

%% 2 Raster plot for single neuron
movement_num = 1;
neuron_num = 1; % needs to be 1 - 98

figure_2 = figure(2); box on; grid on; hold on;
for trial_idx = 1:100
    trial_spike_train = trial(trial_idx, movement_num).spikes(neuron_num, :);
    [~, time_bins] = find(trial_spike_train);
    scatter(time_bins, trial_idx * ones(1, length(time_bins)), [], rand(1, 3), "filled");
end

%% 3 Peri-stimulus time histogram
movement_idx = 1;
neuron_idx = 1;
min_trial_length = Inf;
max_trial_length = 0;
for trial_idx = 1:100
    trial_length = length(trial(trial_idx, movement_idx).spikes(neuron_idx, :));
    if trial_length > max_trial_length
        max_trial_length = trial_length;
    end
    if trial_length < min_trial_length
        min_trial_length = trial_length;
    end
end
neat_spike_trains =  NaN(100, max_trial_length);
for trial_idx = 1:100
    trial_spike_train = trial(trial_idx, movement_idx).spikes(neuron_idx, :);
    trial_length = length(trial_spike_train);
    neat_spike_trains(trial_idx, 1:trial_length) = trial_spike_train;
end

%% Plotting 
PSTT = mean(neat_spike_trains, 1, "omitnan") ./ 0.001;
MAed_PSTT = movmean(PSTT, 10);
alpha = 0.2;
EMAed_PSTT = filter([alpha], [1, alpha-1], PSTT);

clf
figure_3 = figure(3); box on; grid on; hold on;
bar((0:max_trial_length-1) * 0.001, PSTT, "facecolor", [0, 0, 0.8], "displayname", "Raw PSTT");
bar((0:max_trial_length-1) * 0.001, MAed_PSTT, "facecolor", [0.8, 0, 0],  "displayname", "MA PSTT");
bar((0:max_trial_length-1) * 0.001, EMAed_PSTT, "facecolor", [0, 0.8, 0],  "displayname", "EMA PSTT");
legend;

%% 4 Plotting hand position for different trials
trial_idx = 1;
movement_idx = 1;
trajectories = trial(trial_idx, movement_idx).handPos;
x_trajectory = trajectories(1, :);
y_trajectory = trajectories(2, :);
angles = mod(atan2(y_trajectory, x_trajectory), 2*pi);
figure_4 = figure(4);
clf
subplot(311); box on; grid on; hold on;
plot((0:length(trajectories) - 1) * 0.001, x_trajectory, "linewidth", 2, "color", [0, 0, 0.8], "displayname", "x");
plot((0:length(trajectories) - 1) * 0.001, y_trajectory, "linewidth", 2, "color", [0.8, 0, 0], "displayname", "y");
legend;
subplot(312); box on; grid on; hold on;
plot((0:length(trajectories) - 1) * 0.001, angles, "linewidth", 2, "color", [0, 0, 0.8], "displayname", "angle");
subplot(313); box on; grid on; hold on;
plot(x_trajectory, y_trajectory, "linewidth", 2, "color", [0, 0, 0.8], "displayname", "trajectory");
xlim([-200,200]);
ylim([-200, 200]);