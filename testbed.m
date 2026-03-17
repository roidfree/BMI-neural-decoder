load("monkeydata_training.mat");
rates = rates_from_spikes(trial(1,1).spikes, 100, 150);
plot(rates(1,:));