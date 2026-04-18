function binned_spikes = rebin_spikes(spikes, bin_width)
num_neurons = size(spikes, 1);
num_timepoints = size(spikes, 2);
num_bins = floor(num_timepoints / bin_width);

binned_spikes = zeros(num_neurons, num_bins);

for b = 1:num_bins
    start_idx = (b - 1) * bin_width + 1;
    end_idx = b * bin_width;
    binned_spikes(:, b) = sum(spikes(:, start_idx:end_idx), 2);
end
end
