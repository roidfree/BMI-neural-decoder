function [modelParameters] = positionEstimatorTraining(training_data)
    % Team Members: [Names]
    [trials, movements] = size(training_data); 
    n_neurons = 98; 
    
    % Chestek et al. Parameters: 300ms history binned into 50ms chunks
    bin_size = 50; 
    num_bins = 300 / bin_size; 
    
    modelParameters.dirTemplates = zeros(movements, n_neurons);
    
    for m = 1:movements
        X_list = {}; Y_list = {};
        all_starts = [];
        
        for t = 1:trials
            % Classification Template (Initial 320ms) [cite: 88]
            all_starts = [all_starts; mean(training_data(t,m).spikes(:, 1:320), 2)'];
            
            % Causal Rate Smoothing
            rates = rates_from_spikes(training_data(t,m).spikes, 10, 30, 1);
            pos = training_data(t,m).handPos(1:2, :); % mm [cite: 77]
            
            T = size(rates, 2);
            for tt = 320:20:T % Scoring at 20ms intervals [cite: 88]
                hist_feat = [];
                for b = 1:num_bins
                    start_idx = tt - (b * bin_size) + 1;
                    end_idx = tt - ((b-1) * bin_size);
                    if start_idx > 0
                        hist_feat = [hist_feat, mean(rates(:, start_idx:end_idx), 2)'];
                    else
                        hist_feat = [hist_feat, zeros(1, n_neurons)];
                    end
                end
                X_list{end+1} = [hist_feat, 1]; % Features + Bias
                Y_list{end+1} = pos(:, tt)';
            end
        end
        modelParameters.dirTemplates(m, :) = mean(all_starts, 1);
        
        % Solve OLS
        X = cell2mat(X_list');
        Y = cell2mat(Y_list');
        modelParameters.B{m} = X \ Y; 
    end
end

function [rate_trains] = rates_from_spikes(spike_trains, kernel_width, window_width, causal)
    n_s = -window_width:window_width;
    gauss_kernel = exp(-(n_s).^2 / (2 * kernel_width.^2)) ./ (kernel_width * sqrt(2 * pi));
    if causal == 1
        gauss_kernel(n_s > 0) = 0; % Enforce causality [cite: 10]
    end
    rate_trains = conv2(spike_trains, gauss_kernel, "same");
end