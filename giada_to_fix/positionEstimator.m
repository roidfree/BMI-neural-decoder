function [x, y] = positionEstimator(testData, modelParameters)

    persistent trialDir
    
    spikes = testData.spikes;
    T = size(spikes,2);

    % Reset direction when a new trial begins
    if T <= 320
        trialDir = [];
    end

    %% 1. Direction classification (first 320 ms)
    if isempty(trialDir)
        featDir = mean(spikes(:,1:320),2)';
        [~,trialDir] = min(vecnorm(modelParameters.dirTemplates - featDir,2,2));
    end

    %% 2. Extract 300 ms causal history features
    rates = rates_from_spikes(spikes,10,30,1);

    hist_feat = [];
    bin_size = 50;
    num_bins = 6;

    for b = 1:num_bins

        start_idx = T - (b*bin_size) + 1;
        end_idx   = T - ((b-1)*bin_size);

        if start_idx > 0
            hist_feat = [hist_feat mean(rates(:,start_idx:end_idx),2)'];
        else
            hist_feat = [hist_feat zeros(1,98)];
        end

    end

    %% 3. Predict position using trained regression
    pred = [hist_feat 1] * modelParameters.B{trialDir};

    x = pred(1);
    y = pred(2);

end


function rate_trains = rates_from_spikes(spike_trains,kernel_width,window_width,causal)

    n_s = -window_width:window_width;

    gauss_kernel = exp(-(n_s).^2/(2*kernel_width.^2)) ...
                   ./ (kernel_width*sqrt(2*pi));

    if causal == 1
        gauss_kernel(n_s > 0) = 0;
    end

    rate_trains = conv2(spike_trains,gauss_kernel,"same");

end
