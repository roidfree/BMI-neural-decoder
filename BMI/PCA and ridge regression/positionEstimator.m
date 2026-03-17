function [x, y] = positionEstimator(testData, modelParameters)

persistent trialDir

spikes = testData.spikes;
T = size(spikes,2);

if T <= 320
    trialDir = [];
end

% direction classification
if isempty(trialDir)
    
    featDir = mean(spikes(:,1:320),2)';
    
    [~,trialDir] = min(vecnorm(modelParameters.dirTemplates - featDir,2,2));
    
end

% smoothing
rates = rates_from_spikes(spikes,10,30,1);

n_neurons = 98;
num_bins = 6;
bin_size = 50;

hist_feat = zeros(1,n_neurons*num_bins);

for b = 1:num_bins
    
    start_idx = T-(b*bin_size)+1;
    end_idx   = T-((b-1)*bin_size);
    
    if start_idx > 0
        feat = mean(rates(:,max(1,start_idx):end_idx),2)';
    else
        feat = zeros(1,n_neurons);
    end
    
    hist_feat((b-1)*n_neurons+1:b*n_neurons) = feat;
    
end

% normalize
mu = modelParameters.mu{trialDir};
sigma = modelParameters.sigma{trialDir};

feat_norm = (hist_feat - mu) ./ sigma;

% PCA projection
coeff = modelParameters.PCAcoeff{trialDir};
feat_pca = feat_norm * coeff;

% add bias
feat_pca = [feat_pca 1];

% predict
pred = feat_pca * modelParameters.B{trialDir};

x = pred(1);
y = pred(2);

end


function [rate_trains] = rates_from_spikes(spike_trains,kernel_width,window_width,causal)

n_s = -window_width:window_width;

gauss_kernel = exp(-(n_s).^2/(2*kernel_width^2));
gauss_kernel = gauss_kernel./sum(gauss_kernel);

if causal == 1
    gauss_kernel(n_s>0) = 0;
end

rate_trains = conv2(spike_trains,gauss_kernel,"same");

end