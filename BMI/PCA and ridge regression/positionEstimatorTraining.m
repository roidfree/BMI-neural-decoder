function [modelParameters] = positionEstimatorTraining(training_data)

[trials, movements] = size(training_data);
n_neurons = 98;

bin_size = 50;
num_bins = 6; % 300 ms history

num_pcs = 30; % number of PCA components
lambda = 10;  % ridge regularization

modelParameters.dirTemplates = zeros(movements, n_neurons);

for m = 1:movements
    
    X_list = {};
    Y_list = {};
    all_starts = [];
    
    for t = 1:trials
        
        spikes = training_data(t,m).spikes;
        pos = training_data(t,m).handPos(1:2,:);
        
        % direction template (first 320 ms)
        all_starts = [all_starts; mean(spikes(:,1:320),2)'];
        
        % smooth spikes
        rates = rates_from_spikes(spikes,10,30,1);
        T = size(rates,2);
        
        for tt = 320:20:T
            
            hist_feat = zeros(1,n_neurons*num_bins);
            
            for b = 1:num_bins
                
                start_idx = tt-(b*bin_size)+1;
                end_idx   = tt-((b-1)*bin_size);
                
                if start_idx > 0
                    feat = mean(rates(:,start_idx:end_idx),2)';
                else
                    feat = zeros(1,n_neurons);
                end
                
                hist_feat((b-1)*n_neurons+1:b*n_neurons) = feat;
                
            end
            
            X_list{end+1} = hist_feat;
            Y_list{end+1} = pos(:,tt)';
            
        end
        
    end
    
    % store direction template
    modelParameters.dirTemplates(m,:) = mean(all_starts,1);
    
    % build matrices
    X = cell2mat(X_list');
    Y = cell2mat(Y_list');
    
    % normalize features
    mu = mean(X);
    sigma = std(X) + 1e-6;
    X_norm = (X - mu) ./ sigma;
    
    % PCA
    [coeff,score] = pca(X_norm);
    
    X_pca = score(:,1:num_pcs);
    
    % add bias
    X_pca = [X_pca ones(size(X_pca,1),1)];
    
    % ridge regression
    B = (X_pca'*X_pca + lambda*eye(size(X_pca,2))) \ (X_pca'*Y);
    
    % store parameters
    modelParameters.PCAcoeff{m} = coeff(:,1:num_pcs);
    modelParameters.mu{m} = mu;
    modelParameters.sigma{m} = sigma;
    modelParameters.B{m} = B;
    
end

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