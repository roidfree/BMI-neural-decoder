function [modelParameters] = positionEstimatorTraining(training_data, regressionMethod, preprocessingConfig)
    if nargin < 2 || isempty(regressionMethod)
        regressionMethod = 'pcr_ridge';
    end
    regressionMethod = lower(string(regressionMethod));
    if nargin < 3 || isempty(preprocessingConfig)
        preprocessingConfig = struct();
    end

    modelParameters = struct;
    processed_data = training_data;
    trials = size(training_data, 1);
    movements = size(training_data, 2);
    neurons = 98;
    
    % CLASSIFICATION
    A = [];
    B = [];

    % Loop through all directions and trials
    for m = 1:movements
        for t = 1:trials

            % Extract spike data
            spikes = training_data(t,m).spikes;

            % Feature: spike counts in first 320 ms
            feat = sum(spikes(:,1:320),2)';

            % Store feature and corresponding label
            A = [A; feat];
            B = [B; m];

        end
    end

    % Get dimensions
    [n_samples, n_features] = size(A);

    % Compute class means (one per direction)
    class_means = zeros(movements, n_features);

    for k = 1:movements
        class_means(k,:) = mean(A(B==k,:), 1);
    end
    
    % Store baseline template-classifier parameters
    modelParameters.means = class_means;

    % Add PCA+LDA+kNN direction classifier with selected hyperparameters.
    % (from your table: PCs=30, LDA=5, k=7)
    modelParameters.pcaKnnClassifier = positionEstimatorTraining_PCA_LDA_K(training_data, 7, 30, 5);

    % MEAN TRAJECTORY
    % Always provide modelParameters.avgTraj{m} = [mean_x(t); mean_y(t)]
    avgTraj = cell(movements, 1);
    T_pad_absolute = 1000;
    for m = 1:movements
        Tmax = 0;
        for t = 1:trials
            Tmax = max(Tmax, size(training_data(t, m).handPos, 2));
        end

        sumPos = zeros(2, Tmax);
        count = zeros(1, Tmax);

        for t = 1:trials
            pos = training_data(t, m).handPos(1:2, :);
            T_i = size(pos, 2);
            sumPos(:, 1:T_i) = sumPos(:, 1:T_i) + pos;
            count(1:T_i) = count(1:T_i) + 1;
        end

        count(count == 0) = 1;
        mean_path = sumPos ./ repmat(count, 2, 1);
        padded_path = zeros(2, T_pad_absolute);
        padded_path(:, 1:Tmax) = mean_path;
        padded_path(:, Tmax+1:end) = repmat(mean_path(:, end), 1, T_pad_absolute - Tmax);
        avgTraj{m} = padded_path;
    end
    modelParameters.avgTraj = avgTraj;

    % PREPROCESSING
    if ~isfield(preprocessingConfig, 'binWidth') || isempty(preprocessingConfig.binWidth)
        preprocessingConfig.binWidth = 20;
    end
    if ~isfield(preprocessingConfig, 'transform') || isempty(preprocessingConfig.transform)
        preprocessingConfig.transform = "anscombe";
    end
    if ~isfield(preprocessingConfig, 'smoothKernel') || isempty(preprocessingConfig.smoothKernel)
        preprocessingConfig.smoothKernel = "none";
    end
    if ~isfield(preprocessingConfig, 'smoothWidth') || isempty(preprocessingConfig.smoothWidth)
        preprocessingConfig.smoothWidth = 0;
    end
    if ~isfield(preprocessingConfig, 'smoothParam') || isempty(preprocessingConfig.smoothParam)
        preprocessingConfig.smoothParam = 2;
    end
    if ~isfield(preprocessingConfig, 'targetMode') || isempty(preprocessingConfig.targetMode)
        preprocessingConfig.targetMode = "deviation";
    end
    modelParameters.preprocess = preprocessingConfig;
    modelParameters.targetMode = char(lower(string(preprocessingConfig.targetMode)));

    % Initialise bin_width field
    for t = 1:trials
        for m = 1:movements
            processed_data(t, m).bin_width = 1;
        end
    end
    
    % Optional temporal smoothing before binning (e.g., gaussian)
    if string(preprocessingConfig.smoothKernel) ~= "none" && preprocessingConfig.smoothWidth > 0
        processed_data = convolve_data(processed_data, trials, movements, neurons, ...
            string(preprocessingConfig.smoothKernel), preprocessingConfig.smoothParam, preprocessingConfig.smoothWidth);
    end

    % Spike counts in configurable bins
    processed_data = rebin_data(processed_data, trials, movements, neurons, preprocessingConfig.binWidth);

    % Apply Anscombe transform for ~constant variance and Gaussinity
    processed_data = transform_data(processed_data, trials, movements, neurons, string(preprocessingConfig.transform));

    % TRAINING DATA CREATION
    % Produce matrices for average velocity regression -- MINIMUM TRIAL
    % LENGTH IS 571 THUS CAN ONLY GO UP TO 560
    % history_bins = 15;  % should be zero to something reasonable, corresponds to history_bins * bin_width lag in time
    % bin_width = processed_data(1, 1).bin_width;
    % max_iter = floor(571 / bin_width) - history_bins;
    % X = zeros(movements, trials * max_iter, neurons * (history_bins + 1));
    % Y = zeros(movements, trials * max_iter, 2); % X avg velocity in col 1 and Y in 2
    % for m = 1:movements
    %     counter = 1;
    %     for t = 1:trials
    %         for i = 1:max_iter
    %             X(m, counter, :) = reshape(processed_data(t, m).spikes(:, i:history_bins + i), 1, []);
    %             Y(m, counter, 1) = processed_data(t, m).handPos(1, (history_bins + i) * bin_width) - processed_data(t, m).handPos(1, (history_bins + i - 1) * bin_width);
    %             Y(m, counter, 2) = processed_data(t, m).handPos(2, (history_bins + i) * bin_width) - processed_data(t, m).handPos(2, (history_bins + i - 1) * bin_width);
    %             counter = counter + 1;
    %         end
    %     end
    % end

    % Produce matrices for average velocity deviation regression -- MINIMUM TRIAL
    % LENGTH IS 571 THUS CAN ONLY GO UP TO 560
    history_bins = 15;  % corresponds to history_bins * bin_width lag in time
    bin_width = processed_data(1, 1).bin_width;
    max_iter = floor(571 / bin_width) - history_bins;
    X = zeros(movements, trials * max_iter, neurons * (history_bins + 1));
    Y = zeros(movements, trials * max_iter, 2); % X avg velocity in col 1 and Y in 2
    for m = 1:movements
        counter = 1;
        for t = 1:trials
            for i = 1:max_iter
                X(m, counter, :) = reshape(processed_data(t, m).spikes(:, i:history_bins + i), 1, []);
                vRealX = processed_data(t, m).handPos(1, (history_bins + i) * bin_width) - processed_data(t, m).handPos(1, (history_bins + i - 1) * bin_width);
                vRealY = processed_data(t, m).handPos(2, (history_bins + i) * bin_width) - processed_data(t, m).handPos(2, (history_bins + i - 1) * bin_width);
                vMeanX = modelParameters.avgTraj{m}(1, (history_bins + i) * bin_width) - modelParameters.avgTraj{m}(1, (history_bins + i - 1) * bin_width);
                vMeanY = modelParameters.avgTraj{m}(2, (history_bins + i) * bin_width) - modelParameters.avgTraj{m}(2, (history_bins + i - 1) * bin_width);
                if modelParameters.targetMode == "velocity"
                    Y(m, counter, 1) = vRealX;
                    Y(m, counter, 2) = vRealY;
                else
                    Y(m, counter, 1) = vRealX - vMeanX;
                    Y(m, counter, 2) = vRealY - vMeanY;
                end
                counter = counter + 1;
            end
        end
    end

    modelParameters.B = cell(movements, 1);
    modelParameters.mu_X = cell(movements, 1);
    modelParameters.V_reduced = cell(movements, 1);
    modelParameters.B_pls = cell(movements, 1);

    % OLS
    % Have a bias input just in case
    % WON'T HAVE ENOUGH DATA TO TRAIN THE WEIGHTS FOR OLS
    % for m = 1:movements
    %     Xmov = [squeeze(X(m, :, :)), ones(trials * max_iter, 1)];
    %     Ymov = squeeze(Y(m, :, :));
    %     modelParameters.B{m} = Xmov \ Ymov;
    % end

    % PCR
    % for m = 1:movements
    %     Xmov = squeeze(X(m, :, :));
    %     Ymov = squeeze(Y(m, :, :));
    %     mu_X = mean(Xmov, 1);
    %     modelParameters.mu_X{m} = mu_X;
    %     centred_X = Xmov - mu_X;
    %     [U, S, V] = svd(centred_X);
    %     PCs = 100;
    %     U_reduced = U(:, 1:PCs);
    %     S_reduced = S(1:PCs, 1:PCs);
    %     V_reduced = V(:, 1:PCs);  % these are eigenvectors of covariance matrix
    %     modelParameters.V_reduced{m} = V_reduced;
    %     eigen_X = centred_X * V_reduced;
    %     eigen_X = [eigen_X, ones(trials * max_iter, 1)];
    %     modelParameters.B{m} = eigen_X \ Ymov;
    % end

    modelParameters.regressionMethod = char(regressionMethod);
    modelParameters.lambda = 1000;
    modelParameters.nPC = 500;
    modelParameters.nPLS = 30;
    if isfield(preprocessingConfig, 'nPLS') && ~isempty(preprocessingConfig.nPLS)
        modelParameters.nPLS = preprocessingConfig.nPLS;
    end

    % OLS / PCR / PCR+Ridge / PLS
    lambda = modelParameters.lambda;
    for m = 1:movements
        Xmov = squeeze(X(m, :, :));
        Ymov = squeeze(Y(m, :, :));

        switch regressionMethod
            case "avg_only"
                modelParameters.B{m} = [0, 0];
                modelParameters.mu_X{m} = [];
                modelParameters.V_reduced{m} = [];
                modelParameters.B_pls{m} = [];

            case "ols"
                Xaug = [Xmov, ones(size(Xmov, 1), 1)];
                modelParameters.B{m} = Xaug \ Ymov;
                modelParameters.mu_X{m} = [];
                modelParameters.V_reduced{m} = [];
                modelParameters.B_pls{m} = [];

            case {"pcr", "pcr_ridge"}
                mu_X = mean(Xmov, 1);
                modelParameters.mu_X{m} = mu_X;
                centred_X = Xmov - mu_X;
                [~, ~, V] = svd(centred_X, 'econ');
                PCs = min(modelParameters.nPC, size(V, 2));
                V_reduced = V(:, 1:PCs);  % eigenvectors of covariance matrix
                modelParameters.V_reduced{m} = V_reduced;
                eigen_X = centred_X * V_reduced;
                eigen_X = [eigen_X, ones(size(eigen_X, 1), 1)];

                if regressionMethod == "pcr"
                    modelParameters.B{m} = eigen_X \ Ymov;
                else
                    penalty = lambda * eye(PCs + 1);
                    penalty(end, end) = 0; % do not penalize bias
                    modelParameters.B{m} = (eigen_X' * eigen_X + penalty) \ (eigen_X' * Ymov);
                end
                modelParameters.B_pls{m} = [];

            case "pls"
                nComp = min([modelParameters.nPLS, size(Xmov, 1) - 1, size(Xmov, 2)]);
                if nComp < 1
                    error('PLS needs at least 1 component for direction %d.', m);
                end
                [~, ~, ~, ~, BETA] = plsregress(Xmov, Ymov, nComp);
                modelParameters.B_pls{m} = BETA; % [intercept; coeffs]
                modelParameters.B{m} = [];
                modelParameters.mu_X{m} = [];
                modelParameters.V_reduced{m} = [];

            otherwise
                error('Unknown regressionMethod: %s (use avg_only|ols|pcr|pcr_ridge|pls)', regressionMethod);
        end
    end

end


function [training_data] = rebin_data(training_data, trials, movements, neurons, new_bin_width)
    % Take .spikes and rebin by counts
    % i.e. [1 1 1 1 1] -> [5] if bin_width 1 -> 5
    % Fast exit pass zero
    if new_bin_width == 0
    else
        for t = 1:trials
            for m = 1:movements
                unbinned_target = training_data(t, m);
                unbinned_data = unbinned_target.spikes;
                unbinned_length = size(unbinned_data, 2);
                binned_data = zeros(neurons, floor(unbinned_length / new_bin_width));  % drop last bin of wrong size as will confuse training
                counter = 1;
                for i = 1:new_bin_width:unbinned_length - (new_bin_width - 1)
                    binned_data(:, counter) = sum(unbinned_data(:, i:i + (new_bin_width - 1)), 2);
                    counter = counter + 1;
                end
                training_data(t, m).spikes = binned_data;
                training_data(t, m).bin_width = training_data(t, m).bin_width * new_bin_width;
            end
        end
    end
end


function [training_data] = downsample_data(training_data, trials, movements, neurons, downsample_step)
    % Take .spikes and "rebin" by value at start of bin
    % i.e. [1 2 3 4 5] -> [1] if downsample_step == 5
    % Fast exit pass zero
    if downsample_step == 0
    else
        for t = 1:trials
            for m = 1:movements
                training_data(t, m).spikes = training_data(t, m).spikes(:, 1:downsample_step:end);
                training_data(t, m).bin_width = training_data(t, m).bin_width * downsample_step;
            end
        end
    end
end


function [training_data] = maxpool_data(training_data, trials, movements, neurons, maxpool_bin_width)
    % Take .spikes and rebin by max value in bin (inspired by CNNs)
    % i.e. [1 2 3 2 1] -> [3] if maxpool_bin_width == 5
    % Fast exit pass zero
    if maxpool_bin_width == 0
    else
        for t = 1:trials
            for m = 1:movements
                unbinned_target = training_data(t, m);
                unbinned_data = unbinned_target.spikes;
                unbinned_length = size(unbinned_data, 2);
                binned_data = zeros(neurons, floor(unbinned_length / maxpool_bin_width));  % drop last bin of wrong size as will confuse training
                counter = 1;
                for i = 1:maxpool_bin_width:unbinned_length - (maxpool_bin_width - 1)
                    binned_data(:, counter) = max(unbinned_data(:, i:i + (maxpool_bin_width - 1)), [], 2);
                    counter = counter + 1;
                end
                training_data(t, m).spikes = binned_data;
                training_data(t, m).bin_width = training_data(t, m).bin_width * maxpool_bin_width;
            end
        end
    end
end


function [training_data] = transform_data(training_data, trials, movements, neurons, transform)
    % Neuron firing can be modelled as a double Poisson point process
    % (Poisson but with also changing mean)
    % Poisson-ity means that the variance scales with the mean
    % Can be bad for regression / dim reduction that assumes homoscedascity
    % Apply transform (sqrt or anscombe) to make variance more independent of mean
    % Pass "none" for fast exit
    if transform == "none"
    else
        if transform == "sqrt"
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = sqrt(training_data(t, m).spikes);
                end
            end
        end
        if transform == "anscombe"
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = 2 * sqrt(training_data(t, m).spikes + 3 / 8);
                end
            end
        end
    end
end


function [training_data] = convolve_data(training_data, trials, movements, neurons, kernel, kernel_param, kernel_width)
    % Convolve .spikes with a convolution `kernel` (MA, EMA, CGAUSS, AGAUSS)
    % For MA `kernel_param` is irrelevant
    % For EMA `kernel_param` corresponds to alpha
    % For CGAUSS (causal half-gaussian) `kernel_param` corresponds to std
    % For AGAUSS (acausal full-gaussain) `kernel_param` corresponds to std
    % Pass kernel_width == 0 for fast exit
    if kernel_width == 0
    else
        if kernel == "MA"
            ma_kernel = (1 / kernel_width) * ones(1, kernel_width);
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = filter(ma_kernel, [1], training_data(t, m).spikes, [], 2);
                end
            end
        end
        if kernel == "EMA"
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = filter([kernel_param], [1, kernel_param - 1], training_data(t, m).spikes, [], 2);
                end
            end
        end
        if kernel == "CGAUSS"
            n_s = -kernel_width:kernel_width;
            gauss_kernel = exp(-(n_s).^2 / (2 * kernel_param.^2)) ./ (kernel_param * sqrt(2 * pi));
            gauss_kernel(n_s < 0) = 0;  % making it causal -- kernel filled during convolution so zero first half
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = conv2(training_data(t, m).spikes, gauss_kernel, "same");
                end
            end
        end
        if kernel == "AGAUSS"
            n_s = -kernel_width:kernel_width;
            gauss_kernel = exp(-(n_s).^2 / (2 * kernel_param.^2)) ./ (kernel_param * sqrt(2 * pi));
            for t = 1:trials
                for m = 1:movements
                    training_data(t, m).spikes = conv2(training_data(t, m).spikes, gauss_kernel, "same");
                end
            end
        end
    end
end