function predicted_direction = positionEstimator(test_data, modelParameters, alpha)
    % Set default alpha if not passed
    if nargin < 3
        alpha = 0.5;
    end

    time_window = 320;
    spike_data = test_data.spikes;
    firing_rates = sum(spike_data(:, 1:time_window), 2)';

    % === Apply PCA ===
    centered = firing_rates - modelParameters.pca_mean;
    reduced = centered * modelParameters.pca_coeff;

    % === SVM prediction (one-vs-all raw scores) ===
    svm_scores = zeros(1, 8);
    for k = 1:8
        model = modelParameters.svm_models{k};
        svm_scores(k) = reduced * model.w + model.b;
    end
    svm_scores = svm_scores / max(abs(svm_scores));  % Normalize

    % === LDA posterior probabilities ===
    [~, lda_scores] = predict(modelParameters.lda_model, reduced);
    lda_scores = lda_scores / sum(lda_scores);  % Normalize

    % === Score fusion ===
    combined_score = alpha * svm_scores + (1 - alpha) * lda_scores;

    [~, predicted_direction] = max(combined_score);
end
