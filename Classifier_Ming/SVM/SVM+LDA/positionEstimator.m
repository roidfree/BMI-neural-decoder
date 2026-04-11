function predicted_direction = positionEstimator(test_data, modelParameters)
    time_window = 320;
    spike_data = test_data.spikes;
    firing_rates = sum(spike_data(:, 1:time_window), 2)';

    % --- SVM raw scores (distance to boundary) ---
    svm_scores = zeros(1, 8);
    for k = 1:8
        model = modelParameters.svm_models{k};
        svm_scores(k) = firing_rates * model.w + model.b;
    end

    % Normalize SVM scores
    svm_scores = svm_scores / max(abs(svm_scores));

    % --- LDA posterior probabilities ---
    [~, lda_scores] = predict(modelParameters.lda_model, firing_rates); % 1x8

    % Normalize LDA scores
    lda_scores = lda_scores / sum(lda_scores);

    % --- Weighted combination ---
    alpha = 0.5;  % Weighting factor: 0.5 means equal weight
    combined_score = alpha * svm_scores + (1 - alpha) * lda_scores;

    % Final prediction: class with highest combined score
    [~, predicted_direction] = max(combined_score);
end
