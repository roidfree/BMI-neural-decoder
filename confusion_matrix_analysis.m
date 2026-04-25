function [confMat, accuracy, normConfMat] = confusion_matrix_analysis(true_labels, predicted_labels)
% CONFUSION_MATRIX_ANALYSIS  Evaluate an 8-direction classifier.
%
%   [confMat, accuracy, normConfMat] = confusion_matrix_analysis(true_labels, predicted_labels)
%
%   Inputs
%     true_labels      - Nx1 or 1xN vector of true directions (integers 1–8)
%     predicted_labels - Nx1 or 1xN vector of predicted directions (integers 1–8)
%
%   Outputs
%     confMat     - 8x8 raw count confusion matrix  (row = true, col = predicted)
%     accuracy    - overall classification accuracy (0–100 %)
%     normConfMat - 8x8 row-normalised confusion matrix (each row sums to 1)

    NUM_DIRS   = 8;
    DIR_LABELS = {'45°','90°','135°','180°','225°','270°','315°','360°'};

    % When called with no arguments: load real data and run 2-fold cross-validation
    if nargin == 0
        fprintf('Loading monkeydata0.mat ...\n');
        addpath(genpath(fullfile(fileparts(mfilename('fullpath')), 'BMI')));
        trial = []; %#ok<NASGU>
        load('monkeydata0.mat', 'trial');

        rng(2013);
        ix = randperm(size(trial, 1));   % shuffle once; split into two equal halves

        true_labels      = [];
        predicted_labels = [];

        for fold = 1:2
            fprintf('Training fold %d/2 ...\n', fold);
            if fold == 1
                trainIdx = ix(1:50);
                testIdx  = ix(51:end);
            else
                trainIdx = ix(51:end);
                testIdx  = ix(1:50);
            end

            ldaModel = trainDirectionClassifier(trial(trainIdx, :));

            for tr = 1:size(testIdx, 2)
                for d = 1:NUM_DIRS
                    spikes  = trial(testIdx(tr), d).spikes;
                    pred    = lda_predict(spikes, ldaModel);
                    true_labels(end+1)      = d;    %#ok<AGROW>
                    predicted_labels(end+1) = pred; %#ok<AGROW>
                end
            end
        end
        fprintf('Done. 2-fold CV — all 100 trials evaluated (%d samples total).\n\n', ...
                numel(true_labels));
    end

    true_labels      = true_labels(:);
    predicted_labels = predicted_labels(:);

    if numel(true_labels) ~= numel(predicted_labels)
        error('true_labels and predicted_labels must have the same length.');
    end

    % ── 1. Raw confusion matrix ──────────────────────────────────────────────
    confMat = zeros(NUM_DIRS, NUM_DIRS);
    for i = 1:numel(true_labels)
        t = true_labels(i);
        p = predicted_labels(i);
        if t >= 1 && t <= NUM_DIRS && p >= 1 && p <= NUM_DIRS
            confMat(t, p) = confMat(t, p) + 1;
        end
    end

    % ── 2. Overall accuracy ──────────────────────────────────────────────────
    accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;

    % ── 3. Row-normalised confusion matrix ──────────────────────────────────
    row_sums   = sum(confMat, 2);
    normConfMat = confMat ./ max(row_sums, 1);   % avoid /0 for empty classes

    % ── 4. Console report ────────────────────────────────────────────────────
    fprintf('\n==============================================\n');
    fprintf('  Direction Classifier — Confusion Analysis\n');
    fprintf('  Evaluation: 2-fold cross-validation\n');
    fprintf('==============================================\n');
    fprintf('Overall Accuracy: %.2f%%\n\n', accuracy);

    fprintf('Raw Confusion Matrix (rows = true, cols = predicted):\n');
    disp(confMat);

    fprintf('Per-Class Accuracy:\n');
    per_class = diag(normConfMat) * 100;
    for d = 1:NUM_DIRS
        fprintf('  Dir %d (%s): %.1f%%\n', d, DIR_LABELS{d}, per_class(d));
    end

    % ── 5. Most confused class pairs ────────────────────────────────────────
    fprintf('\nMost Confused Class Pairs (off-diagonal):\n');
    tmp = normConfMat;
    tmp(logical(eye(NUM_DIRS))) = 0;   % zero the diagonal

    % Collect all off-diagonal entries
    pairs = [];
    for r = 1:NUM_DIRS
        for c = 1:NUM_DIRS
            if r ~= c && tmp(r,c) > 0
                pairs(end+1,:) = [r, c, tmp(r,c)]; %#ok<AGROW>
            end
        end
    end

    if ~isempty(pairs)
        [~, order] = sort(pairs(:,3), 'descend');
        pairs = pairs(order, :);
        top_n = min(5, size(pairs,1));
        for k = 1:top_n
            r = pairs(k,1);  c = pairs(k,2);
            fprintf('  True %d (%s) → Predicted %d (%s): %.1f%% misclassified\n', ...
                r, DIR_LABELS{r}, c, DIR_LABELS{c}, pairs(k,3)*100);
        end
    else
        fprintf('  None (perfect classification)\n');
    end
    fprintf('\n');

    % ── 6. Heatmap plots ─────────────────────────────────────────────────────
    % — Raw counts —
    figure('Name','Confusion Matrix — Raw Counts','NumberTitle','off');
    imagesc(confMat);
    colormap(flipud(hot));
    colorbar;
    clim([0, max(confMat(:))+1]);

    maxVal = max(confMat(:));
    for r = 1:NUM_DIRS
        for c = 1:NUM_DIRS
            if confMat(r,c) > maxVal * 0.6
                txtColor = 'w';
            else
                txtColor = 'k';
            end
            text(c, r, num2str(confMat(r,c)), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontWeight','bold', ...
                'Color', txtColor);
        end
    end

    set(gca, 'XTick',1:NUM_DIRS, 'XTickLabel',DIR_LABELS, ...
             'YTick',1:NUM_DIRS, 'YTickLabel',DIR_LABELS, ...
             'FontSize',10, 'TickDir','out');
    xlabel('Predicted Direction');
    ylabel('True Direction');
    title(sprintf('Confusion Matrix (2-fold CV) — Accuracy: %.2f%%', accuracy));

    % — Normalised —
    figure('Name','Confusion Matrix — Normalised','NumberTitle','off');
    imagesc(normConfMat, [0, 1]);
    colormap(flipud(hot));
    cb = colorbar;
    cb.Label.String = 'Proportion';

    for r = 1:NUM_DIRS
        for c = 1:NUM_DIRS
            if normConfMat(r,c) > 0.6
                txtColor = 'w';
            else
                txtColor = 'k';
            end
            text(c, r, sprintf('%.2f', normConfMat(r,c)), ...
                'HorizontalAlignment','center', ...
                'VerticalAlignment','middle', ...
                'FontSize',8, ...
                'Color', txtColor);
        end
    end

    set(gca, 'XTick',1:NUM_DIRS, 'XTickLabel',DIR_LABELS, ...
             'YTick',1:NUM_DIRS, 'YTickLabel',DIR_LABELS, ...
             'FontSize',10, 'TickDir','out');
    xlabel('Predicted Direction');
    ylabel('True Direction');
    title('Normalised Confusion Matrix (row sums = 1)');

end
