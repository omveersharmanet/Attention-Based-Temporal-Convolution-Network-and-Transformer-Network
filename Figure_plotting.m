
clc;
clear;
close all;

%% =========================================================
%  1. LOAD PYTHON RESULTS
% ==========================================================

data = load('validation_model_outputs.mat');

% Actual labels (Python labels start from 0)
actual_maneuver = double(data.actual_maneuver(:)) + 1;
actual_speed = double(data.actual_speed(:)) + 1;
actual_combined = double(data.actual_combined(:)) + 1;

% Predicted labels (Python labels start from 0)
predicted_maneuver = double(data.predicted_maneuver(:)) + 1;
predicted_speed = double(data.predicted_speed(:)) + 1;
predicted_combined = double(data.predicted_combined(:)) + 1;

% Predicted probabilities
lateral_prob = double(data.lateral_prob);
longitudinal_prob = double(data.longitudinal_prob);
combined_prob = double(data.combined_prob);


%% =========================================================
%  2. CLASS NAMES
% ==========================================================

lateral_names = {'LCL', 'LK', 'LCR'};

longitudinal_names = {'Slow', 'Medium', 'High'};

combined_names = {
    'LCL-Slow'
    'LCL-Medium'
    'LCL-High'
    'LK-Slow'
    'LK-Medium'
    'LK-High'
    'LCR-Slow'
    'LCR-Medium'
    'LCR-High'
};


%% =========================================================
%  3. CHECK DATA DIMENSIONS
% ==========================================================

N = length(actual_combined);

assert(length(predicted_combined) == N, ...
    'Combined actual and predicted labels have different lengths.');

assert(size(combined_prob,1) == N && size(combined_prob,2) == 9, ...
    'Combined probability dimensions are incorrect.');

assert(size(lateral_prob,1) == N && size(lateral_prob,2) == 3, ...
    'Lateral probability dimensions are incorrect.');

assert(size(longitudinal_prob,1) == N && size(longitudinal_prob,2) == 3, ...
    'Longitudinal probability dimensions are incorrect.');

assert(all(actual_combined >= 1 & actual_combined <= 9) && ...
       all(predicted_combined >= 1 & predicted_combined <= 9), ...
       'Combined labels must be between 1 and 9.');

assert(all(actual_maneuver >= 1 & actual_maneuver <= 3) && ...
       all(predicted_maneuver >= 1 & predicted_maneuver <= 3), ...
       'Lateral labels must be between 1 and 3.');

assert(all(actual_speed >= 1 & actual_speed <= 3) && ...
       all(predicted_speed >= 1 & predicted_speed <= 3), ...
       'Longitudinal labels must be between 1 and 3.');

fprintf('Total validation samples: %d\n', N);


%% =========================================================
%  4. CALCULATE CONFUSION MATRICES
% ==========================================================

% Rows = actual classes
% Columns = predicted classes

C_combined = confusionmat( ...
    actual_combined, predicted_combined, 'Order', 1:9);

C_lateral = confusionmat( ...
    actual_maneuver, predicted_maneuver, 'Order', 1:3);

C_longitudinal = confusionmat( ...
    actual_speed, predicted_speed, 'Order', 1:3);


%% =========================================================
%  5. CALCULATE PRECISION AND RECALL
% ==========================================================

[precision_combined, recall_combined] = ...
    calculate_metrics(C_combined);

[precision_lateral, recall_lateral] = ...
    calculate_metrics(C_lateral);

[precision_longitudinal, recall_longitudinal] = ...
    calculate_metrics(C_longitudinal);


% Macro-average precision and recall

P_combined = mean(precision_combined, 'omitnan');
R_combined = mean(recall_combined, 'omitnan');

P_lateral = mean(precision_lateral, 'omitnan');
R_lateral = mean(recall_lateral, 'omitnan');

P_longitudinal = mean(precision_longitudinal, 'omitnan');
R_longitudinal = mean(recall_longitudinal, 'omitnan');


%% =========================================================
%  6. DISPLAY PER-CLASS PRECISION AND RECALL
% ==========================================================

fprintf('\n============================================\n');
fprintf('       COMBINED BEHAVIOR (9 CLASSES)\n');
fprintf('============================================\n');

disp(table(combined_names, precision_combined, recall_combined, ...
    'VariableNames', {'Class', 'Precision_Percent', 'Recall_Percent'}));

fprintf('Macro Precision: %.4f%%\n', P_combined);
fprintf('Macro Recall:    %.4f%%\n', R_combined);


fprintf('\n============================================\n');
fprintf('          LATERAL BEHAVIOR\n');
fprintf('============================================\n');

disp(table(lateral_names(:), precision_lateral, recall_lateral, ...
    'VariableNames', {'Class', 'Precision_Percent', 'Recall_Percent'}));

fprintf('Macro Precision: %.4f%%\n', P_lateral);
fprintf('Macro Recall:    %.4f%%\n', R_lateral);


fprintf('\n============================================\n');
fprintf('       LONGITUDINAL BEHAVIOR\n');
fprintf('============================================\n');

disp(table(longitudinal_names(:), precision_longitudinal, ...
    recall_longitudinal, ...
    'VariableNames', {'Class', 'Precision_Percent', 'Recall_Percent'}));

fprintf('Macro Precision: %.4f%%\n', P_longitudinal);
fprintf('Macro Recall:    %.4f%%\n', R_longitudinal);


%% =========================================================
%  7. CONFUSION MATRICES: PERCENTAGES
% ==========================================================

% Combined behavior: 9 classes

figure('Name', 'Combined Behavior - Percentages');

cm_combined = confusionchart( ...
    categorical(actual_combined, 1:9, combined_names), ...
    categorical(predicted_combined, 1:9, combined_names));

cm_combined.Title = sprintf( ...
    'Combined Behavior (9 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_combined, R_combined);

cm_combined.RowSummary = 'row-normalized';
cm_combined.ColumnSummary = 'column-normalized';
cm_combined.Normalization = 'row-normalized';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');


% Lateral behavior: 3 classes

figure('Name', 'Lateral Behavior - Percentages');

cm_lateral = confusionchart( ...
    categorical(actual_maneuver, 1:3, lateral_names), ...
    categorical(predicted_maneuver, 1:3, lateral_names));

cm_lateral.Title = sprintf( ...
    'Lateral Behavior (3 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_lateral, R_lateral);

cm_lateral.RowSummary = 'row-normalized';
cm_lateral.ColumnSummary = 'column-normalized';
cm_lateral.Normalization = 'row-normalized';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');

% Longitudinal behavior: 3 classes

figure('Name', 'Longitudinal Behavior - Percentages');

cm_longitudinal = confusionchart( ...
    categorical(actual_speed, 1:3, longitudinal_names), ...
    categorical(predicted_speed, 1:3, longitudinal_names));

cm_longitudinal.Title = sprintf( ...
    'Longitudinal Behavior (3 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_longitudinal, R_longitudinal);

cm_longitudinal.RowSummary = 'row-normalized';
cm_longitudinal.ColumnSummary = 'column-normalized';
cm_longitudinal.Normalization = 'row-normalized';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');


%% =========================================================
%  8. CONFUSION MATRICES: ACTUAL COUNTS
% ==========================================================

% Combined behavior

figure('Name', 'Combined Behavior - Counts');

cm_combined_count = confusionchart( ...
    categorical(actual_combined, 1:9, combined_names), ...
    categorical(predicted_combined, 1:9, combined_names));

cm_combined_count.Title = sprintf( ...
    'Combined Behavior Counts (9 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_combined, R_combined);

cm_combined_count.RowSummary = 'row-normalized';
cm_combined_count.ColumnSummary = 'column-normalized';
cm_combined_count.Normalization = 'absolute';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');


% Lateral behavior

figure('Name', 'Lateral Behavior - Counts');

cm_lateral_count = confusionchart( ...
    categorical(actual_maneuver, 1:3, lateral_names), ...
    categorical(predicted_maneuver, 1:3, lateral_names));

cm_lateral_count.Title = sprintf( ...
    'Lateral Behavior Counts (3 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_lateral, R_lateral);

cm_lateral_count.RowSummary = 'row-normalized';
cm_lateral_count.ColumnSummary = 'column-normalized';
cm_lateral_count.Normalization = 'absolute';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');


% Longitudinal behavior

figure('Name', 'Longitudinal Behavior - Counts');

cm_longitudinal_count = confusionchart( ...
    categorical(actual_speed, 1:3, longitudinal_names), ...
    categorical(predicted_speed, 1:3, longitudinal_names));

cm_longitudinal_count.Title = sprintf( ...
    'Longitudinal Behavior Counts (3 Classes)\nPrecision = %.2f%% | Recall = %.2f%%', ...
    P_longitudinal, R_longitudinal);

cm_longitudinal_count.RowSummary = 'row-normalized';
cm_longitudinal_count.ColumnSummary = 'column-normalized';
cm_longitudinal_count.Normalization = 'absolute';
annotation('textbox',[0.83 0.68 0.15 0.04],'String','Recall (%)', 'EdgeColor','none','FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold', 'HorizontalAlignment','center');
annotation('textbox',[0.40 0.02 0.25 0.04], 'String','Precision (%)',  'EdgeColor','none', 'FontName','Times New Roman', 'FontSize',12, 'FontWeight','bold',  'HorizontalAlignment','center');


%% =========================================================
%  9. ROC CURVES
% ==========================================================

% Combined behavior: 9 classes

plot_multiclass_ROC( ...
    actual_combined, combined_prob, combined_names, ...
    'Combined Behavior ROC (9 Classes)');


% Lateral behavior: 3 classes

plot_multiclass_ROC( ...
    actual_maneuver, lateral_prob, lateral_names, ...
    'Lateral Behavior ROC (3 Classes)');


% Longitudinal behavior: 3 classes

plot_multiclass_ROC( ...
    actual_speed, longitudinal_prob, longitudinal_names, ...
    'Longitudinal Behavior ROC (3 Classes)');


%% =========================================================
%  10. SAVE ALL FIGURES
% ==========================================================

if ~exist('MATLAB_Figures', 'dir')
    mkdir('MATLAB_Figures');
end

figures = findall(0, 'Type', 'figure');

for i = 1:length(figures)

    figure(figures(i));

    filename = sprintf('Figure_%02d.png', i);

    saveas(figures(i), fullfile('MATLAB_Figures', filename));

end

fprintf('\nAll confusion matrices and ROC figures saved.\n');
fprintf('Output folder: MATLAB_Figures\n');

%%
%% Calculate F1-score

F1_combined = 2 .* (precision_combined .* recall_combined) ./ ...
    (precision_combined + recall_combined);

F1_lateral = 2 .* (precision_lateral .* recall_lateral) ./ ...
    (precision_lateral + recall_lateral);

F1_longitudinal = 2 .* (precision_longitudinal .* recall_longitudinal) ./ ...
    (precision_longitudinal + recall_longitudinal);

% Handle undefined values
F1_combined(~isfinite(F1_combined)) = NaN;
F1_lateral(~isfinite(F1_lateral)) = NaN;
F1_longitudinal(~isfinite(F1_longitudinal)) = NaN;

%% Generate classification performance tables

% Combined behavior
Table_Combined = table( ...
    combined_names, ...
    precision_combined, ...
    recall_combined, ...
    F1_combined, ...
    'VariableNames', ...
    {'Behavior_Class', 'Precision', 'Recall', 'F1_Score'});

% Lateral behavior
Table_Lateral = table( ...
    lateral_names(:), ...
    precision_lateral, ...
    recall_lateral, ...
    F1_lateral, ...
    'VariableNames', ...
    {'Behavior_Class', 'Precision', 'Recall', 'F1_Score'});

% Longitudinal behavior
Table_Longitudinal = table( ...
    longitudinal_names(:), ...
    precision_longitudinal, ...
    recall_longitudinal, ...
    F1_longitudinal, ...
    'VariableNames', ...
    {'Behavior_Class', 'Precision', 'Recall', 'F1_Score'});


%% Display tables

disp('COMBINED BEHAVIOR');
disp(Table_Combined);

disp('LATERAL BEHAVIOR');
disp(Table_Lateral);

disp('LONGITUDINAL BEHAVIOR');
disp(Table_Longitudinal);


%% Macro averages

Macro_Combined = {
    'Macro Average'
    mean(precision_combined, 'omitnan')
    mean(recall_combined, 'omitnan')
    mean(F1_combined, 'omitnan')
};

Macro_Lateral = {
    'Macro Average'
    mean(precision_lateral, 'omitnan')
    mean(recall_lateral, 'omitnan')
    mean(F1_lateral, 'omitnan')
};

Macro_Longitudinal = {
    'Macro Average'
    mean(precision_longitudinal, 'omitnan')
    mean(recall_longitudinal, 'omitnan')
    mean(F1_longitudinal, 'omitnan')
};


%% Save tables to Excel

writetable(Table_Combined, 'Classification_Metrics.xlsx', ...
    'Sheet', 'Combined');

writetable(Table_Lateral, 'Classification_Metrics.xlsx', ...
    'Sheet', 'Lateral');

writetable(Table_Longitudinal, 'Classification_Metrics.xlsx', ...
    'Sheet', 'Longitudinal');
%% =========================================================
%  LOCAL FUNCTION 1: PRECISION AND RECALL
% ==========================================================

function [precision, recall] = calculate_metrics(C)

    TP = diag(C);

    % Precision = TP / (TP + FP)
    precision = TP ./ sum(C, 1)';

    % Recall = TP / (TP + FN)
    recall = TP ./ sum(C, 2);

    % Convert to percentages
    precision = precision * 100;
    recall = recall * 100;

    % Handle division by zero
    precision(~isfinite(precision)) = NaN;
    recall(~isfinite(recall)) = NaN;

end


%% =========================================================
%  LOCAL FUNCTION 2: MULTICLASS ROC
% ==========================================================

function plot_multiclass_ROC(actual_labels, probabilities, ...
                              class_names, plot_title)

    num_classes = size(probabilities, 2);

    actual_labels = actual_labels(:);

    if length(actual_labels) ~= size(probabilities, 1)
        error('Actual labels and probability dimensions do not match.');
    end

    if any(~isfinite(probabilities(:)))
        error('Probability matrix contains NaN or Inf values.');
    end

    % One-hot encoding
    actual_binary = zeros(length(actual_labels), num_classes);

    for k = 1:num_classes
        actual_binary(:, k) = (actual_labels == k);
    end

    % Common FPR grid for macro-average
    common_fpr = linspace(0, 1, 1001);

    interpolated_tpr = NaN(num_classes, length(common_fpr));

    class_auc = NaN(num_classes, 1);

    figure('Name', plot_title);

    hold on;
    grid on;
    box on;

    %% Individual class ROC curves

    for k = 1:num_classes

        positive_class = actual_binary(:, k);

        % Check positive and negative samples
        if ~any(positive_class) || all(positive_class)

            warning('Class %s lacks positive or negative samples.', ...
                class_names{k});

            continue;

        end

        [fpr, tpr, ~, auc_value] = perfcurve( ...
            positive_class, probabilities(:, k), 1);

        class_auc(k) = auc_value;

        plot(fpr, tpr, ...
            'LineWidth', 1.8, ...
            'DisplayName', sprintf('%s (AUC = %.4f)', ...
            class_names{k}, auc_value));

        % Interpolate for macro-average
        [unique_fpr, ia] = unique(fpr, 'stable');

        unique_tpr = tpr(ia);

        interpolated_tpr(k, :) = interp1( ...
            unique_fpr, unique_tpr, common_fpr, 'linear', NaN);

        % ROC endpoints
        interpolated_tpr(k, 1) = 0;
        interpolated_tpr(k, end) = 1;

    end


    %% Micro-average ROC

    actual_micro = actual_binary(:);
    score_micro = probabilities(:);

    [micro_fpr, micro_tpr, ~, micro_auc] = perfcurve( ...
        actual_micro, score_micro, 1);

    plot(micro_fpr, micro_tpr, 'k-', ...
        'LineWidth', 2.5, ...
        'DisplayName', sprintf('Micro-average (AUC = %.4f)', ...
        micro_auc));


    %% Macro-average ROC

    valid_classes = all(isfinite(interpolated_tpr), 2);

    if any(valid_classes)

        macro_tpr = mean(interpolated_tpr(valid_classes, :), 1);

        macro_auc = trapz(common_fpr, macro_tpr);

        plot(common_fpr, macro_tpr, 'r--', ...
            'LineWidth', 2.5, ...
            'DisplayName', sprintf('Macro-average (AUC = %.4f)', ...
            macro_auc));

    end


    %% Random classifier reference

    plot([0 1], [0 1], 'k:', ...
        'LineWidth', 1.5, ...
        'DisplayName', 'Random classifier');


    %% Formatting

    xlabel('False Positive Rate', 'FontSize', 12);

    ylabel('True Positive Rate', 'FontSize', 12);

    title(plot_title, ...
        'FontSize', 14, ...
        'FontWeight', 'bold');

    legend('Location', 'southeast', 'FontSize', 9);

    xlim([0 1]);
    ylim([0 1]);

    axis square;

    set(gca, 'FontSize', 11);

    hold off;

end
