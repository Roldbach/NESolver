%{
Script used to plot the performance of vanilla/novel solvers on datasets with
various sizes.

Author: Weixun Luo
Date: 09/05/2024
%}
clear variables;
close all;
clc;


%% Constant
DATA_3_ION_FILE_PATH = '../data/performance_vs_training_sample_size_Na-K-Cl.mat';
THRESHOLD_3_ION = [3.3421871124572866e-06, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];

DATA_5_ION_FILE_PATH = '../data/performance_vs_training_sample_size_Na-K-Mg-Ca-Cl.mat';
THRESHOLD_5_ION = [2.388734730408741e-06, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];


%% Data
load(DATA_3_ION_FILE_PATH, 'vanilla', 'novel');
data_vanilla = -1 * pow2db(vanilla);
data_novel = -1 * pow2db(novel);
threshold = -1 * pow2db(THRESHOLD_5_ION);
training_sample_size = 5:999;


%% Plot
COLOR_PALETTE = [
    130, 176, 210;
    250, 127, 111;
    153, 153, 153;
] / 256;
LINE_STYLE_ALL = {'-', '-.', '--'};
LINE_WIDTH = 2;
TICK_LABEL_FONT_SIZE = 12;
LABEL_FONT_SIZE = 16;
LEGEND_FONT_SIZE = 16;
FONT_NAME = 'Times New Roman';

figure;
plot_handler_all = cell(1,3);
for i = 1:6
    axis = subplot(2,3,i);
    hold on;
    plot_handler_all{1} = plot( ...
        training_sample_size, data_vanilla(:,i), ...
        'Color', COLOR_PALETTE(1,:), ...
        'LineStyle', LINE_STYLE_ALL{1}, 'LineWidth', LINE_WIDTH ...
    );
    plot_handler_all{2} = plot( ...
        training_sample_size, data_novel(:,i), ...
        'Color', COLOR_PALETTE(2,:), ...
        'LineStyle', LINE_STYLE_ALL{2}, 'LineWidth', LINE_WIDTH ...
    );
    plot_handler_all{3} = yline( ...
        threshold(i), ...
        'COLOR', COLOR_PALETTE(3,:), ...
        'LineStyle', LINE_STYLE_ALL{3}, 'LineWidth', LINE_WIDTH ...
    );
    grid minor;
    axis.XAxis.FontName = FONT_NAME;
    axis.YAxis.FontName = FONT_NAME;
    axis.XAxis.FontSize = TICK_LABEL_FONT_SIZE;
    axis.YAxis.FontSize = TICK_LABEL_FONT_SIZE;
    if i == 5
        xlabel('Training Sample Number (a.u.)', ...
               'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
    end
end
y_label = ylabel('Score (dB)', ...
    'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
set(y_label, 'Position', get(y_label,'Position')-[0,0,0]);
legend( ...
    [plot_handler_all{:}], {'Vanilla', 'Novel', 'Threshold'}, ...
    'Position', [0.5,0.97,0,0], 'Orientation', 'horizontal', ...
    'FontName', FONT_NAME, 'FontSize', LEGEND_FONT_SIZE ...
);