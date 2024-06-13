%{
Script used to plot the performance of various solvers on the in-range datasets.

Author: Weixun Luo
Date: 06/05/2024
%}
clear variables;
close all;
clc;


%% Constant
DATA_3_ION = zeros(5,3,4);
DATA_3_ION(1,:,:) = [
    0.40597, 0.69904, 1404.4, 0.0000000000023894;
    0.37425, 1.0453, 37.93, 0.00000000031211;
    0.40146, 1.107, 0.67914, 0.0000000016224;
];  % Forward Accuracy
DATA_3_ION(2,:,:) = [
    1.1545, 1.5671, 905510, 0.00000000016895;
    0.61385, 1.1925, 469380000000000, 0.00000000019065;
    0.27086, 1.3556, 3453000000000, 0.000000001155;
];  % Backward Accuracy
DATA_3_ION(3,:,:) = [
    5.4088, 334.57, 5.4205, 0.000000000078961;
    4.1737, 338.26, 4.1741, 0.00000000019036;
    4.5251, 337.83, 4.5252, 0.0000000029203;
];  % Drift
DATA_3_ION(4,:,:) = [
    0.013648, 0.013648, 0.013682, 0.00000000000023439;
    0.0096315, 0.0096315, 0.0096329, 0.00000000000055887;
    0.010712, 0.010712, 0.010712, 0.0000000000083175;
];  % Slope
DATA_3_ION(5,:,:) = [
    0.33422, 8367, 0.33428, 0.000000000000013163;
    0.33422, 9613.4, 0.33422, 0.00000000000004523;
    69798000000, 9379, 0.33422, 0.00000000000080194;
];  % Selectivity Coefficient
THRESHOLD_3_ION = [1e-3, 1e-3, 3.3421871124572866e-06];

DATA_5_ION = zeros(5,3,4);
DATA_5_ION(1,:,:) = [
    0.59323, 2.3609, 41.434, 0.010875;
    0.48863, 1.7489, 3.3998e+00, 0.024032;
    0.4957, 1.5578, 0.49688, 0.0092015;
];  % Forward Accuracy
DATA_5_ION(2,:,:) = [
    19.749, 97.575, 8295100, 0.11283;
    12.437, 60.762, 2806000000000000, 0.2383;
    14.011, 54.217, 21151000000000, 0.10162;
];  % Backward Accuracy
DATA_5_ION(3,:,:) = [
    101.48, 935.6, 110.05, 2.6218;
    108.18, 932.59, 108.67, 2.4848;
    115.27, 933.66, 115.31, 2.0182;
];  % Drift
DATA_5_ION(4,:,:) = [
    0.17667, 0.17667, 0.19182, 0.0011877;
    0.18333, 0.18333, 0.18417, 0.00079619;
    0.19702, 0.19702, 0.1971, 0.0009651;
];  % Slope
DATA_5_ION(5,:,:) = [
    0.23066, 6230.3, 0.23007, 0.012557;
    0.23309, 6694.6, 0.23378, 0.012819;
    0.2333, 7062.1, 0.23262, 0.013247;
];  % Selectivity Coefficient
THRESHOLD_5_ION = [1e-3, 1e-3, 2.388734730408741e-06];


%% Data
data = -1 * pow2db(DATA_5_ION);
threshold = -1 * pow2db(THRESHOLD_5_ION);
category = categorical({'10','100','1000'});


%% Plot
COLOR_PALETTE = [
    130, 176, 210;
    250, 127, 111;
    142, 207, 201;
    255, 190, 122;
    190, 184, 220;
    153, 153, 153;
] / 256;
THRESHOLD_LINE_STYLE = '--';
THRESHOLD_LINE_WIDTH = 1.5;
TICK_LABEL_FONT_SIZE = 12;
LABEL_FONT_SIZE = 16;
LEGEND_FONT_SIZE = 12;
TITLE_FONT_SIZE = 16;
FONT_NAME = 'Times New Roman';

titles = {'Forward Accuracy', 'Backward Accuracy', 'Response Intercept', 'Response Slope', 'Selectivity Coefficient'};

fig = figure;
set(fig, 'Units', 'inches', 'Position', [1, 1, 8.27, 6]);  % Adjust the figure size to fit an A4 paper width (8.27 inches)

% Adjust subplot position and size to create a two-row layout
subplot_positions = [
    0.08, 0.55, 0.4, 0.35;   % First row, first column
    0.53, 0.55, 0.4, 0.35;   % First row, second column
    0.08, 0.1, 0.25, 0.35;   % Second row, first column
    0.38, 0.1, 0.25, 0.35;   % Second row, second column
    0.68, 0.1, 0.25, 0.35    % Second row, third column
];

plot_handler_all = cell(1, 2);
for i = 1:5
    axis = subplot('Position', subplot_positions(i, :));
    hold on;
    plot_handler_all{1} = bar( ...
        category, squeeze(data(i, :, :)), ...
        'EdgeColor', 'none');
    colororder(axis, COLOR_PALETTE);
    if i >= 3
        plot_handler_all{2} = yline( ...
            threshold(i-2), ...
            'Color', COLOR_PALETTE(6, :), ...
            'LineStyle', THRESHOLD_LINE_STYLE, 'LineWidth', THRESHOLD_LINE_WIDTH ...
        );
    end
    grid minor;
    axis.XAxis.FontName = FONT_NAME;
    axis.YAxis.FontName = FONT_NAME;
    axis.XAxis.FontSize = TICK_LABEL_FONT_SIZE;
    axis.YAxis.FontSize = TICK_LABEL_FONT_SIZE;
    title(titles{i}, 'FontName', FONT_NAME, 'FontSize', TITLE_FONT_SIZE);
    if i == 4
        xlabel('#Training Sample', ...
               'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
    end
end

% Add a shared y-label for the entire figure
han = axes(fig, 'visible', 'off');
han.YLabel.Visible = 'on';
ylabel(han, 'Score (dB)', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
han.Position = [0.07, 0.07, 0, 0.87]; % Position the y-label marginally to the left

% Adjust the legend position to be in the top margin
lgd = legend( ...
    [plot_handler_all{:}], {'OLS', 'PLS', 'BR', 'NESolver', 'Threshold'}, ...
    'Position', [0.5, 0.97, 0, 0], 'Orientation', 'horizontal', ...
    'FontName', FONT_NAME, 'FontSize', LEGEND_FONT_SIZE ...
);
set(lgd, 'Box', 'off');  % Remove the border around the legend