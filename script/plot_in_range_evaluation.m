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
    0.27634, 0.35984, 1.1597, 0.00000000000078955;
    0.16189, 0.34263, 0.16139, 0.0000000000068844;
    0.18541, 0.3818, 0.18547, 0.000000000000033744;
];  % Potential (In-range)
DATA_3_ION(2,:,:) = [
    1.1545, 1.5671, 136410000000000, 0.000000000174;
    0.61385, 1.1925, 1.1925, 0.00000000018269;
    0.27076, 1.3556, 1.3556, 0.000000000174;
];  % Concentration (In-range)
DATA_3_ION(3,:,:) = [
    2.1668, 126.97, 2.3214, 0.0000000000012915;
    1.5857, 129.25, 1.6071, 0.000000000000014798;
    1.7388, 128.93, 1.7409, 0.00000000000011271;
];  % Drift
DATA_3_ION(4,:,:) = [
    0.00095815, 0.00095815, 0.0010264, 7.0083E-16;
    0.00068295, 0.00068295, 0.00069251, 0.000000000000014798;
    0.00075269, 0.00075269, 0.00075361, 5.3198E-17;
];  % Slope
DATA_3_ION(5,:,:) = [
    0.33422, 8367, 0.33428, 0.0000000000000010063;
    0.33422, 9613.4, 0.33422, 0.000000000000015819;
    69798000000, 9379, 0.33422, 4.5233e-17;
];  % Selectivity Coefficient
THRESHOLD_3_ION = [1e-3, 1e-3, 3.3421871124572866e-06];

DATA_5_ION = zeros(5,3,4);
DATA_5_ION(1,:,:) = [
    1.2255, 3.6914, 114.61, 0.00042978;
    1.4526, 4.5274, 11.473, 0.0004691;
    1.3239, 4.0709, 1.3, 0.00045742;
];  % Potential (In-range)
DATA_5_ION(2,:,:) = [
    26.788, 53.311, 5190200, 0.0027182;
    17.594, 54.858, 3191600000000000, 0.0022909;
    15.212, 1710200, 83019000000000, 0.00211;
];  % Concentration (In-range)
DATA_5_ION(3,:,:) = [
    13.69, 99.794, 14.367, 0.0040953;
    17.083, 99.854, 17.126, 0.0039526;
    16.442, 99.726, 16.446, 0.0034937;
];  % Drift
DATA_5_ION(4,:,:) = [
    0.0066705, 0.0066705, 0.0069805, 0.0000019657;
    0.0082646, 0.0082646, 0.0082839, 0.0000018717;
    0.0079364, 0.0079364, 0.0079385, 0.0000016368;
];  % Slope
DATA_5_ION(5,:,:) = [
    0.2405, 7266.3, 0.22942, 0.015232;
    0.23315, 6875.1, 0.24081, 0.013146;
    0.23357, 7080.2, 0.23285, 0.012247;
];  % Selectivity Coefficient
THRESHOLD_5_ION = [1e-3, 1e-3,2.388734730408741e-06];


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