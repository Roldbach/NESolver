%{
Script used to plot the performance of vanilla/novel solvers on the out-range
dataset.

Author: Weixun Luo
Date: 07/05/2024
%}
clear variables;
close all;
clc;


%% Constant
DATA_3_ION = zeros(3,4,2);
DATA_3_ION(1,:,:) = [
    1.1545, 8.9822;
    1.5671, 9.2452;
    136410000000000, 1181300;
    0.000000000174, 0.000000000057744;
];  % Concentration (10)
DATA_3_ION(2,:,:) = [
    0.61385, 6.9672;
    1.1925, 6.1165;
    150590000000, 98.328;
    0.00000000018269, 0.00000000017036;
];  % Concentration (100)
DATA_3_ION(3,:,:) = [
    0.27076, 5.7864;
    1.3556, 6.7807;
    805590000, 2621300000000;
    0.000000000174, 0.000000000054064;
];  % Concentration (1000)

DATA_5_ION = zeros(3,4,2);
DATA_5_ION(1,:,:) = [
    26.788, 50590;
    53.311, 18171;
    5190200, 15522000000;
    0.0027182, 0.13645;
];  % Concentration (10)
DATA_5_ION(2,:,:) = [
    17.594, 4578.9;
    54.858, 3662.6;
    3191600000000000, 504710000000000000;
    0.0022909, 0.11604;
];  % Concentration (100)
DATA_5_ION(3,:,:) = [
    15.212, 1938.7;
    37.195, 1247.2;
    83019000000000, 10671000000000000;
    0.00211, 0.10477;
];  % Concentration (1000)


%% Data
data = -1.0 * pow2db(DATA_5_ION);
category = categorical({'OLS','PLS','BR', 'NESolver'});
category = reordercats(category,{'OLS','PLS','BR', 'NESolver'});


%% Plot
COLOR_PALETTE = [
    130, 176, 210;
    250, 127, 111;
    142, 207, 201;
    190, 184, 220;
    255, 190, 122;
    153, 153, 153;
] / 256;
TICK_LABEL_FONT_SIZE = 12;
LABEL_FONT_SIZE = 16;
LEGEND_FONT_SIZE = 16;
TITLE_FONT_SIZE = 12;
FONT_NAME = 'Times New Roman';

figure;
set(gcf, 'Units', 'inches', 'Position', [1, 1, 8.27, 6]);  % Set the figure size to fit an A4 paper width (8.27 inches)

% Adjust subplot position and size to create square subplots
subplot_positions = [
    0.1, 0.55, 0.2, 0.35;   % First row, first column
    0.4, 0.55, 0.2, 0.35;   % First row, second column
    0.7, 0.55, 0.2, 0.35    % First row, third column
];

plot_handler_all = cell(1, 1);
TITLE = {
    '#Training Sample = 10', ...
    '#Training Sample = 100', ...
    '#Training Sample = 1000',
};
for i = 1:3
    axis = subplot('Position', subplot_positions(i, :));
    hold on;
    plot_handler_all{1} = bar( ...
        category, squeeze(data(i,:,:)), ...
        'EdgeColor', 'none');
    colororder(axis, COLOR_PALETTE);
    hold off;
    grid minor;
    axis.XAxis.FontName = FONT_NAME;
    axis.YAxis.FontName = FONT_NAME;
    axis.XAxis.FontSize = TICK_LABEL_FONT_SIZE;
    axis.YAxis.FontSize = TICK_LABEL_FONT_SIZE;
    title(TITLE{i}, 'FontName', FONT_NAME, 'FontSize', TITLE_FONT_SIZE);
    pbaspect([1 1 1]);  % Set the aspect ratio to be square
    if i == 1
        ylabel('Score (dB)', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
    end
    if i == 2
        xlabel('Numerical Method', 'FontName', FONT_NAME, 'FontSize', LABEL_FONT_SIZE);
    end
end

lgd = legend( ...
    [plot_handler_all{:}], {'Intrapolability', 'Extrapolability'}, ...
    'Position', [0.5, 0.94, 0, 0], 'Orientation', 'horizontal', ...
    'FontName', FONT_NAME, 'FontSize', LEGEND_FONT_SIZE ...
);
set(lgd, 'Box', 'off');  % Remove the border around the legend
set(lgd, 'Color', 'white');  % Ensure the legend background is white