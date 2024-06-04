%{
Script used to plot the performance of various solvers in the ablation study.

Author: Weixun Luo
Date: 06/05/2024
%}
clear variables;
close all;
clc;


%% Constant
DATA_3_ION = zeros(6,3,5);
DATA_3_ION(1,:,:) = [
    0.28576, 0.36617, 0.24577, 0.00087428, 0.0000000000000010063;
    0.25322, 0.35112, 0.21767, 0.00079668, 0.000000000000015819;
    0.25788, 0.21909, 0.21772, 0.00079529, 4.5233e-17;
];  % Selectivity Coefficient
DATA_3_ION(2,:,:) = [
    0.00000037537, 0.000002042, 0.00000054135, 0.0000000023548, 7.0083E-16;
    0.00000067186, 0.00000028592, 0.00000018148, 0.0000000020211, 0.000000000000014798;
    0.00000017557, 0.000000084508, 0.00000017118, 0.0000000019211, 5.3198E-17;
];  % Slope
DATA_3_ION(3,:,:) = [
    13.288, 10.234, 3.8, 0.017811, 0.0000000000012915;
    14.53, 10.276, 3.4399, 0.016209, 0.000000000033025;
    6.7256, 8.15, 3.479, 0.016013, 0.00000000000011271;
];  % Drift
DATA_3_ION(4,:,:) = [
    0.00019743, 0.00028846, 0.00023045, 0.0000007021, 0.00000000000078955;
    0.000036279, 0.00010888, 0.000039655, 0.00000060649, 0.0000000000068844;
    0.000038219, 0.000029049, 0.000039802, 0.00000065652, 0.000000000000033744;
];  % Potential (In-range)
DATA_3_ION(5,:,:) = [
    0.00063897, 0.0072015, 0.0010746, 0.0000015957, 0.0000000000019496;
    0.00015075, 0.00048527, 0.00024479, 0.0000013593, 0.000000000015001;
    0.00016154, 0.00010656, 0.00023838, 0.000001442, 0.00000000000011835;
];  % Activity (In-range)
DATA_3_ION(6,:,:) = [
    0.00064047, 0.0072007, 0.0010748, 0.0000016022, 0.000000000174;
    0.00015131, 0.00048556, 0.00024503, 0.0000013634, 0.00000000018269;
    0.00016168, 0.00010653, 0.00023859, 0.0000014448, 0.000000000174;
];  % Concentration (In-range)
THRESHOLD_3_ION = [3.3421871124572866e-06, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];

DATA_5_ION = zeros(6,3,5);
DATA_5_ION(1,:,:) = [
    0.42468, 0.50215, 0.5908, 0.056568, 0.015232;
    0.49239, 0.55074, 0.63957, 0.051413, 0.013146;
    0.44812, 0.59826, 0.68613, 0.052332, 0.012247;
];  % Selectivity Coefficient
DATA_5_ION(2,:,:) = [
    0.038821, 0.046895, 0.010104, 0.0000064061, 0.0000019657;
    0.038937, 0.047701, 0.006299, 0.0000018774, 0.0000018717;
    0.026647, 0.039641, 0.0047074, 0.0000020308, 0.0000016368;
];  % Slope
DATA_5_ION(3,:,:) = [
    75.136, 81.376, 40.147, 1.0739, 0.0040953;
    66.682, 83.461, 34.341, 0.91744, 0.0039526;
    57.516, 75.088, 29.216, 0.94507, 0.0034937;
];  % Drift
DATA_5_ION(4,:,:) = [
    10.403, 12.04, 10.201, 0.0021005, 0.00042978;
    10.136, 12.506, 9.0232, 0.00073384, 0.0004691;
    10.08, 12.997, 8.8678, 0.00085295, 0.00045742;
];  % Potential (In-range)
DATA_5_ION(5,:,:) = [
    372850000, 2314300000000000000, 2533.6, 0.014575, 0.0027115;
    196460, 581630000000000000, 2755.5, 0.0087003, 0.0022844;
    106480000, 1746200, 4009.1, 0.010074, 0.0021055;
];  % Activity (In-range)
DATA_5_ION(6,:,:) = [
    365680000, 2271500000000000000, 2648.1, 0.014657, 0.0027182;
    202580, 567250000000000000, 2819.4, 0.0087705, 0.0022909;
    104830000, 1710200, 4196.4, 0.010157, 0.00211;
];  % Concentration (In-range)
THRESHOLD_5_ION = [2.388734730408741e-06, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3];


%% Data
data = -1 * pow2db(DATA_3_ION);
threshold = -1 * pow2db(THRESHOLD_3_ION);
category = categorical({'10','100','1000'});


%% Plot
COLOR_PALETTE = [
    130, 176, 210;
    250, 127, 111;
    142, 207, 201;
    190, 184, 220;
    255, 190, 122;
    153, 153, 153;
] / 256;
THRESHOLD_LINE_STYLE = '--';
THRESHOLD_LINE_WIDTH = 1.5;
TICK_LABEL_FONT_SIZE = 12;
LABEL_FONT_SIZE = 16;
LEGEND_FONT_SIZE = 12;
FONT_NAME = 'Times New Roman';

figure;
plot_handler_all = cell(1,2);
for i = 1:6
    axis = subplot(2,3,i);
    hold on;
    plot_handler_all{1} = bar( ...
        category, squeeze(data(i,:,:)), ...
        'EdgeColor', 'none');
    colororder(axis, COLOR_PALETTE);
    plot_handler_all{2} = yline( ...
        threshold(i), ...
        'COLOR', COLOR_PALETTE(6,:), ...
        'LineStyle', THRESHOLD_LINE_STYLE, 'LineWidth', THRESHOLD_LINE_WIDTH ...
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
set(y_label, 'Position', get(y_label,'Position')-[10,-70,0]);
legend( ...
    [plot_handler_all{:}], {'Vanilla', 'Vanilla (1)', 'Vanilla(2)', 'Vanilla(3)', 'Novel', 'Threshold'}, ...
    'Position', [0.5,0.97,0,0], 'Orientation', 'horizontal', ...
    'FontName', FONT_NAME, 'FontSize', LEGEND_FONT_SIZE ...
);