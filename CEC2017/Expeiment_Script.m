clc;
clear;
close all;

%% ==================== 参数设置 ====================
% Func_IDs = [1 3 5 10 14 15 20 24 26]; 
Func_IDs = [1]; % 需要测试的函数编号
dim = 30;                               % 问题维度
nPop = 30;                              % 种群大小
MaxFEs = 30000;                         % 最大函数评价次数
Run_Times = 30;                         % 独立运行次数

%% ==================== 算法列表 ====================
Algorithm_Names = {'TGWO',  'HSO'};
nAlgorithms = length(Algorithm_Names);

%% ==================== 结果目录 ====================
Result_Folder = fullfile(pwd, '2026-4-14');
if ~exist(Result_Folder, 'dir')
    mkdir(Result_Folder);
end

Max_Curve_Length = ceil(MaxFEs / nPop);

%% ==================== 开始实验（仅输出收敛曲线） ====================
for func_idx = 1:length(Func_IDs)
    Func_ID = Func_IDs(func_idx);
    [lb, ub, dim, fobj] = Get_Functions_cec2017(Func_ID, dim);

    Convergence_All = cell(nAlgorithms, 1);
    for alg = 1:nAlgorithms
        Convergence_All{alg} = zeros(Run_Times, Max_Curve_Length);
    end

    for run = 1:Run_Times
        for alg = 1:nAlgorithms
            alg_name = Algorithm_Names{alg};

            switch alg_name
                case 'TGWO'
                    [~, ~, curve] = TGWO(nPop, MaxFEs, lb, ub, dim, fobj);
              
                case 'HSO'
                    [~, ~, curve] = HSO(nPop, MaxFEs, lb, ub, dim, fobj);
            end
            curve_len = length(curve);
            if curve_len > Max_Curve_Length
                idx = round(linspace(1, curve_len, Max_Curve_Length));
                curve = curve(idx);
                curve_len = Max_Curve_Length;
            end
            if curve_len < Max_Curve_Length
                curve = [curve, repmat(curve(end), 1, Max_Curve_Length - curve_len)];
            end

            Convergence_All{alg}(run, :) = curve;
        end
    end

    %% ==================== 绘制并保存收敛曲线 ====================
    figure('Position', [100, 100, 900, 600], 'Color', 'w');

    colors = [
        1.0, 0.0, 0.0;
        0.0, 0.5, 0.0;
        0.0, 0.0, 1.0;
        0.0, 0.8, 0.8;
        0.6, 0.6, 0.6;
        1.0, 0.5, 0.0;
        0.8, 0.0, 0.8;
        0.4, 0.2, 0.0;
        0.0, 0.6, 0.3;
        0.5, 0.5, 0.0;
        0.0, 0.4, 0.8;
        0.7, 0.3, 0.7;
        0.9, 0.6, 0.0
    ];
    line_styles = {'-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':', '--'};
    line_widths = [2, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5];

    hold on;
    for alg = 1:nAlgorithms
        mean_curve = mean(Convergence_All{alg}, 1);
        FEs_axis = (1:length(mean_curve)) * nPop;

        valid_idx = FEs_axis <= MaxFEs;
        FEs_axis = FEs_axis(valid_idx);
        mean_curve = mean_curve(valid_idx);

        plot(FEs_axis, mean_curve, ...
            'LineWidth', line_widths(alg), ...
            'Color', colors(alg, :), ...
            'LineStyle', line_styles{alg});
    end
    hold off;

    xlabel('FEs', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('Fitness', 'FontSize', 16, 'FontWeight', 'bold');
    title(sprintf('CEC2017 F%d (Dim=%d)', Func_ID, dim), 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'YScale', 'log', 'FontWeight', 'bold', 'LineWidth', 1.2, ...
        'Color', 'w', 'XColor', 'k', 'YColor', 'k');

    % 不显示图例
    grid off;
    box on;

    saveas(gcf, fullfile(Result_Folder, sprintf('CEC2017_F%d_Dim%d_Convergence.png', Func_ID, dim)));
    savefig(gcf, fullfile(Result_Folder, sprintf('CEC2017_F%d_Dim%d_Convergence.fig', Func_ID, dim)));
    close(gcf);
end
