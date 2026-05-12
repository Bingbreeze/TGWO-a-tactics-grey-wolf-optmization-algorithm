% Parameter Sensitivity Experiment Script for TGWO
% Tests the sensitivity of parameters k and c on CEC2017 functions
clear all;
close all;
clc;

% ==================== Experiment Settings ====================
Func_IDs = [10 15 20 21 23 ];      % Run F1-F30
dim = 50;             % Dimension
nPop = 30;            % Population size
MaxFEs = 1000 * dim;  % Maximum function evaluations
Runs = 30;            % Number of independent runs (建议正式发论文时改为 30)

% Parameter ranges to test
k_values = [15];                  % Fixed k=10
c_values = [0.6];                 % Fixed c=0.8
p_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7]; % Elite Proportion to test

% Create results directory
Result_Folder = 'Parameter_Sensitivity_Results_Elite';
if ~exist(Result_Folder, 'dir')
    mkdir(Result_Folder);
end

% 定义最终文件名
Final_Excel_Filename = fullfile(Result_Folder, 'Parameter_Sensitivity_Results_All_Elite.xlsx');

% 【关键修改】运行前检查并删除旧的 Excel 文件，防止数据混杂
if exist(Final_Excel_Filename, 'file')
    try
        delete(Final_Excel_Filename);
        fprintf('Old results file deleted: %s\n', Final_Excel_Filename);
    catch
        fprintf('Warning: Could not delete old file. Please close Excel if it is open.\n');
    end
end

% Initialize global storage for Excel data
All_Excel_Data = {'Func_ID', 'k', 'c', 'p', 'Mean', 'Std', 'Friedman_Rank'};

% ==================== Main Loop ====================
for f_idx = 1:length(Func_IDs)
    Func_ID = Func_IDs(f_idx);
    
    fprintf('\n========================================\n');
    fprintf('Processing Function F%d [%d/%d]\n', Func_ID, f_idx, length(Func_IDs));
    fprintf('========================================\n');
    
    try
        [lb, ub, dim, fobj] = Get_Functions_cec2017(Func_ID, dim);
    catch ME
        fprintf('Error loading F%d: %s. Skipping.\n', Func_ID, ME.message);
        continue;
    end
    
    % ==================== Initialization ====================
    num_k = length(k_values);
    num_c = length(c_values);
    num_p = length(p_values);
    num_combinations = num_k * num_c * num_p;
    
    % Store all run scores for Friedman test
    % Rows: Runs, Cols: Combinations (k, c, p)
    All_Run_Scores = zeros(Runs, num_combinations);
    Combination_Labels = cell(1, num_combinations);
    Combination_Params = zeros(num_combinations, 3); % [k, c, p]
    
    Results_Mean = zeros(num_k, num_c, num_p);
    Results_Std = zeros(num_k, num_c, num_p);
    
    fprintf('Running Parameter Sensitivity Experiment for TGWO on F%d (Dim=%d)...\n', Func_ID, dim);
    
    total_start_time = tic;
    
    comb_idx = 0;
    for i = 1:num_k
        for j = 1:num_c
            for m = 1:num_p
                comb_idx = comb_idx + 1;
                current_k = k_values(i);
                current_c = c_values(j);
                current_p = p_values(m);
                
                Combination_Params(comb_idx, :) = [current_k, current_c, current_p];
                Combination_Labels{comb_idx} = sprintf('k=%d, c=%.1f, p=%.2f', current_k, current_c, current_p);
                
                Run_Scores = zeros(1, Runs);
                
                for r = 1:Runs
                    % Run TGWO with specific k, c, p
                    [Alpha_score, ~, ~] = TGWO(nPop, MaxFEs, lb, ub, dim, fobj, current_k, current_c, current_p);
                    Run_Scores(r) = Alpha_score;
                end
                
                All_Run_Scores(:, comb_idx) = Run_Scores';
                
                % Store results in 3D matrix
                Results_Mean(i, j, m) = mean(Run_Scores);
                Results_Std(i, j, m) = std(Run_Scores);
            end
        end
    end
    
    total_time = toc(total_start_time);
    fprintf('F%d Experiment completed in %.2f seconds.\n', Func_ID, total_time);
    
    % ==================== Friedman Ranking (Per Function) ====================
    Ranks = zeros(Runs, num_combinations);
    for r = 1:Runs
        current_scores = All_Run_Scores(r, :);
        % 使用 tiedrank 处理排名 (处理平局情况)
        % 如果没有统计工具箱，使用简单的排序替代
        try
            Ranks(r, :) = tiedrank(current_scores);
        catch
            [~, ~, rank_vals] = unique(current_scores);
            Ranks(r, :) = rank_vals'; 
        end
    end
    
    Friedman_Mean_Ranks = mean(Ranks, 1);
    
    % Prepare data for Excel
    for idx = 1:num_combinations
        k_val = Combination_Params(idx, 1);
        c_val = Combination_Params(idx, 2);
        p_val = Combination_Params(idx, 3);
        
        % 反向获取 i 和 j 和 m 的索引
        r_idx = find(abs(k_values - k_val) < 1e-9, 1);
        c_idx = find(abs(c_values - c_val) < 1e-9, 1);
        p_idx = find(abs(p_values - p_val) < 1e-9, 1);
        
        % Add row to global data
        new_row = {Func_ID, k_val, c_val, p_val, ...
            Results_Mean(r_idx, c_idx, p_idx), ...
            Results_Std(r_idx, c_idx, p_idx), ...
            Friedman_Mean_Ranks(idx)};
            
        All_Excel_Data = [All_Excel_Data; new_row];
    end
    
    % Save individual function results backup (Optional)
    save_path_mat = fullfile(Result_Folder, sprintf('Parameter_Sensitivity_Results_F%d.mat', Func_ID));
    save(save_path_mat, 'Results_Mean', 'Results_Std', 'k_values', 'c_values', 'p_values', 'Func_ID', 'dim');
    
    clear Results_Mean Results_Std Run_Scores All_Run_Scores Ranks
    
end

% Save aggregated results to a single Excel file
fprintf('\nWriting all results to Excel...\n');
writecell(All_Excel_Data, Final_Excel_Filename);
fprintf('All results saved to: %s\n', Final_Excel_Filename);


%% ==================== New Section: Overall Analysis & Boxplot ====================
% 这一部分会读取刚刚生成的数据，绘制所有函数汇总的箱型图

fprintf('\nStarting Overall Analysis for Parameter p (Elite Proportion)...\n');

% 1. 将保存的 Excel 数据转换为矩阵 (去除表头)
Raw_Data = cell2mat(All_Excel_Data(2:end, :)); 

% 提取列: [Func_ID, k, c, p, Mean, Std, Friedman_Rank]
Col_p    = Raw_Data(:, 4);
Col_Rank = Raw_Data(:, 7); % 提取每一行的 Friedman Rank

% 2. 准备箱型图数据
unique_p_list = sort(unique(Col_p));
num_p_types = length(unique_p_list);
% 初始化矩阵：行=函数数量，列=参数种类
Rank_Matrix = NaN(length(Func_IDs), num_p_types); 

for i = 1:num_p_types
    target_p = unique_p_list(i);
    indices = (abs(Col_p - target_p) < 1e-9); % 找出所有等于当前p的行
    ranks = Col_Rank(indices);
    
    % 填充矩阵 (使用 1:length 防止数据长度不一致导致报错)
    if length(ranks) <= length(Func_IDs)
        Rank_Matrix(1:length(ranks), i) = ranks;
    end
end

% 3. 打印统计结果
Mean_Friedman_Ranks = mean(Rank_Matrix, 1, 'omitnan'); % 忽略NaN求均值
[~, best_idx] = min(Mean_Friedman_Ranks);

fprintf('\n--- Final Statistical Results (Overall Friedman Rank) ---\n');
for i = 1:num_p_types
    fprintf('p = %.2f : Average Rank = %.4f\n', unique_p_list(i), Mean_Friedman_Ranks(i));
end
fprintf('>>> Best parameter is p = %.2f (Lowest Rank) <<<\n', unique_p_list(best_idx));

% 4. 绘制箱型图
figure('Position', [300, 300, 800, 600]);
boxplot(Rank_Matrix, 'Labels', num2cell(unique_p_list));
xlabel('Parameter p (Elite Proportion)');
ylabel('Friedman Rank (Lower is Better)');
title('Overall Friedman Rank Distribution across all Functions');
grid on;

% 保存最终的箱型图
save_path_overall = fullfile(Result_Folder, 'Overall_Parameter_Sensitivity_Boxplot_Elite.png');
saveas(gcf, save_path_overall);
fprintf('Overall Boxplot saved to %s\n', save_path_overall);

fprintf('\nDone.\n');