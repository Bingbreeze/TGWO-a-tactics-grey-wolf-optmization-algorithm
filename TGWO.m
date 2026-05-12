function [Alpha_score, Alpha_pos, Convergence_curve] = TGWO(nPop, MaxFEs, lb, ub, dim, fobj, varargin)
    % ==================== 1. Initialization ====================
    % Default parameters
    para_k = 15;
    para_c = 0.5;
    para_p = 0.2; % Default Elite Proportion
    
    % Parse optional parameters
    if ~isempty(varargin)
        if length(varargin) >= 1, para_k = varargin{1}; end
        if length(varargin) >= 2, para_c = varargin{2}; end
        if length(varargin) >= 3, para_p = varargin{3}; end
    end
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    
    Positions = lb + (ub - lb) .* rand(nPop, dim);
    Fitness = zeros(nPop, 1);
    
    FEs = 0;
    Convergence_curve = zeros(1, MaxFEs); 
    
    for i = 1:nPop
        Fitness(i) = fobj(Positions(i, :));
        FEs = FEs + 1;
        if i == 1
            Record_Best_Score = Fitness(i);
        else
            if Fitness(i) < Record_Best_Score
                Record_Best_Score = Fitness(i);
            end
        end
        Convergence_curve(FEs) = Record_Best_Score;
    end
    
    [Fitness, sort_idx] = sort(Fitness);
    Positions = Positions(sort_idx, :);
    
    Alpha_pos = Positions(1, :); Alpha_score = Fitness(1);
    Beta_pos  = Positions(2, :);
    Delta_pos = Positions(3, :);
    
    % ==================== 2. Main Loop ====================
    while FEs < MaxFEs
        
        progress = FEs / MaxFEs;
        
        Energy = cos(pi/2 * progress);
        % Sigmoid 收敛因子
        k = para_k;
        c = para_c;
        Prob_SRA = 1 ./ (1 + exp(k * (progress - c)));
        % a = 2 .* Sigmoid_Factor;
      a=(1 - progress) * 2; 
        Exploration_Leaders = (Alpha_pos + Beta_pos + Delta_pos) / 3;
       % Prob_SRA = (1 - progress) * 2; 
       
        for i = 1:nPop
            if FEs >= MaxFEs, break; end
            
            % =======================================================
            % [分支 A: 先锋侦察 - 优势导向  柯西变异]
            % =======================================================
            if rand() < Prob_SRA
                Target_Anchor = Exploration_Leaders;
               
                idx_r1 = randi(nPop); idx_r2 = randi(nPop);
                if Fitness(idx_r1) < Fitness(idx_r2)
                    Pos_Better = Positions(idx_r1, :); Pos_Worse  = Positions(idx_r2, :);
                else
                    Pos_Better = Positions(idx_r2, :); Pos_Worse  = Positions(idx_r1, :);
                end
                Cauchy_Scale = 0.5.* tan(pi * 0.9 * (rand(1, dim) - 0.5));
                Diff_Vector = (Pos_Better - Pos_Worse) .* Cauchy_Scale;
                Perturbed_Component = Positions(i, :) + Diff_Vector;
              % =======================================================
              % =======================================================
                % [策略修正] 双层级精英协同策略 (Two-Stage Elite Synergy)
                % =======================================================
                
                % 1. 定义阶层分界线 (前 para_p 比例为精英)
                Num_Elites = round(nPop * para_p);
                
                % =======================================================
                % [阶段一] 精英组 (Elites): 自身保留 + 自适应柯西微调
                % 逻辑：X_New = Self + Cauchy * (Alpha - Self)
                % 特点：不抛弃自身，利用与最优解的距离来控制步长
                % =======================================================
                if i <= Num_Elites
                    
                    % 计算自适应步长控制向量 (距离 Alpha 越近，步长越小)
                    % 如果是 Alpha 本身(i=1)，为了防止静止，让它参考 Beta 或做极小震荡
                    if i == 1
                        Dist_To_Best = (Beta_pos - Alpha_pos) * 0.1; 
                    else
                        Dist_To_Best = Alpha_pos - Positions(i, :);
                    end
                    
                    % 柯西变异算子 (不做 0.9 截断，保留长尾能力，因为有 Dist 拉住)
                    Cauchy_Val = tan(pi * (rand(1, dim) - 0.5));
                    
                    % 施加变异 (Energy 是能量衰减因子，建议加上以保证后期稳定)
                    % 0.5 是基础比例
                    Mutation_Step = 0.5 * Cauchy_Val .* Dist_To_Best;
                    
                    % 直接叠加 (Additive Update)
                    X_New = Positions(i, :) + Mutation_Step;
                    
                % =======================================================
                % [阶段二] 探索组 (Explorers): 激进的三元维度交叉
                % 逻辑：放弃自身位置，在"领袖"和"柯西变异"之间做维度选择
                % 特点：探索力度大，负责在大范围内寻找更好解
                % =======================================================
                else
                    % 对于后 80% 的个体，我们认为它们的位置不好，不需要"保留自我"
                    % 只需要在"跟班(Leader)"和"捣乱(Mutant)"之间选
                    
                    % 动态概率：后期多跟班，前期多捣乱
                    P_Leader = 0.4 + 0.4 * progress; 
                    % P_Mutant = 1 - P_Leader; (剩下的概率给变异)
                    
                    % 生成掩码
                    Rand_Prob = rand(1, dim);
                    
                    % 掩码逻辑 (互斥)
                    % 这里 P_Self 设为 0 (或者极低)，强制它们动起来
                    Mask_Leader = Rand_Prob < P_Leader;
                    Mask_Mutant = Rand_Prob >= P_Leader;
                    
                    % 组装部件
                    % 1. Leader 部分
                    Part_Leader = Target_Anchor; 
                    
                    % 2. Mutant 部分: 使用前面计算好的柯西变异体
                    Part_Mutant = Perturbed_Component;
                    
                    % 维度交叉组装 (Crossover Assembly)
                    X_New = Mask_Leader .* Part_Leader + ...
                            Mask_Mutant .* Part_Mutant;
                end
            is_Out = X_New < lb | X_New > ub;
                
                % 2. 只有当存在越界时才执行，节省计算资源
                if any(is_Out)
                    % 生成全维度的随机位置 (在 [lb, ub] 范围内)
                    Rand_Pos = lb + (ub - lb) .* rand(1, dim);
                    
                    % 3. 只替换掉越界的那些维度，没越界的保留原值
                    % 这种"逻辑索引"写法在 MATLAB 中运行速度极快
                    X_New(is_Out) = Rand_Pos(is_Out);
                end
   
            else
               % =======================================================
                % [策略 B] 分层异构 GWO (Rank-Based Stratified GWO)
                % =======================================================
                
                % 1. 定义分层界限 (和上面 SRA 策略保持一致)
                Num_Elites = round(nPop * para_p);

                % 通用参数计算 (Alpha/Beta/Delta 的向量)
                r1=rand(1,dim); r2=rand(1,dim); A1=2*a*r1-a; C1=2*r2;
                D_alpha = abs(C1.*Alpha_pos - Positions(i,:));
                X1 = Alpha_pos - A1 .* D_alpha;
                
                r1=rand(1,dim); r2=rand(1,dim); A2=2*a*r1-a; C2=2*r2;
                D_beta = abs(C2.*Beta_pos - Positions(i,:));
                X2 = Beta_pos - A2 .* D_beta;
                
                r1=rand(1,dim); r2=rand(1,dim); A3=2*a*r1-a; C3=2*r2;
                D_delta = abs(C3.*Delta_pos - Positions(i,:));
                X3 = Delta_pos - A3 .* D_delta;
                
                % =======================================================
                % [层级一] 精英狼 (Top 20%): 惯性加权 GWO
                % 特点：引入权重和平滑惯性，"走弧线"，拒绝阶跃
                % =======================================================
                if i <= Num_Elites
                    
                
                    
                    % 加权共识位置
                    X_New =  (X1 +  X2 +  X3)/3;
                    
                    % 2. 惯性更新 (核心平滑机制)
                    % c1 (自我保留): 0.2 ~ 0.4，随时间增加，越后期越稳
                    % c2 (社会跟随): 1 - c1
                    c1 = 0.2 + 0.2 * progress; 
                    c2 = 1 - c1;
                    
                    % 新位置 = 惯性 * 旧位置 + 学习 * 共识位置
                    X_New = c1 * Positions(i, :) + c2 * X_New;
                    
                % =======================================================
                % [层级二] 普通狼 (Bottom 80%): 标准 GWO (快速收敛)
                % 特点：直接取平均，响应速度快，负责提供收敛压力
                % =======================================================
                else
                    % 传统 GWO 的简单平均
                    % 这里不需要惯性，因为它们的位置本来就不好，不需要保留
                    % 它们需要的是"瞬移"到好的区域
                    X_New = (X1 + X2 + X3) / 3;
                end
            end
            
            % --- 边界控制 ---
            X_New = max(min(X_New, ub), lb);
            
            % --- 第一次评估 ---
            fit_new = fobj(X_New);
            FEs = FEs + 1;

            if fit_new < Record_Best_Score
                Record_Best_Score = fit_new;
            end
            Convergence_curve(FEs) = Record_Best_Score;

            % =======================================================
            % [AGR 策略: Alpha 引导的末期收缩]
            % % =======================================================
            Prob_AGR = progress^4;

            % 判断条件：完全平滑，没有任何硬阈值
            if FEs < MaxFEs && (rand() < Prob_AGR)

                % --- 下面是原来的 AGR 逻辑，保持不变 ---
                Contract_Factor = 0.5 * Energy * (2 * rand() - 1);
                X_Refined = Alpha_pos + Contract_Factor .* (Alpha_pos - X_New);
                X_Refined = max(min(X_Refined, ub), lb);

                fit_refined = fobj(X_Refined);
                FEs = FEs + 1;

                if fit_refined < Record_Best_Score
                    Record_Best_Score = fit_refined;
                end
                Convergence_curve(FEs) = Record_Best_Score;

                if fit_refined < fit_new
                    X_New = X_Refined;
                    fit_new = fit_refined;
                end
            end
            %
            % --- 贪婪更新 ---
            if fit_new < Fitness(i)
                Positions(i, :) = X_New;
                Fitness(i) = fit_new;
            end
        end

        % --- 代末排序 ---
        [Fitness, sort_idx] = sort(Fitness);
        Positions = Positions(sort_idx, :);
        Alpha_pos = Positions(1, :); Alpha_score = Fitness(1);
        Beta_pos  = Positions(2, :); Beta_score = Fitness(2);
        Delta_pos = Positions(3, :); Delta_score = Fitness(3);

    end
    
    Convergence_curve = Convergence_curve(1:min(length(Convergence_curve), MaxFEs));
end