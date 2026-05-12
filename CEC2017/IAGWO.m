function [Alpha_score, Alpha_pos, Convergence_curve] = IAGWO(SearchAgents_no, Max_iter, lb, ub, dim, fobj)

    % ==================== 初始化阶段 ====================
    if isscalar(lb)
        lb = repmat(lb, 1, dim);
    end
    if isscalar(ub)
        ub = repmat(ub, 1, dim);
    end

    % 初始化种群位置
    Positions = initialization(SearchAgents_no, dim, ub, lb);
    
    % 初始化 PSO 相关的速度 V 和个体历史最优 pBest
    V = zeros(SearchAgents_no, dim);
    pBestPosition = Positions;
    pBestScore = inf(SearchAgents_no, 1);
    
    % 初始化 GWO 的 Alpha, Beta, Delta
    Alpha_pos = zeros(1, dim);
    Alpha_score = inf; 
    Beta_pos = zeros(1, dim);
    Beta_score = inf; 
    Delta_pos = zeros(1, dim);
    Delta_score = inf; 

    Convergence_curve = zeros(1, Max_iter);
    
    % 计算初始适应度
    for i = 1:SearchAgents_no
        fitness = fobj(Positions(i, :));
        
        % 更新 pBest
        if fitness < pBestScore(i)
            pBestScore(i) = fitness;
            pBestPosition(i, :) = Positions(i, :);
        end
        
        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score
            Alpha_score = fitness; 
            Alpha_pos = Positions(i, :);
        elseif fitness > Alpha_score && fitness < Beta_score
            Beta_score = fitness; 
            Beta_pos = Positions(i, :);
        elseif fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
            Delta_score = fitness; 
            Delta_pos = Positions(i, :);
        end
    end

    % ==================== 主循环 ====================
    for t = 1:Max_iter
        
        % 更新参数 a (从 2 线性递减到 0)
        a = 2 - 2 * t / Max_iter; 
        
        % --- 策略 2: IMF 惯性权重 (Eq. 9) ---
        % 参数设定来自论文: a=0.6, b=0.02, c=0.05, d=0.3
        % 注意：论文中的 t 可能是当前迭代数，c=0.05 会导致指数衰减极快
        % 此处严格按照公式 (9) 实现
        omega = 0.6 * exp(-0.02 * exp(-0.05 * t)) + 0.3;
        
        % 计算种群平均适应度 (用于策略 3)
        current_fitness = zeros(SearchAgents_no, 1);
        
        for i = 1:SearchAgents_no
            
            % --- 策略 1: PSO 搜索机制 (Eq. 8) ---
            % 在每次迭代早期更新位置
            r_pso = rand();
            % 更新速度
            V(i, :) = V(i, :) + r_pso * (Alpha_pos - Positions(i, :)) + ...
                      r_pso * (pBestPosition(i, :) - Positions(i, :));
            % 限制速度（可选，防止飞出太远，通常设为边界的 10%）
            V(i, :) = max(min(V(i, :), 0.1*(ub-lb)), -0.1*(ub-lb));
            
            % 基于 PSO 更新位置
            Positions(i, :) = Positions(i, :) + V(i, :);
            
            % 边界检查
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
            
            % --- 策略 2: GWO 更新 (Eq. 2, 10, 4) ---
            % 引入了 omega 权重的 GWO 公式
            for j = 1:dim
                r1 = rand(); r2 = rand();
                A1 = 2 * a * r1 - a;
                C1 = 2 * r2;
                D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
                X1 = Alpha_pos(j) - omega * A1 * D_alpha; % 引入 omega
                
                r1 = rand(); r2 = rand();
                A2 = 2 * a * r1 - a;
                C2 = 2 * r2;
                D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
                X2 = Beta_pos(j) - omega * A2 * D_beta;   % 引入 omega
                
                r1 = rand(); r2 = rand();
                A3 = 2 * a * r1 - a;
                C3 = 2 * r2;
                D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
                X3 = Delta_pos(j) - omega * A3 * D_delta; % 引入 omega
                
                Positions(i, j) = (X1 + X2 + X3) / 3;
            end
            
            % GWO 更新后的边界检查
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
            
            % 临时计算适应度用于策略 3
            current_fitness(i) = fobj(Positions(i, :));
        end
        
        f_ave = mean(current_fitness);
        
        % --- 策略 3: 自适应更新机制 (Adaptive Updating Mechanism) ---
        % 公式 11 & 12
        for i = 1:SearchAgents_no
            fi = current_fitness(i);
            
            % Eq. 11: 计算 Phi (基于 Sigmoid)
            % 聚合系数 (Aggregation Factor) = fi / f_ave
            theta = 0.5; % 论文设定指数系数
            if f_ave == 0
                ratio = 1; 
            else
                ratio = fi / f_ave;
            end
            
            % Phi: 适应度越差(ratio大) -> Phi趋近1；适应度越好(ratio小) -> Phi趋近0.5
            Phi = 1 / (1 + (exp(-ratio))^theta);
            
            % Eq. 12: Yi = Yi * Phi
            % Yi 定义为基于搜索空间的随机扰动向量
            % (rand - 0.5) * (ub - lb) 产生一个覆盖整个搜索范围的随机向量
            Yi = (rand(1, dim) - 0.5) .* (ub - lb);
            
            % 应用公式 12
            Yi = Yi * Phi; 
            
            % 更新位置
            Positions(i, :) = Positions(i, :) + Yi;
            
            % 边界检查
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
        end
        
        % ==================== 更新全局信息 ====================
        for i = 1:SearchAgents_no
            % 计算最终适应度
            fitness = fobj(Positions(i, :));
            
            % 更新 pBest (PSO 需要)
            if fitness < pBestScore(i)
                pBestScore(i) = fitness;
                pBestPosition(i, :) = Positions(i, :);
            end
            
            % 更新 Alpha, Beta, Delta
            if fitness < Alpha_score
                Alpha_score = fitness;
                Alpha_pos = Positions(i, :);
            elseif fitness > Alpha_score && fitness < Beta_score
                Beta_score = fitness;
                Beta_pos = Positions(i, :);
            elseif fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
                Delta_score = fitness;
                Delta_pos = Positions(i, :);
            end
        end
        
        Convergence_curve(t) = Alpha_score;
        
        % 可选：打印迭代进度
        % disp(['Iteration ' num2str(t) ': Best Cost = ' num2str(Alpha_score)]);
    end
end

% 辅助函数：初始化
function Positions = initialization(SearchAgents_no, dim, ub, lb)
    Boundary_no= size(ub, 2); 
    if Boundary_no==1
        Positions = rand(SearchAgents_no, dim) .* (ub - lb) + lb;
    else
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(:, i) = rand(SearchAgents_no, 1) .* (ub_i - lb_i) + lb_i;
        end
    end
end