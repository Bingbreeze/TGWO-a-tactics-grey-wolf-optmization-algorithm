function [Alpha_score, Alpha_pos, Convergence_curve] = IGWO(nPop, Max_iter, lb, ub, dim, fobj)

    % ==================== 1. Initialization ====================
    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end
    
    % 初始化种群
    Positions = lb + (ub - lb) .* rand(nPop, dim);
    Fitness = zeros(nPop, 1);
    
    FEs = 0;
    MaxFEs = Max_iter * nPop;
    
    for i = 1:nPop
        Fitness(i) = fobj(Positions(i, :));
        FEs = FEs + 1;
    end
    
    % 排序并初始化 Alpha, Beta, Delta
    [Fitness, sort_idx] = sort(Fitness);
    Positions = Positions(sort_idx, :);
    
    Alpha_pos = Positions(1, :); Alpha_score = Fitness(1);
    Beta_pos  = Positions(2, :); Beta_score  = Fitness(2);
    Delta_pos = Positions(3, :); Delta_score = Fitness(3);
    
    Convergence_curve = [];
    next_record_FEs = nPop;
    Record_Best_Score = Alpha_score;
    
    if FEs >= next_record_FEs
        Convergence_curve(end+1) = Record_Best_Score;
        next_record_FEs = next_record_FEs + nPop;
    end
    
    % ==================== 2. Main Loop ====================
    l_iter = 0;
    while FEs < MaxFEs
        l_iter = l_iter + 1;
        progress = FEs / MaxFEs;
        
        a = 2 * (1 - progress); % GWO 线性收敛因子 a 从 2 降到 0
        
        % 这一步是为了 DLH 策略计算半径
        % Radius 计算：基于当前迭代次数和搜索空间范围
        % R = norm(ub - lb) * (1 - (l / Max_iter)); % (Unused variable removed/commented) 
        
        % 记录上一代的种群（用于 DLH 比较和同步更新）
        Positions_Old = Positions;
        Fitness_Old = Fitness;
        
        for i = 1:nPop
            if FEs >= MaxFEs, break; end
            
            % ---------------------------------------------------------
            % Phase 1: GWO Strategy (标准 GWO 更新)
            % ---------------------------------------------------------
            % 使用 Positions_Old 保持同步更新逻辑
            r1 = rand(1, dim); r2 = rand(1, dim);
            A1 = 2*a*r1 - a; C1 = 2*r2;
            D_alpha = abs(C1 .* Alpha_pos - Positions_Old(i,:));
            X1 = Alpha_pos - A1 .* D_alpha;
            
            r1 = rand(1, dim); r2 = rand(1, dim);
            A2 = 2*a*r1 - a; C2 = 2*r2;
            D_beta = abs(C2 .* Beta_pos - Positions_Old(i,:));
            X2 = Beta_pos - A2 .* D_beta;
            
            r1 = rand(1, dim); r2 = rand(1, dim);
            A3 = 2*a*r1 - a; C3 = 2*r2;
            D_delta = abs(C3 .* Delta_pos - Positions_Old(i,:));
            X3 = Delta_pos - A3 .* D_delta;
            
            X_GWO = (X1 + X2 + X3) / 3;
            X_GWO = max(min(X_GWO, ub), lb);
            Fit_GWO = fobj(X_GWO);
            FEs = FEs + 1;
            
            if Fit_GWO < Record_Best_Score, Record_Best_Score = Fit_GWO; end
            if FEs >= next_record_FEs
                Convergence_curve(end+1) = Record_Best_Score;
                next_record_FEs = next_record_FEs + nPop;
            end
            
            % ---------------------------------------------------------
            % Phase 2: DLH Strategy (基于维度的学习猎杀)
            % ---------------------------------------------------------
            % 选择邻居：基于 Positions_Old 计算欧氏距离
            dist = sum((repmat(Positions_Old(i,:), nPop, 1) - Positions_Old).^2, 2);
            [~, sorted_dist_idx] = sort(dist);
            
            % 排除自己，选择最近的邻居
            neighbor_idx = sorted_dist_idx(2); 
            X_Neighbor = Positions_Old(neighbor_idx, :);
            
            % 随机选择一只狼作为伙伴
            random_idx = randi(nPop);
            while random_idx == i || random_idx == neighbor_idx
                random_idx = randi(nPop);
            end
            X_Random = Positions_Old(random_idx, :);
            
            % DLH 位置更新公式 (使用 Positions_Old)
            X_DLH = Positions_Old(i, :);
            for d = 1:dim
                 X_DLH(d) = Positions_Old(i, d) + rand * (X_Neighbor(d) - X_Random(d));
            end
            X_DLH = max(min(X_DLH, ub), lb);
            Fit_DLH = fobj(X_DLH);
            FEs = FEs + 1;
            
            if Fit_DLH < Record_Best_Score, Record_Best_Score = Fit_DLH; end
            if FEs >= next_record_FEs
                Convergence_curve(end+1) = Record_Best_Score;
                next_record_FEs = next_record_FEs + nPop;
            end
            
            % ---------------------------------------------------------
            % Phase 3: Selection (择优保留)
            % ---------------------------------------------------------
            % 比较 GWO 候选解、DLH 候选解 和 当前解
            
            best_candidate_pos = X_GWO;
            best_candidate_fit = Fit_GWO;
            
            if Fit_DLH < best_candidate_fit
                best_candidate_pos = X_DLH;
                best_candidate_fit = Fit_DLH;
            end
            
            % 如果候选解比当前解好，则更新
            if best_candidate_fit < Fitness_Old(i)
                Positions(i, :) = best_candidate_pos;
                Fitness(i) = best_candidate_fit;
            end
        end
        
        if FEs >= MaxFEs, break; end
        
        % --- 更新 Leaders ---
        % IGWO 原文逻辑：每次迭代结束重新评估和排序整个种群
        for i = 1:nPop
            if Fitness(i) < Alpha_score
                Delta_score = Beta_score; Delta_pos = Beta_pos;
                Beta_score  = Alpha_score; Beta_pos  = Alpha_pos;
                Alpha_score = Fitness(i); Alpha_pos = Positions(i, :);
            elseif Fitness(i) > Alpha_score && Fitness(i) < Beta_score
                Delta_score = Beta_score; Delta_pos = Beta_pos;
                Beta_score  = Fitness(i); Beta_pos  = Positions(i, :);
            elseif Fitness(i) > Beta_score && Fitness(i) < Delta_score
                Delta_score = Fitness(i); Delta_pos = Positions(i, :);
            end
        end
        
        % Convergence_curve(l) = Alpha_score;
    end
end