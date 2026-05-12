function [gBestScore, gBest, Convergence_curve] = PSO(nPop, Max_iter, lb, ub, dim, fobj)

    % 1. 初始化
    if isscalar(lb)
        lb = lb * ones(1, dim);
        ub = ub * ones(1, dim);
    end
    
    % PSO 参数
    w = 0.9;      % 惯性权重初始值
    w_damp = 0.99; % 惯性权重衰减系数
    c1 = 1.5;     % 个体学习因子
    c2 = 1.5;     % 社会学习因子
    
    % 速度限制 (通常设为搜索范围的 10%-20%)
    VelMax = 0.1 * (ub - lb);
    VelMin = -VelMax;
    
    % 初始化种群
    Positions = rand(nPop, dim) .* (ub - lb) + lb;
    Velocities = zeros(nPop, dim);
    
    % 初始化个体最优 (pBest) 和 全局最优 (gBest)
    pBest = Positions;
    pBestScore = inf(nPop, 1);
    
    gBest = zeros(1, dim);
    gBestScore = inf;
    
    Convergence_curve = zeros(1, Max_iter);
    
    % 计算初始适应度
    for i = 1:nPop
        fit = fobj(Positions(i, :));
        pBestScore(i) = fit;
        if fit < gBestScore
            gBestScore = fit;
            gBest = Positions(i, :);
        end
    end
    
    % 2. 主循环
    for l = 1:Max_iter
        
        for i = 1:nPop
            % 更新速度
            r1 = rand(1, dim);
            r2 = rand(1, dim);
            
            Velocities(i, :) = w * Velocities(i, :) ...
                + c1 * r1 .* (pBest(i, :) - Positions(i, :)) ...
                + c2 * r2 .* (gBest - Positions(i, :));
            
            % 速度边界处理
            Velocities(i, :) = max(min(Velocities(i, :), VelMax), VelMin);
            
            % 更新位置
            Positions(i, :) = Positions(i, :) + Velocities(i, :);
            
            % 位置边界处理
            Positions(i, :) = max(min(Positions(i, :), ub), lb);
            
            % 计算适应度
            fit = fobj(Positions(i, :));
            
            % 更新个体最优
            if fit < pBestScore(i)
                pBestScore(i) = fit;
                pBest(i, :) = Positions(i, :);
                
                % 更新全局最优
                if fit < gBestScore
                    gBestScore = fit;
                    gBest = Positions(i, :);
                end
            end
        end
        
        % 惯性权重衰减
        w = w * w_damp;
        
        Convergence_curve(l) = gBestScore;
    end
    
end
