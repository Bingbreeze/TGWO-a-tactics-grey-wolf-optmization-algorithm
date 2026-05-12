function [Best_score, Best_pos, Convergence_curve] = GSA(N, max_iter, lb, ub, dim, fobj)

% N: 种群大小
% max_iter: 最大迭代次数
% lb: 下界
% ub: 上界
% dim: 维度
% fobj: 目标函数句柄

% GSA 参数
G0 = 100;
alpha = 20;
Power_Exponent = 1; % 论文中通常设为 1

% 1. 初始化
if isscalar(lb)
    lb = lb * ones(1, dim);
    ub = ub * ones(1, dim);
end

X = rand(N, dim) .* (ub - lb) + lb; % 位置
V = zeros(N, dim); % 速度
a = zeros(N, dim); % 加速度

Best_score = inf;
Best_pos = zeros(1, dim);
Convergence_curve = zeros(1, max_iter);

fitness = inf(N, 1);

% 主循环
for t = 1:max_iter
    
    % 2. 检查边界
    for i = 1:N
        X(i, :) = max(min(X(i, :), ub), lb);
        fitness(i) = fobj(X(i, :));
    end
    
    % 3. 更新最优解
    for i = 1:N
        if fitness(i) < Best_score
            Best_score = fitness(i);
            Best_pos = X(i, :);
        end
    end
    
    Convergence_curve(t) = Best_score;
    
    % 4. 计算质量 (Mass)
    best = min(fitness);
    worst = max(fitness);
    
    % 防止分母为0
    if best == worst
        M = ones(N, 1);
    else
        m = (fitness - worst) ./ (best - worst);
        M = m ./ sum(m);
    end
    
    % 5. 计算引力常数 G
    G = G0 * exp(-alpha * t / max_iter);
    
    % 6. 计算加速度 a
    % Kbest: 随着迭代减少，只计算最好的 Kbest 个个体的引力
    % 初始 Kbest = N, 最后 Kbest = 1 (或者保持百分比)
    % 此处简化为全连接，或者线性递减
    final_per = 2; % 最后 2%
    kbest = round(N * (final_per/100 + (1-final_per/100) * (max_iter - t) / max_iter));
    
    [~, sorted_idx] = sort(M, 'descend');
    
    a = zeros(N, dim);
    for i = 1:N
        for ii = 1:kbest
            j = sorted_idx(ii);
            if j ~= i
                R = norm(X(i, :) - X(j, :), 2); % 欧氏距离
                
                % 随机数 rand(1, dim) 增加随机性 (Elitist check)
                % Force = G * (Mi * Mj) / (R + eps) * (Xj - Xi)
                % a = Force / Mi = G * Mj / (R + eps) * (Xj - Xi)
                
                for d = 1:dim
                    force = G * M(j) * (X(j, d) - X(i, d)) / (R^Power_Exponent + eps);
                    a(i, d) = a(i, d) + rand() * force;
                end
            end
        end
    end
    
    % 7. 更新速度和位置
    V = rand(N, dim) .* V + a;
    X = X + V;
    
end

end
