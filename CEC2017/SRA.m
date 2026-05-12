function [Best_Cost, Best_X, convergence_curve] = SRA(nPop, MaxFEs, lb, ub, dim, fobj)
    % 初始化边界
    if numel(lb) == 1
        lb = repmat(lb, 1, dim);
        ub = repmat(ub, 1, dim);
    end
    
    % 参数初始化
    L = 0.5;
    h = 6.625e-34; % 普朗克常数
    Cost = inf(1, nPop);
    pos = zeros(nPop, dim);
    Psai = zeros(nPop, dim);
    
    % 种群初始化
    for i = 1:dim
        pos(:, i) = rand(1, nPop) .* (ub(i) - lb(i)) + lb(i);
    end
    
    FEs = 0;
    % 计算初始适应度
    for i = 1:nPop
        pos(i, :) = max(min(pos(i, :), ub), lb);
        % 初始化波函数 Psi
        Psai(i, :) = sqrt(2/L) * sin(pos(i, :)) * exp(2);
        Cost(i) = fobj(pos(i, :));
        FEs = FEs + 1;
        if FEs >= MaxFEs, break; end
    end
    
    % 初始排序
    [SmellOrder, SmellIndex] = sort(Cost);
    Best_Cost = SmellOrder(1);
    Best_X = pos(SmellIndex(1), :);
    
    convergence_curve = [];
    next_record_FEs = nPop;
    Record_Best_Score = Best_Cost;
    
    if FEs >= next_record_FEs
        convergence_curve(end+1) = Record_Best_Score;
        next_record_FEs = next_record_FEs + nPop;
    end
    
    % 主循环
    while FEs < MaxFEs
        progress = FEs / MaxFEs;
        
        % ---------------------------------------------------------
        % [修改处 1] 引入 TF(t) 动态概率因子
        % TF 从 1 线性递减到 0
        TF = 1 - progress; 
        % 也可以使用非线性: TF = (1 - progress)^(1/2); 视具体论文公式而定
        % ---------------------------------------------------------
        
        b = 1 - (progress^(1.0 / 5)); 
        
        [~, SmellIndex] = sort(Cost);
        sorted_Psai = Psai(SmellIndex, :);
        Best_Psai = sorted_Psai(1, :);
        Worst_Psai = sorted_Psai(end, :);
        
        % 计算排名相关的概率 p (用于波函数更新中的参数 h2)
        Seq = 1:nPop;
        R = nPop - Seq;
        p = (R / nPop).^2;
        
        for i = 1:nPop
            if FEs >= MaxFEs, break; end
            
            h2 = p(i);
            vc = unifrnd(-b, b, [1, dim]);
            Z = Levy(dim);
            k = 1;
            
            % 随机重置机制 (保持不变)
            if rand() < 0.03
                Xnew = rand(1, dim) .* (ub - lb) + lb;
            else
                ids_except_current = 1:nPop;
                ids_except_current(i) = [];
                id_12 = randsample(ids_except_current, 2);
                id_1 = id_12(1);
                id_2 = id_12(2);
                
                % ---------------------------------------------------------
                % [修改处 2] 使用 dynamic TF 替换静态的 abs(p(i)) >= 0.5
                % 
                % 逻辑说明:
                % rand() >= TF : 随着 TF 变小(后期)，这个条件更容易满足 -> 粒子行为(开发)
                % rand() < TF  : 随着 TF 变大(前期)，这个条件更容易满足 -> 波动行为(探索)
                % ---------------------------------------------------------
                if rand() >= TF 
                    % === Classical Particle-like Updating (Exploitation) ===
                    % 经典牛顿力学更新
                    if rand() < 0.5
                        if i > 1
                            Xnew = k * rand() + 2 * pos(i, :) - pos(i-1, :);
                        else
                            Xnew = k * rand() + pos(i, :);
                        end
                    else
                        Xnew = Best_X - 0.1 * Z + rand() * ((ub - lb) * rand() + lb);
                    end
                else
                    % === Quantum Wave-like Updating (Exploration) ===
                    % 薛定谔波动方程更新
                    pos_1 = Best_X + rand() * vc .* (h * (Best_Psai - Worst_Psai) + (h2 * (Psai(id_1, :) - 2 * Psai(i, :) + Psai(id_2, :)))) ./ Psai(i, :);
                    pos_2 = pos(i, :) + rand() * vc .* (h * (Best_Psai - Worst_Psai) + (h2 * (Psai(id_1, :) + 2 * Psai(i, :) + Psai(id_2, :)))) ./ Psai(i, :);
                    if rand() < 0.5
                        Xnew = pos_1;
                    else
                        Xnew = pos_2;
                    end
                end
            end
            
            % 边界处理
            Xnew = max(min(Xnew, ub), lb);
            Xnew_Cost = fobj(Xnew);
            FEs = FEs + 1;
            
            % 更新全局最优
            if Xnew_Cost < Record_Best_Score, Record_Best_Score = Xnew_Cost; end
            if FEs >= next_record_FEs
                convergence_curve(end+1) = Record_Best_Score;
                next_record_FEs = next_record_FEs + nPop;
            end
            
            % 贪婪选择更新个体
            if Cost(i) > Xnew_Cost
                Cost(i) = Xnew_Cost;
                pos(i, :) = Xnew;
                if Cost(i) < Best_Cost
                    Best_X = pos(i, :);
                    Best_Cost = Cost(i);
                end
            end
            
            % 更新波函数 Psi
            Psai(i, :) = sin(rand() * pos(i, :));
        end
    end
end

% Levy 飞行辅助函数
function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1,d)*sigma;
    v = randn(1,d);
    step = u./abs(v).^(1/beta);
    o = step;
end