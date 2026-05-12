function [Best_score, Best_pos, Convergence_curve] = SSA(nPop, Max_iter, lb, ub, dim, fobj) 
    PD = 0.2; SD = 0.1; 
    PDNumber = round(nPop * PD); 
    SDNumber = round(nPop * SD); 
    
    if isscalar(lb), ub = ub*ones(1,dim); lb = lb*ones(1,dim); end 
    
    X = lb + (ub - lb) .* rand(nPop, dim); 
    fitness = arrayfun(@(i) fobj(X(i,:)), 1:nPop); 
    
    [fitness, idx] = sort(fitness); 
    X = X(idx,:); 
    BestF = fitness(1); BestX = X(1,:); 
    
    Convergence_curve = zeros(1, Max_iter); 
    ST = 0.8;  % 安全阈值 
    
    for t = 1:Max_iter 
        % 1. 发现者 
        for i = 1:PDNumber 
            R2 = rand();  % ✅ 独立生成 
            if R2 < ST 
                X(i,:) = X(i,:) .* exp(-i ./ (rand() * Max_iter)); 
            else 
                X(i,:) = X(i,:) + randn() * ones(1,dim); 
            end 
        end 
        
        % 2. 跟随者 
        for i = PDNumber+1:nPop 
            if i > nPop/2 
                X(i,:) = randn(1,dim) .* exp((X(end,:) - X(i,:)) ./ i^2); 
            else 
                A = sign(randn(1,dim));  % ✅ 简化A⁺ 
                X(i,:) = BestX + abs(X(i,:) - BestX) .* A; 
            end 
        end 
        
        % 3. 警戒者 
        c = randperm(nPop, SDNumber); 
        for j = 1:SDNumber 
            i = c(j); 
            if abs(fitness(i) - BestF) > 1e-10  % ✅ 最优个体 
                X(i,:) = BestX + randn(1,dim) .* abs(X(i,:) - BestX); 
            else 
                K = 2*rand() - 1; 
                X(i,:) = X(i,:) + K .* (abs(X(i,:) - X(end,:)) ./ (fitness(i)-fitness(end)+eps)); 
            end 
        end 
        
        % 4. 统一边界处理 + 适应度更新 
        X = max(min(X, ub), lb); 
        fitness = arrayfun(@(i) fobj(X(i,:)), 1:nPop); 
        
        [fitness, idx] = sort(fitness); 
        X = X(idx,:); 
        
        % 更新全局最优 (防止震荡)
        if fitness(1) < BestF
            BestF = fitness(1); 
            BestX = X(1,:); 
        end
        
        Convergence_curve(t) = BestF; 
    end 
    
    Best_score = BestF; Best_pos = BestX; 
end
