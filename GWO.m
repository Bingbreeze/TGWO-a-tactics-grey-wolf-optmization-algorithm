function [Alpha_score, Alpha_pos, Convergence_curve] = GWO(nPop, MaxFEs, lb, ub, dim, fobj)
% =========================================================================
% 灰狼优化算法 (GWO) - FEs 驱动版 (用于公平对比)
% =========================================================================

% 参数校验
if nPop < 3
    error('GWO算法要求种群数量 nPop >= 3');
end

% 初始化 alpha, beta, and delta
Alpha_pos = zeros(1, dim);
Alpha_score = inf; 

Beta_pos = zeros(1, dim);
Beta_score = inf;

Delta_pos = zeros(1, dim);
Delta_score = inf;

% 初始化种群
Positions = initialization(nPop, dim, ub, lb);

Convergence_curve = [];
next_record_FEs = nPop;
Record_Best_Score = inf;

FEs = 0;

% 1. 评估初始种群
for i = 1:nPop
    % 边界检查
    Flag4ub = Positions(i,:) > ub;
    Flag4lb = Positions(i,:) < lb;
    Positions(i,:) = (Positions(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
    
    % 计算适应度
    fitness = fobj(Positions(i,:));
    FEs = FEs + 1;
    
    % 更新全局最优记录
    if fitness < Record_Best_Score
        Record_Best_Score = fitness;
    end
    
    % 精英保留机制更新
    if fitness < Alpha_score
        Alpha_score = fitness; 
        Alpha_pos = Positions(i,:);
    end
    
    if fitness > Alpha_score && fitness < Beta_score
        Beta_score = fitness; 
        Beta_pos = Positions(i,:);
    end
    
    if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
        Delta_score = fitness; 
        Delta_pos = Positions(i,:);
    end
    
    % 检查是否达到 MaxFEs
    if FEs >= MaxFEs
        break;
    end
end

% 记录初始点
if FEs >= next_record_FEs
    Convergence_curve(end+1) = Alpha_score;
    next_record_FEs = next_record_FEs + nPop;
end

% 主循环
while FEs < MaxFEs
    
    % 计算进度
    progress = FEs / MaxFEs;
    a = 2 - 2 * progress; % a 从2线性下降到0
    
    % 2. 基于 Alpha, Beta, Delta 更新所有狼的位置
    for i = 1:nPop
        if FEs >= MaxFEs, break; end
        
        for j = 1:dim
            r1=rand(); r2=rand();
            A1=2*a*r1-a; C1=2*r2;
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j));
            X1=Alpha_pos(j)-A1*D_alpha;
            
            r1=rand(); r2=rand();
            A2=2*a*r1-a; C2=2*r2;
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j));
            X2=Beta_pos(j)-A2*D_beta;
            
            r1=rand(); r2=rand();
            A3=2*a*r1-a; C3=2*r2;
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j));
            X3=Delta_pos(j)-A3*D_delta;
            
            Positions(i,j)=(X1+X2+X3)/3;
        end
        
        % 边界检查
        Flag4ub = Positions(i,:) > ub;
        Flag4lb = Positions(i,:) < lb;
        Positions(i,:) = (Positions(i,:).*(~(Flag4ub+Flag4lb))) + ub.*Flag4ub + lb.*Flag4lb;
        
        % 计算适应度
        fitness = fobj(Positions(i,:));
        FEs = FEs + 1;
        
        if fitness < Record_Best_Score
            Record_Best_Score = fitness;
        end
        
        % 更新 Alpha, Beta, Delta
        if fitness < Alpha_score
            Alpha_score = fitness; 
            Alpha_pos = Positions(i,:);
        end
        
        if fitness > Alpha_score && fitness < Beta_score
            Beta_score = fitness; 
            Beta_pos = Positions(i,:);
        end
        
        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
            Delta_score = fitness; 
            Delta_pos = Positions(i,:);
        end
        
        % 记录收敛曲线 (每 nPop 次 FEs)
        if FEs >= next_record_FEs
            Convergence_curve(end+1) = Record_Best_Score;
            next_record_FEs = next_record_FEs + nPop;
        end
    end
end

end

%% 初始化函数
function Positions = initialization(nPop, dim, ub, lb)
    Boundary_no = size(ub, 2);
    if Boundary_no == 1
        Positions = rand(nPop, dim) * (ub - lb) + lb;
    else
        Positions = zeros(nPop, dim);
        for i = 1:dim
            Positions(:, i) = rand(nPop, 1) * (ub(i) - lb(i)) + lb(i);
        end
    end
end
