function [Alpha_score, Alpha_pos, Convergence_curve] = TGWO1(nPop, MaxFEs, lb, ub, dim, fobj, varargin)

    params = defaultParams();
    if ~isempty(varargin)
        if isstruct(varargin{1})
            params = mergeParams(params, varargin{1});
        else
            if length(varargin) >= 1, params.k = varargin{1}; end
            if length(varargin) >= 2, params.c = varargin{2}; end
            if length(varargin) >= 3, params.p = varargin{3}; end
        end
    end

    para_k = params.k;
    para_c = params.c;
    para_p = params.p;

    if isscalar(lb), lb = repmat(lb, 1, dim); end
    if isscalar(ub), ub = repmat(ub, 1, dim); end

    Positions = lb + (ub - lb) .* rand(nPop, dim);
    Fitness = zeros(nPop, 1);

    FEs = 0;
    Convergence_curve = zeros(1, MaxFEs);

    % 初始评估
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

    while FEs < MaxFEs

        progress = FEs / MaxFEs;
        Energy = cos(pi/2 * progress);
        k = para_k;
        c = para_c;
        Prob_SRA = 1 ./ (1 + exp(k * (progress - c)));
        a = (1 - progress) * 2;

        Space_Span = ub - lb;
        Exploration_Leaders = (Alpha_pos + Beta_pos + Delta_pos) / 3;

        for i = 1:nPop
            if FEs >= MaxFEs, break; end

            % =======================================================
            % 分支 A: 探索（已改为 GTO 风格 Levy 步长）
            % =======================================================
            if rand() < Prob_SRA
                Target_Anchor = Exploration_Leaders;

                idx_r1 = randi(nPop); idx_r2 = randi(nPop);
                if Fitness(idx_r1) < Fitness(idx_r2)
                    Pos_Better = Positions(idx_r1, :); Pos_Worse = Positions(idx_r2, :);
                else
                    Pos_Better = Positions(idx_r2, :); Pos_Worse = Positions(idx_r1, :);
                end

                Cauchy_Scale = tan(pi * (rand(1, dim) - 0.5));
                Base_Diff = (Pos_Better - Pos_Worse) .* Cauchy_Scale;
                Space_Compensation = Cauchy_Scale .* Space_Span .* params.space_comp .* (1 - progress);
                Diff_Vector = params.base_diff .* Base_Diff + Space_Compensation;

                % ===================== 【关键修改 1】 =====================
                % 去掉原来 20% 固定步长 → 替换为 GTO 风格 Levy 步长
                beta = 1.5;
                sigma = (gamma(1+beta)*sin(pi*beta/2) / (gamma((1+beta)/2)*beta*2^((beta-1)/2))) .^ (1/beta);
                levy = 0.01 * (randn(1,dim) .* sigma) ./ abs(randn(1,dim)).^(1/beta);
                Max_Allowed_Step = abs(Space_Span .* levy);
                Diff_Vector = max(min(Diff_Vector, Max_Allowed_Step), -Max_Allowed_Step);
                % ==========================================================

                Perturbed_Component = Positions(i, :) + Diff_Vector;

                Num_Elites = round(nPop * para_p);
                if i <= Num_Elites
                    if i == 1
                        Dist_To_Best = (Beta_pos - Alpha_pos) * params.elite_beta_ref;
                    else
                        Dist_To_Best = Alpha_pos - Positions(i, :);
                    end
                    Cauchy_Val = tan(pi * (rand(1, dim) - 0.5));
                    Mutation_Step = params.mutation_scale * Cauchy_Val .* Dist_To_Best;
                    X_New = Positions(i, :) + Mutation_Step;
                else
                    w_leader = params.w_leader_min + (params.w_leader_max - params.w_leader_min) * progress;
                    w_leader = min(max(w_leader, 0), 1);
                    X_New = w_leader .* Target_Anchor + (1 - w_leader) .* Perturbed_Component;
                end

                % ===================== 【关键修改 2】 =====================
                % 边界反弹 → 改为 GTO 风格 边界裁剪
                X_New = max(min(X_New, ub), lb);
                % ==========================================================

            else
                % =======================================================
                % 分支 B: GWO 开发
                % =======================================================
                Num_Elites = round(nPop * para_p);

                r1=rand(1,dim); r2=rand(1,dim); A1=2*a*r1-a; C1=2*r2;
                D_alpha = abs(C1.*Alpha_pos - Positions(i,:));
                X1 = Alpha_pos - A1 .* D_alpha;

                r1=rand(1,dim); r2=rand(1,dim); A2=2*a*r1-a; C2=2*r2;
                D_beta = abs(C2.*Beta_pos - Positions(i,:));
                X2 = Beta_pos - A2 .* D_beta;

                r1=rand(1,dim); r2=rand(1,dim); A3=2*a*r1-a; C3=2*r2;
                D_delta = abs(C3.*Delta_pos - Positions(i,:));
                X3 = Delta_pos - A3 .* D_delta;

                if i <= Num_Elites
                    X_New = (X1 + X2 + X3)/3;
                    c1 = 0.2 + 0.2 * progress;
                    c2 = 1 - c1;
                    X_New = c1 * Positions(i, :) + c2 * X_New;
                else
                    X_New = (X1 + X2 + X3) / 3;
                end

                % ===================== 【关键修改 2】 =====================
                X_New = max(min(X_New, ub), lb);
                % ==========================================================
            end

            fit_new = fobj(X_New);
            FEs = FEs + 1;

            if fit_new < Record_Best_Score
                Record_Best_Score = fit_new;
            end
            if FEs <= MaxFEs
                Convergence_curve(FEs) = Record_Best_Score;
            end

            Prob_AGR = progress^params.agr_prob_pow;
            if FEs < MaxFEs && rand() < Prob_AGR
                Contract_Factor = params.contract_scale * Energy * (2*rand()-1);
                X_Refined = Alpha_pos + Contract_Factor .* (Alpha_pos - X_New);
                X_Refined = max(min(X_Refined, ub), lb);

                fit_refined = fobj(X_Refined);
                FEs = FEs + 1;

                if fit_refined < Record_Best_Score
                    Record_Best_Score = fit_refined;
                end
                if FEs <= MaxFEs
                    Convergence_curve(FEs) = Record_Best_Score;
                end

                if fit_refined < fit_new
                    X_New = X_Refined;
                    fit_new = fit_refined;
                end
            end

            if fit_new < Fitness(i)
                Positions(i, :) = X_New;
                Fitness(i) = fit_new;
            end
        end

        [Fitness, sort_idx] = sort(Fitness);
        Positions = Positions(sort_idx, :);
        Alpha_pos = Positions(1, :); Alpha_score = Fitness(1);
        Beta_pos  = Positions(2, :);
        Delta_pos = Positions(3, :);
    end
end

% ==================== 默认参数 ====================
function params = defaultParams()
    params.k = 10;
    params.c = 0.5;
    params.p = 0.2;
    params.space_comp = 0.01;
    params.base_diff = 0.5;
    params.mutation_scale = 0.5;
    params.elite_beta_ref = 0.1;
    params.w_leader_min = 0.5;
    params.w_leader_max = 0.9;
    params.max_step_a = 0.2;
    params.max_step_b = 0.01;
    params.max_step_pow = 2;
    params.agr_prob_pow = 4;
    params.contract_scale = 0.5;
end

function out = mergeParams(base, in)
    out = base;
    keys = fieldnames(in);
    for i = 1:numel(keys)
        key = keys{i};
        out.(key) = in.(key);
    end
end