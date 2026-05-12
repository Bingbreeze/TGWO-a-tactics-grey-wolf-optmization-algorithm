function [Best_score, Best_pos, Convergence_curve] = HSO(nPop, MaxFEs, lb, ub, dim, fobj)

    %% Algorithm Parameters
    alpha = 3;         % Scaling factor for position updates
    
    % Simulated Annealing parameters
    initialTemp = 10000;
    coolingRate = 0.995;
    
    % Adaptive Mutation parameters
    initialMutationRate = 0.5;
    finalMutationRate = 0.1;
    initialMutationStep = 0.3;
    finalMutationStep = 0.1;
    
    %% Initialization
    if isscalar(lb)
        lb = repmat(lb, 1, dim);
    end
    if isscalar(ub)
        ub = repmat(ub, 1, dim);
    end
    
    % Initialize positions
    positions = zeros(nPop, dim);
    for i = 1:nPop
        positions(i, :) = lb + (ub - lb) .* rand(1, dim);
    end
    positionsNew = positions;
    
    % Initialize fitness
    fitness = inf(1, nPop);
    fitnessNew = fitness;
    
    FEs = 0;
    
    % Evaluate initial population
    for i = 1:nPop
        fitness(i) = fobj(positions(i, :));
        FEs = FEs + 1;
        if FEs >= MaxFEs, break; end
    end
    
    % Initialize Best Solution
    [bestCost, bestIdx] = min(fitness);
    bestPos = positions(bestIdx, :);
    
    Convergence_curve = [];
    iter = 0;
    
    %% Main Loop
    while FEs < MaxFEs
        iter = iter + 1;
        progress = FEs / MaxFEs;
        
        temp = initialTemp * (coolingRate ^ (iter-1)); % Use iter for cooling is fine
    
        % Update adaptive mutation parameters based on FEs progress
        mutationRate = initialMutationRate - progress * (initialMutationRate - finalMutationRate);
        mutationStep = initialMutationStep - progress * (initialMutationStep - finalMutationStep);
    
        for i = 1:nPop
            if FEs >= MaxFEs, break; end
            
            % Ensure positions remain within bounds
            positionsNew(i, :) = max(min(positionsNew(i, :), ub), lb);
    
            % Evaluate fitness of each agent
            fitnessNew(i) = fobj(positionsNew(i, :));
            FEs = FEs + 1;
    
            % Selection using greedy and SA-based criteria
            if fitnessNew(i) < fitness(i)
                positions(i, :) = positionsNew(i, :);
                fitness(i) = fitnessNew(i);
            else
                delta = fitnessNew(i) - fitness(i);
                if exp(-delta / temp) > rand
                    positions(i, :) = positionsNew(i, :);
                    fitness(i) = fitnessNew(i);
                end
            end
        end
    
        % Sort and archive best solution
        [fitness, idx] = sort(fitness);
        positions = positions(idx, :);
    
        if fitness(1) < bestCost
            bestCost = fitness(1);
            bestPos = positions(1, :);
        end
    
        %% Coefficient Calculation (simplified, without sign)
        fitnessVector = fitness(:);
        differences = rms(fitnessVector) - fitnessVector;
        sum_diff = sum(abs(differences));
        if sum_diff == 0
            updateCoef = zeros(size(differences));
        else
            updateCoef = differences / sum_diff;
        end
    
        %% Position Update
        for i = 1:nPop
            for j = 1:dim
                randWeights = rand(nPop, 1);
                displacement = alpha * sum(randWeights .* updateCoef .* (positions(:, j) - positions(i, j)));
                positionsNew(i, j) = positions(i, j) + displacement;
            end
        end
    
        %% Adaptive Mutation
        for i = 1:nPop
            if rand < mutationRate
                mutationVector = mutationStep * randn(1, dim);
                positionsNew(i, :) = positionsNew(i, :) + mutationVector;
            end
        end
        
        Convergence_curve(iter) = bestCost;
    end
    
    Best_score = bestCost;
    Best_pos = bestPos;
    
end
