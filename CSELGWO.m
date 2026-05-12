function [gbestfitness,gbest,Convergence_curve]= CSELGWO(n,MaxFEs,lb,ub,dim,fobj)
    Alpha_pos = zeros(1, dim);
    Alpha_score = inf;
    Beta_pos = zeros(1, dim);
    Beta_score = inf;
    Delta_pos = zeros(1, dim);
    Delta_score = inf;
    CR=0.9;
    new_fitness=zeros(1,n);
    archive_pos=zeros(n,dim);
    levy_pos=zeros(n,dim);
    X1=[];
    X2=[];
    X3=[];
    ns=n;
    nd=5;
    jishu=0;
    op=[];
    pos = initialization(n,dim,ub,lb);
    
    Iter_Curve = [];
    Convergence_curve = [];
    next_record_FEs = n;
    FEs = 0;
    
    for i=1:n
        fitness(i)=fobj(pos(i,:));
        FEs = FEs + 1;
        if FEs >= MaxFEs, break; end
    end
    
    [sortfitness,indexsort]=sort(fitness);
    Alpha_pos=pos(indexsort(1),:);
    Alpha_score=sortfitness(1);
    Beta_pos=pos(indexsort(2),:);
    Beta_score=sortfitness(2);
    Delta_pos=pos(indexsort(3),:);
    Delta_score=sortfitness(3);
    gbest=Alpha_pos;
    gbestfitness=Alpha_score;
    archive_fitness=fitness;
    archive_pos=pos;
    
    Record_Best_Score = Alpha_score;
    if FEs >= next_record_FEs
        Convergence_curve(end+1) = Record_Best_Score;
        next_record_FEs = next_record_FEs + n;
    end
    
    t=0;
    
    while FEs < MaxFEs
        t = t + 1;
        progress = FEs / MaxFEs;
        
        a = 2 - progress * 2; % Adjusted using FEs progress
        sizepop=size(pos(:,1));
        v=zeros(sizepop(1),dim);
        u=zeros(sizepop(1),dim);
        
        for i=1:sizepop(1)
            if FEs >= MaxFEs, break; end
            
            Flag4ub = pos(i, :) > ub;
            Flag4lb = pos(i, :) < lb;
            pos(i, :) = (pos(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
            for j=1:dim
                r1 = rand();
                r2 = rand();
                A1 = 2*a*r1 - a;
                C1 = 2*r2;
                D_alpha = abs(C1*Alpha_pos(j) - pos(i, j));
                X1(j) = (Alpha_pos(j)  - A1*D_alpha);
                r1 = rand();
                r2 = rand();
                A2 = 2*a*r1 - a;
                C2 = 2*r2;
                D_beta = abs(C2*Beta_pos(j) - pos(i, j));
                X2(j) = (Beta_pos(j) - A2*D_beta);
                r1 = rand();
                r2 = rand();
                A3 = 2*a*r1 - a;
                C3 = 2*r2;
                D_delta = abs(C3*Delta_pos(j) - pos(i, j));
                X3(j) =(Delta_pos(j)  - A3*D_delta);
            end
            pos(i, :) = ((X1 + X2 + X3) / 3);
            if fitness(i)<Alpha_score
                Alpha_score=fitness(i);
                Alpha_pos=pos(i,:);
            end
            if fitness(i) > Alpha_score && fitness(i) < Beta_score
                Beta_score = fitness(i);
                Beta_pos = pos(i, :);
            end
            if fitness(i) > Alpha_score && fitness(i) > Beta_score && fitness(i) < Delta_score
                Delta_score = fitness(i);
                Delta_pos = pos(i, :);
            end
    
            fitness(i)=fobj(pos(i,:));
            FEs = FEs + 1;
            
            if fitness(i) < Record_Best_Score, Record_Best_Score = fitness(i); end
            if FEs >= next_record_FEs
                Convergence_curve(end+1) = Record_Best_Score;
                next_record_FEs = next_record_FEs + n;
            end
        end
        
        if FEs >= MaxFEs, break; end
        
        for i=1:sizepop(1)
            if archive_fitness(i)<fitness(i)
                pos(i,:)=archive_pos(i,:);
                fitness(i)=archive_fitness(i);
            end
        end
        
        % F(t)=1-abs(normrnd(n*atan(t)/maxiter,0.01,[1 1])-1/2); 
        % Replaced t/maxiter with progress
        F_t=1-abs(normrnd(n*atan(progress),0.01,[1 1])-1/2); 
        
        [m, ~] = size(pos);
        random_matrix = rand(m, n);
        [~, indices] = sort(random_matrix(:,sizepop(1)));
        for i=1:sizepop(1)
            rand_pos(i,:)=pos(indices(i),:);
            rand_fitness(i)=fitness(indices(i));
        end
        
        % app(t) = ceil(ns - (ns - nd) * (t / maxiter));
        app_t = ceil(ns - (ns - nd) * progress);
        now_app=app_t;
        
        % NOTE: Subpop might depend on t/maxiter internally or purely on parameters
        % We pass t and maxiter as placeholder or we assume Subpop handles logic
        % Given Subpop signature: (sizepop,dim,pos,rand_fitness,now_app,jishu,t,maxiter)
        % We can pass FEs and MaxFEs if we modify Subpop, but Subpop is a .p file (protected).
        % We must maintain t and maxiter semantic if possible or simulate them.
        % Since we cannot change Subpop.p, we just pass t (iteration count) and an estimated maxiter
        % Estimated maxiter = MaxFEs / nPop?
        % Let's use t and a large number, or just keep incrementing t.
        % Ideally Subpop uses t/maxiter ratio.
        % Let's assume t is just used for some internal ratio.
        % We can pass t=FEs and maxiter=MaxFEs to preserve ratio? 
        % Or pass t=progress*1000 and maxiter=1000.
        
        fake_maxiter = 1000;
        fake_t = ceil(progress * fake_maxiter);
        if fake_t == 0, fake_t = 1; end
        
        [pos_1,pos_2,pos_3,pos_4,jishu] = Subpop(sizepop(1),dim,pos,rand_fitness,now_app,jishu,fake_t,fake_maxiter);
        
        sz1 = size(pos_1, 1);
        sz2 = size(pos_2, 1);
        sz3 = size(pos_3, 1);
        sz4 = size(pos_4, 1);
        
        for i=1:sizepop(1)
             r1=randi([1,sizepop(1)],1,1);
             while(r1==i)
                 r1=randi([1,sizepop(1)],1,1);
             end
             % new_r1
             if sz1 > 0
                 new_r1=randi([1,sz1],1,1);
                 try_count = 0;
                 while(new_r1==i) && try_count < 5
                     new_r1=randi([1,sz1],1,1);
                     try_count = try_count + 1;
                 end
             else
                 new_r1 = 1; % Fallback
             end
     
             % new_r2
             if sz2 > 0
                 new_r2=randi([1,sz2],1,1);
                 try_count = 0;
                 while((new_r2==new_r1)||(new_r2==i)) && try_count < 5
                     new_r2=randi([1,sz2],1,1);
                     try_count = try_count + 1;
                 end
             else
                 new_r2 = 1;
             end
     
             % new_r3
             if sz3 > 0
                 new_r3=randi([1,sz3],1,1);
                 try_count = 0;
                 while((new_r3==i)||(new_r3==new_r2)||(new_r3==new_r1)) && try_count < 5
                     new_r3=randi([1,sz3],1,1);
                     try_count = try_count + 1;
                 end
             else
                 new_r3 = 1;
             end
     
             % new_r4
             if sz4 > 0
                 new_r4=randi([1,sz4],1,1);
                 try_count = 0;
                 while((new_r4==i)||(new_r4==new_r2)||(new_r4==new_r1)||(new_r4==new_r3)) && try_count < 5
                     new_r4=randi([1,sz4],1,1);
                     try_count = try_count + 1;
                 end
             else
                 new_r4 = 1;
             end
     
             h=rand;
             R=0.5;
             if h>R
                 if sz1>0 && sz2>0
                    v(i,:)=pos(r1,:)+F_t*(pos_1(new_r1,:) -pos_2(new_r2,:));
                 else
                    v(i,:)=pos(r1,:);
                 end
             else
                 if sz3>0 && sz4>0
                    v(i,:)=pos(r1,:)+F_t*(pos_3(new_r3,:) -pos_4(new_r4,:));
                 else
                    v(i,:)=pos(r1,:);
                 end
             end
        end
        r=randi([1,dim],1,1);
        for j=1:dim
            cr=rand;
            if (cr<=CR)||(n==r)
                u(:,j)=v(:,j);
            else
                u(:,j)=pos(:,j);
            end
        end
        for i=1:sizepop(1)
            if FEs >= MaxFEs, break; end
            
            new_fitness(i)=fobj(u(i,:));
            FEs = FEs + 1;
            
            if new_fitness(i) < Record_Best_Score, Record_Best_Score = new_fitness(i); end
            if FEs >= next_record_FEs
                Convergence_curve(end+1) = Record_Best_Score;
                next_record_FEs = next_record_FEs + n;
            end
            
            if new_fitness(i)<fitness(i)
                pos(i,:)=u(i,:);
                fitness(i)=new_fitness(i);
            end
            if fitness(i)<Alpha_score
                Alpha_score=fitness(i);
                Alpha_pos=pos(i,:);
            end
        end
        
        if FEs >= MaxFEs, break; end
        
        for i=1:sizepop(1)
            Flag4ub = pos(i, :) > ub;
            Flag4lb = pos(i, :) < lb;
            pos(i, :) = (pos(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        end
        
        is_t = (1 - progress);
        deta_t = is_t * progress;
        
        if t>=2
            average_fitness=sum(Iter_Curve)/t;
            op_t=(Iter_Curve(t-1)-(Alpha_score))/(abs(average_fitness-Alpha_score) + eps);
            if 0<=op_t && op_t<=deta_t
                for i=1:sizepop(1)
                    if FEs >= MaxFEs, break; end
                    
                    step=Levy1(dim);
                    levy_pos(i,:)=pos(i,:)+step.*(Alpha_pos-pos(i,:));
                    levy_fitness(i)=fobj(levy_pos(i,:));
                    FEs = FEs + 1;
                    
                    if levy_fitness(i) < Record_Best_Score, Record_Best_Score = levy_fitness(i); end
                    if FEs >= next_record_FEs
                        Convergence_curve(end+1) = Record_Best_Score;
                        next_record_FEs = next_record_FEs + n;
                    end
                    
                    if levy_fitness(i)<fitness(i)
                        pos(i,:)=levy_pos(i,:);
                        fitness(i)=levy_fitness(i);
                    end
                    if fitness(i)<Alpha_score
                        Alpha_score=fitness(i);
                        Alpha_pos=pos(i,:);
                    end
                end
            end
        end
        
        archive_fitness=fitness;
        archive_pos=pos;
        gbest=Alpha_pos;
        gbestfitness=Alpha_score;
        Iter_Curve(t)=gbestfitness;
    end
end
