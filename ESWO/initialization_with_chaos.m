% This function initialize the first population of search agents
function Positions = initialization_with_chaos(SearchAgents_no, dim, ub, lb)
    Boundary_no = length(ub);

    r = 4; 

    x = zeros(SearchAgents_no, dim);
    x(1,:) = rand(1,dim);

    for j = 2:SearchAgents_no
        alpha = 0.7 + (0.9 - 0.7) * rand(1,dim);

        for k = 1:dim
            if x(j-1,k) < 0.5
                fTent = 2 * x(j-1,k);
            else
                fTent = 2 * (1 - x(j-1,k));
            end
            fLogistic = r * x(j-1,k) * (1 - x(j-1,k));

            x(j,k) = alpha(k) * fTent + (1 - alpha(k)) * fLogistic;
        end
    end

    Positions = x;

    if Boundary_no == 1
        Positions = Positions .* (ub - lb) + lb;
    end

    if Boundary_no > 1
        for i = 1:dim
            ub_i = ub(i);
            lb_i = lb(i);
            Positions(i, :) = Positions(i, :) .* (ub_i - lb_i) + lb_i;
        end
    end
end

