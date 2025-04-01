function [x_opt] = SolveConeProg_CvxCcv(A_in, x_prev, max_it, tolerance)
    [m,n] = size(A_in);
    f = [zeros(n,1);ones(m,1)];
    socConstraints = secondordercone([eye(n) zeros(n,m)],zeros(n,1), zeros(n+m,1), -1);
    b_ineq = [zeros(2*m,1);-1];
    Aeq = [];
    beq = [];
    options = optimoptions('coneprog','OptimalityTolerance',1e-2, 'ConstraintTolerance',1e-2, 'Display','none', 'LinearSolver','normal');
    %options = optimoptions('coneprog', 'Display','none', 'LinearSolver','normal');
    A_ineq = [ [A_in -eye(m)]; [-A_in -eye(m)]; [-x_prev' zeros(1,m)] ];
    var_opt = coneprog(f, socConstraints, A_ineq, b_ineq, Aeq, beq,[],[],options);
    try
        x_opt = var_opt(1:n);
    catch
        x_opt = x_prev;
    end
end


