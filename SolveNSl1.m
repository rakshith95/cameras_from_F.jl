function [x_opt] = SolveNSl1(A_in)
    [m,n] = size(A_in);
    f = [zeros(n,1);zeros(m,1);ones(m,1)];
    A_ineq = [ [zeros(m,n),eye(m),-eye(m)] ; [zeros(m,n),-eye(m),-eye(m)] ];
    b_ineq = zeros(2*m,1);
    A_eq = [ [-A_in,eye(m),zeros(m,m)] ; [ones(1,n),zeros(1,m),zeros(1,m)] ];
    b_eq = [zeros(m,1);1];
    sSolverOptions = optimoptions('linprog', 'Display', 'off');
    vX = linprog(f, A_ineq, b_ineq, A_eq, b_eq, [], [], sSolverOptions);
    x_opt = vX(1:n);
end