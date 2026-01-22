function x=global_optimizer(x0, F)
    %objective=@(x) obj(previous_variables,x);
    obj = @(x) global_obj(x, F);
    % x = fmincon(obj, x0, zeros(1,length(x0)), 0.0, zeros(1,length(x0)), 0.0, -inf(size(x0)), inf(size(x0)), @constr );
    x = fmincon(obj, x0, zeros(1,length(x0)), 0.0);

end

function [c,ceq] = constr(x)
    n = length(x)/12;
    c = zeros(n);
    ceq = zeros(n);
    for i=1:n
        ceq(i) = n - x'*x;
    end
end