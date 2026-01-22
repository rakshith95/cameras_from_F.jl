function f = global_obj(x, FN)
% x: vector of cameras p1..pn
% FN: Multiview fundamental matrix

ncams = length(x)/12;
err = 0;
%K34 = get_commutation_matrix(3,4);
for i=1:ncams-1
    for j=i+1:ncams
        Fij = FN((i-1)*3+1:i*3, (j-1)*3+1:j*3);
        if ~any(any(Fij))
            continue
        end
        Pi = reshape(x((i-1)*12 + 1: i*12), 3,4);
        Pj = reshape(x((j-1)*12 + 1: j*12), 3,4);
        %Aj = kron(Pj'*Fij', eye(4))*K34 + kron(eye(4),Pj'*Fij') ;
        %err = err + norm(Aj*reshape(Pi,12,1));
        err = err + norm(Pi'*Fij*Pj + Pj'*Fij'*Pi, 2)^2;
    end
end
f = err;
end

function K = get_commutation_matrix(m, n)
    K = zeros(m*n, m*n);
    block_m = n;
    block_n = m;
    for i=1:m
        for j=1:n
            K((i-1)*block_m+j, (j-1)*block_n+i) = 1; 
        end
    end
end
