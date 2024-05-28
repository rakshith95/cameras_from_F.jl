function T = relative_projectivity( Pi, Pj )
% P and PP are cell arrays of PPMs
% T is a 4x4 projectivity s.t. Pj{k}  = Pi{k} * T 

% 
% for  k = 1:length(Pi)
%     % same norm of a matrix made of ones
%     Pi{k}  = Pi{k}/norm(Pi{k},'fro')*3.4641;
%     Pj{k} = Pj{k}/norm(Pj{k},'fro')*3.4641;
% 
% end

L = [];

for  k = 1:length(Pi)

    a =  Pj{k}(:);

    L = [L; (a'*a*eye(12)-a*a')*kron(eye(4),Pi{k});];


    % if (rank(L)~=15)
    %   warning('wrong rank!');
    % end

end

[~, ~, V] = svd(L);

T = reshape( V(:,end),4,4);

% fprintf('relative_projectivity error: %0.5g \n',norm(maxone(P{1}*T) - maxone(PP{1}),'fro'));

end



