function [P, train_outputs, test_outputs] = SURE(train_data, train_p_target, test_data, test_target, optmParameter)

max_iter = optmParameter.maxIter;
lambda = optmParameter.lambda;
beta = optmParameter.beta;

par = optmParameter.par; %mean(pdist(train_data));
ker = optmParameter.ker; %'rbf';

l = size(test_target, 2);
Aeq = ones(1, l);
beq = 1;
opts = optimoptions('quadprog',...
    'Algorithm','interior-point-convex','Display','off');
lb = sparse(l, 1);
H = 2*speye(l, l);
P = train_p_target;

for iter = 1:max_iter
    [train_outputs, test_outputs] = kernelRidgeRegression(train_data, P, test_data, beta, par, ker);
    [P] = solveQP(train_p_target, train_outputs, H, Aeq, beq, lb, opts, lambda);
end

% [Ytrain] = kernelRidgeRegression(train_data, P, test_data, beta, par, ker);
% [accuracy_tr1] = CalAccuracy(P.*train_p_target, train_target);%training accuracy
% [accuracy_tr2] = CalAccuracy(Ytrain.*train_p_target, train_target);%training accuracy (predict-specifc)
% [accuracy_tr3] = CalAccuracy(Ytrain, train_target);%training accuracy (predict)

end