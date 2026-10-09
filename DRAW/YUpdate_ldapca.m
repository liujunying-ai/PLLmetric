function [Y_new,Iterflag] = YUpdate_aggd_ldapca(lower_train_data, Y_ori, k,partial_target,InsSet)

Iterflag = 0;

M = size(lower_train_data, 2); % dxM
Q = size(Y_ori, 2);

% parameter for PL-AGGD
ker  = 'rbf'; %Type of kernel function
aggd.k=k; % Number of neighbors
lambda = 1;
mu = 1;
gama = 0.05;

data= lower_train_data'; 
data = normr(data);
par = 1*mean(pdist(data)); %Parameters of kernel function
%training
Maxiter=1;

F0= Initialization_labelConfidence(partial_target);
[W] = w_update(data,partial_target,aggd.k,Maxiter,lambda,mu,Y_ori);

% 根据得到的W进行标记置信度更新
%   find knn neighbours
Mdl = KDTreeSearcher(data); 
[Idx,Dis] = knnsearch(Mdl, data, 'k', aggd.k + 1);
Idx = Idx(:, 2:end);

%   calculate matrix Z and V
Z = zeros(M, Q);
V = zeros(M, Q);
for i = 1:M
    neighbours = Idx(i, :);
    for nei = 1:aggd.k
        ind = neighbours(nei);
        
        nLabels = InsSet(ind, 1);
        Labels = InsSet(ind, 2:(nLabels + 1));
        
        for it = 1:nLabels
            j = Labels(it);
            Z(i, j) = Z(i, j) + Y_ori(ind, j) * W(i,ind);
            V(i, j) = V(i, j) + 1;
        end 
    end
end



Z_New = Y_ori;
for i = 1:M
    nLabels = InsSet(i, 1);
    Labels = InsSet(i, 2:(nLabels + 1));
    
    for it = 1:nLabels
        j = Labels(it);
        Z_New(i, j) = Z_New(i, j) + Z(i, j);
    end
    
    if sum(Z_New(i, :)) ~= 0
        Z_New(i, :) = Z_New(i, :) ./ sum(Z_New(i, :));
    else
        Z_New(i, :) = Y_ori(i, :);
    end
end
Y_new = Z_New;
end