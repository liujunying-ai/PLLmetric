function [projection_matrix, lower_train_data] = Draw(data,partial_target,selfpara)

%default parameter setting
if nargin<3 || isempty(selfpara)
    num_features = size(data,2);
    num_label = size(partial_target,1);
    reduced_dim = min(num_label-1,num_features);
    selfpara.k = 8;
    selfpara.beta = 0.5;
    selfpara.decay = 0.95;
    selfpara.r = reduced_dim;
    selfpara.T = 100;
end

% combine pca and lda as dimension reduction objective
[n,d] = size(data);
q = size(partial_target,1);
k = selfpara.k;
decay = selfpara.decay;

miu = mean(data,1);
Q = size(partial_target,1);
M = size(partial_target,2);
LabelSet = zeros(Q, M + 6);     %   LabelSet(i, 1) represent the number of instances regarding label i and LableSet(i, j + 1) represent the index of j_th instance
InsSet = zeros(M, Q + 6);       %   InsSet(i, 1) represent the number of labels regarding ins i and InsSet(i, j + 1) represent the j_th label

%   Initialize LabelSet and InsSet
train_p_target = partial_target';
for i = 1:M
    for j = 1:Q
        if train_p_target(i, j) == 1
            LabelSet(j, 1) = LabelSet(j, 1) + 1;
            t = LabelSet(j, 1);
            LabelSet(j, t + 1) = i;
            
            InsSet(i, 1) = InsSet(i, 1) + 1;
            t = InsSet(i, 1);
            InsSet(i, t + 1) = j;
        end
    end
end

%   Initialize label confidence matrix Y
Y = zeros(M, Q);
for i = 1:M
    tot = InsSet(i, 1);
    if tot ~= 0
        for j = 1:tot
            la = InsSet(i, j + 1);
            Y(i, la) = 1 / tot;
        end
    end
    Y(i, :) = Y(i, :) / norm(Y(i, :), 1);
end


Y_0 = Initialization_labelConfidence(partial_target);
beta = selfpara.beta;
Yconfidence = Y_0;
for T=1:selfpara.T
    
    % PCA
    St_pca = zeros(d,d);
    St_pca = (data-repmat(miu,n,1))'*(data-repmat(miu,n,1));
    Sw_pca = diag(ones(1,d));
    
    [St_LDA,Sw_LDA,Sb_LDA] = cal_scatterM_ldapca(data,Yconfidence,selfpara.r);
    St_rlb = Sb_LDA;
    if beta>1
        Sw_rlb = beta*Sw_pca;        
    else
        Sw_rlb = (1-beta)*Sw_LDA + beta*Sw_pca;
    end
    
%     Sw_rlb = (1-beta)*Sw_LDA + beta*Sw_pca;
    
    % 基于混合St与Sb求解，得到对应的映射矩阵
    [V,D] = eig(St_rlb,Sw_rlb);

    eigenVectors = V;
    eigenValues= diag(D);
    nr_dimension = size(eigenValues);
    for i=1:nr_dimension
        if((isreal(eigenValues(i))==1) & (eigenValues(i) > 0.0) & (isinf(eigenValues(i)) == 0)) %real positive eigenvalue
            continue;
        end
        eigenValues(i)=0.0;
    end
    [eigenValues,order] = sort(eigenValues, 'descend');
    eigenVectors = eigenVectors(:,order);
    
    reduced_dimension = selfpara.r;
    % 用当前的映射矩阵得到结果
    P = eigenVectors(:, 1:reduced_dimension);
    
    X1=P'*data';
    X1=X1';
    lower_train_data= X1;
    
    % update label confidence
    temp = Yconfidence;
    [Y_new,Iterflag] = YUpdate_ldapca(lower_train_data', Yconfidence,k,partial_target,InsSet);
    Yconfidence = Y_new;
    beta = beta*decay^T;
    projection_matrix = P;
end
end


