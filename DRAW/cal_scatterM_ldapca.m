function  [St,Sw,Sb] = cal_scatterM_ldapca(train_data,Yconfidence,r)
% r:d-->r

train_data=train_data';
[N,M] = size(train_data); % N--the number of features
Q = size(Yconfidence,2); % Q--the number of labels， MxQ
m1=0;

for j=1:Q
    for i=1:M
        m1=m1+Yconfidence(i,j)*train_data(:,i);
    end
end

sumYconfidence=0;
for j=1:Q
    for i=1:M
        sumYconfidence=sumYconfidence+Yconfidence(i,j);
    end
end

m=m1/sumYconfidence; % miu
e=ones(1,M)';
Xm=train_data-m*e';

% the sum of weight for each label and the total sum of weights
% Nk -- a Q-dimensonal row vector
Wk = sum(Yconfidence); % 1xQ，每一行求和为1
Wt = sum(Wk);  % 就等于M

%N1=1./Nk; C的逆阵
for i=1:Q
    if(Wk(i)==0)
        W1(i)=0;
    else
        W1(i) = 1/Wk(i);
    end
end

sumL = sum(Yconfidence');
Ln=diag(sumL);
Hb = Yconfidence*diag(W1)*Yconfidence';
St=Xm*Ln*Xm';

Sb=Xm*Hb*Xm';
Sw=St-Sb;

% 类内散度，类间散度一定要对称吗？
Sw = (Sw + Sw')/2.0;
Sb = (Sb + Sb')/2.0;

end