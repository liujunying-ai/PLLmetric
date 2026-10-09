function model = PLSVM_train(train_data,train_p_target,lambda,T)

% A maximum margin approach to partial label learning.
% This function is the training phase of the algorithm. 
%
%    Syntax
%
%      model = PLSVM_train( trainData,trainTarget,k,alpha )
%
%    Description
%
%       PLSVM_train takes,
%           train_data                  - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target              - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           lambda                      - The regularization parameter (defalut 1)
%           T                           - The  maximum number of iterations 
%      and returns,
%           model is a structure continues following elements
%           model.w                      -A A QxN array, the parameters of the svm model, w(j,:) is the paramters for class j.
%           model.fea_num               - # features of training data


%  [1]N. Nguyen and R. Caruana, ¡°Classification with partial labels,¡±in Proceedings of the 14th ACM SIGKDD International Conference onKnowledge Discovery and Data Mining, Las Vegas, NV, 2008, pp.381¨C389.

if nargin<4
    T = 1000;
end
if nargin<3
    lambda = 1;
end
if nargin<2
    error('Not enough input parameters, please check again.');
end
if size(train_data,1)~=size(train_p_target,2)
    error('Length of label vector does match the number of instances');
end


label_num = size(train_p_target,1);


    [ins_num,fea_num] = size(train_data);

   loss_old = Inf;
    w = rand(label_num,fea_num);
    normw = norm(reshape(w,label_num*fea_num,1));
    if normw>1/lambda^0.5
        w = w/(lambda^0.5*normw);
    end
    insList = 1:ins_num;
    for t=1:T
        if mod(t,10)==0
            disp(['PL_SVM iteration: ',num2str(t)]);
        end

        value = w*train_data';
        [maxPosVal,maxPosIdx] = max(value-1e5*(train_p_target==0));
        [maxNegVal,maxNegIdx] = max(value-1e5*(train_p_target==1));
        violateIdx = insList((maxPosVal-maxNegVal)<1);
%         size(violateIdx)
        violateLabelPos = maxPosIdx(violateIdx);
        violateLabelNeg = maxNegIdx(violateIdx);
        wtemp = zeros(label_num,fea_num);
        for label=1:label_num
            violate_label = violateIdx(violateLabelPos==label);
            wtemp(label,:) = sum(train_data(violate_label,:));
            violate_label = violateIdx(violateLabelNeg==label);
            wtemp(label,:) = wtemp(label,:) - sum(train_data(violate_label,:));
        end
        niu = 1/(lambda*t);
%         wold = w;
        wtemp = wtemp*niu/ins_num+(1-niu*lambda)*w;
        normw = norm(reshape(wtemp,label_num*fea_num,1));
        if 1/(lambda^0.5*normw)<1
            w = wtemp*(1/(lambda^0.5*normw));
        else
            w = wtemp;
        end
        xi = 1-maxPosVal+maxNegVal;
        xi(xi<0) = 0;
        loss = norm(w)*lambda/2+mean(xi);
        if abs(loss_old-loss)<=0.001
            break
        end
        loss_old = loss;
    end
    model.w = w;
    model.fea_num = fea_num;
    model.type = 'PLSVM';
end

