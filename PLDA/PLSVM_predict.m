function [accuarcy,predictLabel,outputValue] = PLSVM_predict(test_data,test_target,model)


% A maximum margin approach to partial label learning.
% This function is the prediction phase of the algorithm. 
%    Syntax
%
% [accuarcy,predictLabel,outputValue] = PLSVM_predict(test_data,test_target,model)
% 
%    Description
%
%       CLPL_predict takes,
%           model                       - the model which returned in the training phase
%           test_data                   - An MxN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target                 - A QxM array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
% 
%      and returns,
%            accuarcy                     - Predictive accuracy on the test set
%            predictLabel                 - A QxM array, if the ith test instance is predicted to have the jth class label, then predictLabel(j,i) is 1, otherwise predictLabel(j,i) is 0
%            outputValue                  - A QxM array, the numerical output of the ith test instance on the jth class label is stored in Outputs(j,i)
%  [1]N. Nguyen and R. Caruana, ¡°Classification with partial labels,¡±in Proceedings of the 14th ACM SIGKDD International Conference onKnowledge Discovery and Data Mining, Las Vegas, NV, 2008, pp.381¨C389.
if strcmp(model.type,'PLSVM')==0
    error('The input model does not match the prediction model')
end
fea_num = size(test_data,2);
if fea_num ~= model.fea_num
    error('feature size of test data does not match the feature size of training data');
end

if size(test_data,1)~=size(test_target,2)
    error('Length of label vector does match the number of instances');
end

    outputValue = model.w*test_data';
    [~,predictLabel] = max(outputValue);
    [~,real] = max(full(test_target));
    [label_num,test_num] = size(test_target);
    accuarcy = sum(predictLabel==real)/size(test_data,1);
        LabelMat = repmat((1:label_num)',1,test_num);
    predictLabel = repmat(predictLabel,label_num,1)==LabelMat;