function accuracy = kNN_pl(lower_data,partial_target,target,tr_idx,te_idx)
%PL_kNN:A k-nearest neighbor approach to partial label learning
%
%    Syntax
%
%       [ output_args ] = PL_kNN( train_data,train_p_target,test_data,test_target,k)
%
%    Description
%
%       PLL_MIL takes,
%           train_data     - An MxN array, the ith instance of training instance is stored in train_data(i,:)
%           train_p_target - A QxM array, if the jth class label is one of the partial labels for the ith training instance, then train_p_target(j,i) equals +1, otherwise train_p_target(j,i) equals 0
%           test_data      - An M2xN array, the ith instance of testing instance is stored in test_data(i,:)
%           test_target    - A QxM2 array, if the jth class label is the ground-truth label for the ith test instance, then test_target(j,i) equals 1; otherwise test_target(j,i) equals 0
%           k              - the number of the neighboors
%      and returns,
%           Outputs        - A QxM2 array, the numerical output of the ith test instance on the jth class label is stored in Outputs(j,i)
%           Pre_Labels     - A QxM2 array, if the ith test instance is predicted to have the jth class label, then Pre_Labels(j,i) is 1, otherwise Pre_Labels(j,i) is 0
%           Accuracy       - Predictive accuracy on the test set
%
% E. H ¡§ ullermeier and J. Beringer, ¡°Learning from ambiguously labeled examples,¡± Intelligent Data Analysis, vol. 10, no. 5, pp. 419¨C439, 2006


num_kfold = size(tr_idx,1);
acc_lowerdvec = [];
for w=1:num_kfold
    train_idx = tr_idx{w,1};
    test_idx = te_idx{w,1};
    
    train_ptarget = partial_target(:,train_idx);
    test_target = target(:,test_idx);
    
    % using dimension reduction
    lower_train_data = lower_data(train_idx',:);
    lower_test_data = lower_data(test_idx',:);
    pl_knn.k = 10;
    [accuracy_lowerd,predictLabel,outputValue] = PL_kNN(lower_train_data,train_ptarget,lower_test_data,test_target,pl_knn.k);
    acc_lowerdvec = [acc_lowerdvec,accuracy_lowerd];
end
accuracy =mean(acc_lowerdvec);

end

