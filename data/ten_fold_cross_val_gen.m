function ten_fold_cross_val_gen(data_name)
%This function is used to generate the index of ten-fold cross validation

    load(data_name);%load the partial label data set
    numFolds = 10;%ten-fold cross validation
    numInstances = size(data,1);
    
    %fix random seed
    rng(1,'v5uniform');%rand('state', 1);
    index_rand = randperm(numInstances);%shuffle
    index_train = cell(numFolds,1);
    index_test = cell(numFolds,1);
    for numFold=1:numFolds
        [idx_tr,idx_te] = CV_data_partition(numInstances,numFolds,numFold);
        index_train{numFold} = index_rand(idx_tr);%shuffle
        index_test{numFold} = index_rand(idx_te);%shuffle
    end
    save(data_name, 'data', 'target', 'partial_target', 'index_train', 'index_test');
end