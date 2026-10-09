function [ index_train,index_test,index_val ] = CV_data_partition( numInstances,numFolds,numFold, modes )
% Author: JiaBinBin
% Version: 1.0@2018-05-04 14:20
% Version: 2.0@2018-11-29 14:55
% Description: data set partition for n-fold Cross Validation
% Modification2.0: data set partition for Train:Val:Test = (numFolds-2):1:1
% Reference: weka.core.Instances (trainCV and testCV)
% Inputs:
%       numInstances: number of instances in dataset
%       numFolds    : n-folds
%       numFold     : n-th fold
%       modes       : 1-only Train&Test(default), 0-Train&Val&Test (V2.0)
% Outputs:
%       index_train : training set index (n-1 folds with modes = 1) or (n-2 folds with modes = 0)
%       index_test  : testing set index (1 fold)
%       index_val   : validation set index (1 fold) when modes = 0
    if nargin <4
        modes = 1;
    end
    if numFolds < 2
        error('Number of folds must be at least 2!');
    end
    if numFolds > numInstances
        error('Can not have more folds than instances!');
    end
    if numFold > numFolds
        error('numFold can not be bigger than numFolds!');
    end
    if modes==1
        index_all = 1:numInstances;
        numInstForFold = floor(numInstances/numFolds);
        if numFold < mod(numInstances,numFolds)+1
            numInstForFold = numInstForFold + 1;
            offset = numFold;
        else
            offset = mod(numInstances,numFolds)+1;
        end
        first = (numFold-1) * floor(numInstances/numFolds) + offset;
        last = first+numInstForFold-1;
        index_test = index_all(first:last);
        index_train = setdiff(index_all,index_test);
        index_val = [];
    else
        index_all = 1:numInstances;
        %Test index
        numInstForFold = floor(numInstances/numFolds);
        if numFold < mod(numInstances,numFolds)+1
            numInstForFold = numInstForFold + 1;
            offset = numFold;
        else
            offset = mod(numInstances,numFolds)+1;
        end
        test_first = (numFold-1) * floor(numInstances/numFolds) + offset;
        test_last = test_first+numInstForFold-1;
        index_test = index_all(test_first:test_last);
        %Validation index
        if numFold == numFolds
            numFold = 1;
        else
            numFold = numFold + 1;
        end
        numInstForFold = floor(numInstances/numFolds);
        if numFold < mod(numInstances,numFolds)+1
            numInstForFold = numInstForFold + 1;
            offset = numFold;
        else
            offset = mod(numInstances,numFolds)+1;
        end
        val_first = (numFold-1) * floor(numInstances/numFolds) + offset;
        val_last = val_first+numInstForFold-1;
        index_val = index_all(val_first:val_last);
        index_train = setdiff(index_all,[index_test,index_val]);
    end
%% The end!
end

