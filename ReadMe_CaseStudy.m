%This is an exemplar file on how to obtain the experimental result in Section 4 (A case study)
clear;clc;close all;fclose('all');

%include the DRAW & PLDA package, some modifications have been made to the software packages for convenience
addpath('DRAW');%avaiable at: https://palm.seu.edu.cn/zhangml/files/DRAW.zip
addpath('PLDA');%avaiable at: https://github.com/wwangwitsel/PLDA
algo_name = 'casestudy';

%More data sets are publicly available at: 
%https://palm.seu.edu.cn/zhangml/Resources.htm#partial_data
data_name = 'MSRCv2';
load(['data\',data_name]);%load the partial label data set
X_load = zscore(data);%help zscore
y_load_p = transpose(full(partial_target));%partial target
y_load_r = transpose(full(target));%real target

%create log file
title_str = ['log_',algo_name, '_',data_name,'_',TimeStr(clock,1)];
temp_str = [title_str,'.txt'];
all_fid = fopen(temp_str,'w');
% all_fid = 1;%for standard output, the screen

numFolds = 10;%ten-fold cross validation
num_algos = 6;
disACCt = zeros(numFolds,num_algos);
teACC = zeros(numFolds,num_algos);
oraACC = zeros(numFolds,num_algos);
for numFold=1:numFolds
    temp_str = ['Fold-', num2str(numFold), ' begins (',TimeStr(clock,0),') ...\n'];
    fprintf(all_fid,temp_str);
    
    %split dataset into training set and testing set
    X_train = X_load(index_train{numFold},:);
    y_train = y_load_p(index_train{numFold},:);%partial target
    y_train_r = y_load_r(index_train{numFold},:);%real target
    X_test = X_load(index_test{numFold},:);
    y_test = y_load_r(index_test{numFold},:);%real target
    
    %feature augmentation
    [X_train_PLDA, X_test_PLDA] = PLDA(X_train, y_train, X_test); 
    
    %dimensionality reduction
    P_DRAW= Draw(X_train, y_train');
    X_train_DRAW = X_train*P_DRAW;
    X_test_DRAW = X_test*P_DRAW; 

    %(1)Base learner: PLKNN
    i_algo = 1;%PLKNN
    teACC(numFold,i_algo) = PL_kNN(X_train,y_train',X_test,y_test');
    [~, ~, train_outputs] = PL_kNN(X_train,y_train',X_train,y_train_r');
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs, y_train_r, y_train);
    oraACC(numFold,i_algo) = PL_kNN(X_train,y_train_r',X_test,y_test');
    
    i_algo = 2;%PLKNN-PLDA
    teACC(numFold,i_algo) = PL_kNN(X_train_PLDA,y_train',X_test_PLDA,y_test');
    [~, ~, train_outputs] = PL_kNN(X_train_PLDA,y_train',X_train_PLDA,y_train_r');
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs, y_train_r, y_train);
    oraACC(numFold,i_algo) = PL_kNN(X_train_PLDA,y_train_r',X_test_PLDA,y_test');
    
    i_algo = 3;%PLKNN-DRAW
    teACC(numFold,i_algo) = PL_kNN(X_train_DRAW,y_train',X_test_DRAW,y_test');
    [~, ~, train_outputs] = PL_kNN(X_train_DRAW,y_train',X_train_DRAW,y_train_r');
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs, y_train_r, y_train);
    oraACC(numFold,i_algo) = PL_kNN(X_train_DRAW,y_train_r',X_test_DRAW,y_test');
    
    %(2)Base learner: PLSVM
    i_algo = 4;%PLSVM
    model = PLSVM_train(X_train,y_train');
    teACC(numFold,i_algo) = PLSVM_predict(X_test,y_test',model);%test accuracy
    [~,~,train_outputs] = PLSVM_predict(X_train,y_train_r',model);%training accuracy
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs', y_train_r, y_train);
    model = PLSVM_train(X_train,y_train_r');
    oraACC(numFold,i_algo) = PLSVM_predict(X_test,y_test',model);%supervised accuracy
    
    i_algo = 5;%PLSVM-PLDA
    model = PLSVM_train(X_train_PLDA,y_train');
    teACC(numFold,i_algo) = PLSVM_predict(X_test_PLDA,y_test',model);%test accuracy
    [~,~,train_outputs] = PLSVM_predict(X_train_PLDA,y_train_r',model);%training accuracy
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs', y_train_r, y_train);
    model = PLSVM_train(X_train_PLDA,y_train_r');
    oraACC(numFold,i_algo) = PLSVM_predict(X_test_PLDA,y_test',model);%supervised accuracy
    
    i_algo = 6;%PLSVM-DRAW
    model = PLSVM_train(X_train_DRAW,y_train');
    teACC(numFold,i_algo) = PLSVM_predict(X_test_DRAW,y_test',model);
    [~,~,train_outputs] = PLSVM_predict(X_train_DRAW,y_train_r',model);
    disACCt(numFold,i_algo) = calc_acc_train(train_outputs', y_train_r, y_train);
    model = PLSVM_train(X_train_DRAW,y_train_r');
    oraACC(numFold,i_algo) = PLSVM_predict(X_test_DRAW,y_test',model);
end
%display
fprintf(all_fid,['Metric ', ' & ', 'PLKNN', ' & ', 'PLKNN-PLDA', ' & ', 'PLKNN-DRAW', ...
    ' & ', 'PLSVM', ' & ', 'PLSVM-PLDA', ' & ', 'PLSVM-DRAW', '\n']);
temp_str = 'disACCt';
for i_algo=1:num_algos
    temp_str_i = [num2str(mean(disACCt(:,i_algo)),'%4.3f'),'¡À', num2str(std(disACCt(:,i_algo)),'%4.3f')];
    temp_str = [temp_str, ' & ', temp_str_i];
end
fprintf(all_fid,[temp_str, '\n']);

temp_str = ' oraACC';
for i_algo=1:num_algos
    temp_str_i = [num2str(mean(oraACC(:,i_algo)),'%4.3f'),'¡À', num2str(std(oraACC(:,i_algo)),'%4.3f')];
    temp_str = [temp_str, ' & ', temp_str_i];
end
fprintf(all_fid,[temp_str, '\n']);

temp_str = '  teACC';
for i_algo=1:num_algos
    temp_str_i = [num2str(mean(teACC(:,i_algo)),'%4.3f'),'¡À', num2str(std(teACC(:,i_algo)),'%4.3f')];
    temp_str = [temp_str, ' & ', temp_str_i];
end
fprintf(all_fid,[temp_str, '\n']);
save(title_str,'disACCt','teACC','oraACC','algo_name','data_name');