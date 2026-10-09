%This is an exemplar file on how to compute the defined evaluation metrics with PL-AGGD (rbf kernel) as the base learner
clear;clc;close all;fclose('all');

%include the PL-AGGD package, avaiable at: https://palm.seu.edu.cn/zhangml/files/PL-AGGD.rar
addpath('PLAGGD');%Note that we have made some necessary modifications to the software package.
algo_name = 'PLAGGDrbf';

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
disACCd = zeros(numFolds,1);
disACCt = zeros(numFolds,1);
teACC = zeros(numFolds,1);
oraACC = zeros(numFolds,1);
time_traditional = zeros(numFolds,1);
time_additional = zeros(numFolds,1);
for numFold=1:numFolds
    temp_str = ['Fold-', num2str(numFold), ' begins (',TimeStr(clock,0),') ...\n'];
    fprintf(all_fid,temp_str);
    
    %split dataset into training set and testing set
    X_train = X_load(index_train{numFold},:);
    y_train = y_load_p(index_train{numFold},:);%partial target
    y_train_r = y_load_r(index_train{numFold},:);%real target
    X_test = X_load(index_test{numFold},:);
    y_test = y_load_r(index_test{numFold},:);%real target
    
    %hyper-parameters
    ker = 'rbf'; %Type of kernel function
    par = 1*mean(pdist(X_train)); %Parameters of kernel function
    k = 10; %Number of neighbors
    lambda = 1;
    mu = 1;
    gama = 0.05;
    Maxiter = 10;
    
    %train & test
    start_clock1 = clock;
    [label_recover, train_outputs, test_outputs] = PL_AGGD(X_train,y_train,X_test,k,ker,par,Maxiter,lambda,mu,gama);
    finish_clock1 = clock;
    time_traditional(numFold) = etime(finish_clock1,start_clock1);
    
    start_clock2 = clock;
    [~, ~, test_outputs_ora] = PL_AGGD(X_train,y_train_r,X_test,k,ker,par,Maxiter,lambda,mu,gama);%replace y_train with y_train_r
    finish_clock2 = clock;
    time_additional(numFold) = etime(finish_clock2,start_clock2);
    
    %calculate accuracy
    disACCd(numFold) = calc_acc_train(label_recover, y_train_r, y_train);
    disACCt(numFold) = calc_acc_train(train_outputs, y_train_r, y_train);
    teACC(numFold) = calc_acc_test(test_outputs, y_test);
    oraACC(numFold) = calc_acc_test(test_outputs_ora, y_test);
end
%display
temp_str = [ 'disACCd = ', num2str(mean(disACCd),'%4.3f'),'¡À', num2str(std(disACCd),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
temp_str = [ 'disACCt = ', num2str(mean(disACCt),'%4.3f'),'¡À', num2str(std(disACCt),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
temp_str = [ ' oraACC = ', num2str(mean(oraACC),'%4.3f'),'¡À', num2str(std(oraACC),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
temp_str = [ '  teACC = ', num2str(mean(teACC),'%4.3f'),'¡À', num2str(std(teACC),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
temp_str = [ 'total time (traditional) = ', num2str(sum(time_traditional),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
temp_str = [ 'total time (additional)  = ', num2str(sum(time_additional),'%4.3f'), '\n'];fprintf(all_fid,temp_str);
save(title_str,'disACCd','disACCt','teACC','oraACC','time_traditional','time_additional','algo_name','data_name');