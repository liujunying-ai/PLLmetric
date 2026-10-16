clear;clc;close all;fclose('all');
start_clock = clock;

%load dataset
load('lost.mat');%preprocessed by ten_fold_cross_val_gen.m in the "data" folder 
X_load = zscore(data);%zero mean and unit standard deviation
y_load_p = transpose(partial_target);%partial target
y_load_r = transpose(target);%real target
data_name = 'Lost';

%create log file
title_str = ['log_deepPLL_',data_name,'_',TimeStr(clock,1)];   
temp_str = [title_str,'.txt'];
all_fid = fopen(temp_str,'w');

%parameters
num_loop = 10;%10 runs

disACCd = zeros(num_loop,1); 
disACCt = zeros(num_loop,1);
oraACC = zeros(num_loop,1); 
teACC = zeros(num_loop,1); 

%split dataset into training set and testing set
X_train = X_load(index_train{1},:);
y_train = y_load_p(index_train{1},:);
y_train_r = y_load_r(index_train{1},:);
X_test = X_load(index_test{1},:);
y_test = y_load_r(index_test{1},:);

% main
label_recover = build_label_manifold(X_train,y_train,10);%disambiguation
hiddenLayerSize = size(y_train,2);%the number of labels
for loop=1:num_loop
    temp_str = ['Loop-', num2str(loop), ' begins (',TimeStr(clock,0),') ...\n'];
    fprintf(all_fid,temp_str);
    
    %learn with disambiguated labeling information
    net = fitnet(hiddenLayerSize);
    net = train(net, X_train', label_recover');
    train_outputs = net(X_train');train_outputs=train_outputs';
    test_outputs = net(X_test');test_outputs=test_outputs';
    disACCd(loop) = calc_acc_train(label_recover, y_train_r, y_train);
    disACCt(loop) = calc_acc_train(train_outputs, y_train_r, y_train);
    teACC(loop) = calc_acc_test(test_outputs, y_test);

    %learn with ground-truth labeling information
    net = fitnet(hiddenLayerSize);
    net = train(net, X_train', y_train_r');
    train_outputs = net(X_train');train_outputs=train_outputs';
    test_outputs = net(X_test');test_outputs=test_outputs';
    oraACC(loop) = calc_acc_test(test_outputs, y_test);
end

%display
fprintf(all_fid,'\nExperimental results:\n');
temp_str = 'disACCd: ';
for loop=1:num_loop
    temp_str = [temp_str, num2str(disACCd(loop),'%4.3f'), ' & '];
end
temp_str = [temp_str, num2str(mean(disACCd),'%4.3f'), '¡À', num2str(std(disACCd),'%4.3f')];
fprintf(all_fid,temp_str);fprintf(all_fid,'\n');

temp_str = 'disACCt: ';
for loop=1:num_loop
    temp_str = [temp_str, num2str(disACCt(loop),'%4.3f'), ' & '];
end
temp_str = [temp_str, num2str(mean(disACCt),'%4.3f'), '¡À', num2str(std(disACCt),'%4.3f')];
fprintf(all_fid,temp_str);fprintf(all_fid,'\n');

temp_str = ' oraACC: ';
for loop=1:num_loop
    temp_str = [temp_str, num2str(oraACC(loop),'%4.3f'), ' & '];
end
temp_str = [temp_str, num2str(mean(oraACC),'%4.3f'), '¡À', num2str(std(oraACC),'%4.3f')];
fprintf(all_fid,temp_str);fprintf(all_fid,'\n');

temp_str = '  teACC: ';
for loop=1:num_loop
    temp_str = [temp_str, num2str(teACC(loop),'%4.3f'), ' & '];
end
temp_str = [temp_str, num2str(mean(teACC),'%4.3f'), '¡À', num2str(std(teACC),'%4.3f')];
fprintf(all_fid,temp_str);fprintf(all_fid,'\n');

finish_clock = clock;
temp_str = ['Start Time: ',TimeStr(start_clock,0),'\n'];fprintf(all_fid,temp_str);
temp_str = ['Finish Time: ',TimeStr(finish_clock,0),'\n'];fprintf(all_fid,temp_str);
temp_str = [title_str,'.mat'];
save(temp_str,'start_clock','finish_clock','data_name','disACCd','disACCt','oraACC','teACC');
fclose(all_fid);
