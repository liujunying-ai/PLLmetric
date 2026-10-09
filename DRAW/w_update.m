function [W] = w_update(train_data,train_p_target,k,Maxiter,lambda,mu,Y_ori)


train_p_target= train_p_target';
for i = 1:Maxiter
% 	fprintf('The %d-th iteration\n',i);
	W = obtain_W(train_data,Y_ori,k,lambda,mu);
% 	fprintf('Generate the labeling confidence...\n');
end

end