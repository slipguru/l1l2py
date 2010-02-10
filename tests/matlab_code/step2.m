function [beta_opt] = step2(X, Y, Xtest, Ytest, tau_opt, lambda_opt, mu_range, err_type, norm_mean, norm_col)

[n, d] = size(X);

err_test = zeros(1,1); 
selected = zeros(d, length(mu_range));
beta_opt = cell(1, length(mu_range));

[X,Y,Xtest,Ytest,meanY] = normalization(X,Y,norm_mean,norm_col,Xtest,Ytest);
beta = l1l2_algorithm(X,Y,tau_opt,mu_range(1));
selected(:,1) = (beta~=0); % mark selected variables
beta_opt{1} = rls_algorithm(X(:,logical(selected(:,1))),Y,lambda_opt);
% calculate kcv and training errors
err_test = linear_test(Xtest(:,logical(selected(:,1))),Ytest,beta_opt{1},err_type,meanY);

% for fixed lambda opt find solutions with different correlation
% paramters epsilon
for m = 2:length(mu_range);
    beta = l1l2_algorithm(X,Y,tau_opt,mu_range(m));
    selected(:,m) = (beta~=0);    
    beta_opt{m} = rls_algorithm(X(:,logical(selected(:,m))),Y,lambda_opt);
end
    
