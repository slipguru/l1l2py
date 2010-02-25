function [selected,sparsity,k] = l1l2_regpath(X,Y,tau_values,mu_fact,sigma0,kmax)
% L2L2_REGPATH acceleration of l1l2 through cascade of l1l2 w. decreasing values of TAU
% 
% [SELECTED] = L1L2_REGPATH(X,Y,TAU_VALUES) for each value in TAU_VALUES
%   evaluates l1l2 solution with l2 parameter 0, and builds cell of indexes 
%   of selected features
% 
% [SELECTED,SPARSITY] = L1L2_REGPATH(X,Y,TAU_VALUES) also returns a vector
%   with the number of selected features for each value in TAU_VALUES
% 
% [SELECTED,SPARSITY,K] = L2L2_REGPATH(X,Y,TAU_VALUES) also returns a 
%   vector with the number of iterations for each value in TAU_VALUES
% 
% [...] = L1L2_REGPATH(X,Y,TAU_VALUES,MU_FACT) sets l2 parameter equal to
%   MU_FACT*SIGMA0
% 
% [...] = L1L2_REGPATH(X,Y,TAU_VALUES,MU_FACT,SIGMA0) uses SIGMA0+MU as 
%   step size
% 
% [...] = L1L2_REGPATH(X,Y,TAU_VALUES,MU_FACT,SIGMA0,KMAX) stops after at 
%   most KMAX iterations
 
if nargin<3; error('too few inputs!'); end
if nargin<4, mu_fact = 0; end
if nargin<5, sigma0 = []; end
if nargin<6, kmax = 1e5; end
if nargin>6; error('too many inputs!'); end

%%%%%%%%%%%%%%%%%%%%%%% CASCATA %%%%%%%%%%%%%%%%%%%%%%%%
T = length(tau_values);
[n,d] = size(X);
beta = cell(T,1);
selected = cell(T,1);
k = zeros(T,1);

if isempty(sigma0);
    if d>n; 
        a = normest(X*X');
        b = 0; 
    else
        aval = svd(X'*X);
        a = aval(1);
        b = aval(end);
    end
    sigma0 = (a+b)/(n*2); %step size for mu_fact=0
end
beta0 = zeros(d,1);
sparsity = zeros(T,1);
sparsity_prev = 0;
for t = 1:T;
    % when mu=0, if for larger value of tau l1l2 selected less than n 
    % variables, then keep running l1l2 for smaller values, 
    % o.w. take rls solution
    if and(mu_fact==0,sparsity_prev>=n);
        beta{T+1-t} = beta_ls;
    else
        [beta{T+1-t},k(T+1-t)] = l1l2_algorithm(X,Y,tau_values(T+1-t),mu_fact,beta0,sigma0,kmax);
    end
    beta0 = beta{T+1-t}; 
    selected{T+1-t} = beta{T+1-t}~=0; % selected variables
    sparsity(T+1-t) = sum(selected{T+1-t}); % number of selected variables
    sparsity_prev = sparsity(T+1-t);
end