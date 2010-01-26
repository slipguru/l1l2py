function [beta,k] = l1l2_algorithm(X,Y,tau,mu_fact,beta0,sigma0,kmax)
% L1L2_ALGORITHM Argmin of the least squares error with l1 and l2 penalties. 
%   [BETA] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT) returns the solution of the l1l2
%   regularization with l1 parameter TAU and l2 parameter MU_FACT*SIGMA0.
%   If the input data X is a NxD matrix, and the labels Y are a Nx1 vector,
%   BETA is the Dx1 vector. 
%   The step size is (A+B)/(N*2)+MU, where A and B are the largest and smaller
%   eigenvalues of X'*X and N is the number of training samples
%   The algorithm stops when each element of BETA reached convergence.
%
%   [BETA,K] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT) also returns the number of iterations.
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT,BETA0) uses BETA0 as initialization vector
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT,BETA0,SIGMA0) uses the step size
%   SIGMA0+MU. If SIGMA0=[], the step size is (A+B)/(N*2)+MU.
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT,BETA0,SIGMA0,KMAX) the algorithm stops after
%   KMAX iterations or when each element of BETA reached convergence.
%
    if nargin<3; error('too few inputs!'); end
    if nargin<4; mu_fact = 0; end
    if nargin<5; beta0 = []; end
    if nargin<6; sigma0=[]; end
    if nargin<7; kmax = 1e5; end
    if nargin>7; error('too many inputs!'); end
    
    [n,d] =size(X);
    XT = X';

% if sigma0 is not specified in input, evaluates it as (a+b)/(n*2)
    if isempty(sigma0);
        if d>n; 
            a = normest(X*XT);
            b = 0; 
        else
            aval = svd(XT*X);
            a = aval(1);
            b = aval(end);
        end
        sigma0 = (a+b)/(n*2); %step size for mu_fact=0
    end
    
    tol = 0.01; % tolerance for stopping rule
    kmin = 100; % minimum number of iterations
    
% initialization 
    mu = mu_fact*sigma0;
    sigma = sigma0+mu;
    mu_s = mu/sigma;
    tau_s = tau/sigma;    % twice the threshold
    XT = XT./(n*sigma);
    if isempty(beta0); % initialize with RLS solution
        beta0 = rls_algorithm(X,Y); 
    end   
% l1l2 algorithm
    k = 0;
    beta = thresholding(beta0.*(1-mu_s)+XT*(Y-X*beta0),tau_s); % first iteration
    log=1;
    while or(and(k<kmax,log),k<kmin)
        % the threshold grows with the iteration
        if all(abs(beta-beta0)<=(abs(beta0)*tol/(k+1))), log=0; end
        beta0 = beta;
        beta = thresholding(beta0.*(1-mu_s)+XT*(Y-X*beta0),tau_s);
        k = k+1;
    end
