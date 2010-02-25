function [beta,k] = l1l2_algorithm(X,Y,tau,mu_fact,beta0,sigma0,kmax)
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
    %mu = mu_fact*sigma0;
    %sigma = sigma0+mu;
    mu = mu_fact;
    sigma = sigma0; % C = n*sigma
    
    mu_s = mu/sigma; % mu/sigma = (mu*n)/C
    tau_s = tau/sigma; % tau/sigma = (tau*n)/C
    XT = XT./(n*sigma); % XT/C
    if isempty(beta0); % initialize with RLS solution
        beta0 = rls_algorithm(X,Y); 
    end   
% l1l2 algorithm
    k = 0;
    beta = thresholding(beta0+XT*(Y-X*beta0),tau_s) ./ (1+mu_s); % first iteration
    log=1;
    while or(and(k<kmax,log),k<kmin)
        % the threshold grows with the iteration
        if all(abs(beta-beta0)<=(abs(beta0)*tol/(k+1))), log=0; end
        beta0 = beta;
        beta = thresholding(beta0+XT*(Y-X*beta0),tau_s) ./ (1+mu_s);
        k = k+1;
    end
