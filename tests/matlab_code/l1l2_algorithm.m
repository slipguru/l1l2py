function [beta,k] = l1l2_algorithm(X,Y,tau,mu_fact,beta0,sigma0,kmax,tol)
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
%   KMAX iterations or when each element of BETA reached convergence 
%   (default tolernace is 1e-6).
%
%   [...] = L1L2_ALGORITHM(X,Y,TAU,MU_FACT,BETA0,SIGMA0,KMAX,TOL) uses TOL
%   as tolerance for stopping.
%

    if nargin<3; error('too few inputs!'); end
    if nargin<4; mu_fact = 0; end
    if nargin<5; beta0 = []; end
    if nargin<6; sigma0=[]; end
    if nargin<7; kmax = 1e5; end
    if nargin<8; tol = 1e-6; end
    if nargin>8; error('too many inputs!'); end
    
    [n,d] =size(X);
    XT = X';

    % if sigma is not specified in input, evaluates it as a/n
    if isempty(sigma0);
        a = normest(X*XT);
        sigma0 = a/n; %step size for mu_fact=0
    end
    mu = mu_fact;%*sigma0; % l2 parameter
    sigma = sigma0+mu; % step size

    
    % useful quantities
    mu_s = mu/sigma;
    tau_s = tau/sigma;    % twice the threshold
    XT = XT./(n*sigma);
    XY = XT*Y;

% l1l2 algorithm
    
    % initialization 
    if isempty(beta0);
        beta0 = zeros(d,1); 
    end
    k = 0;
    kmin = 10; % minimum number of iterations
    log=1;
    beta = beta0;
    h = beta0;
    t = 1;
    beta_prev = zeros(d,1);
    
    while or(and(k<kmax,log),k<kmin)
        if norm(beta-beta_prev)<=(norm(beta_prev)*tol), log=0; else log =1; end
        
        beta_prev = beta;                    

        k = k+1;
        
        beta = thresholding(h.*(1-mu_s) + XY-XT*(X*h),tau_s);
               
        t_new = .5*(1+sqrt(1+4*t^2));
        h = beta + (t-1)/(t_new)*(beta-beta_prev);
        t = t_new;
    end
    
    beta = beta_prev;

function [beta] = thresholding(beta0,tau)
%THRESHOLDING Soft-thresholding
%   BETA = THRESHOLDING(BETA,TAU) returns the soft-thresholding of vector 
%   BETA with threshold TAU.
%
    ind = logical(abs(beta0)<tau);
    beta = beta0-sign(beta0).*tau;
    beta(ind) = 0;
