function [beta] =rls_algorithm(x,y,lambda)
%RLS_ALGORITHM Regularized Least Squares
%   BETA = RLS_ALGORITHM(X,Y) evaluates the Least Squares estimates of
%       ||Y-X*BETA||^2
%   BETA = RLS_ALGORITHM(X,Y,LAMBDA) evaluates the Regularized Least 
%   Squares estimates of 
%       1/N||Y-X*BETA||^2 +LAMBDA||BETA||^2
%
if nargin<3, lambda = 0; end
[n,d] = size(x);
if n<d;    beta = x'*pinv(x*x'+lambda*n*eye(n))*y;
else    beta = pinv(x'*x+lambda*n*eye(d))*x'*y;
end