function [beta] = thresholding(beta0,tau)
%THRESHOLDING Soft-thresholding
%   BETA = THRESHOLDING(BETA,TAU) returns the soft-thresholding of vector 
%   BETA with threshold TAU/2.
%
    ind = logical(abs(beta0)<tau/2);
    beta = beta0-sign(beta0).*tau/2;
    beta(ind) = 0;