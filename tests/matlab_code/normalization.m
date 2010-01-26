function [varargout] = normalization(X,Y,norm_mean,norm_col,Xts,Yts)
%Normalization of a data set
%   [XNORM,YNORM] = NORMALIZATION(X,Y,NORM_MEAN,NORM_COL) normalizes a data
%       set X,Y where X is a matrix NxD and Y an array Nx1 amd returns the 
%       normalized matrix XNORM, and array Ynorm. Matrix X is normalized 
%       column by column by subtracting the mean (only if NORM_MEAN=1) and
%       by setting the norm to 1 (only if NORM_COL=1). Y is normalized by
%       subtracting its mean. NORMALIZATION(X,Y,0,0) = [X,Y].
%
%   [XNORM,YNORM,XTS_NORM,YTS_NORM] = NORMALIZATION(X,Y,NORM_MEAN,NORM_COL,Xts,Yts)
%       normalizes matrices XNORM,XTS_NORM, and arrays Ynorm,YTS_NORM. 
%       Test data is normalized with respect to the mean and
%       standard deviation estimated from the training data.
%   [XNORM,YNORM,XTS_NORM,YTS_NORM,MEANY] = NORMALIZATION(...) also returns
%   the mean for Y.

if nargin<3; error('too few inputs!'); end
if nargin<4; norm_col = 0; end    
if nargin>6; error('too many inputs!'); end
    
[n,xdim] = size(X);
if norm_col, stdevs = std(X); else stdevs = ones(1,xdim); end 
if norm_mean;
    means = mean(X); 
    meanY = mean(Y);
else
    means = zeros(1,xdim);
    meanY= 0;
end
Xnorm = zeros(n,xdim);
for i = 1:n;
    Xnorm(i,:) = (X(i,:)-means)./stdevs;   
end
Ynorm = Y-meanY;

varargout{1} = Xnorm;
varargout{2} = Ynorm;
if nargout>2;
    if nargin<5; error('second data set is missing!');
    else
        nd_ts = size(Xts);
        nts = nd_ts(1);
        Xts_norm = zeros(nts,xdim);
        for i = 1:nts;
            Xts_norm(i,:) = (Xts(i,:)-means)./stdevs;   
        end
        Yts_norm = Yts-meanY;
        varargout{3} = Xts_norm;
        varargout{4} = Yts_norm;
    end
end
if nargout==5; varargout{5} = meanY; end