function [test_err,Ytest_learned] = linear_test_weighted(Xtest,Ytest,beta,type,meanY)
% LINEAR_TEST prediction error for linear model
%   [TEST_ERR] = LINEAR_TEST(XTEST,YTEST,BETA) returns the classification 
%   error committed by BETA on the test set XTEST, YTEST.
% 
%   [TEST_ERR,YTEST_LEARNED] = LINEAR_TEST(XTEST,YTEST,BETA) also returns the
%   estimated value for the test samples.
%
%   [...] = LINEAR_TEST(XTEST,YTEST,BETA,TYPE) if TYPE='REGR'(default) 
%   evaluates the regression error; if TYPE='CLASS'evaluates the 
%   classification error.
%
%   [...] = LINEAR_TEST(XTEST,YTEST,BETA,TYPE,MEANY) adds offset MEANY to 
%   regression function: Y = X*BETA + MEANY.
%
if nargin<3; error('too few inputs!'); end
if nargin<4; type = 'regr'; end
if nargin<5; meanY = 0; end
if nargin>5; error('too many inputs!'); end

Ytest_learned = Xtest*beta+meanY;
if isequal(type,'class')
    test_err = sum(((sign(Ytest_learned).*sign(Ytest+meanY))~=1).*abs(Ytest))/length(Ytest);
elseif isequal(type,'regr')
    test_err = norm(Ytest_learned-(Ytest+meanY))^2/length(Ytest);
end