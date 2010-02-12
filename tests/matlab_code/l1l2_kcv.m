function varargout = l1l2_kcv(X,Y,tau_range,range_LS,mu_fact,K,type,split,norm_mean,norm_col)
%L1L2_KCV Parameters choice through K-fold or LOO cross validation for 
%the l1l2-algorithm (used as variables selector) followed by Regularized Least
%Squares (used as estimator).
%   [TAU_OPT,LA_LS_KCV] = L1L2_KCV(X,Y,TAU_RANGE) returns the regularization
%   parameters minimizing the K-fold cross-validation error of the training
%   set X,Y for the l1 algorithm with l2 parameter definded by MU_FACT. 
%   If TAU_RANGE is a 1xL vector TAU_OPT will be
%   chosen among the values in TAU_RANGE, if TAU_RANGE is a 3x1 vector TAU_OPT will be
%   chosen among the TAU_RANGE(3) values of the geometric series ranging from
%   TAU_RANGE(1) to TAU_RANGE(2), if TAU_RANGE is a 1x3 vector TAU_OPT will be
%   chosen among the equispaced values
%   TAU_RANGE(1):TAU_RANGE(2):TAU_RANGE(3).
% 
%   [TAU_OPT,LA_LS_KCV,SPARSITY] = L1L2_KCV(X,Y,TAU_RANGE) also 
%   returns the # of selected features for eavh value of the l1 parameter
%
%   [TAU_OPT,LA_LS_KCV,SPARSITY,ERR_KCV] = L1L2_KCV(X,Y,TAU_RANGE) 
%   also returns the cross-validation error on validation set
%
%   [TAU_OPT,LA_LS_KCV,SPARSITY,ERR_KCV,ERR_TRAIN] = L1L2_KCV(X,Y,TAU_RANGE) 
%   also returns the cross-validation error on training set
% 
%   [TAU_OPT,LA_LS_KCV,SPARSITY,ERR_KCV,ERR_TRAIN,SELECTED] = 
%   L1L2_KCV(X,Y,TAU_RANGE,MU_FACT) also returns the indexes of the selected 
%   features for the optimal l1 parameter.
% 
%   [TAU_OPT,LA_LS_KCV] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS) returns the regularization
%   parameters minimizing the K-fold cross-validation error of the training
%   set X,Y for the l1 algorithm followed by regularized least squares.
%   RANGE_LS has the same choices as TAU_RANGE. Default for RANGE_LS is [ ]:
%   L1L2_KCV(X,Y,TAU_RANGE) = L1L2_KCV(X,Y,TAU_RANGE,[])
% 
%   [TAU_OPT,LA_LS_KCV] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT) implements 
%  the l1 algorithm with l2 parameter definded by MU_FACT. 
% 
%   [...] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT,K) performs sequential K-fold cross-validation.
%   If K=0 or K=length(Y) it performs LOO cross-validation
% 
%   [...] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT,K,TYPE) select the kind of
%   error to be considered for KCV. TYPE= 'CLASS' misclassification error
%   is considered (default), if TYPE='REGR' the squared error is considered. 
%
%   [...] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT,K,TYPE,SPLIT) performs sequential(SPLIT=0)
%   K-fold cross-validation, or random(SPLIT=1) K-fold cross-validation.
%
%   [...] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT,K,TYPE,SPLIT,NORM_MEAN) 
%   if NORM_MEAN=1, the training data will be recentered.
%
%   [...] = L1L2_KCV(X,Y,TAU_RANGE,RANGE_LS,MU_FACT,K,TYPE,SPLIT,NORM_MEAN,NORM_COL)
%   if NORM_COL=1, the input training data will have columns normalized to 1.
%
%   See also L1L2_ALGORITHM, RLS_ALGORITHM, SPLITTING, LINEAR_TEST
%

if nargin<3, error('too few input!'), end
if nargin<4, range_LS = []; end; 
if nargin<5, mu_fact = 0; end; 
if nargin<6, K = 0; end; 
if nargin<7, type = 'class'; end; 
if nargin<8, split = 0; end
if nargin<9, norm_mean = 0; end
if nargin<10, norm_col = 0; end
if nargin>10, error('too many input!'), end


% given the parameters range, determines possibe values
if isequal(size(tau_range),[3,1]); 
    tau = [tau_range(1) tau_range(1)*((tau_range(2)/tau_range(1))^(1/(tau_range(3)-1))).^(1:(tau_range(3)-1))]; 
elseif isequal(size(tau_range),[1,3]), tau = tau_range(1):tau_range(2):tau_range(3); 
else tau = tau_range;
end
if isequal(size(range_LS),[3,1]); 
    lambda_LS = [range_LS(1) range_LS(1)*((range_LS(2)/range_LS(1))^(1/(range_LS(3)-1))).^(1:(range_LS(3)-1))]; 
elseif isequal(size(range_LS),[1,3]), tau = range_LS(1):range_LS(2):range_LS(3); 
else lambda_LS = range_LS;
end

sets = splitting(Y,K,split); %splits the training set in K subsets
d = size(X,2);
% initialization
err_KCV = zeros(length(sets),length(tau),length(lambda_LS));
err_train = zeros(length(sets),length(tau),length(lambda_LS));
selected = cell(length(sets),length(tau));
sparsity = zeros(length(sets),length(tau));


for i = 1:length(sets);
    
    ind = setdiff(1:length(Y),sets{i}); %indexes of training set
        
    % normalization
    [Xtr,Ytr,Xts,Yts,meanY] = normalization(X(ind,:),Y(ind),norm_mean,norm_col,X(sets{i},:),Y(sets{i}));
    ntr = length(Ytr);
    if d>ntr; 
        a = normest(Xtr*Xtr');
        b = 0; 
    else
        aval = svd(Xtr'*Xtr);
        a = aval(1);
        b = aval(end);
    end
    sigma0 = (a+b)/(ntr*2); %step size for mu_fact=0
    
    % evaluate all betas for all taus concurrently
    [selected(i,:), sparsity(i,:)] = l1l2_regpath(Xtr,Ytr,tau,mu_fact,sigma0); 
    
    % for each value of the l1 parameter, use the l1l2 solution for
    % selection and train rls on the selected features, then evaluate error
    % on validation set (err_KCV)
    for t = 1:length(tau);
        for j = 1:length(lambda_LS);
            beta = rls_algorithm(Xtr(:,selected{i,t}),Ytr,lambda_LS(j));
            err_KCV(i,t,j) = linear_test(Xts(:,selected{i,t}),Yts,beta,type,meanY);       
            err_train(i,t,j) = linear_test(Xtr(:,selected{i,t}),Ytr,beta,type,meanY);       
        end    
    end
    
end

% evaluate avg. error over the splits
err_KCV = reshape(mean(err_KCV,1),length(tau),length(lambda_LS));
err_train = reshape(mean(err_train,1),length(tau),length(lambda_LS));

% for each value of the l1 parameter, find rls parameter minimizing the
% error
lambda_LS_opt = zeros(length(tau),1);
for t = 1:length(tau);
    lambda_LS_opt(t) = lambda_LS(find(err_KCV(t,:)==min(err_KCV(t,:)),1,'last'));
end
min_err_KCV = min(err_KCV,[],2);

% find l1 parameter minimizing the error
t_opt = find(min_err_KCV==min(min_err_KCV),1,'last');

varargout{1} = tau(t_opt);
varargout{2} = lambda_LS_opt(t_opt);
varargout{3} = mean(sparsity);
varargout{4} = err_KCV;
varargout{5} = err_train;
varargout{6} = selected;
