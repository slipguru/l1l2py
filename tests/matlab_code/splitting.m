function sets = splitting(Y,K,type)
%SPLITTING Splitting in balanced subsets
%   SETS = SPLIT(Y,K) return a cell array of K subsets of 1:n where
%   n=length(Y). The elements 1:n are split so that in each subset the
%   ratio between indexes corresponding to positive elements of array Y and
%   indexes corresponding to negative elements of Y is the about same as in
%   1:n. The subsets are obtained  by sequentially distributing the
%   elements of 1:n.
%
%   SETS = SPLIT(Y,K,TYPE) return a cell array of K subsets of 1:n where
%   n=length(Y) according to splitting type TYPE which can be either
%   0 (sequential) or 1 (random). SPLIT(Y,K,0) = SPLIT(Y,K)
%
if nargin<2; error('too few inputs!'); end
if nargin==2, type = 0; end; 
if nargin>3; error('too many inputs!'); end

n = length(Y);
if or(K==0,K==n);
    sets = cell(1,n);
    for i = 1:n, sets{i} = i; end
else
    c1 = find(Y>=0);
    c2 = find(Y<0);
    l1 = length(c1);
    l2 = length(c2);
    if type==0;
        perm1=1:l1;
        perm2=1:l2;
    elseif type==1;
        perm1 = randperm(l1);
        perm2 = randperm(l2);
    end;
    sets = cell(1,K);
    i = 1;
    while i<=l1;
        for v = 1:K;
            if i<=l1;
                sets{v} = [sets{v}; c1(perm1(i))];
                i = i+1;
            end;
        end;
    end;
    i = 1;
    while i<=l2;
        for v = 1:K;
            if i<=l2;
                sets{v} = [sets{v}; c2(perm2(i))];
                i = i+1;
            end;
        end;
    end;
end    