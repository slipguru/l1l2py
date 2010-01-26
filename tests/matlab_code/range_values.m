function values = range_values(range)
% given the parameters range, determines possibe values
if isequal(size(range),[3,1]);
    values = [range(1) range(1)*((range(2)/range(1))^(1/(range(3)-1))).^(1:(range(3)-1))];
elseif isequal(size(range),[1,3]);
     values = range(1):range(2):range(3);
end

