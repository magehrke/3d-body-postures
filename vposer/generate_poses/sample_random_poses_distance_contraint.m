sd=2;
w=6;
clear data 
data(1,:) = randn(1,32)*sd;
for i = 2:100
    data(i,:) = randn(1,32)*sd;
    backind = max(1,i-w):i;
    D = pdist(data(backind,:));
    while any(D<20) 
        data(i,:) = randn(1,32)*sd;
        D = pdist(data(backind,:));
    end
end
D = pdist(data);
M = squareform(D);
Ms = diag(M,1);