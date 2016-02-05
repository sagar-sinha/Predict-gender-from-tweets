function [yhat] = kernel_libsvm(X, Y, Xtest, Ytest, kernel)

addpath('./libsvm');
K = kernel(X, X);
Ktest = kernel(X, Xtest);

crange = 10.^[-10:2:3];
for i = 1:numel(crange)
    acc(i) = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -v 10 -c %g', crange(i)));
end
[~, bestc] = max(acc);
fprintf('Cross-val chose best C = %g\n', crange(bestc));

model = svmtrain(Y, [(1:size(K,1))' K], sprintf('-t 4 -c %g', crange(bestc)));
[yhat, ~] = svmpredict(Ytest, [(1:size(Ktest,1))' Ktest], model);
