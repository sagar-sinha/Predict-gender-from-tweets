function [residual, w] = get_residual_and_weight(Y,  X, lambda)

w = ridge(Y,X,lambda, 0);
X2 = [ones(size(X,1),1) X];
prediction = X2*w;
residual = Y - prediction;