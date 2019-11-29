function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

[n1,n2]=size(theta);
h=sigmoid(X*theta);
theta2=theta(2:n1,1);
J=-1/m*sum((log(h)'*y+log(1-h)'*(1-y)))+lambda/(2*m)*sum(theta2.^2);

grad_orig=1/m*(X'*(h-y));
grad=1/m*(X'*(h-y))+lambda/m*theta;
grad(1)=grad_orig(1);

grad = grad(:);

end