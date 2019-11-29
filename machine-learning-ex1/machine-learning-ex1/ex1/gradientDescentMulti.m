function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
thetaLength = length(theta);
J_history = zeros(num_iters, 1);
tic
for iter = 1:num_iters
    xTransp = X';
    for i = 1 : m
        for j = 1: thetaLength
            theta(j) = theta(j) - alpha/m*(xTransp(j,i) * (X(i,j) * theta(j) - y(i))); 
        end
    end
    
    J_history(iter) = computeCostMulti(X, y, theta);

end
time = toc;
fprintf('Time %f\n', time);
end
