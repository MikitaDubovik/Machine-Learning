function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

deltaL1 = 0;
deltaL2 = 0;

for i=1:m
    a1 = [1 X(i,:)];
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    
    yk = zeros(num_labels, 1);
    yk(y(i)) = 1;
    
    J = J + (-yk' * log(a3') - (1 - yk') * log(1 - a3'));
    J = J + lambda / (2 * m) * ...
        (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)));
    
    delta3 = a3' - yk;
    delta2 = Theta2' * delta3 .* sigmoidGradient([1 z2])';
    deltaL1 = deltaL1 + delta2(2:end) * a1;
    deltaL2 = deltaL2 + delta3 * a2;
end

J = J / m;

Theta1_grad = 1 / m * deltaL1;
Theta2_grad = 1 / m * deltaL2;

nonBiasedTheta1 = Theta1;
nonBiasedTheta1(:,1) = 0;
Theta1_grad = Theta1_grad + lambda / m * nonBiasedTheta1;

nonBiasedTheta2 = Theta2;
nonBiasedTheta2(:,1) = 0;
Theta2_grad = Theta2_grad + lambda / m * nonBiasedTheta2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients

grad = [Theta1_grad(:) ; Theta2_grad(:)];


end