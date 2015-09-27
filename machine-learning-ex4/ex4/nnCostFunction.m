function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


partx = [ones(m,1) X];
activation2 = sigmoid(Theta1*partx');
activation2bis = [ones(m,1) activation2'];
activation3 = sigmoid(Theta2*activation2bis');
 
for i = 1:m
  for k = 1:num_labels
    if y(i) == k
      val_y = 1;   
    else
      val_y = 0;
    end
    J += (1/m) * ((-val_y*log(activation3(k, i))) - (1 - val_y)*log(1 - activation3(k, i)));
  end
end

regularization_part = 0;

for j=1:size(Theta1,1)
  for k=2:size(Theta1,2)
    regularization_part += (lambda/(2*m))*Theta1(j,k)^2;
  end
end

for j=1:size(Theta2,1)
  for k=2:size(Theta2,2)
    regularization_part += (lambda/(2*m))*Theta2(j,k)^2;
  end
end

J += regularization_part;

reconstructed_y = zeros(m, num_labels);
for i = 1:m
  reconstructed_y(i, y(i)) = 1;
end
% BACKPROPAGATION ALGORITHM
for i = 1:m
  % 1. Get the input activation values
  z2 = partx(i,:)*Theta1';
     % already done before in vars activation2 and activation3
  % 2. Get the layer 3 delta
  delta3 = activation3'(i, :) - reconstructed_y(i,:);
  % 3. Get layer 2 delta
  delta2 = (Theta2'*delta3')'(2:end).*sigmoidGradient(z2);
  % 4. Accumulate deltasize
  Theta2_grad += delta3' * activation2bis(i, :);
  Theta1_grad += delta2' * partx(i,:);
end
 % 5. Obtain the unregularized gradient
 Theta2_grad = 1/m * Theta2_grad;
 Theta1_grad = 1/m * Theta1_grad;
 
 % Regularize gradient
 reg_Theta1 = Theta1;
 reg_Theta1(:, 1) = zeros(size(reg_Theta1(:, 1)));
 reg_Theta2 = Theta2;
 reg_Theta2(:, 1) = zeros(size(reg_Theta2(:, 1)));
 Theta1_grad += (lambda/m)*reg_Theta1;
 Theta2_grad += (lambda/m)*reg_Theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
