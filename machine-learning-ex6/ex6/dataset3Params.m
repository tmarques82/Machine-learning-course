function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
values_to_iter = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

error_min = 0;
C_final = 0;
sigma_final = 0;

for cindex = 1:8
  for sigmaindex = 1:8
    C = values_to_iter(cindex);
    sigma = values_to_iter(sigmaindex);
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    prediction = svmPredict(model, Xval);
    error = mean(double(prediction ~= yval));
    if cindex == 1 && sigmaindex == 1
      C_final = C;
      sigma_final = sigma;
      error_min = error;
    end
    if error < error_min
      C_final = C;
      sigma_final = sigma;
      error_min = error;
    end 
  end
end

C = C_final;
sigma = sigma_final;


% =========================================================================

end
