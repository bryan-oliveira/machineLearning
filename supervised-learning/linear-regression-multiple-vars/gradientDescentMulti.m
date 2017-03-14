function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% number of training examples
m = length(y);

% save Cost function values over iterations to track convergence
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % calculate hypothesis
    %   - X (mxn), theta (nx1), hypothesis (mx1)
    hypothesis = X * theta;
    
    % calculate error
    errors = hypothesis .- y;   % (mx1)


    % note that we transpose the result of errors' * X
    theta = theta - ( (alpha * (1/m)) * (errors' * X))';

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
