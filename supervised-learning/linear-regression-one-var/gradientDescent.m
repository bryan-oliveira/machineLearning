function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

% gradientDescent() - Performs gradient descent to learn theta
%   theta = gradientDescent(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% number of training examples
m = length(y);

% save Cost function values over iterations
J_history = zeros(num_iters, 1);

% fprintf("\nTheta 0: %.2f Theta 1: %.2f N: %d Alpha: %d\n\n", theta(1), theta(2), m, alpha);

for iter = 1:num_iters

    % hypothesis or predictions w/ current theta
    hypothesis = X * theta;

    % sum of errors
    %   - hypothesis and y are mx1 vectors
    %   - the result is also a mx1 vector
    errors = hypothesis .- y;

    % version 1: simultaneous update of thetas independently
    %theta(1) = theta(1) - (alpha * (1/m) * sum(errors) );
    %theta(2) = theta(2) - (alpha * (1/m) * errors' * X(:,2) );

    % version 2: vectorization formula
    % note that we transpose the result of errors' * X
    
    theta = theta - ((alpha * (1/m)) * (errors' * X))';

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
