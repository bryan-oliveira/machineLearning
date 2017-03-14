function J = computeCost(X, y, theta)

%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% number of training examples
m = length(y);

% predictions/hypothesis on all m examples
predictions = X*theta;

% calculate squared error using theta
sqrErrors = (predictions-y).^2;

% set J - value of cost function with given theta
J = 1/(2*m) * sum(sqrErrors);

end
