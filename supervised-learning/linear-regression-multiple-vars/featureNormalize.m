function [X_norm, mu, sigma] = featureNormalize(X)

%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
 

% compute mean and standard deviation values of X - return 1x3 vector
mu = mean(X);
sigma = std(X);

% subtract mean from each element of dataset
X_norm = X .- mu;

% scale (divide) each element of dataset by the standard deviation
X_norm = X_norm ./ sigma;
	

end
