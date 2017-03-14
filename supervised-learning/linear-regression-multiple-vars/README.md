Multi-Variable Linear Regression for Prediction
===============================================

The code in this section contains a dataset to predict housing prices. Suppose you are selling
your house and would like to know what a good market price is. The file '''ex1data2.txt''' contains
a training set of housing prices in Portland, Oregon. The first column is the size of the house
in square feet, the second is the number of bedrooms, and the third the price of the house.


## Linear Regression In Multiple Variables Review


- Multiple features / variables
  + n = number of features
  + h(x) = theta_0 + theta_1 * x_1 + theta_2 * x_2 + ... theta_n * x_n
  + h(x) = X * theta - Our hypothesis / prediction can be rewritten like so
  + J(theta) = (1/2m)(Sum i=1:m) ( h(x^(i)) - y^(i) )^2

- Gradient Descent
  + formula = Then, one iteration of updating theta_j is given as:  
  ![alt text](imgsg/update_eq.gif)
  + problems with gradient descent is shown here:  
  ![alt text](imgs/grad.jpg)

- Feature Scaling / Mean Normalization
  + Scale features so that they are on a similar scale
    - Examples of good ranges
      + -1 < x < 1
      + -2 < x < 0
      + 0 < x < 1
    - Examples of bad ranges
      + 0 < x < 10,000
      + -500 < x < 1,000
      + 25,000 < x < 1,000,000

  + In general subtract the mean of the feature, and scale (divide) it by the standard deviation


- Learning Rate (alpha):
   + alpha too small - slow convergence
   + alpha too large - J(theta) may not converge

- Polynomial Regressioin
   + for non linearly distributed training data, the use of polynomial (quadratic, cubic) functions may help to fit the data better
   + h_theta(x) = theta_0 + theta_1 * x_1 + theta_2 * (x_1)^2  - quadratic example

- Normal Equation
  + algaibric method to solve for theta
  + solves for theta by computing equation as opposed to iteratively
  + to min J(theta), take the derivative of J(theta), set it to 0, and then solve for theta
  + theta = (X^T * X)^-1 * X^T * y  --- '''pinv(X' * X) * X' * y''' (Octave/Matlab equivalent)

- Normal Equation vs. Gradient Descent
  + m = # of training examples
  + n = # of features
  + alpha = learning rate

  |Normal Eq vs. Gradient Descent|Pros                              |Cons                                  |
  |:----------------------------:|:--------------------------------:|:------------------------------------:|
  |Gradient Descent              |works well even when n is large   |need to choose alpha                  |
  |                              |                                  |needs many iterations                 |
  |Normal Equation               |no need to choose alpha           |slow if n is very large               |
  |                              |don't need to iterate             |doesn't work for logistic regression  |

   + <b>General Rule:</b> # of features is > 10,000 use Normal Equation, otherwise use Gradient Descent
   + NOTE: Normal equation slows down when n > 10,000 because we need to compute (X^T * X)^-1 which will take O(n^3) time for inversing the matrix.
