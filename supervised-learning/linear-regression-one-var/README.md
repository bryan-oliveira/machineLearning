Linear Regression (one variable) for Prediction
====================================================

The code in this section contains a dataset of business profits according to
population size. The exercise is to predict the profit of a food truck. Say
we are the CEO of a restaurant franchise already operating in multiple cities, and pretend to open a new outlet. We have the data in the '''ex1data1.txt''' file.

The code does the following:

1. Plot the data to visualize

2. Use the gradient descent algorithm to determine theta_0 and theta_1 values

3. Plot the minimal values for theta (fit the data)


+Notes:
	- m = number of training examples
    - x's = "input" variable / features
    - y's = "output" variable
    - (x,y) = one training example
    - h(x) = theta_0 + theta_1 * x where h maps from x's to y's
    - We want to find theta_0 & theta_1 so that h(x) minimizes our Cost Function J(theta) to our training examples
    - min (1/2m) (Sum i=1:m) (h(x^(i))-y^(i))^2)
    - Errors are squared to simplify

    ![alt text](https://raw.github.com/quinnliu/MachineLearning/master/imagesForExplanation/CostFunctionExampleWithTheta_0AndTheta_1.jpg)

    - We can then perform predictions on the data by plugging in theta_0 and theta_1: prediction = theta_0 + theta_1 * X
    - Two versions of Gradient Descent are implemented - Batch Gradient Descent and Normal Equation
    
