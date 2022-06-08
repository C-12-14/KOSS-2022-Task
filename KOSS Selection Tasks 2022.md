# KOSS Selection Tasks 2022

# Data classification

I used logistic regression with a linear activation function and mean square error as the loss function to predict the results.

I have implemented two models. With and without using the scikit-learn's inbuilt training modules.

#### Libraries used

* pandas for data manipulation
* seaborn for data visualisation
* matplotlib for plotting simple data
* scikit-learn  for training the 2nd model
* numpy

#### Maths

* Input $X$ : It is a $4\times m$ matrix where the number of rows are the characteristics of one object and number of columns represents the number of input objects.

* Weight $W$ : It is a $4\times1$ matrix containing weights for different characteristics

* Bias $b$ : constant number acting as a bias.

* Activation function:  $W^T \cdot A + b $
* Loss Function J :  $\frac{1}{m}\sum (A-Y)^2$ (Mean Square Error)
* $\frac{dJ}{dw}$ or simply $dw$ = $\frac{2}{m} \sum A^T\cdot(A-Y)$

* $\frac{dJ}{db}$ or simple $db$ = $\frac{2}{m}\sum(A-Y)$

* Giving numerical representation for different classes:

  * Iris-setosa = 0

  * Iris-versicolor = 1

  * Iris-virginica = 2

#### Approach

For my model: 

* `split_data(data = df, test = 0.3)` is used to split the given data-set into two parts, train and test data-set.
* `propogate(w, b, X, Y)` function calculates outputs and gradients for 1 iteration.
* `optimize(w, b, X, Y, iterations, learning_rate)` runs the calculations for `iterations` number of times with a given learning rate.
* `predict(w, b, X)` is used to predict the output given the final weights and biases. The results are rounded off to give integer answers.
* `model()` function finally combines all the above functions.

For scikit-learn's model:

* `sklearn.linear_model.LogisticRegression` was used to train the model
* `sklearn.metrics` was used to calculate the accuracy of the model

In both models, I used `sklearn.model_selection.train_test_split` to split the data-set into two data-sets.

## Results

#### My Model using logistic regression with linear activation function

Average of $86.22\%$ accuracy on test cases

#### Scikit-learn's logistic regression model

$93.33\%$ accuracy



- [x] Bonus Task 1: Data Visualisation.

- [ ] Bonus Task 2: Setting up REST api.