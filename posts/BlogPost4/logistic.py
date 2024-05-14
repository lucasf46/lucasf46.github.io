
# IMPLEMENTING LOGISTIC REGRESSION

import torch


class LinearModel:

    def __init__(self):
        self.w = None 
        self.prev_w = None

    def score(self, X):
        """
        Compute the scores for each data point in the feature matrix X. 
        The formula for the ith entry of s is s[i] = <self.w, x[i]>. 

        If self.w currently has value None, then it is necessary to first initialize self.w to a random value. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            s torch.Tensor: vector of scores. s.size() = (n,)
        """
        if self.w is None: 
            self.w = torch.rand((X.size()[1]))

        if self.prev_w is None:
            self.prev_w = torch.zeros((X.size()[1]))

        s = X@(self.w)

        return s
        

    def predict(self, X):
        """
        Compute the predictions for each data point in the feature matrix X. The prediction for the ith data point is either 0 or 1. 

        ARGUMENTS: 
            X, torch.Tensor: the feature matrix. X.size() == (n, p), 
            where n is the number of data points and p is the 
            number of features. This implementation always assumes 
            that the final column of X is a constant column of 1s. 

        RETURNS: 
            y_hat, torch.Tensor: vector predictions in {0.0, 1.0}. y_hat.size() = (n,)
        """
        scores = self.score(X)
        y_hat = (scores > 0.0) * 1.0

        return y_hat


class LogisticRegression(LinearModel):

    def logistic_loss(self, X, y):

        """
        Calculate the logistic loss for a dataset given its feature matrix X and labels y. 

        ARGUMENTS:
            X, torch.Tensor: The feature matrix where each row is a feature vector. The size of X is (n, p),
                               where n is the number of data points and p is the number of features.
            y, torch.Tensor: The true labels for each data point in X.

        RETURNS:
            torch.Tensor: The mean logistic loss for the given data.
        """

        s_i = self.score(X)
        sigmoid_si = torch.sigmoid(s_i)
        loss = (-y * torch.log(sigmoid_si)) - ((1 - y) * torch.log(1 - sigmoid_si))

        return torch.mean(loss)

    def logistic_grad(self, X, y):

        """
        Compute the gradient of the logistic loss function.

        ARGUMENTS:
            X, torch.Tensor: The feature matrix where each row is a feature vector. The size of X is (n, p),
                               where n is the number of data points and p is the number of features.
            y, torch.Tensor: The true labels for each data point in X.

        RETURNS:
            torch.Tensor: The gradient of the logistic loss.
        """

        s_i = self.score(X)
        # [:,None] converts a tensor v with shape (n,) into a tensor v_ with shape (n,1)
        # borrowed from Logistic Regression blog post page
        delta = ((torch.sigmoid(s_i) - y)[:,None]* X)
        gradient = torch.mean(delta, dim = 0)

        return gradient

class GradientDescentOptimizer:

    def __init__(self, model):
        self.model = model 


    def step(self, X, y, alpha, beta):

        """
        Perform one step of gradient descent optimization on the logistic regression model's parameters.

        ARGUMENTS:
            X, torch.Tensor: The feature matrix. Size of X is (n, p), where n is the number of data points,
                              and p is the number of features.
            y, torch.Tensor: The true labels for each data point in X. 
            alpha, float: The learning rate, a scalar that scales the gradient.
            beta, float: The momentum factor, a scalar that scales the contribution of the previous weight update.

        RETURNS:
            torch.Tensor: The logistic loss after updating the model's parameters.
        """

        gradient = self.model.logistic_grad(X, y)
        weight = self.model.w
        self.model.w = weight - (alpha * gradient) + beta * (weight - self.model.prev_w)
        self.model.prev_w = self.model.w

        return self.model.logistic_loss(X,y)


    
  



        

    