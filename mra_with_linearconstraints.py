import numpy as np
import pandas as pd

class ConstrainedRegression:
    def __init__(self, C, t):
        self.C = C
        self.t = t
    
    """
    @staticmethod
    def add_ones(X):
        n = len(X)
        X2 = np.hstack((np.ones(n).reshape(-1,1), X))
        return X2
    """

    def fit(self, X, y):
        #X = ConstrainedRegression.add_ones(X)
        b = np.linalg.pinv(X.T@X)@X.T@y
        self.coef_ = b -  np.linalg.pinv(X.T@X)@self.C.T@np.linalg.pinv(self.C@np.linalg.pinv(X.T@X)@self.C.T)@(self.C@b-self.t)
        
    def predict(self, X):
        #X = ConstrainedRegression.add_ones(X)
        return X@self.coef_

# Example usage:
if __name__ == "__main__":
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([3, 7, 11])
    
    C_matrix = np.array([[1, -1]])
    t_vector = np.array([2])
    
    regression_model = ConstrainedRegression(C=C_matrix, t=t_vector)
    regression_model.fit(X, y)
    coefficients = regression_model.coef_
    print("Linear Constraint: b1 - b2 = 2")
    print("Regression coefficients:", coefficients)
    print("check whether the constraint is satisfied or not")
    print(coefficients[0], "-", coefficients[1], "=", coefficients[0]-coefficients[1])