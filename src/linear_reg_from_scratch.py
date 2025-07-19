import numpy as np


class LinearRegFromScratch:

    def __init__(
            self,
            learning_rate= 0.01,
            n_iterations= 1000,
            tol_loss= 1e-7,
            print_every= 100
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tol_loss = tol_loss
        self.print_every = print_every

        self.weights = None
        self.bias = None
        self.cost_log = []

    def fit(self, X, y):

        self.cost_log = []

        X = np.array(X, dtype= float)
        y = np.array(y, dtype= float)

        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        previous_cost = float("inf")

        for i in range(1, self.n_iterations + 1):
            # Calculate prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            error = y_pred - y
            dw = (1 / n_samples) * np.dot(X.T, error)
            db = (1 / n_samples) * np.sum(error)

            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Calculate (MSE / 2) cost
            cost = (1 / (2 * n_samples)) * np.sum(error ** 2)

            # Add cost to cost log
            self.cost_log.append(cost)
            
            # Check for convergence based on specified loss tolerance
            if abs(previous_cost - cost) < self.tol_loss:
                print(f"Converged at iteration {i} with cost = {cost}")
                break
            previous_cost = cost

            # Print out iteration & cost progress
            if i % self.print_every == 0:
                print(f"Iteration {i}, cost = {cost}")

        return self

    def predict(self, X):
        X = np.array(X, dtype= float)
        return np.dot(X, self.weights) + self.bias
    
    def __repr__(self):
        
        w = np.round(self.weights, 4) if self.weights is not None else None
        b = round(self.bias, 4) if self.bias is not None else None
        return(
            f"{self.__class__.__name__}(learning rate = {self.learning_rate}, "
            f"n iterations = {self.n_iterations}, loss tolerance = {self.tol_loss})\n"
            f"weights = {w}\n"
            f"bias = {b}"
        )
    
    __str__ = __repr__