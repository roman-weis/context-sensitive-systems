import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.optimize import minimize
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class ExpectedImprovement:
    def __init__(self, surrogate_model, current_best_value, xi=0.01):
        """
        Initialize the Expected Improvement acquisition function.

        Parameters:
        - surrogate_model: The surrogate model (GaussianProcessRegressor) trained on observed data.
        - current_best_value: The current best value observed in the objective function.
        - xi: Exploration-exploitation trade-off parameter (default is 0.01).
        """
        self.surrogate_model = surrogate_model
        self.current_best_value = current_best_value
        self.xi = xi

    def evaluate(self, x):
        """
        Calculate the Expected Improvement (EI) for a single point.

        Parameters:
        - x: The point in the search space for which EI is calculated.

        Returns:
        - ei: The Expected Improvement for the given point x.
        """
        mean, std = self.surrogate_model.predict(np.array(x).reshape(1, -1), return_std=True)
    
        if std is None or std == 0:
            return 0  # Prevent division by zero or handle cases where std is None
    
        z = (mean - self.current_best_value - self.xi) / std
        ei = (mean - self.current_best_value - self.xi) * norm.cdf(z) + std * norm.pdf(z)
    
        return ei


    def optimize(self, search_space):
        """
        Optimize the acquisition function to find the next query point.

        Parameters:
        - search_space: The bounds of the search space for optimization.

        Returns:
        - opt_result: The result of the optimization process, including the next query point.
        """
        def objective_function(x):
            return -self.evaluate(x)

        opt_result = minimize(
            objective_function,
            x0=np.random.uniform(search_space[0][0], search_space[0][1], len(search_space)),
            bounds=search_space,
            method='L-BFGS-B'
        )

        return opt_result




class BayesianOptimizer:
    def __init__(self, n_iter=50, random_state=None):
        self.n_iter = n_iter
        self.random_state = random_state
        self.surrogate_model = None
        self.best_params = None
        self.best_value = 0
        self.X = None
        self.y = None

    def fit(self, objective_function, search_space):
        np.random.seed(self.random_state)

        for _ in range(self.n_iter):
            # Fit the surrogate model to the data
            self.surrogate_model = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                n_restarts_optimizer=25,
                random_state=self.random_state
            )
            self.surrogate_model.fit(self.X, self.y)

            acquisition = ExpectedImprovement(self.surrogate_model, self.best_value, xi=0.01)

            opt_result = acquisition.optimize(search_space)

            next_query_point = opt_result.x
            next_value = -opt_result.fun  # Invert the sign as we minimized the negative EI

            # Update the dataset with the new observation
            self.X = np.vstack([self.X, next_query_point])
            self.y = np.append(self.y, objective_function(next_query_point))

            # Update the best observed value and corresponding input
            if self.best_value is None or next_value > self.best_value:
                self.best_value = next_value
                self.best_params = next_query_point

    def get_best_params(self):
        return self.best_params

