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
