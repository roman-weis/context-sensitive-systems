import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from BayesianOptimizer import BayesianOptimizer
from sklearn import datasets
from sklearn.model_selection import train_test_split
data = datasets.load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the SVM objective function
def objective_function(params):
    C = params[0] 
    coef0 = params[1]
    # Create an SVM classifier with the specified hyperparameters
    svm_classifier = SVC(C=C, coef0=coef0, random_state=42)
    
    # Evaluate the SVM using cross-validation and return the mean accuracy
    accuracy_scores = cross_val_score(svm_classifier, X_train, y_train, cv=5)
    mean_accuracy = np.mean(accuracy_scores)
    
    # Since Bayesian optimization maximizes, we need to negate the accuracy
    return -mean_accuracy

# Define the search space for optimization
search_space = [(0.1, 10.0), (0.0, 0.05, 0.5)]  # Range for gamma if using 'rbf' kernel

# Create an instance of BayesianOptimizer and perform optimization
optimizer = BayesianOptimizer(n_iter=50, random_state=42)
optimizer.X = np.random.rand(5, len(search_space))  # Initialize with random points
optimizer.y = np.array([objective_function(x) for x in optimizer.X])
optimizer.fit(objective_function, search_space)

# Get the best hyperparameters found during optimization
best_params = optimizer.get_best_params()
print("Best Hyperparameters:", best_params)
