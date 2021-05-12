# Hyperparameter-Optimization

Hyperparameters can be thought of as model settings. These settings need to be tuned for each problem because the best model hyperparameters for one particular dataset will not be the best across all datasets. The process of hyperparameter tuning (also called hyperparameter optimization) means finding the combination of hyperparameter values for a machine learning model that performs the best - as measured on a validation dataset - for a problem.

- A hyperparameter is a parameter whose value is used to control the learning process.
- Hyperparameter optimization finds a tuple of hyperparameters that yields an optimal model which minimizes a predefined loss function on given independent data.

**Search Space:** Volume to be searched where each dimension represents a hyperparameter and each point represents one model configuration.
**Grid Search:** Define a search space as a grid of hyperparameter values and evaluate every position in the grid.
**Random Search:** Define a search space as a bounded domain of hyperparameter values and randomly sample points in that domain.

- Grid search is great for spot-checking combinations that are known to perform well generally. 
- Random search is great for discovery and getting hyperparameter combinations that you would not have guessed intuitively, although it often requires more time to execute.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

*GRID SEARCH:* Grid search or a parameter sweep, which is simply an exhaustive searching through a manually specified subset of the hyperparameter space of a learning algorithm. A grid search algorithm must be guided by some performance metric, typically measured by cross-validation on the training set or evaluation on a held-out validation set.

*RANDOM SEARCH:* Replaces the exhaustive enumeration of all combinations by selecting them randomly. 

*BAYESIAN OPTIMIZATION:* Bayesian optimization is a global optimization method for noisy black-box functions. Applied to hyperparameter optimization, Bayesian optimization builds a probabilistic model of the function mapping from hyperparameter values to the objective evaluated on a validation set. {Smartly explores the space}

*GRADIENT BASED OPTIMIZATION:* For specific learning algorithms, it is possible to compute the gradient with respect to hyperparameters and then optimize the hyperparameters using gradient descent. Mainly for CNN and later implemented for Logistic Regression and SVM.

Evolutionary Optimization:
Population Based:
Early Stopping Based:

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

scikit-learn provides techniques to tune model hyperparameters. It provides RandomizedSearchCV for random search and GridSearchCV for grid search. Both techniques evaluate models for a given hyperparameter vector using cross-validation, hence the “CV” suffix of each class name.

Both classes require two arguments: 
1. Model that we are optimizing
2. Search Space (defined as a dictionary where the names are the hyperparameter arguments to the model and the values are discrete values or a distribution of values to sample)

# define model
model = LogisticRegression()
# define search space
space = dict()
...
# define search
search = GridSearchCV(model, space)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Arguments:

1. cv - allows either an integer number of folds to be specified, e.g. 5, or a configured cross-validation object.(Recommended: defining and specifying cross-validation object) 
2.  - RepeatedStratifiedKFold for Classification
    - RepeatedKFold for Regression
3. "scoring” argument that takes a string indicating the metric to optimize. 
    - For Classification: Accuracy
    - For Regression    : neg_mean_absolute_error
4. n_jobs - To make search parallel. Set it to be -1 to automatically use all of the cores in your system.

Fit and Execute the Search and Summarize the result.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



    




