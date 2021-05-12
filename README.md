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

*define model*
model = LogisticRegression()
*define search space*
space = dict()
...
*define search*
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

Eg: For RandomForest

n_estimators - No.of trees in Random Forest
max_features - No.of features to consider at every split
max_depth    - Maximum No.of levels in tree
min_samples_split - Minimum no.of samples required to split a node
min_samples_leaves- Minimum no.of samples required at each node


from sklearn.model_selection import RandomizedSearchCV
*Number of trees in random forest*
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
*Number of features to consider at every split*
max_features = ['auto', 'sqrt','log2']
*Maximum number of levels in tree*
max_depth = [int(x) for x in np.linspace(10, 1000,10)]
*Minimum number of samples required to split a node*
min_samples_split = [2, 5, 10,14]
*Minimum number of samples required at each leaf node*
min_samples_leaf = [1, 2, 4,6,8]
*Create the random grid*
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)

rf=RandomForestClassifier()
rf_randomcv=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=100,cv=3,verbose=2,
                               random_state=100,n_jobs=-1)
*fit the randomized model*
rf_randomcv.fit(X_train,y_train)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### Automated Hyperparameter Tuning
Automated Hyperparameter Tuning can be done by using techniques such as
- Bayesian Optimization
- Gradient Descent
- Evolutionary Algorithms

1. Bayesian Optimization:
- Bayesian optimization uses probability to find the minimum of a function. 
- The final aim is to find the input value to a function which can gives us the lowest possible output value.
- It usually performs better than random,grid and manual search providing better performance in the testing phase and reduced optimization time. 
- In Hyperopt, Bayesian Optimization can be implemented giving 3 three main parameters to the function fmin.  
    -   Domain Space = defines the range of input values to test (in Bayesian Optimization this space creates a probability distribution for each of the used Hyperparameters).
    -   Objective Function = defines the loss function to minimize.
    -   Optimization Algorithm = defines the search algorithm to use to select the best input values to use in each new iteration.  
    
#pip install hyperopt
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials

Then apply the function for 'Trials'. From the O/P to get back all the values, do a reverse mapping.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Genetic Algorithms:
- Try to apply Natural Selection
- Let's immagine we create a population of N Machine Learning models with some predifined Hyperparameters. We can then calculate the accuracy of each model and decide to keep just half of the models (the ones that performs best).
- We can now generate some offsprings having similar Hyperparameters to the ones of the best models so that go get again a population of N models.
- At this point we can again caltulate the accuracy of each model and repeate the cycle for a defined number of generations. In this way, just the best models will survive at the end of the process.

Use all RF parameters and use a TPOT clssifier. To use TPOT classifier, We need Tensor Flow to be installed. 


Optuna:
- pip install optuna
- 


