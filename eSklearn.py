# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 12:37:49 2017

@author: msiwek
"""

import numpy as np

# data
X, y = [[-1, 0, 2, -2], [0, 0, 3, 5], [1, 1, 4, -2],[-2,-3,1,1]], [-1, 0, -1,0]

# sparse data
from sklearn.datasets import make_sparse_coded_signal
n_components, n_features = 512, 100
n_nonzero_coefs = 17
# generate the data
###################
# y = Xw
# |x|_0 = n_nonzero_coefs
y, X, w = make_sparse_coded_signal(n_samples=1,
                                   n_components=n_components,
                                   n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs,
                                   random_state=0)
"""



>>> 1. Supervised learning



"""
"""


>> 1.1. Generalized Linear Models


"""
"""

> 1.1.1. Ordinary Least Squares

sklearn.linear_model.LinearRegression

(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
"""
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit ([[0, 0], [1, 1], [2, 2]], [0, 1, 2])
reg.coef_
"""

> 1.1.2. Ridge Regression

sklearn.linear_model.Ridge
	(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=’auto’, random_state=None)
Regularized linear regression (L2-norm).
"""
from sklearn import linear_model
reg = linear_model.Ridge (alpha = .5)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1]) 
reg.coef_
reg.intercept_
"""
sklearn.linear_model.RidgeCV
	(alphas=(0.1, 1.0, 10.0), fit_intercept=True, normalize=False, scoring=None, cv=None, gcv_mode=None, store_cv_values=False)
	# Generalized Cross-Validation for alpha = a form of efficient LOOCV
"""
from sklearn import linear_model
reg = linear_model.RidgeCV(alphas=(0.1,0.5,1,10), store_cv_values=True)
reg.fit ([[0, 0], [0, 0], [1, 1]], [0, .1, 1])
reg.cv_values_  # rows are LOO folds, columns are alphas
reg.cv_values_.mean(axis=0) # the last alpha (10) has the highest CV score
reg.alpha_ # and alpha=10 is chosen
reg.coef_
reg.intercept_
"""

> 1.1.3 Lasso

sklearn.linear_model.Lasso
	(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)

Linear Model trained with L1 prior as regularizer
uses coordinate descent as the algorithm to fit the coefficients. See Least Angle Regression for another implementation
"""
from sklearn import linear_model
reg = linear_model.Lasso(alpha = 0.1)
reg.fit([[0, 0], [1, 1]], [0, 1])
reg.predict([[1, 1]])
reg.n_iter_
"""
sklearn.linear_model.lasso_path

	(X, y, eps=0.001, n_alphas=100, alphas=None, precompute=’auto’, Xy=None, copy_X=True, coef_init=None, verbose=False,	return_n_iter=False, positive=False, **params)
    
Paths of coefficients over a range of alphas.
"""
"""
Lasso: selecting value of alpha:
    LassoCV - CV
    LassoLarsCV - Least Angle Regression algorithm for CV
    LassoLarsIC - information critiria (AIC or BIC)
The equivalence between alpha and the regularization parameter of SVM, C is given by alpha = 1 / C
"""
# see Lasso Model Selection:
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html
# done
"""

> 1.1.4. Multi-task Lasso

sklearn.linear_model.MultiTaskLasso
	(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False,
	random_state=None, selection=’cyclic’)

estimates sparse coefficients for multiple regression problems jointly: y is a 2D array, of shape (n_samples, n_tasks). The constraint is that the selected features are the same for all the regression problems, also called tasks
L1/L2 mixed-norm as regularizer
X argument of the fit method should be directly passed as a Fortran-contiguous numpy array.
"""
from sklearn import linear_model
clf = linear_model.MultiTaskLasso(alpha=0.1)
clf.fit([[0,0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 2]])
print(clf.coef_)
print(clf.intercept_)
clf.n_iter_
# see multi task lasso
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_multi_task_lasso_support.html
# done
"""

> 1.1.5. Elastic Net

sklearn.linear_model.ElasticNet

(alpha=1.0, l1_ratio=0.5, fit_intercept=True, normalize=False, precompute=False, max_iter=1000, copy_X=True, tol=0.0001, warm_start=False, positive=False, random_state=None, selection=’cyclic’)

Linear regression with combined L1 and L2 priors as regularizer.
Learning a sparse model where:
    few of the weights are non-zero like Lasso, while still
    maintaining the regularization properties of Ridge (inherit some of Ridge’s stability under rotation)
We control the convex combination of L1 and L2 using the l1_ratio parameter.
Useful when there are multiple features which are correlated with one another. Lasso is likely to pick one of these at random, while elastic-net is likely to pick both.
"""
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression
X, y = make_regression(n_features=2, random_state=0)
regr = ElasticNet(random_state=0)
regr.fit(X, y)
print(regr.coef_) 
print(regr.intercept_) 
print(regr.predict([[0, 0]]))
# see Lasso and Elastic Net
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html
# done
"""
sklearn.linear_model.enet_path

Path of coefficients.
"""
"""
sklearn.linear_model.ElasticNetCV

	(l1_ratio=0.5, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute=’auto’,	max_iter=1000, tol=0.0001, cv=None, copy_X=True, verbose=0, n_jobs=1, positive=False, random_state=None, selection=’cyclic’)
	# set the parameters alpha (\alpha) and l1_ratio (\rho) by cross-validation
"""
"""

> 1.1.7. Least Angle Regression

sklearn.linear_model.Lars

	(fit_intercept=True, verbose=False, normalize=True, precompute=’auto’, n_nonzero_coefs=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)

regression algorithm for high-dimensional data - efficient, but sensitive to the effects of noise.
At each step, it finds the predictor most correlated with the response. When there are multiple predictors having equal correlation, instead of continuing along the same predictor, it proceeds in a direction equiangular between the predictors.
"""
from sklearn import linear_model
reg = linear_model.Lars(n_nonzero_coefs=1)
reg.fit([[-1, 1], [0, 0], [1, 1]], [-1.1111, 0, -1.1111])
print(reg.coef_) 
reg.intercept_
"""
sklearn.linear_model.lars_path

	(X, y, Xy=None, Gram=None, max_iter=500, alpha_min=0, method=’lar’, copy_X=True, eps=2.2204460492503131e-16, copy_Gram=True, verbose=0, return_path=True, return_n_iter=False, positive=False)

Computes Lasso Path along the regularization parameter using the LARS algorithm
"""
# see
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_lars.html
# done
"""
> 1.1.8. LARS Lasso

sklearn.linear_model.LassoLars
	(alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute=’auto’, max_iter=500, eps=2.2204460492503131e-16, copy_X=True, fit_path=True, positive=False)
a lasso model implemented using the LARS algorithm
exact solution
full path
"""
from sklearn import linear_model
reg = linear_model.LassoLars(alpha=0.01)
reg.fit([[-1, 1], [0, 0], [1, 1]], [-1, 0, -1])
print(reg.coef_)
reg.intercept_

# see an example for more

"""
> 1.1.9. Orthogonal Matching Pursuit (OMP)

sklearn.linear_model.OrthogonalMatchingPursuit
	(n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute=’auto’)
    
sklearn.linear_model.orthogonal_mp
	(X, y, n_nonzero_coefs=None, tol=None, precompute=False, copy_X=True, return_path=False, return_n_iter=False)

sklearn.linear_model.OrthogonalMatchingPursuitCV
	(copy=True, fit_intercept=True, normalize=True, max_iter=None, cv=None, n_jobs=1, verbose=False)

Approximates the optimal solution with forward feature selection by taregting:
    fix number of non-zero elements (coefs)
    a specific error
Includes at each step the atom most highly correlated with the current residual. It is similar to the simpler matching pursuit (MP) method, but better in that at each iteration, the residual is recomputed using an orthogonal projection on the space of the previously chosen dictionary elements.
"""
from sklearn import linear_model
reg = linear_model.OrthogonalMatchingPursuit() # score for 1: 0.75; for 2: 0.98, for 3: 1
X, y = [[-1, 0, 2, -2], [0, 0, 3, 5], [1, 1, 4, -2],[-2,-3,1,1]], [-1, 0, -1,0]
reg.fit(X, y)
print(reg.coef_)
reg.intercept_
reg.predict(X)
reg.score(X, y)

reg = linear_model.OrthogonalMatchingPursuit(tol=0.1) # tol=0.1 -> 2 non-zero coefs
X, y = [[-1, 0, 2, -2], [0, 0, 3, 5], [1, 1, 4, -2],[-2,-3,1,1]], [-1, 0, -1,0]
reg.fit(X, y)
print(reg.coef_)
reg.intercept_
reg.predict(X)
reg.score(X, y)

"""
> 1.1.10. Bayesian Regression

Estimates the regularization parameter from the data as a random variable:
    The L_2 regularization used in Ridge Regression is equivalent to finding a maximum a posteriori estimation under a Gaussian prior over the parameters w with precision \lambda^{-1}. 

The advantages of Bayesian Regression are:
    It adapts to the data at hand.
    It can be used to include regularization parameters in the estimation procedure.

The disadvantages of Bayesian regression include:
    Inference of the model can be time consuming.
"""
"""
1.1.10.1. Bayesian Ridge Regression
sklearn.linear_model.BayesianRidge
	(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True,
	normalize=False, copy_X=True, verbose=False)
"""
from sklearn import linear_model
X = [[0., 0.], [1., 1.], [2., 2.], [3., 3.]]
y = [0., 1., 2., 3.]
reg = linear_model.BayesianRidge()
reg.fit(X, y)
reg.coef_
reg.predict(X)
reg.score(X, y)

# see example

""" 
1.1.10.2. Automatic Relevance Determination - ARD
sklearn.linear_model.ARDRegression
	(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0,
	 fit_intercept=True, normalize=False, copy_X=True, verbose=False)
"""
# <--------------------------------------------------------------------------------------------------------

"""

> 1.1.11. Logistic regression

sklearn.linear_model.LogisticRegression

(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='liblinear', max_iter=100, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

multiclass case - the training algorithm uses
- the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’
- the cross- entropy loss if the ‘multi_class’ option is set to ‘multinomial’ (Currently the ‘multinomial’ option is supported only by the ‘lbfgs’, ‘sag’ and ‘newton-cg’ solvers.)
"""
# example code





"""


>> 1.2. Linear and Quadratic Discriminant Analysis


"""
"""
Linear Discriminant Analysis

sklearn.discriminant_analysis.LinearDiscriminantAnalysis

(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001)

A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes’ rule.
The model fits a Gaussian density to each class, assuming that all classes share the same covariance matrix.
The fitted model can also be used to reduce the dimensionality of the input by projecting it to the most discriminative directions.
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)
NX = [[-0.8, -1]]
clf.predict(NX)
clf.predict_proba(NX)
clf.predict_log_proba(NX)
clf.score(X,y) # in sample accuracy
clf.get_params()
clf.transform(X) # Project data to maximize class separation.
X
"""
Quadratic Discriminant Analysis

sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis

(priors=None, reg_param=0.0, store_covariances=False, tol=0.0001)
"""
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
y = np.array([1, 1, 1, 2, 2, 2])
clf = QuadraticDiscriminantAnalysis()
clf.fit(X, y)
NX = [[-0.8, -1]]
clf.predict(NX)
clf.predict_proba(NX)
clf.predict_log_proba(NX)
clf.get_params()

"""


>> 1.4. Support Vector Machines


"""
"""
C-Support Vector Classification

sklearn.svm.SVC

(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=None, random_state=None)

The implementation is based on libsvm. The fit time complexity is more than quadratic with the number of samples which makes it hard to scale to dataset with more than a couple of 10000 samples.
The multiclass support is handled according to a one-vs-one scheme.
For details on the precise mathematical formulation of the provided kernel functions and how gamma, coef0 and degree affect each other, see the corresponding section in the narrative documentation: Kernel functions.
"""
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC
clf = SVC()
clf.fit(X, y) 
print(clf.predict([[-0.8, -1]]))
clf.decision_function(X) # distance of the samples X to the separating hyperplane
"""
sklearn.svm.NuSVC

(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,	tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1,	decision_function_shape=None, random_state=None)
"""
import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import NuSVC
clf = NuSVC()
clf.fit(X, y) 
print(clf.predict([[-0.8, -1]]))
"""
sklearn.svm.LinearSVC

(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)
"""
"""


>> 1.5. Stochastic Gradient Descent


"""
"""

sklearn.linear_model.SGDClassifier

(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True,	n_iter=5, shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal',	eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)

Linear classifiers (SVM, logistic regression, a.o.) with SGD training
data should have zero mean and unit variance
Much faster training than SVM
supports multi-class classification by combining multiple binary classifiers in a “one versus all” (OVA) scheme

The concrete loss function can be set via the loss parameter. SGDClassifier supports the following loss functions:
- loss="hinge": (soft-margin) linear Support Vector Machine,
- loss="modified_huber": smoothed hinge loss,
- loss="log": logistic regression,
- and all regression losses below.

Using loss="log" or loss="modified_huber" enables the predict_proba method, which gives a vector of probability estimates P(y|x) per sample x.

The concrete penalty can be set via the penalty parameter. SGD supports the following penalties:
- penalty="l2": L2 norm penalty on coef_.
- penalty="l1": L1 norm penalty on coef_.
- penalty="elasticnet": Convex combination of L2 and L1; (1 - l1_ratio) * L2 + l1_ratio * L1.

In the case of multi-class classification
- coef_ is a two-dimensionally array of shape=[n_classes, n_features]
- intercept_ is a one dimensional array of shape=[n_classes]
- The i-th row of coef_ holds the weight vector of the OVA classifier for the i-th class
- classes are indexed in ascending order (see attribute classes_).
Note that, in principle, since they allow to create a probability model, loss="log" and loss="modified_huber" are more suitable for one-vs-all classification.
"""
# 1
from sklearn import linear_model
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
Y = np.array([1, 1, 2, 2])
clf = linear_model.SGDClassifier()
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))
# 2
from sklearn.linear_model import SGDClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = SGDClassifier(loss="hinge", penalty="l2")
clf.fit(X, y)
clf.predict([[2., 2.]])
clf.coef_
clf.intercept_
clf.decision_function([[2., 2.]])
# 3
clf = SGDClassifier(loss="log").fit(X, y)
clf.predict_proba([[1., 1.]])                      

# <-------------------------------------------------------------------------------------
"""


>> 1.6. Nearest Neighbors


"""
"""
sklearn.neighbors.NearestNeighbors

(n_neighbors=5, radius=1.0, algorithm='auto', leaf_size=30, metric='minkowski', p=2, metric_params=None, n_jobs=1, **kwargs)

Unsupervised learner for implementing neighbor searches.
"""
# <--------------------------------------------------------------------------------------
"""
KNeighborsClassifier

sklearn.neighbors.KNeighborsClassifier

(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
"""
from sklearn.neighbors import KNeighborsClassifier
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
neigh = KNeighborsClassifier(n_neighbors=3) # n_jobs=-1
neigh.fit(X, y) 
print(neigh.predict([[1.1]]))
print(neigh.predict_proba([[0.9]]))

# In the following example, we construct a NeighborsClassifier class from an array representing our data set and ask who’s the closest point to [1,1,1]
from sklearn.neighbors import NearestNeighbors
samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
neigh = NearestNeighbors(n_neighbors=1)
neigh.fit(samples) 
print(neigh.kneighbors([[1., 1., 1.]]))
neigh.kneighbors([[1., 1., 1.]])[1][0][0]
# -> As you can see, it returns [[0.5]], and [[2]], which means that the element is at distance 0.5 and is the third element of samples (indexes start at 0)
print('The nearest point is ', samples[neigh.kneighbors([[1., 1., 1.]])[1][0][0]])
# You can also query for multiple points
X = [[0., 1., 0.], [1., 0., 1.]]
neigh.kneighbors(X, return_distance=False) 

# returning an array showing the nearest neighbours
from sklearn.neighbors import NearestNeighbors
X = [[0], [3], [1]]
neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(X) 
A = neigh.kneighbors_graph(X)
A.toarray()
"""
sklearn.neighbors.KNeighborsRegressor

(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs)
"""
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(X, y) 
print(neigh.predict([[1.5]]))
# <-------------------------------------------------------------------------------------






"""


>> 1.10. Decision Trees


"""
"""
Decision Tree Classifier

sklearn.tree.DecisionTreeClassifier

(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)

if many features & few samples -> reduce dims not to overfit (PCA, ICA, Feature selection)
tune: max_depth, min_samples_split / min_samples_leaf
max_depth=3 -> visualise (export) & check -> increase depth
if the classes are unbalanced -> use class_weight & replace min_samples_leaf with min_weight_fraction_leaf
"""
# toy
from sklearn import tree
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
clf.predict([[2., 2.]])
clf.predict_proba([[2., 2.]])

# iris
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
clf.predict(iris.data[:1, :])
clf.predict_proba(iris.data[:1, :]) # fractions of each class in the leaf

# export the tree in Graphviz format -> Linux only
import os
os.getcwd()
os.chdir('G:\\Dropbox\\cooperation\\_python\\Extracts')
os.chdir('/home/michal/Dropbox/cooperation/_python/Extracts')

# Graphviz file
with open("data/iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
# using dot tool to create pdf file (or any other supported file type) (you need to install it first)
# dot -Tpdf iris.dot -o iris.pdf
os.unlink('data/iris.dot')

# PDF from Graphviz
# generate a PDF file (or any other supported file type) directly in Python -> this will work only under linux
import pydotplus
# -> conda: PackageNotFoundError: Package missing in current win-64 channels
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("data/iris.pdf")

# Image in IPython from Graphviz
from IPython.display import Image  
dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())  

# another example
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
cross_val_score(clf, iris.data, iris.target, cv=10)



"""
sklearn.tree.DecisionTreeRegressor

(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, presort=False)
"""
from sklearn import tree
X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
"""

> 1.10.3. Multi-output problems

"""
# [...]

# <--------------------------------------------------------------------------------------------------------

"""


>> 1.11. Ensemble methods


"""
"""

> 1.11.1. Bagging meta-estimator


How to draw random subsets of the training set:
When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting [B1999].
When samples are drawn with replacement, then the method is known as Bagging [B1996].
When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces [H1998].
Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches [LG2012].
"""
"""
sklearn.ensemble.BaggingClassifier

(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)

an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction
"""
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),
                            max_samples=0.5, max_features=0.5)
# -> Random Patches with Bagging
# -> see 'plot_bias_variance.py' to see the Regression example
"""
sklearn.ensemble.BaggingRegressor

(base_estimator=None, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)
"""
# -> see 'plot_bias_variance.py'
"""

> 1.11.2. Forests of randomized trees

sklearn.ensemble module includes two averaging algorithms based on randomized decision trees:
    the RandomForest algorithm
    the Extra-Trees method.
Both algorithms are perturb-and-combine techniques specifically designed for trees. This means a diverse set of classifiers is created by introducing randomness in the classifier construction. The prediction of the ensemble is given as the averaged prediction of the individual classifiers.
Like decision trees, forests of trees also extend to multi-output problems.
"""
from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(X, Y)
"""
Random Forest Classifier

sklearn.ensemble.RandomForestClassifier

(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default)
- each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap sample)
- the split that is picked is the best split among a random subset of the features (not among all features as for std trees)
"""
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
iris = load_iris()
clf.fit(iris.data, iris.target)
clf.feature_importances_
# generating predictions
X_test = iris.data[1:5]
clf.predict(X_test)
clf.predict_proba(X_test)
clf.predict_log_proba(X_test)
clf.get_params()





"""
sklearn.ensemble.RandomForestRegressor

(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
"""
# <--------------------------------------------------------------
# get example code
"""
Extremely Randomized Trees

sklearn.ensemble.ExtraTreesClassifier

(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)

Parameters
n_estimators - larger better, but no improvemnt afer some point; computing time
max_features - +variance, -bias
max_features=n_features for regression problems
max_features=sqrt(n_features) for classification tasks
usually good results for max_depth=None & min_samples_split=1(=full trees) but not optimal & lot of RAM
to asses generalization accuracy when using bootstrap: oob_score=True

Parellelization: n_jobs=-1

Feature importances
- the relative rank (i.e. depth) of a feature used as a decision node in a tree
- the expected fraction of the samples they contribute to classify
- feature_importances_ on the fitted model
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
X, y = make_blobs(n_samples=10000, n_features=10, centers=100, random_state=0)
# the classifiers show improving scores
clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()
clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()                             
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
scores.mean()
scores.mean() > 0.999
"""
sklearn.ensemble.ExtraTreesRegressor

(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)
"""
# <--------------------------------------------------------------
# get example code
"""
sklearn.ensemble.RandomTreesEmbedding

(n_estimators=10, max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_leaf_nodes=None, min_impurity_split=1e-07, sparse_output=True, n_jobs=1, random_state=None, verbose=0, warm_start=False)

An unsupervised transformation of a dataset to a high-dimensional sparse representation
The dimensionality of the resulting representation is n_out <= n_estimators * max_leaf_nodes.
If max_leaf_nodes == None, the number of leaf nodes is at most n_estimators * 2 ** max_depth.
"""
# <--------------------------------------------------------------
# get example code
"""

> 1.11.3. AdaBoost

Fit a sequence of weak learners (i.e., models that are only slightly better than random guessing, such as small decision trees) on repeatedly modified versions of the data. Iterations put more weight on wrongly classified samples (difficult to predict samples receive ever-increasing influence).
Can be used both for classification and regression problems.
"""
"""
sklearn.ensemble.AdaBoostClassifier

(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

default=DecisionTreeClassifier
learning rate shrinks the contribution of each classifier (trade-off with n_estimators)
The main parameters to tune to obtain good results are
n_estimators and the complexity of the base estimators (e.g., its depth max_depth or minimum required number of samples at a leaf min_samples_leaf in case of decision trees).
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100)
scores = cross_val_score(clf, iris.data, iris.target)
scores.mean()                
"""
sklearn.ensemble.AdaBoostRegressor

(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None)

default=DecisionTreeRegressor
"""
# <--------------------------------------------------------------
# get example code
"""

> 1.11.4. Gradient Tree Boosting

Gradient Boosted Regression Trees (GBRT) is a generalization of boosting to arbitrary differentiable loss functions.
Classification and Regression
The advantages of GBRT are:
Natural handling of data of mixed type (= heterogeneous features)
Predictive power
Robustness to outliers in output space (via robust loss functions)
The disadvantages of GBRT are:
Scalability, due to the sequential nature of boosting it can hardly be parallelized.
"""
"""
sklearn.ensemble.GradientBoostingClassifier

(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')

for binary and multiclass
warm_start=True allows you to add more estimators to an already fitted model.
"""
# fit a gradient boosting classifier with 100 decision stumps as weak learners
from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier
X, y = make_hastie_10_2(random_state=0)
X_train, X_test = X[:2000], X[2000:]
y_train, y_test = y[:2000], y[2000:]
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
    max_depth=1, random_state=0).fit(X_train, y_train)
clf.score(X_test, y_test)   
# <--------------------------------------------------------------
# read more
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# get example code
"""
sklearn.ensemble.GradientBoostingRegressor

(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_split=1e-07, init=None, random_state=None, max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')
"""
# <--------------------------------------------------------------
# get example code
"""

> 1.11.5. VotingClassifier

combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels.
"""
"""
sklearn.ensemble.VotingClassifier

(estimators, voting='hard', weights=None, n_jobs=1)
"""
# Majority Class Labels (Majority/Hard Voting)
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
# Loading some example data
iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target
# Training classifiers
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), 2*scores.std(ddof=1), label))
# Weighted Average Probabilities (Soft Voting)
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0,2]]
y = iris.target
# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
for clf, label in zip([clf1, clf2, clf3, eclf], ['Decision Tree', 'KNN', 'SVC-RBF', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), 2*scores.std(ddof=1), label))
# -> here ensamble is the worst of all classifiers
# Using the VotingClassifier with GridSearch
from sklearn.model_selection import GridSearchCV
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')
params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(iris.data, iris.target, n_jobs=-1)
# adding weights
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[2,5,1])

"""


>> 1.12. Multiclass and multilabel algorithms


"""
"""

> 1.12.2. One-Vs-The-Rest

"""
"""
OneVsRestClassifier

estimator, n_jobs=1

efficiency + interpretability, default choice for muliticlass classification
"""
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
# multiclass learning
iris = datasets.load_iris()
X, y = iris.data, iris.target
OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y).predict(X)
# multilabel learning
# To use this feature, feed the classifier an indicator matrix, in which cell [i, j] indicates the presence of label j in sample i



"""


>> 1.13 Feature selection


"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape








"""



>>> 2. Unsupervised learning



"""
# ...
"""


>> 2.3 Clustering


"""
# Comparing different clustering algorithms on toy datasets
# http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
"""
> 2.3.2. K-means
sklearn.cluster.KMeans
	(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm=’auto’)

"""
from sklearn.cluster import KMeans
import numpy as np
X = np.array([[1, 2], [1, 4], [1, 0],
              [4, 2], [4, 4], [4, 0]])
X = np.array([[1.5, 2], [1, 3], [1, 1],
              [3.5, 2], [4, 4], [4, 0]])
plt.scatter(*X.T)
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
kmeans.labels_
kmeans.predict([[0, 0], [4, 4]])
kmeans.cluster_centers_
# plot the results
col_dict = {0: 'r', 1:'g'}
col = [col_dict[l] for l in kmeans.labels_]
plt.scatter(*X.T, s=64, color=col)
plt.scatter(*kmeans.cluster_centers_.T, s=512, marker='+', c=['r','g'])


"""


>> 2.5. Decomposing signals in components (matrix factorization problems)


"""
"""
> 2.5.1. Principal component analysis (PCA)
"""
"""
sklearn.decomposition.PCA

(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
"""
import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_) 
pca = PCA(n_components=2, svd_solver='full')
pca.fit(X)                 
print(pca.explained_variance_ratio_) 
pca = PCA(n_components=1, svd_solver='arpack')
pca.fit(X)
print(pca.explained_variance_ratio_) 



















"""



>>> 3. Model selection and evaluation



"""
"""


>> 3.1. Cross-validation: evaluating estimator performance


"""
"""
train_test_split
"""
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
iris.data.shape, iris.target.shape
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
X_train.shape, y_train.shape
X_test.shape, y_test.shape
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_train, y_train) # in-sample
clf.score(X_test, y_test) # test set

"""
cross_val_score
"""
from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf, iris.data, iris.target, cv=5)
scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# -> shouldn't we set ddof=1 for std()?
# customizing the scoring metric
scores = cross_val_score(
    clf, iris.data, iris.target, cv=5, scoring='f1_macro')
scores
# customizing the CV iterator
from sklearn.model_selection import ShuffleSplit
n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=3, test_size=0.3, random_state=0)
cross_val_score(clf, iris.data, iris.target, cv=cv)

"""
corss_val_predict

Only cross-validation strategies that assign all elements to a test set exactly once can be used (otherwise, an exception is raised). -> maybe you could use it in model ensambling, where we need a mapping from predicted responses into true responses, and want to teach a second-level model on this data.
"""
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics.accuracy_score(iris.target, predicted) 

"""

> 3.1.3. Cross-validation iterators for i.i.d. data

"""
"""
KFold
"""
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2)
kf.get_n_splits(X)
print(kf)  
for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
"""
LeaveOneOut
"""
from sklearn.model_selection import LeaveOneOut
X = np.array([[1, 2], [3, 4]])
y = np.array([1, 2])
loo = LeaveOneOut()
loo.get_n_splits(X)
print(loo)
for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)

"""
LeavePOut
"""
from sklearn.model_selection import LeavePOut
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
lpo = LeavePOut(2)
lpo.get_n_splits(X)
print(lpo)
for train_index, test_index in lpo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

"""
ShuffleSplit
"""
from sklearn.model_selection import ShuffleSplit
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
rs = ShuffleSplit(n_splits=3, test_size=.25, random_state=0)
rs.get_n_splits(X)
print(rs)
for train_index, test_index in rs.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
# -> you can reuse it i.e. rerun this loop again and again
# if you set a fixed random_state -> in each run you will get the same splits
# if you set random_state to None -> in each run you will get different random splits
rs = ShuffleSplit(n_splits=3, train_size=0.5, test_size=.25, random_state=0)
for train_index, test_index in rs.split(X): # strange, in each split one observation is left out
   print("TRAIN:", train_index, "TEST:", test_index)

"""

> 3.1.4. Cross-validation iterators with stratification based on class labels

"""
"""
StratifiedKFold
"""
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
print(skf)  
for train_index, test_index in skf.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
"""
StratifiedShuffleSplit

n_splits=10, test_size=0.1, train_size=None, random_state=None

Provides train/test indices to split data in train/test sets.
This cross-validation object is a merge of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds. The folds are made by preserving the percentage of samples for each class.
Note: like the ShuffleSplit strategy, stratified random splits do not guarantee that all folds will be different, although this is still very likely for sizeable datasets.
"""
from sklearn.model_selection import StratifiedShuffleSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
sss = StratifiedShuffleSplit(n_splits=3, test_size=0.5, random_state=0)
sss.get_n_splits(X, y)
print(sss)       
for train_index, test_index in sss.split(X, y):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
## [deprecated] (almost) the same in cross validation
#from sklearn.cross_validation import StratifiedShuffleSplit
#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#y = np.array([0, 0, 1, 1])
#sss = StratifiedShuffleSplit(y, 3, test_size=0.5, random_state=0)
#len(sss)
#print(sss)
#for train_index, test_index in sss:
#   print("TRAIN:", train_index, "TEST:", test_index)
#   X_train, X_test = X[train_index], X[test_index]
#   y_train, y_test = y[train_index], y[test_index]

"""

> 3.1.5. Cross-validation iterators for grouped data

"""
"""
Grouped data

we would like to know if a model trained on a particular set of groups generalizes well to the unseen groups. To measure this, we need to ensure that all the samples in the validation fold come from groups that are not represented at all in the paired training fold.
"""
"""
GroupKFold
"""
from sklearn.model_selection import GroupKFold
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 3, 4])
groups = np.array([0, 0, 2, 2])
group_kfold = GroupKFold(n_splits=2)
group_kfold.get_n_splits(X, y, groups)
print(group_kfold)
for train_index, test_index in group_kfold.split(X, y, groups):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print(X_train, X_test, y_train, y_test)
# 2
from sklearn.model_selection import GroupKFold
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]
gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
"""
LeaveOneGroupOut
"""
from sklearn.model_selection import LeaveOneGroupOut
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 2, 1, 2])
groups = np.array([1, 1, 2, 2])
logo = LeaveOneGroupOut()
logo.get_n_splits(X, y, groups)
print(logo)
for train_index, test_index in logo.split(X, y, groups):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)
"""
LeavePGroupsOut
"""
from sklearn.model_selection import LeavePGroupsOut
X = np.arange(6)
y = [1, 1, 1, 2, 2, 2]
groups = [1, 1, 2, 2, 3, 3]
lpgo = LeavePGroupsOut(n_groups=2)
for train, test in lpgo.split(X, y, groups=groups):
    print("%s %s" % (train, test))
"""
GroupShuffleSplit

This class is useful when the behavior of LeavePGroupsOut is desired, but the number of groups is large enough that generating all possible partitions with P groups withheld would be prohibitively expensive.
"""
from sklearn.model_selection import GroupShuffleSplit
X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))

"""

> 3.1.6. Predefined Fold-Splits / Validation-Sets

"""
"""
PredefinedSplit
"""
from sklearn.model_selection import PredefinedSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
test_fold = [0, 1, -1, 1]
ps = PredefinedSplit(test_fold)
ps.get_n_splits()
print(ps)
for train_index, test_index in ps.split():
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
# -> I don't understand this

"""

> 3.1.7. Cross validation of time series data

"""
"""
Time-series data

characterised by the correlation between observations that are near in time (autocorrelation)
evaluate our model for time series data on the “future” observations least like those that are used to train the model
"""
"""
TimeSeriesSplit
In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.
"""
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  
for train_index, test_index in tscv.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
# 2
from sklearn.model_selection import TimeSeriesSplit
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4, 5, 6])
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  
for train, test in tscv.split(X):
    print("%s %s" % (train, test))

"""
Shuffling

If the data ordering is not arbitrary (e.g. samples with the same class label are contiguous), shuffling it first may be essential to get a meaningful cross- validation result. However, the opposite may be true if the samples are not independently and identically distributed. For example, if samples correspond to news articles, and are ordered by their time of publication, then shuffling the data will likely lead to a model that is overfit and an inflated validation score: it will be tested on samples that are artificially similar (close in time) to training samples.
Some cross validation iterators, such as KFold, have an inbuilt option to shuffle the data indices before splitting them. Note that:
This consumes less memory than shuffling the data directly.
By default no shuffling occurs, including for the (stratified) K fold cross- validation performed by specifying cv=some_integer to cross_val_score, grid search, etc. Keep in mind that train_test_split still returns a random split.
The random_state parameter defaults to None, meaning that the shuffling will be different every time KFold(..., shuffle=True) is iterated. However, GridSearchCV will use the same shuffling for each set of parameters validated by a single call to its fit method.
To ensure results are repeatable (on the same platform), use a fixed value for random_state.
"""





"""


>> 3.2. Tuning the hyper-parameters of an estimator


"""
"""

A search consists of:
an estimator (regressor or classifier such as sklearn.svm.SVC());
a parameter space;
a method for searching or sampling candidates;
a cross-validation scheme; and
a score function.
"""

"""
GridSearchCV

estimator, param_grid, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score='raise', return_train_score=True

Exhaustive search over specified parameter values for an estimator.
Important members are fit, predict.
GridSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
"""
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svr = svm.SVC()
clf = GridSearchCV(svr, parameters, n_jobs=-1)
clf.fit(iris.data, iris.target)
clf.best_estimator_
clf.best_score_
clf.best_params_
clf.cv_results_
sorted(clf.cv_results_.keys())
clf.cv_results_['rank_test_score']
print("The best parameters are %s with a score of %0.3f" % (clf.best_params_, clf.best_score_))
# generating predictions
X_test = iris.data[1:5]
# Using gs directly
pred = clf.predict(X_test)
# Using best_estimator_.
pred = clf.best_estimator_.predict(X_test)
# Calling each step in the pipeline individual (this code works for CVS)
#X_test_fs = clf.best_estimator_.named_steps['fs'].transform(X_test)
#pred = clf.best_estimator_.named_steps['clf'].predict(X_test_fs)

"""
RandomizedSearchCV

estimator, param_distributions, n_iter=10, scoring=None, fit_params=None, n_jobs=1, iid=True, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score='raise', return_train_score=True

Randomized search on hyper parameters.
RandomizedSearchCV implements a “fit” and a “score” method. It also implements “predict”, “predict_proba”, “decision_function”, “transform” and “inverse_transform” if they are implemented in the estimator used.
- a fixed number (n_iter) of parameter settings is sampled from the specified distributions
- If all parameters are presented as a list, sampling without replacement is performed
- If at least one parameter is given as a distribution, sampling with replacement is used
- It is highly recommended to use continuous distributions for continuous parameters.
"""
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
# get some data
digits = load_digits()
X, y = digits.data, digits.target
# build a classifier
clf = RandomForestClassifier(n_estimators=20)
# Utility function to report best scores
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# specify parameters and distributions to sample from
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11), # this had to be changed from 1 to 2
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_jobs=-1,
                                   n_iter=n_iter_search)
start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates"
      " parameter settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)
################# comparison with exhaustive GridSearchCV
from sklearn.model_selection import GridSearchCV
# use a full grid over all parameters
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10], # this had to be changed from 1 to 2
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# run grid search
grid_search = GridSearchCV(clf, param_grid=param_grid, n_jobs=-1)
start = time()
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)


"""
FeautreUnion & Pipline & GridSearchCV
"""
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
iris = load_iris()
X, y = iris.data, iris.target
combined_features = FeatureUnion([("pca", PCA()), ("univ_select", SelectKBest())])
pipeline = Pipeline([("features", combined_features), ("svm", SVC(kernel="linear"))])
param_grid = dict(features__pca__n_components=[1, 2, 3],
                  features__univ_select__k=[1, 2],
                  svm__C=[0.1, 1, 10])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X, y)
grid_search.best_score_
grid_search.best_params_




"""


>> 3.3. Model evaluation: quantifying the quality of predictions


"""
"""
There are 3 different approaches to evaluate the quality of predictions of a model
- Estimator score method: Estimators have a score method providing a default evaluation criterion for the problem they are designed to solve. This is not discussed on this page, but in each estimator’s documentation.
- Scoring parameter: Model-evaluation tools using cross-validation (such as model_selection.cross_val_score and model_selection.GridSearchCV) rely on an internal scoring strategy. This is discussed in the section The scoring parameter: defining model evaluation rules.
- Metric functions: The metrics module implements functions assessing prediction error for specific purposes. These metrics are detailed in sections on Classification metrics, Multilabel ranking metrics, Regression metrics and Clustering metrics.
- Finally, Dummy estimators are useful to get a baseline value of those metrics for random predictions.
"""

from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf = svm.SVC(probability=True, random_state=0)
cross_val_score(clf, X, y, scoring='neg_log_loss') # correct
model = svm.SVC()
cross_val_score(model, X, y, scoring='wrong_choice') # prints the list of functions
# the full list
from sklearn.metrics import SCORERS
SCORERS

"""
make_scorer
"""
# 1. wrap an existing metric function from the library with non-default values for its parameters
from sklearn.metrics import fbeta_score, make_scorer
ftwo_scorer = make_scorer(fbeta_score, beta=2)
ftwo_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
grid = GridSearchCV(LinearSVC(), param_grid={'C': [1, 10]}, scoring=ftwo_scorer)
# 2. build a completely custom scorer object from a simple python function
def my_custom_loss_func(ground_truth, predictions):
    diff = np.abs(ground_truth - predictions).max()
    return np.log(1 + diff)
# loss_func will negate the return value of my_custom_loss_func,
#  which will be np.log(2), 0.693, given the values for ground_truth
#  and predictions defined below.
loss  = make_scorer(my_custom_loss_func, greater_is_better=False)
score = make_scorer(my_custom_loss_func, greater_is_better=True)
ground_truth = [[1, 1]]
predictions  = [0, 1]
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy='most_frequent', random_state=0)
clf = clf.fit(ground_truth, predictions)
loss(clf,ground_truth, predictions) 
score(clf,ground_truth, predictions) 
"""
3.3.2. Classification metrics
"""
# [...]

# accuracy
from sklearn.metrics import accuracy_score
y_pred = [0, 2, 1, 3]
y_true = [0, 1, 2, 3]
accuracy_score(y_true, y_pred)
accuracy_score(y_true, y_pred, normalize=False)

# Log loss, aka logistic loss or cross-entropy loss.
# the negative log-likelihood of the true labels given a probabilistic classifier’s predictions
from sklearn.metrics import log_loss
log_loss(["spam", "ham", "ham", "spam"],  
         [[.1, .9], [.9, .1], [.8, .2], [.35, .65]])
"""
3.3.2 -> Multiclass and more
"""
# Multiclass
from sklearn import metrics
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]
metrics.precision_score(y_true, y_pred, average='macro')
metrics.recall_score(y_true, y_pred, average='micro')
metrics.f1_score(y_true, y_pred, average='weighted')
metrics.fbeta_score(y_true, y_pred, average='macro', beta=0.5)
metrics.precision_recall_fscore_support(y_true, y_pred, beta=0.5, average=None)
"""
3.3.3. Multilabel ranking metrics
"""
'''
roc_curve

y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True
'''
import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
scores = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
fpr
tpr
thresholds
'''
roc_auc_score

(y_true, y_score, average='macro', sample_weight=None)

ROC doesn’t require optimizing a threshold for each label
roc_auc_score can also be used in multi-class classification, if the predicted outputs have been binarized
In multi-label classification, the roc_auc_score function is extended by averaging over the labels
'''
import numpy as np
from sklearn.metrics import roc_auc_score
y_true = np.array([0, 0, 1, 1])
y_scores = np.array([0.1, 0.4, 0.35, 0.8])
roc_auc_score(y_true, y_scores)

# [...]

"""
3.3.4. Regression metrics
"""
# [...]

"""
3.3.5. Clustering metrics
"""
# [...]

"""
3.3.6. Dummy estimators

simple rules of thumb
"""
"""
DummyClassifier
- stratified generates random predictions by respecting the training set class distribution.
- most_frequent always predicts the most frequent label in the training set.
- prior always predicts the class that maximizes the class prior (like most_frequent`) and ``predict_proba returns the class prior.
- uniform generates predictions uniformly at random.
- constant always predicts a constant label that is provided by the user.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X, y = iris.data, iris.target
y[y != 1] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# compare SVC against most_frequent on accuracy
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
clf = SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) # 0.63
clf = DummyClassifier(strategy='most_frequent',random_state=0)
clf.fit(X_train, y_train)
clf.score(X_test, y_test) # 0.58
# -> SVC isn't much better; let's change the kernel
clf = SVC(kernel='rbf', C=1).fit(X_train, y_train)
clf.score(X_test, y_test) # 0.97
# -> much better now
"""
DummyRegressor
- mean always predicts the mean of the training targets.
- median always predicts the median of the training targets.
- quantile always predicts a user provided quantile of the training targets.
- constant always predicts a constant value that is provided by the user.
(predict ignores input data for these strategies)
"""



"""


>> 3.4 Model Persistance


"""
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  
# preserving the model
import pickle
s = pickle.dumps(clf) # saved as a string
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
y[0]
# using joblib
import os
os.getcwd()
from sklearn.externals import joblib
joblib.dump(clf, 'filename.pkl') # saved as a file (only)
clf = joblib.load('filename.pkl') 

"""


3.5. Validation curves: plotting scores to evaluate models


"""
"""
It is sometimes helpful to plot the influence of a single hyperparameter on the training score and the validation score to find out whether the estimator is overfitting or underfitting for some hyperparameter values.

validation_curve

estimator, X, y, param_name, param_range, groups=None, cv=None, scoring=None, n_jobs=1, pre_dispatch='all', verbose=0

If the training score and the validation score are both low, the estimator will be underfitting. If the training score is high and the validation score is low, the estimator is overfitting and otherwise it is working very well. 
"""
import numpy as np
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices] # shuffled
train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha",
                                              np.logspace(-7, 3, 3))
# -> alpha is the regularization parameter for Ridge regression
# -> default 3-fold CV
np.logspace(-7, 3, 3) # values of alpha 10^-7, 10^-2, 10^3
train_scores # rows = values of alpha (see 10^3 is the worst), columns = folds of CV
valid_scores # ditto but for validation
"""
learning_curve

(estimator, X, y, groups=None, train_sizes=array([ 0.1, 0.33, 0.55, 0.78, 1. ]), cv=None, scoring=None, exploit_incremental_learning=False, n_jobs=1, pre_dispatch='all', verbose=0)

shows the validation and training score of an estimator for varying numbers of training samples
to find out how much we benefit from adding more training data and whether the estimator suffers more from a variance error or a bias error
"""
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
train_sizes, train_scores, valid_scores = learning_curve(
    SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
train_sizes       
train_scores # rows = increasing train sizes; cols = folds of the CV
valid_scores




"""



>>> 4. Dataset transformations



"""
"""


4.1. Pipeline and FeatureUnion: combining estimators


"""
"""

> 4.1.1. Pipeline: chaining estimators

"""
# Pipeline()
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
pipe = Pipeline(estimators)
pipe
# make_pipeline()
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Binarizer
make_pipeline(Binarizer(), MultinomialNB()) 
# steps
pipe.steps[0]
pipe.named_steps['reduce_dim']
# accessing parameters
pipe.get_params()
pipe.set_params(clf__C=10)
# grid search
from sklearn.model_selection import GridSearchCV
params = dict(reduce_dim__n_components=[2, 5, 10],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)
# Individual steps may also be replaced as parameters, and non-final steps may be ignored by setting them to None
from sklearn.linear_model import LogisticRegression
params = dict(reduce_dim=[None, PCA(5), PCA(10)],
              clf=[SVC(), LogisticRegression()],
              clf__C=[0.1, 10, 100])
grid_search = GridSearchCV(pipe, param_grid=params)
# example
from sklearn import svm
from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
# generate some data to play with
X, y = samples_generator.make_classification(
    n_informative=5, n_redundant=0, random_state=42)
# ANOVA SVM-C
anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
# You can set the parameters using the names issued
# For instance, fit using a k of 10 in the SelectKBest
# and a parameter 'C' of the svm
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)                        
# getting the selected features chosen by anova_filter
anova_svm.named_steps['anova'].get_support()



# Pipeline for multiclass encoding
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import make_pipeline

class CustomEncoder():
    def __init__(self):
        self.le = LabelEncoder()
        
    def fit(self, y):
        self.le.fit(y)
        
    def transform(self, y):
        return self.le.transform(y).reshape(-1,1)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, y):
        return self.le.inverse_transform(y) #.reshape(-1,1)

response = pd.Series(['a','b','c','a','a','b','b','c','c'])
pipe = make_pipeline(CustomEncoder(), LabelBinarizer())
result = pipe.fit_transform(response)
pipe.inverse_transform(result)


"""

> 4.1.2. FeatureUnion: composite feature spaces

"""
"""

>> 4.2 Feature extraction


"""
"""
sklearn.feature_extraction.DictVectorizer

(dtype=<type 'numpy.float64'>, separator='=', sparse=True, sort=True)

This transformer turns lists of mappings (dict-like objects) of feature names to feature values into Numpy arrays or scipy.sparse matrices for use with scikit-learn estimators.
"""
from sklearn.feature_extraction import DictVectorizer
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]
vec = DictVectorizer()
vec.fit_transform(measurements).toarray()
vec.get_feature_names()

# extended example
import numpy as np
import pandas as pd, os
from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)
df2, _, _ = one_hot_dataframe(df, ['state'], replace=True)
print(df2)

# in sparse format
import numpy as np
import pandas as pd, os
import scipy.sparse as sps
import itertools
def one_hot_column(df, cols, vocabs):
    mats = []; df2 = df.drop(cols,axis=1)
    mats.append(sps.lil_matrix(np.array(df2)))
    for i,col in enumerate(cols):
        mat = sps.lil_matrix((len(df), len(vocabs[i])))
        for j,val in enumerate(np.array(df[col])):
            mat[j,vocabs[i][val]] = 1.
        mats.append(mat)
    res = sps.hstack(mats)   
    return res
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': ['2000', '2001', '2002', '2001', '2002'],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
df = pd.DataFrame(data)
print(df)
vocabs = []
vals = ['Ohio','Nevada']
vocabs.append(dict(itertools.izip(vals,range(len(vals)))))
vals = ['2000','2001','2002']
vocabs.append(dict(itertools.izip(vals,range(len(vals)))))
print vocabs
print one_hot_column(df, ['state','year'], vocabs).todense()



"""


>> 4.3 Preprocessing data


"""

'''
LabelBinarizer
'''

'''
label_binarize
'''
from sklearn.preprocessing import label_binarize
label_binarize([1, 6], classes=[1, 2, 4, 6])
label_binarize([1, 6], classes=[1, 6, 4, 2]) # order of classes is preserved
label_binarize(['yes', 'no', 'no', 'yes'], classes=['no', 'yes']) # binary classes transformed into a column vector
label_binarize([1,2,2,1], [1,2]) # result: 4 * 1
label_binarize([1,2,2,1], [1,2,3]) # result: 4* 3

"""
applying the preprocessing to the held-out data
"""

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
iris = datasets.load_iris()
iris.data.shape, iris.target.shape
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
# preprocessing
scaler = preprocessing.StandardScaler().fit(X_train) # scaling on X_train
X_train_transformed = scaler.transform(X_train) # transforming X_train
# training
clf = svm.SVC(C=1).fit(X_train_transformed, y_train) # fitting the model
# preprocessing the test set
X_test_transformed = scaler.transform(X_test) # transforming X_test (based on X_train fit)
clf.score(X_test_transformed, y_test)
# same using Pipline
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
clf.fit(X_train, y_train)
clf.score(X_test, y_test) # the same!
# now adding cross-validation
from sklearn.model_selection import cross_val_score
cross_val_score(clf, iris.data, iris.target, cv=5)
"""
sklearn.preprocessing.OneHotEncoder

(n_values='auto', categorical_features='all', dtype=<type 'numpy.float64'>, sparse=True, handle_unknown='error')

transforms each categorical feature with m possible values into m binary features, with only one active

"""
from sklearn.preprocessing import OneHotEncoder
# 1
enc = OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
# variable: possible values -> dummies (positions)
# 1: 0,1 -> 0,1
# 2: 0,1,2 -> 2,3,4
# 3: 0,1,2,3 -> 5,6,7,8
enc.transform([[0, 1, 3]]).toarray()
# 2
enc = OneHotEncoder(n_values=[2, 3, 4])
# Note that there are missing categorical values for the 2nd and 3rd
# features
enc.fit([[1, 2, 3], [0, 2, 0]])  
enc.transform([[1, 0, 0]]).toarray()

"""


>> 4.4. Unsupervised dimensionality reduction


"""


"""


>> 4.5. Random Projection


a simple and computationally efficient way to reduce the dimensionality of the data by trading a controlled amount of accuracy (as additional variance) for faster processing times and smaller model sizes
"""
"""

> 4.5.1. The Johnson-Lindenstrauss lemma

a small set of points in a high-dimensional space can be embedded into a space of much lower dimension in such a way that distances between the points are nearly preserved. The map used for the embedding is at least Lipschitz, and can even be taken to be an orthogonal projection

sklearn.random_projection.johnson_lindenstrauss_min_dim

(n_samples, eps=0.1)

Knowing only the number of samples estimates conservatively the minimal size of the random subspace to guarantee a bounded distortion introduced by the random projection
"""
from sklearn.random_projection import johnson_lindenstrauss_min_dim
johnson_lindenstrauss_min_dim(n_samples=1e6, eps=0.5)
johnson_lindenstrauss_min_dim(n_samples=1e6, eps=[0.5, 0.1, 0.01])
johnson_lindenstrauss_min_dim(n_samples=[1e4, 1e5, 1e6], eps=0.1)
"""

> 4.5.2. Gaussian random projection

sklearn.random_projection.GaussianRandomProjection

(n_components='auto', eps=0.1, random_state=None)
"""
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.GaussianRandomProjection()
X.shape # 10,000 features
X_new = transformer.fit_transform(X)
X_new.shape # 4,000 features
X_new2 = random_projection.GaussianRandomProjection(eps=0.2).fit_transform(X)
X_new2.shape # 1,000 features
"""

> 4.5.3. Sparse random projection

sklearn.random_projection.SparseRandomProjection

(n_components='auto', density='auto', eps=0.1, dense_output=False, random_state=None)

Reduce dimensionality through sparse random projection.
Sparse random matrix is an alternative to dense random projection matrix that guarantees similar embedding quality while being much more memory efficient and allowing faster computation of the projected data.
"""
from sklearn import random_projection
X = np.random.rand(100, 10000)
transformer = random_projection.SparseRandomProjection()
X.shape # 10,000 features
X_new = transformer.fit_transform(X)
X_new.shape # 4,000 features
X_new2 = random_projection.SparseRandomProjection(eps=0.2).fit_transform(X)
X_new2.shape # 1,000 features
# -> Gaussian & Spares both give exactly the same number of new features



"""


>> 4.6 Kernel Approximation


- functions that approximate the feature mappings that correspond to certain kernels, as they are used for example in support vector machines
- perform non-linear transformations of the input, which can serve as a basis for linear classification or other algorithms

The advantage of using approximate explicit feature maps compared to the kernel trick, which makes use of feature maps implicitly, is that explicit mappings can be better suited for online learning and can significantly reduce the cost of learning with very large datasets. Standard kernelized SVMs do not scale well to large datasets, but using an approximate kernel map it is possible to use much more efficient linear SVMs. In particular, the combination of kernel map approximations with SGDClassifier can make non-linear learning on large datasets possible.
"""
"""

> 4.6.1. Nystroem Method for Kernel Approximation

sklearn.kernel_approximation.Nystroem

(kernel='rbf', gamma=None, coef0=1, degree=3, kernel_params=None, n_components=100, random_state=None)

a general method for low-rank approximations of kernels. It achieves this by essentially subsampling the data on which the kernel is evaluated
"""
# [...]

"""

> 4.6.2. Radial Basis Function Kernel

sklearn.kernel_approximation.RBFSampler

(gamma=1.0, n_components=100, random_state=None)

Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform.
It implements a variant of Random Kitchen Sinks.
This transformation can be used to explicitly model a kernel map, prior to applying a linear algorithm, for example a linear SVM.
For a given value of n_components RBFSampler is often less accurate as Nystroem.
- fit function performs the Monte Carlo sampling
- the transform method performs the mapping of the data
"""
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]
rbf_feature = RBFSampler(gamma=1, random_state=1)
X_features = rbf_feature.fit_transform(X)
X # 4 x 2
X_features.shape # 4 x 100
X_features
clf = SGDClassifier()
clf.fit(X_features, y)
clf.score(X_features, y)
"""

> 4.6.3. Additive Chi Squared Kernel
[...]

> 4.6.4. Skewed Chi Squared Kernel
[...]

> 4.6.5. Mathematical Details
[...]

"""
"""

>> 4.7. Pairwise metrics, Affinities and Kernels


evaluate pairwise distances or affinity of sets of samples
contains both distance metrics and kernels
Similarity measure
1. d(a, b) < d(a, c) if objects a and b are considered “more similar” than objects a and c
2. Two objects exactly alike would have a distance of zero
Metrics
1. d(a, b) >= 0, for all a and b
2. d(a, b) == 0, if and only if a = b, positive definiteness
3. d(a, b) == d(b, a), symmetry
4. d(a, c) <= d(a, b) + d(b, c), the triangle inequality
Kernels
- are measures of similarity
- must also be positive semi-definite
Convert between a distance metric (D) and a similarity measure (kernel S)
1. S = np.exp(-D * gamma), where one heuristic for choosing gamma is 1 / num_features
2. S = 1. / (D / np.max(D))
3. more
"""
"""

> 4.7.1. Cosine similarity
[...]

> 4.7.1. Cosine similarity
[...]

> 4.7.3. Polynomial kernel
[...]

> 4.7.4. Sigmoid kernel
[...]

> 4.7.5. RBF kernel
[...]

> 4.7.6. Laplacian kernel
[...]

> 4.7.7. Chi-squared kernel
[...]

"""
"""

>> 4.8. Transforming the prediction target (y)


"""
"""

> 4.8.1. Label binarization

sklearn.preprocessing.LabelBinarizer

(neg_label=0, pos_label=1, sparse_output=False)

Binarize labels in a one-vs-all fashion.
Several regression and binary classification algorithms are available in the scikit. A simple way to extend these algorithms to the multi-class classification case is to use the so-called one-vs-all scheme.
At learning time
- this simply consists in learning one regressor or binary classifier per class. In doing so, one needs to convert multi-class labels to binary labels (belong or does not belong to the class). LabelBinarizer makes this process easy with the transform method.
At prediction time
- one assigns the class for which the corresponding model gave the greatest confidence. LabelBinarizer makes this easy with the inverse_transform method.
"""
from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
lb.fit([1, 2, 6, 4, 2])
lb.classes_
lb.transform([1, 6]) # create a label indicator matrix from a list of multi-class labels
"""
sklearn.preprocessing.MultiLabelBinarizer

(classes=None, sparse_output=False)

Transform between iterable of iterables and a multilabel format
"""
lb = preprocessing.MultiLabelBinarizer()
lb.fit_transform([(1, 2), (3,)]) # For multiple labels per instance
lb.classes_
"""

> 4.8.2. Label encoding

sklearn.preprocessing.LabelEncoder

Normalize labels: Encode labels with value between 0 and n_classes-1
This is sometimes useful for writing efficient Cython routines
"""
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0, 0, 1, 2])
# transform non-numerical labels (as long as they are hashable and comparable) to numerical labels
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
list(le.classes_)
le.transform(["tokyo", "tokyo", "paris"])
list(le.inverse_transform([2, 2, 1]))

""" LabelEncoder
Encode labels with value between 0 and n_classes-1.
"""
# Normalizing labels
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit([1, 2, 2, 6]) # changed into (0,1,1,2)
le.classes_
le.transform([1, 1, 2, 6]) 
le.inverse_transform([0, 0, 1, 2])

# ransform non-numerical labels (as long as they are hashable and comparable) to numerical labels
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"]) # becomes (1,1,2,0), alphabetical order
list(le.classes_)
le.transform(["tokyo", "tokyo", "paris", "amsterdam"]) 
list(le.inverse_transform([2, 2, 1]))















