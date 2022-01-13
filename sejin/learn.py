from sklearn.experimental import enable_halving_search_cv
from sklearn.base import clone

from sklearn.model_selection import train_test_split, HalvingGridSearchCV, RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import SGDClassifier, SGDRegressor, LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest, SelectPercentile, chi2, f_classif
from sklearn.metrics import get_scorer

import numpy as np
import pandas as pd
# import pickle
import json
import os, warnings

from joblib import dump, load

from datetime import datetime as dt
from copy import deepcopy

# TO-DO
# * add bootstrapping for results
# * feature seleciton as part of pipeline

class AutoML():
    def __init__(self, 
                 task: str = "classifier",
                 cv: int = 4,
                 multitask: bool=False,
                 scaling_method: str='standard',
                 search: str='gridsearch',
                 fs: str='none',
                 save_dir: str='./automl'):
        """
        params
        ------
        task
            What kind of task is being handled? One of: ['classifier', 'regressor', 'clustering'] with loose string matching (default = 'classifier')
        cv
            Number of cross-validation folds (default = 4)
        multitask
            Boolean of whether there are more than 1 endpoints
        scaling_method
            Normalization method. One of: ['standard', 'minmax', 'none'] with loose string matching
        search
            Hyperparameter search method. One of ['halvinggrid', 'grid', 'random'] 
        fs
            Feature selection method. One of: ['lasso', ]
        save_dir
            Directory to save model pickles / dataset normalization profile
        """
        self.cv = cv
        self.multitask = multitask  # TO-DO implement mult-task wrapper
        self.task = task.lower()    # TO-DO auto-detect task
        self.search = search.lower()
        self.feature_selection = fs.lower()
        self.scaling_method = scaling_method.lower()

        # checking if it needs new 
        if os.path.exists(os.path.join(save_dir, "profile.json")) or save_dir is None: 
            self.save_dir = save_dir
        else:
            self.save_dir = os.path.join(save_dir, dt.now().strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.save_dir)

        try:
            if "class" in task: 
                self.models = {'logr': LogisticRegression(), 
                               'logr_pen': LogisticRegression(), 
                               'boost_c': GradientBoostingClassifier(), 
                               'rf_c': RandomForestClassifier(), 
                               'tree_c': DecisionTreeClassifier(), 
                               'svc': SVC(), 
                               'mlp_c': MLPClassifier(), 
                               'knn_c': KNeighborsClassifier()}
                self.metrics = ['accuracy', 'average_precision', 'recall', 'f1', 'roc_auc']
            elif "reg" in task:
                self.models = {'linr': LinearRegression(), 
                               'linr_pen': LinearRegression(), 
                               'boost_r': GradientBoostingRegressor(), 
                               'rf_r': RandomForestRegressor(), 
                               'tree_r': DecisionTreeRegressor(), 
                               'svr': SVR(), 
                               'knn_r': KNeighborsRegressor()}
                self.metrics = ['neg_mean_squared_error', 'r2']
            elif 'clust' in task:
                self.models = {'km': KMeans(), 
                               'ap': AffinityPropagation(), 
                               'sc': SpectralClustering()}
                self.metrics = ['silhouette', 'adjusted_rand_score', 'adjusted_mutual_info_score']
            else:
                raise ValueError
            
        except ValueError:
            warnings.warn("Parameter 'task' was invalid. Select one of: ['classifier', 'regressor', 'clustering']")
    
    def scale(self, scaler, X, name='y'):
        """
        Scale data `X` using `scaler` and save profile of mean/variance for each feature
        """

        # convert to numpy array
        if isinstance (X, (pd.DataFrame, pd.Series)):
            X_arr = X.to_numpy()
        else:
            X_arr = X

        # add 2nd dimension
        if X.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # transform
        X_arr = scaler.fit_transform(X_arr)

        # record scaling profile
        if isinstance(X, pd.DataFrame):
            for n, feature in enumerate(X.columns):
                self.profile[feature] = {'mean': scaler.mean_[n], 'var': scaler.var_[n]}
        elif isinstance(X, pd.Series) or X.ndim == 1:
            self.profile[X.name] = {'mean': scaler.mean_[0], 'var': scaler.var_[0]}
        else:
            for n in range(X.shape[1]):
                self.profile[name + "_" + str(n)] = {'mean': scaler.mean_[n], 'var': scaler.var_[n]}

        return X_arr

    def normalize(self, X, y, 
                  method: str = 'standard'):
        """
        params
        ------
        method
            Which noramlization method to use? One of ['standard', 'minmax']

        returns
        -------
        X
        """
        # initialize self.profile containing mean and variance for each feature
        self.profile = {} 

        # set method
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return X, y

        # scale using the chosen scaler
        X = self.scale(scaler, X, name='X')
        if 'class' not in self.task:
            y = self.scale(scaler, y, name='y')
        elif isinstance(X, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype='float64')

        assert 0 < y.sum() < len(y)

        if self.save_dir is not None:
            with open(os.path.join(self.save_dir, f"profile.json"), "w") as f:
                    json.dump(self.profile, f)

        return X, y
    
    def get_hparams(self, model: str):
        """
        Returns hparam dict to finetune
        """
        # TO-DO add regression/clustering hparams

        if model == 'logr':
            return {'penalty': ['none'],
                    'C': np.logspace(-4, 0, num=5),
                    'class_weight': [None, 'balanced'],
                    'solver': ['newton-cg', 'lbfgs', 'saga']}
        elif model == 'logr_pen':
            return [{'penalty': ['l1', 'elasticnet'],
                     'C': np.logspace(-4, 0, num=5),
                     'solver':['saga']},
                    {'penalty': ['l2'],
                    'C': np.logspace(-4, 0, num=5),
                    'l1_ratio': np.linspace(0.1, 0.5, num=5),
                    'solver': ['newton-cg', 'lbfgs', 'saga']}]
        elif model == 'boost_c':
            return {'loss': ['exponential'],
                    'learning_rate': np.logspace(-3, -1, num=3),
                    'n_estimators': np.logspace(2, 4, num=5, base=5, dtype=int),
                    'subsample': np.linspace(0.5, 1., num=3),
                    'criterion': ['friedman_mse', 'mse'],
                    'min_samples_split': np.linspace(2, 8, num=4, dtype=int),
                    'min_samples_leaf': np.linspace(2, 5, num=4, dtype=int),
                    'max_depth': np.linspace(3, 7, num=3, dtype=int),
                    'max_features': ['sqrt', 'log2']}
        elif model == 'rf_c':
            return {'n_estimators': np.logspace(3, 10, num=16, base=2, dtype=int),
                    'criterion': ['gini', 'entropy'],
                    'min_samples_split': np.logspace(1, 3, num=3, base=2, dtype=int),
                    'min_samples_leaf': np.linspace(2, 6, num=5, dtype=int),
                    'max_depth': np.linspace(3, 7, num=3, dtype=int),
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True],
                    'oob_score': [True],
                    'class_weight': ['balanced', 'balanced_subsample']}
        elif model == 'tree_c':
            return {'criterion': ['gini', 'entropy'],
                    'min_samples_split': np.linspace(2, 10, num=5, dtype=int),
                    'min_samples_leaf': np.linspace(1, 6, num=6, dtype=int),
                    'max_depth': np.linspace(2, 9, num=8, dtype=int),
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', None]}
        elif model == 'svc':
            return {'C': np.logspace(-4, 0, num=4),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto'], 
                    'probability': [True],
                    'class_weight': ['balanced', None]}
        elif model == 'mlp_c':
            return {'hidden_layer_sizes': [(128,), (128, 64), (128, 64, 16)],
                    'activation': ['relu', 'tanh'],
                    'solver': ['sgd', 'adam'],
                    'alpha': np.logspace(-1, -4, num=4),
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                    'learning_rate_init': np.logspace(-2, -5, num=8)}
        elif model == 'knn_c':
            return {'n_neighbors': np.logspace(1, 5, num=5, base=2, dtype=int),
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2, 3]}
        else:
            raise NotImplementedError   

    def get_nparams(self,
                    hparams: dict):

        if isinstance(hparams, dict):
            n = 1
            for h in hparams:
                n *= len(hparams[h])
        elif isinstance(hparams, list):
            n = [1. for i in range(len(hparams))]
            for i, hparam in enumerate(hparams):
                for h in hparam:
                    n[i] *= len(hparam[h])
            n = sum(n)
        
        return n

    def tune_hparams(self, model, hparams, kwargs, X, y):
        model = self.get_model(model, hparams, kwargs)
        return model.fit(X, y)
    
    def get_model(self, model, hparams, kwargs):
        if 'halvinggrid' in self.search:
            return HalvingGridSearchCV(model,
                                       hparams,
                                       factor=2,
                                       scoring=self.metrics[-1],
                                       cv=self.cv,
                                       error_score=0.)
        elif 'grid' in self.search:
            return GridSearchCV(model, 
                                hparams, 
                                **kwargs)
        elif 'random' in self.search:
            return RandomizedSearchCV(model, 
                                      hparams, 
                                      **kwargs)
        
    def fit(self, X, y, 
            test_size: float = 0.2,
            n_iter: int = 25,
            normalize: bool = True,
            test: bool = True):
        """
        params
        ------
        X
            independent variables
        y
            dependent variable / endpoint to predict
        test_size
            float representing fraction of dataset to hold-out as the test set
        """
        assert X.ndim == 2

        # TO-DO: use ColumnTransformer to handle numerical vs categorical >> integrate into Pipeline()
        if normalize:
            print("Normalizing model with StandardScaler")
            X_train, y_train = self.normalize(X, y, method='standard') 

        # train/test split
        if test:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=1129)
            print(f"Event rates: {y_train.sum()/len(y_train):.3f} (train) {y_test.sum()/len(y_test):.3f} (test)")
            print(f"Train/test: {len(y_train)}/{len(y_test)}")
        else:
            X_train, y_train = X, y
            print(f"Event rates: {y_train.sum()/len(y_train):.3f}")
            print(f"n = {len(y_train)}")
        
        # feature selection
        if self.feature_selection == 'lasso' or self.feature_selection == 'l1':
            lasso = LogisticRegression(C=1e-1, penalty='l1', solver='saga').fit(X_train, y_train)
            fs = SelectFromModel(lasso, prefit=True)
        elif self.feature_selection == 'kbest':
            fs = SelectKBest(f_classif, k=10).fit(X_train, y_train)
        elif 'percent' in self.feature_selection:
            fs = SelectPercentile(f_classif, percentile=20).fit(X_train, y_train)
        elif 'rfe' in self.feature_selection or 'recursive' in self.feature_selection:
            raise NotImplementedError

        if self.feature_selection != 'none':
            print("original:", X_train.shape)
            X_train = fs.transform(X_train)
            if test:
                X_test = fs.transform(X_test)
            print("transformed:", X_train.shape)

        # initialize saving best_models, best_params
        self.best_models, self.best_params, self.scores, self.auroc, self.cv_results = {}, {}, {}, {}, {}
        
        kwargs = {'scoring': self.metrics, 
                  'n_jobs': -1, 
                  'cv': self.cv, 
                  'refit': self.metrics[-1], 
                  'error_score': 0.,
                  'return_train_score': True}
        
        if 'random' in self.search:
            kwargs['n_iter'] = n_iter

        for k in self.models:
            model = self.models[k]
            
            # get+count hparams
            hparams = self.get_hparams(k)
            n_params = self.get_nparams(hparams)
            
            # perform hparam search
            print(f"\nIterating search through {n_params} hyperparameters for {k}.")
            clf = self.tune_hparams(model, hparams, kwargs, X_train, y_train)
            
            self.cv_results[k] = clf.cv_results_
            self.best_models[k] = clone(clf.best_estimator_)
            # self.best_params[k] = clf.best_params_

            print(f"<< {k} >> -- took {clf.refit_time_:.5f}s to refit.")
            print(f"metric              |      train |     test |")

            #save metrics
            self.scores[k] = {}
            for metric in clf.scorer_: 
                # y_train_hat = clf.predict(X_train)
                # print(clf.scorer_[metric])
                train_score = clf.scorer_[metric](clf, X_train, y_train)
                if test:
                    test_score = clf.scorer_[metric](clf, X_test, y_test)
                else:
                    test_score = train_score
                self.scores[k][metric] = test_score
                if "roc_auc" == metric:
                    self.auroc[k] = test_score
                print(f"{metric:<20}:   {train_score:>8.4f} {test_score:>10.4f}")
            
        
        # fitting final best models across entire dataset
        print("\n\n")
        self.fitted_models = {}
        for k in self.best_models:
            self.fitted_models[k] = self.best_models[k].fit(X, y)
            
            print(f"Training {k} on entire dataset yields AUROC: {get_scorer('roc_auc')(self.fitted_models[k], X, y):.3f}")
            if self.save_dir is not None:
                dump(self.fitted_models[k], os.path.join(self.save_dir, f"{k}.pkl")) 
        
        # if no validation/test set, override self.auroc with cv_results
        if not test: 
            for k in self.cv_results:
                results = self.cv_results[k]
                best_idx = np.argmin(results['rank_test_roc_auc'])
                self.auroc[k] = results["mean_test_roc_auc"][best_idx]

        best_idx = np.argmax(list(self.auroc.values()))            
        model_k = list(self.auroc.keys())[best_idx]
        self.best_model = self.best_models[model_k]
        print(f"\n\nBest model: {self.best_model} with AUROC {self.auroc[model_k]:.4f}")
        print(f"Parameters: {self.best_model.get_params()}")

        return
    
    def predict(self, X):
        """
        Returns probability of each class for each sample
        """
        return self.best_model.predict_proba(X)
    
    def infer(self, X):
        """
        Returns final prediction for each sample
        """
        return self.best_model.predict(X)

    def score(self, X, y):
        return self.best_model.score(X, y)

    def report(self):
        raise NotImplementedError

    def features(self):
        raise NotImplementedError