from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier, SGDRegressor 
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.pipeline import Pipeline

class AutoML():
    def __init__(self, 
                 task: str = "classifier",
                 cv: int = 4,
                 multitask: bool = False):
        """
        params
        ------
        task
            What kind of task is being handled? One of: ['classifier', 'regressor', 'clustering'] with loose string matching (default = 'classifier')
        cv
            Number of cross-validation folds (default = 4)
        multitask
            Boolean of whether there are more than 1 endpoints
        """
        self.cv = cv
        self.multitask = multitask
        self.task = task

        try:
            if "class" in task: 
                models = ['logr', 'boost_c', 'rf_c', 'tree_c', 'svc', 'mlp']
            elif "reg" in task:
                models = ['linr', 'boost_r', 'rf_r', 'tree_r', 'svr'] #make sure to use l2, l1, elastic for lin/log regressions
            elif 'clust' in task:
                models = ['knn', 'km', 'sc', 'ap']
            else:
                raise TypeInvalidError
        except TypeInvalidError:
            warnings.warn("Parameter 'task' was invalid. Select one of: ['classifier', 'regressor', 'clustering']")
        
        self.models = self.init_models(models)

    def init_models(self, models):
        scikit_models = []
        for model in models:
            if model == 'logr':
                scikit_models.append()
        
        return scikit_models

    def scale(self, scaler, X, name='y'):
        """
        Scale data `X` using `scaler` and save profile of mean/variance for each feature
        """

        if X.ndim == 1:
            if isinstance (X, [pd.DataFrame, pd.Series]):
                name = X.name
                X_arr = X_arr.to_numpy()
            X_arr = X_arr.reshape(-1, 1)

        X_arr = scaler.fit_transform(X_arr)

        if isinstance(X, pd.DataFrame):
            for n, feature in enumerate(X.columns):
                self.profile[feature] = {'mean': scaler.mean_[n], 'var': scaler.var_[n]}
        elif isinstance(X, pd.Series) or X.ndim == 1:
            self.profile[name] = {'mean': scaler.mean_[0], 'var': scaler.var_[0]}
        else:
            for n in range(X.shape[1]):
                self.profile[name + "_" + str(n)] = {'mean': scaler.mean_[n], 'var': norm.var_[n]}

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

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()

        # scale using the chosen scaler
        X = self.scale(scaler, X, name='X')
        y = self.scale(scaler, y, name='y')

        return X, y
        

    def fit(self, X, y, 
            test_size: float = 0.2):
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

        X, y = self.normalize(X, y, scaler='standard')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        for model in self.models:
            pass

        return self.trained_models
