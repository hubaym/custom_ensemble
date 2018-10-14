from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LogisticRegression

try:
    import xgboost as xgb
except ImportError:
    IMPORT_XGB=False


if IMPORT_XGB:
    xgb_regressor = xgb.XGBRegressor()
    xgb_regressor_def_par = {'max_depth': [2, 4, 6],
                              'n_estimators': [50, 100, 200]}
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier_def_par = {'max_depth': [2, 4, 6],
                               'n_estimators': [50, 100, 200]}

svc = SVC(kernel="linear", C=0.025)
svc_def_par = {'C': [0.025, .5, 1],
                'kernel': ["linear"],
                'gamma': [1, 2]}

gaussian_process = GaussianProcessClassifier()
gaussian_process_def_par = {"max_iter_predict": [100, 500, 1000]}

gaussian_naivebayes = GaussianNB()

decesiontree_classifier = DecisionTreeClassifier(max_depth=5)
decesiontree_classifier_def_par = {'max_depth': [3, 4, 5, 6]}

randomforest_classifier = RandomForestClassifier()
randomforest_classifier_def_par = {'max_depth': [3, 4, 5, 6],
                                    'n_estimators': [50, 100, 200],
                                    "criterion": ["gini", "entropy"]}

logistic_regression = LogisticRegression()
logistic_regression_def_par = {'solver':['lbfgs', 'liblinear']}
mlp_classifier = MLPClassifier()
mlp_classifier_def_par = {'hidden_layer_sizes': [(5, 2), (10, 2), (15, 2)]}

lasso = Lasso()

lasso_def_par = {'max_iter': [100, 500, 1000]}

ridge = Ridge()
ridge_def_par = {'solver': ['auto', 'lsqr', 'svd']}

elasticnet = ElasticNet()
elasticnet_def_par = {'max_iter': [100, 500, 1000]}

