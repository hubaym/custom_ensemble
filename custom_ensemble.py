
# coding: utf-8

# In[ ]:

from sklearn.base import BaseEstimator
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Lasso, Ridge, ElasticNet


class CustomEnsemble(object):
    '''
    Xarg = np.array([range(10),range(0,20,2)])
    yarg = np.array([0,0,0,0,0,1,1,1,1,1])
    
    ce = CustomEnsemble(Xarg.T, yarg.T)
    ce.grid_search_layer1('mod1', DefinedModels._xgb_classifier, DefinedModels._xgb_classifier_def_par)
    ce.grid_search_layer1('mod2', DefinedModels._svc ,  DefinedModels._svc_def_par  )
    ce.grid_search_layer1('mod3', DefinedModels._gaussian_process,  DefinedModels._gaussian_process_def_par)
    ce.grid_search_layer1('mod4', DefinedModels._randomforest_classifier, DefinedModels._randomforest_classifier_def_par  )
    
    ce.set_layer1_from_gridsearch('mod1')
    ce.set_layer1_from_gridsearch('mod2')
    ce.set_layer1_from_gridsearch('mod3')
    ce.set_layer1_from_gridsearch('mod4')
    
    
    ce.grid_search_layer2('mod5', DefinedModels._decesiontree_classifier, DefinedModels._decesiontree_classifier_def_par  )
    ce.set_layer2_from_gridsearch('mod5')
    
    Xtest = np.array([range(5, 15),range(10,30,2)])
    ce.predict(Xtest.T)
    
    '''
    
    def __init__(self, X = None, y= None,Xtest = None, ytest= None, n_jobs = 1):
        self._model_layer1 = {}
        self._model_layer2 = {}
        self._grid_searches = {}
        self._X = X
        self._y = y
        self._Xtest = Xtest
        self._ytest = ytest
        self._n_jobs = n_jobs
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value
    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        self._y = value
         
    def predict(self, X_to_predict):
        
        return self._predict_layer_2(self._predict_layer1(X_to_predict))
    
    def grid_search_layer1(self, name, model, parameter = {}, scoring = None):
        
        self._grid_searches[name] = GridSearchCV(estimator = model, 
                                                 param_grid = parameter, 
                                                 cv=5, 
                                                 scoring = scoring,
                                                n_jobs = self._n_jobs)
        if self._X is not None and self._y is not None:
            self._grid_searches[name].fit(self.X, self.y)
            return self._grid_searches[name].cv_results_
        else: 
            raise Exception('No any data was given')
            
    def grid_search_layer2(self, name, model, parameter = {}, scoring = None):
        
        self._predict_layer1()
        self._grid_searches[name] = GridSearchCV(estimator = model, param_grid = parameter,scoring = scoring)
        if self._dataset_blend_train is not None and self._y is not None:
            self._grid_searches[name].fit(self._dataset_blend_train, self.y)
            return self._grid_searches[name].cv_results_
        else: 
            raise Exception('No any data was given')
            
    def set_layer1(self,*model):
        self._set_layer(self._model_layer1, model)
        
    def set_layer2(self,*model):
        self._set_layer(self._model_layer2, model)
        
    def set_layer1_from_gridsearch(self, *names):
        
        args = self._check_arguments(names)
        self.set_layer1(*args)
        
    def set_layer2_from_gridsearch(self, *names):
        
        args = self._check_arguments(names)
        self.set_layer2(*args)
        
    def best_score(self, name):
        
        if name in self._grid_searches:
            return self._grid_searches[name].best_score_
        else:
            raise Exception('this name is not valid'+name)
    
    def feature_importance(self, name, print_list = True, plot = False):
        if name in self._grid_searches:
            model = self._grid_searches[name].best_estimator_
            if hasattr(model,"feature_importances_"):
                if print_list:
                    print(model.feature_importances_)
                if plot:
                    import matplotlib.pyplot as plt
                    get_ipython().magic('matplotlib inline')
                    x = range(len(model.feature_importances_))
                    plt.figure(figsize=(15,5))
                    plt.bar(x, model.feature_importances_)
                    plt.xticks(x, self._X.columns, rotation=80)
            else:
                raise Exception('this name is not valid'+name)
        else:
                print("this model has no feature importances function")
            
    def confusion_matrix(self, name, xtest=None, ytest=None ):
        if name in self._grid_searches:
            from sklearn.metrics import confusion_matrix
            prediction = self._grid_searches[name].best_estimator_.predict(xtest if xtest is not None else self._Xtest)
            CM = confusion_matrix(ytest if ytest is not None else self._ytest, prediction)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            print("TN", TN,"\tFP", FP)
            print("FN", FN,"\tTP", TP)
            print(CM)
            print(TN/(TN+FP), TP/(TP+FN))
        else:
            raise Exception('this name is not valid'+name)
        
    def get_report(self, name, x_test= None, y_test=None):
        
        #base score print
        print("best trained score: {0}".format(self.best_score(name)))
        print("########################")
        try:
            #print feature importance report
            self.feature_importance(name, True, True)
        except Exception as e:
            print("feature_importance has some problems",e)
           
        print("########################")
        #print confusion matrix is data is exist
        try:
            self.confusion_matrix(name,x_test ,y_test)
        except Exception as e:
            print("confusion_matrix has some problems",e)
    


            
    ####################
    
    def _predict_layer_2(self, dataset_blend_train):
        
        for i,keymodel in enumerate(self._model_layer2.items()):
            print('predict for model: ', keymodel[0])
            return keymodel[1].predict(dataset_blend_train)
        
    
    def _predict_layer1(self, X_to_predict= None):
        
        X = X_to_predict if X_to_predict is not None else self._X
        
        for i,keymodel in enumerate(self._model_layer1.items()):
            print('predict for model: ', keymodel[0])
            self._dataset_blend_train = np.zeros((X.shape[0], len(self._model_layer1.items())))
            self._dataset_blend_train[:, i] = keymodel[1].predict(X)
        print(self._dataset_blend_train.shape)
        if X_to_predict is not None:
            return self._dataset_blend_train
        
            
            
    def _set_layer(self,layer, *model):
        iterate = model
        print("set_layer ", model)
        if model[0][0]:
            if isinstance(model[0][0], tuple):
                iterate = list(*model)
        else:
             raise Exception('No any argument')
        
        for model_i_name, model_i in iterate:
            if isinstance(model_i, BaseEstimator):
                layer[model_i_name] = model_i
            else:
                raise Exception('The given model is not from BaseEstimator')
        

    def _check_arguments(self,names):
        for i in names:
            if i not in self._grid_searches:
                raise Exception('this name is not valid'+i)
                
        return [(name, self._grid_searches[name].best_estimator_) for name in names]
    
    


class DefinedModels(object):
    
    _xgb_regressor = xgb.XGBRegressor()
    _xgb_regressor_def_par = {'max_depth': [2,4,6],
                                'n_estimators': [50,100,200]}
    _xgb_classifier = xgb.XGBClassifier()
    _xgb_classifier_def_par =  {'max_depth': [2,4,6],
                                'n_estimators': [50,100,200]}
    
    _svc = SVC(kernel="linear", C=0.025)
    _svc_def_par =  {'C': [0.025, .5, 1],
                        'kernel': ["linear"],
                        'gamma': [1,2]}
    
    
    _gaussian_process = GaussianProcessClassifier()
    _gaussian_process_def_par =  {"max_iter_predict": [100,500,1000]}
    
    _gaussian_naivebayes = GaussianNB()
    
    _decesiontree_classifier = DecisionTreeClassifier(max_depth=5)   
    _decesiontree_classifier_def_par = {'max_depth': [3,4,5,6]}
    
    _randomforest_classifier = RandomForestClassifier()
    _randomforest_classifier_def_par = {'max_depth': [3,4,5,6],
                                'n_estimators': [50,100,200],
                                "criterion": ["gini", "entropy"]}
    
    
    _mlp_classifier = MLPClassifier()
    _mlp_classifier_def_par = {'hidden_layer_sizes': [(5, 2), (10, 2), (15, 2)]}
    
    _lasso = Lasso()
    
    _lasso_def_par =  {'max_iter': [100, 500, 1000]}
    
    _ridge = Ridge()
    _ridge_def_par =  {'solver': ['auto','lsqr','svd']}
    
    _elasticnet = ElasticNet()
    _elasticnet_def_par =  {'max_iter': [100, 500, 1000]}
    
    
    
       
    def __init__(self):
        pass
    
if __name__=='__main__':
    Xarg = np.array([range(10),range(0,20,2)])
    yarg = np.array([0,0,0,0,0,1,1,1,1,1])
    
    ce = CustomEnsemble(Xarg.T, yarg.T)
    ce.grid_search_layer1('mod1', DefinedModels._xgb_classifier, DefinedModels._xgb_classifier_def_par)
    ce.grid_search_layer1('mod2', DefinedModels._svc ,  DefinedModels._svc_def_par  )
    ce.grid_search_layer1('mod3', DefinedModels._gaussian_process,  DefinedModels._gaussian_process_def_par)
    ce.grid_search_layer1('mod4', DefinedModels._randomforest_classifier, DefinedModels._randomforest_classifier_def_par  )
    
    ce.set_layer1_from_gridsearch('mod1')
    ce.set_layer1_from_gridsearch('mod2')
    ce.set_layer1_from_gridsearch('mod3')
    ce.set_layer1_from_gridsearch('mod4')
    
    
    ce.grid_search_layer2('mod5', DefinedModels._decesiontree_classifier, DefinedModels._decesiontree_classifier_def_par  )
    ce.set_layer2_from_gridsearch('mod5')
    
    Xtest = np.array([range(5, 15),range(10,30,2)])
    print(ce.predict(Xtest.T))


# In[ ]:



