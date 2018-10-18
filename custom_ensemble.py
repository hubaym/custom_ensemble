
import predefined_models as pm
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class CustomEnsemble(object):

    
    def __init__(self, X = None, y= None,Xtest = None, ytest= None, n_jobs = 1):
        self._model_layer1 = {}
        self._model_layer2 = {}
        self._grid_searches = {}
        self._grid_searches2 = {}
        self._X = X
        self._y = y
        self._Xtest = Xtest
        self._ytest = ytest
        self._n_jobs = n_jobs

         
    def predict(self, X_to_predict, name = None):
        if name is None:
            return self._predict_layer_2(self._predict_layer1(X_to_predict))
        else:
            return self.best_estimator(name).predict(X_to_predict)
    
    def grid_search_layer1(self, name, model, parameter={}, scoring=None):

        self._grid_searches[name] = GridSearchCV(estimator=model, param_grid=parameter, cv=5, scoring=scoring)
        if self._X is not None and self._y is not None:
            self._grid_searches[name].fit(self._X, self._y)
            return self._grid_searches[name].cv_results_
        else:
            raise Exception('No any data was given')

    def grid_search_layer2(self, name, model, parameter={}, scoring=None):
        self._grid_searches2[name] = GridSearchCV(estimator=model, param_grid=parameter, scoring=scoring)
        if self._dataset_blend_train is not None and self._y is not None:
            self._grid_searches2[name].fit(self._dataset_blend_train, self._y)
            return self._grid_searches2[name].cv_results_
        else:
            raise Exception('No any data was given')

    def set_layer1(self, *model):
        self._set_layer(self._model_layer1, model)

    def set_layer2(self, *model):
        self._set_layer(self._model_layer2, model)

    def set_layer1_from_gridsearch(self, *names):

        args = [(name, self.best_estimator(name)) for name in names]
        self.set_layer1(*args)
        self._predict_layer1()

    def set_layer2_from_gridsearch(self, *names):

        args = [(name, self.best_estimator(name)) for name in names]
        self.set_layer2(*args)

    def _get_gridsearch_model(self, name):
        if name in self._grid_searches:
            return self._grid_searches[name]
        elif name in self._grid_searches2:
            return self._grid_searches2[name]
        else:
            raise Exception('this name is not valid' + name)

    def best_score(self, name):
        return self._get_gridsearch_model(name).best_score_

    def best_estimator(self, name):
        return self._get_gridsearch_model(name).best_estimator_

    def feature_importance(self, name, print_list=True, plot=False):

        model = self.best_estimator(name)
        if not hasattr(model, "feature_importances_"):
            print("Feature doesnt have feature_importances_ attr")
            return
        if print_list:
            print(model.feature_importances_)
        if plot:

            x = range(len(model.feature_importances_))
            plt.figure(figsize=(15, 5))
            plt.bar(x, model.feature_importances_)
            xdataset = self._get_x_train(name)
            cols = xdataset.columns if hasattr(xdataset, "columns") else list(range(0, len(xdataset)))
            plt.xticks(x, cols, rotation=80)

    def confusion_matrix(self, name):
        try:
            from sklearn.metrics import confusion_matrix
            x = self._get_x_test(name)
            print(len(x))
            prediction = self.predict(x, name)
            print(len(prediction))
            print(len(self._ytest))
            CM = confusion_matrix(self._ytest, prediction)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            print("TN", TN, "\tFP", FP)
            print("FN", FN, "\tTP", TP)
            # print(CM)
            # print(TN/(TN+FP), TP/(TP+FN))
        except Exception as e:
            raise Exception('Name is not valid or problem at validation' + name, e)

    def get_report(self, name):

        # base score print
        print("best trained score: {0}".format(self.best_score(name)))

        # print feature importance report
        self.feature_importance(name, False, True)

        # print confusion matrix
        self.confusion_matrix(name)

    ####################

    def _get_x_test(self, name):
        if name in self._grid_searches:
            return self._Xtest
        elif name in self._grid_searches2:
            return self._dataset_blend_test if self._dataset_blend_test is not None else None
        else:
            return None

    def _get_x_train(self, name):
        if name in self._grid_searches:
            return self._X
        elif name in self._grid_searches2:
            return self._dataset_blend_train if self._dataset_blend_train is not None else None
        else:
            return None

    def _predict_layer_2(self, dataset_blend_train):

        for i, keymodel in enumerate(self._model_layer2.items()):
            print('predict for model: ', keymodel[0])
            return keymodel[1].predict(dataset_blend_train)

    def _predict_layer1(self, x=None):

        if x is None:

            self._dataset_blend_train = np.zeros((self._X.shape[0], len(self._model_layer1.items())))
            for i, keymodel in enumerate(self._model_layer1.items()):
                print('predict for model: ', keymodel[0])
                if hasattr(keymodel[1], "predict_proba"):
                    self._dataset_blend_train[:, i] = keymodel[1].predict_proba(self._X)[:, 1]
                else:
                    self._dataset_blend_train[:, i] = keymodel[1].predict(self._X)
                print(keymodel[1].predict(self._X)[:10])

            self._dataset_blend_test = np.zeros((self._Xtest.shape[0], len(self._model_layer1.items())))
            for i, keymodel in enumerate(self._model_layer1.items()):
                print('predict for model: ', keymodel[0])
                if hasattr(keymodel[1], "predict_proba"):
                    self._dataset_blend_test[:, i] = keymodel[1].predict_proba(self._Xtest)[:, 1]
                else:
                    self._dataset_blend_test[:, i] = keymodel[1].predict(self._Xtest)
        else:

            result = np.zeros((x.shape[0], len(self._model_layer1.items())))
            for i, keymodel in enumerate(self._model_layer1.items()):
                print('predict for model: ', keymodel[0])
                if hasattr(keymodel[1], "predict_proba"):
                    result[:, i] = keymodel[1].predict_proba(x)[:, 1]
                else:
                    result[:, i] = keymodel[1].predict(x)
            return result

    def _set_layer(self, layer, *model):
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


    
if __name__=='__main__':
    df = pd.read_csv("example_df.csv", header=None, names=["c" + str(each) for each in range(1, 21)])
    train_cols = ["c" + str(each) for each in range(1, 20)]
    test_col = "c20"
    df.head()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(df[train_cols], df[test_col], test_size=0.3, random_state=42)
    print(len(X_train), len(X_test))
    ce = CustomEnsemble(X_train, y_train, X_test, y_test)
    ce.grid_search_layer1('mod4', pm.randomforest_classifier,
                          pm.randomforest_classifier_def_par)
    ce.get_report('mod4')
    ce.grid_search_layer1('mod2', pm.logistic_regression, pm.logistic_regression_def_par)
    ce.get_report('mod2')
    ce.grid_search_layer1('mod3', pm.gaussian_process, pm.gaussian_process_def_par)
    ce.get_report('mod3')

    ce.set_layer1_from_gridsearch('mod4', 'mod2', 'mod3')

    ce.grid_search_layer2('mod5', pm.decesiontree_classifier,
                          pm.decesiontree_classifier_def_par)
    ce.get_report('mod5')

    ce.grid_search_layer2('mod6', pm.svc, pm.svc_def_par)
    ce.get_report('mod6')




