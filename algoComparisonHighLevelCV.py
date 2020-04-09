import itertools
import pandas as pd
import time
import random
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, learning_curve, train_test_split
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import gc


class AlgoComparison(BaseEstimator):
    """This is an sklearn type class for sequentially testing multiple sklearn
    interface models on data with hyper-parameter tuning for the individual
    models. Hyper-parameter tuning is done using grid search.
    Cross validation is used to choose between models. Hyperparameter
    search space and sklearn interface compatible models are supplied by the
    user. Cross validation and grid search settings can be user configurable.

    NOTE: Difference between the algoComparisonHighLevelCV.py and the algoComparison.py:
    Their difference is that the algoComparisonHighLevelCV.py used the more high level functions of sklearn for cross
    validation and hyper parameter tuning (cross_val_score and GridSearchCV). The pros of this is that it allows you
    to directly parallely process  the cross validaton in these two cases using the n_jobs input of the above two
    functions. The con of this is that those two functions do not allow for fit_params as inputs and therefore extra
    inputs in of the fit function like sample_weights are not possible. On the other hand algoComparison allows for
    fit_params to go into the fit function by breaking the process into lower level steps, but parallel processing will
    need to be done with other packages.

    ----------
    models : List of user supplied sklearn interface models

    model_params: Dictionary of default model parameters

    model_param_search_grid : Dictionary of parameter space
    """

    def __init__(self, models, model_params = None, model_param_search_grid = None, seed = 1,
                 scoring = 'neg_mean_squared_error', cv_n_folds = 5, tuning_type = 'RandomSearch', path_to_save = None,
                 n_jobs = 1):
        self.models = models
        self.mode_params = model_params
        self.model_param_search_grid = model_param_search_grid
        self.seed = seed
        self.scoring =scoring
        self.cv_n_folds = cv_n_folds
        self.tuning_type = tuning_type
        self.path_to_save= path_to_save
        self.n_jobs = n_jobs

    def _fit_cross_val(self, X, y, model, name, loss_weights = None, shuffle = False):
        start = time.time()

        kfold = KFold(n_splits =self.cv_n_folds, shuffle = shuffle)
        if type(model) == Pipeline:
            fit_params = {str(model.steps[-1][0]) + '__sample_weight': loss_weights}
        else:
            fit_params = {'sample_weight': loss_weights}
        cv_results = cross_val_score(model,X,y,cv = kfold, scoring = self.scoring, n_jobs = self.n_jobs, fit_params = fit_params)
        end = time.time()

        if self.scoring[0:3] == 'neg':
            cv_results = cv_results * -1

        print(str(self.cv_n_folds) + '-fold cross validation time for ' + str(name) + ' is ' +
              "{:.2f}".format((end-start)/60, 2) + ' mins for a total of ' + str(len(y)) + ' data points.')
        return cv_results

    def _fit_tuning(self, X, y, model, name, n_iter, loss_weights = None):
        if type(model) == Pipeline:
            fit_params = {str(model.steps[-1][0]) + '__sample_weight': loss_weights}
        else:
            fit_params = {'sample_weight': loss_weights}
        if self.tuning_type == 'GridSearch':
            search = GridSearchCV(model, self.model_param_search_grid[name], scoring=self.scoring, cv = self.cv_n_folds,
                                  n_jobs = self.n_jobs, verbose=1, return_train_score=True)
        elif self.tuning_type == 'RandomSearch':
            search = RandomizedSearchCV(model, self.model_param_search_grid[name], n_iter = n_iter, scoring=self.scoring,
                                  cv = self.cv_n_folds, n_jobs = self.n_jobs, verbose=1, return_train_score=True)

        search_fit = search.fit(X,y, **fit_params)
        print('best estimator is %s' %(search_fit.best_estimator_))
        return search

    def _chk_dict_key(self, dic, key):
        #checks if keys belong in the dict. if not creates an empty dict
        if key not in dic.keys():
            return {}
        else:
            return dic[key]

    def fit(self, X, y, loss_weights = None):
        #Fits all models on training set
        if self.models_params is None:
            self.mode_params = {}
        for name, model in self.models.items():
            #Fitting
            start = time.time()
            X_, y_ = X.copy, y.copy()
            init_param = self.check_dict_key(self.mode_params, name)
            model.set_params(**init_param)
            if type(model) == Pipeline:
                fit_params = {str(model.steps[-1][0]) + '__sample_weight': loss_weights}
            else:
                fit_params = {'sample_weight': loss_weights}
            print('running fit: ' + name + '\n' + str(model))
            model.fit(X_, y_, **fit_params)
            end = time.time()
            print('fitting time for ' + str(name) + ' is ' + '{:.2f'.format((end-start)/60, 2) +
                  ' mins for ' + str(len(y)) + 'data points')
        return self

    def score(self, X, y, loss_weights = None):
        #Evaluates all fitted models on dataset providing for all self.scoring functions in the class

        y_pred ={}
        scores = {}
        for name, model in self.models.items():
            X_ = X.copy()
            y_pred[name] = self.models[name].predict(X_)
            # for multiple scoring functions can do: for i, metric in enumarate(self.scoring) where self.scoring = [mean absolute error, mean squared error, etc]
            scores[name] = self.scoring(y, y_pred[name], sample_weight = loss_weights)
        print(scores)
        return scores

    def fit_and_score(self, X, y, loss_weights=None, test_split = 0.0):
        scores = {}
        indices = np.arrange(len(X))
        X_train, X_test, y_train, y_test, index_train, index_test = train_test_split(
            X, y, indices, test_size = test_split, shuffle = False)
        if loss_weights is None:
            loss_weights_train, loss_weights_test = None, None
        else:
            loss_weights_train, loss_weights_test = loss_weights[index_train], loss_weights[index_test]
        self.fit(X_train, y_train, loss_weights[index_train])
        scores['train_' + self.scoring._name_] = self.score(X_train, y_train, loss_weights_train)
        if test_split != 0.0:
            scores['test_' + self.scoring,__name__] = self.score(X_test, y_test, loss_weights_test)
        print(scores)
        return scores

    def predict(self,X):
        # Not used in anywhere yet, might be useful if just want to asses the predictions, or look into anything specific in them
        result = {}
        for name, model in self.models:
            result[name] = self.model_fitted[name].predict(X)
        return result

    def CV(self, X, y, loss_weights=None, shuffle = False):
        # doing CV and report results for all models

        cv_results = []
        cv_results_dict = {}
        names = []
        # cv_results = {x: {} for x in self.scoring()}
        for name, model in self.models.items():
            print('running CV: ' + name + '\n' + str(model))
            init_param = self._chk_dict_key(self.model_params,
                                            names) #maybe should add if model == 'NN' if there is a problem, or can create initializeModel Function and not have model as an input (just name)
            model.set_params(**init_param)
            cv_result =self._fit_cross_val(X, y, model, name , loss_weights, shuffle)
            print('%s: %f (%f)' % (name, np.mean(cv_results), np.std(cv_result)))
            # for metric in self.scoring.keys():
            #   cv_results[metric] = cv_result[metric]
            cv_results.append(cv_result)
            cv_results_dict[name] = {'mean': np.mean(cv_result), 'std' : np.std(cv_result), 'all_folds' : cv_result}
            names.append(name)

            # for metric in self.scoring.keys(): (run the below boxplots)
            fig = plt.figure()
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(cv_results)
            ax.set_xticklabels(names)
            plt.ylabel('Error')
            plt.show()
            return cv_results_dict

        def tune(self, X, y, loss_weights, n_iter = 10, save=False, fileDirPath = None):
            # tunes by CV all models and returns table tuned object (can access best params ad results from that)
            # also print and plot results
            if self.model_param_search_grid is None:
                self.model_param_search_grid = {}
            self.model_tuned = {}

            for name, model in self.models.items():
                if name in self.model_param_search_grid.keys():
                    print('running tuning: ' + name + '\n' + str(model))
                    init_param = self._chk_dict_key(self.model_params, name)
                    model.set_params(**init_param) # or initialise model
                    self.model_tuned[name] = self._fit_tuning(X, y, model, name, n_iter = n_iter,
                                                              loss_weights = loss_weights) # if no model_para_search_grid it does 5-fold CV)
                    if save == True:
                        self.model_tuned[name].to_csv(fileDirPath + '/tuning_results_%s.csv' % (name))
            return self.model_tuned

        def cross_val_with_learning_curves(self, X, y, scoring = 'neg_mean_absolute_error', n_jobs = 1,
                                           train_size=np.linspace(.1, 1.0, 5), axes=None, ylim=None):
            # Scoring for this function should be different than the self.scoring ones (in self.scoring use
            # metrics.mean_absolute_error(y, y_pred, sample_weight) type of format, here use 'neg_mean_absolute_error'
            # type of format (i.e. from sklearn scoring options) to be compatible with learning_curves()
            # Note: for these learning curves there is no options to optimize weights
            # Note2: for NN can also use the keras.fit function for learning curves which saves the scores as the algorithm is fitted
            cv_result = {}
            for name, model in self.models.items():
                cv_results[name] = {}
                if axes is None:
                    _, axes = plt.subplots(1, 1, figsize=(20, 5))

                    title = 'Learning curves of ' + str(name)
                    axes.set_title(title)
                    if ylim is not None:
                        axes.set_ylim(*ylim)
                    axes.set_xlabel('Training examples')
                    axes.set_ylable('Score')
                    X_, y_ = X.copy(), y.copy()


                    train_sizes, train_scores, test_scores = \
                        learning_curve(model, X_, y_, cv=self.cv_n_folds, n_jobs = n_jobs,
                                       train_sizes=train_size, scoring=scoring)
                    train_scores_mean = np.mean(train_scores, axis=1)
                    train_scores_std = np.std(train_scores, axis=1)
                    test_scores_mean = np.mean(test_scores, axis=1)
                    test_scores_std = np.std(test_scores, axis=1)
                    cv_result[name] = {'train_cv_scores': train_scores, 'train_scores_mean' : train_scores_mean,
                                       'train_scores_std': train_scores_std,
                                       'test_scores_mean': test_scores_mean, 'test_scores_std': test_scores_std,
                                       'test_scores': test_scores}

                    # Plot learning curve
                    axes.grid()
                    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                                      train_scores_mean + train_scores_std, alpha = 0.1,
                                      color='r')
                    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                                      test_scores_mean + test_scores_std, alpha = 0.1,
                                      color='g')
                    axes.plot(train_sizes, train_scores_mean, 'o-', color='r',
                              label='Training score')
                    axes.plot(train_sizes, test_scores_mean, 'o-', color='g',
                              label='Cross-validation score')
                    axes.legend(loc='best')

                    plt.savefig(None % (title))
                    plt.show()

                return cv_results

            def errorAnalysis(self, X, y, dataSetType, rank_by='MAE', sample_weight=None, percentile=None,
                              no_of_plots=30, normalize_by=None, save=True):
                # Plots no of plots number examples around different percentiles of error, e.g. if percentile = 0.5, and no_of_plots=30, find the index of 50th percentile deal and plots the 15 examples below it and the 15 examples above it
                # Percentile = 'top', 'bottom', 'random', or numerical value between 0 and 1
                # percentile = 30, plots no_of_plots number of random examples
                # models should be fitted before this function is being run
                # Input in only the training set that was to assess on (can use train test with same test split as in fit and score)

                for model, name in self.models.items():
                    y_pred = model.predict(X).reshape(-1, 1)
                    dict_temp = {'y_true': list(y[:, 0]), 'y_pred': list(y_pred[:, 0]),
                                'sample_weights': list(sample_weight)}
                    df = pd.DataFrame(dict_temp)
                    df['squared_error'] = ((df['y'] - df['Y_pred'])**2) * df['sample_weights']

                    if normalize_by == 'Maturity':
                            df['error'] = df['sqrd_error_wtd'] / df['loss_weightsInYears']

                    if rank_by == 'MSE':
                        df_sorted = df.sort_values('squared_error')

                    all_examples = list(df_sorted.index.values)
                    if percentile == 'top':
                        examples = all_examples[-no_of_plots:]
                    elif percentile == 'bottom':
                        examples = all_examples[:no_of_plots]
                    elif percentile == 'random' or percentile == None:
                        examples = random.sample(all_examples, no_of_plots)
                    else:
                        percentile_index = int(len(list(df.index.values)) * percentile)
                        examples = all_examples[((percentile_index - no_of_plots) / 2): ((percentile_index + no_of_plots) / 2)]

                    if save ==True:
                        df_sorted.to_csv(None % (dataSetType, name))
                    #review or plot examples somehow?
                    print(examples, df_sorted)