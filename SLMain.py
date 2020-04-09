
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from SupervisedLearning import kerasNN
from SupervisedLearning.keras_wrapper import KerasRegressor
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def main():
    #------------Data Configurations
    #-------------Load data
    #-------------Data Preparation
    X = None
    y = None
    #-------------Modelling Configurations
    scorer = metrics.mean_absolute_error
    n_iter_randomized_search  = 25
    tuning_type = 'RandomSearch'
    cv_n_folds = 5
    path_to_save = '../SupervisedLearning'

    models = {'LinearRegression': LinearRegression(),
              'XGBoost': GradientBoostingRegressor(),
              'NN': Pipeline([('scale', preprocessing.StandardScaler()), ('reg', KerasRegressor(build_fn = kerasNN))])}

    modelParameters = {'LinearRegression': {},
                       'XGBoost': {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2, 'learning_rate': 0.01,
                                   'loss': 'huber', 'alpha': 0.95},
                       'NN': {'reg__activation': 'elu', 'reg__learning_rate': 0.001, 'reg__input_shape': (X.shape[1],)}
                       }

    tuningParameters = {'LinearRegression': {'alpha': list(np.logspace(np.log10(0.0001), np.log10(1), base = 10, num = 24))},
                        'NN': {'reg__learning_rate': list(np.logspace(np.log10(0.0001), np.log10(1), base = 10, num = 8)),
                               'reg__optimizer': ['adam', 'rmsprop'],
                               'reg__hidder_layers': [1,2,3,4,5],
                               'reg__neurons': [10, 20, 30, 40, (32,16,8,4,2)],
                               'reg__activation': ['elu', 'relu', 'tanh'],
                               'reg__batch_size': [16,32,64,128],
                               'reg__epochs': [100,200],
                               'reg__l2_penatly': list(np.append(np.logspace(np.log10(0.0001), np.log10(1), base = 10, num = 8),0)),
                               'reg__weight_initializer': ['glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'],
                               'reg__dropout': [True,False],
                               'reg__batch_normalization': [True,False]
                               }
                        }
    from SupervisedLearning import algoComparison

    algoComparison = algoComparison(models, model_params = modelParameters, model_param_search_grid = tuningParameters,
                                    cv_n_folds = cv_n_folds, tuning_type = tuning_type, path_to_save = path_to_save)

    scores = algoComparison.fit_and_score(X,y, loss_weights=None, test_split=0.2)
    cv_results = algoComparison.CV(X, y, loss_weights= None)
    tuning_results = algoComparison.tuning(X, y, loss_weights= None, save = True, n_iter = n_iter_randomized_search)

