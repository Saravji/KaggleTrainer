import abc
import Model.Model as Model
import numpy as np
import pandas as pd
import sklearn.model_selection

class SklearnModel(Model.Model, metaclass=abc.ABCMeta):
    def __init__(self, properties_dict):

        self._sweep_parameters = False
        if 'sweep_parameters' in properties_dict:
            self._sweep_parameters = True

        if self._sweep_parameters:
            if 'hyperparameter_grid' not in properties_dict:
                hyperparameter_grid = self.get_default_hyperparameter_grid()
            else:
                hyperparameter_grid = properties_dict['hyperparameter_grid']

            if 'hyperparameter_scoring_function' in properties_dict:
                hyperparameter_grid['scoring_function'] = properties_dict['hyperparameter_scoring_function']
            else:
                hyperparameter_grid['scoring_function'] = 'neg_log_loss'

            if 'max_concurrent_threads' in properties_dict:
                hyperparameter_grid['max_concurrent_threads'] = properties_dict['max_concurrent_threads']
            else:
                hyperparameter_grid['max_concurrent_threads'] = -1

            if 'random_state' in properties_dict:
                random_state = properties_dict['random_state']
            else:
                random_state = 42

            if 'number_of_models' in properties_dict:
                n_iter = properties_dict['number_of_models']
            else:
                n_iter = 50

            self._learner = sklearn.model_selection.RandomizedSearchCV(self._get_default_model(), param_distributions = hyperparameter_grid['param_dist'], scoring = hyperparameter_grid['scoring_function'], n_jobs = hyperparameter_grid['max_concurrent_threads'], random_state = random_state, n_iter = n_iter)

        else:
            self._learner = self._get_default_model()

        self._trained_model = None
        self._best_parameters = None
        self._column_names = None
        self._class_names = None

    def train(self, train_data, train_labels):
        self._column_names = train_data.columns
        self._class_names = [str(i) for i in range(0, len(train_labels.unique()), 1)]
        
        #If the user asks for more iterations than an exhaustive grid search would yield, then RandomizedSearchCV will throw. 
        try:
            self._learner.fit(train_data, train_labels)
        except ValueError:
            if self._sweep_parameters and isinstance(self._learner, sklearn.model_selection.RandomizedSearchCV):
                self._learner = sklearn.model_selection.GridSearchCV(self._get_default_model(), param_grid = self._learner.param_distributions, scoring = self._learner.scoring, n_jobs = self._learner.n_jobs)
                self._learner.fit(train_data, train_labels)
            else:
                raise

        if (self._sweep_parameters):
            self._trained_model = self._learner.best_estimator_
            self._best_parameters = self._learner.best_params_
            self._sweep_parameters = False #so that future runs won't sweep parameters
        else:
            self._trained_model = self._learner
            self._best_parameters = self._learner.get_params(deep=True)


    def predict_proba(self, data):
        if self._trained_model is None:
            raise ValueError('Model has not been trained. Call train() to train the model.')

        return pd.DataFrame(data = self._trained_model.predict_proba(data), columns = self._class_names)

    def describe():
        additional_description = '''
        This model supports parameter sweeping. The following parameters can be passed in the specification:

        sweep_parameters: if true, a parameter sweep is performed. Default is false
        hyperparameter_grid: the grid on which to sweep (see param_grid here: http://scikit-learn.org/stable/modules/grid_search.html). If not provided, a default grid will be used.
        max_concurrent_threads: the maximum number of concurrent threads to use to train. If not specified, then one thread for each core in the host CPU will spawn. THIS CAN MAKE YOUR COMPUTER UNUSABLE WHILE MODEL IS TRAINING.
        random_state: the seed for the RNG. If not specified the default value '42' will be used
        number_of_models: the number of models to train. If not specified, the default value 50 will be used.
        '''
        return additional_description

    def get_hyperparameters(self):
        return self._best_parameters


    @abc.abstractmethod
    def get_default_hyperparameter_grid(self):
        pass

    def get_hyperparameters(self):
        return self._get_default_model().get_params()

    @abc.abstractmethod
    def summarize(self, file_path):
        """
        Summarize the trained model
        """
        pass

    @abc.abstractmethod
    def _get_default_model(self):
        """
        Gets the underlying sklearn model used for training
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name():
        pass