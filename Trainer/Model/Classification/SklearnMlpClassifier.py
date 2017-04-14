import json
import Model.SklearnModel as SklearnModel
import scipy.stats
import sklearn.neural_network

class SklearnMlpClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn Logistic Regression classifier
    """
    def __init__(self, properties_dict):
        super(SklearnMlpClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn MultilevelPerceptron Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html"
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {"activation": ["relu", "logistic", "tanh"],
                           "hidden_layer_sizes": list(range(10, 20, 1)),
                           "solver": ["sgd", "lbfgs", "adam"],
                           "alpha": scipy.stats.uniform(loc=0.000001, scale=0.001),
                           "learning_rate": ["constant", "adaptive"],
                           "max_iter": [200, 500]}

        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature parameters
        with open(file_path + 'sklearn_mlp_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

    def _get_default_model(self):
        return sklearn.neural_network.MLPClassifier()

    def get_name():
        return "SklearnMlpClassifier"
