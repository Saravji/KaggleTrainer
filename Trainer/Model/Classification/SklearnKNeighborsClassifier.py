import json
import Model.SklearnModel as SklearnModel
import scipy.stats
import sklearn.neighbors

class SklearnKNeighborsClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn K Nearest Neighbors classifier
    """
    def __init__(self, properties_dict):
        super(SklearnKNeighborsClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn K Nearest Neighbors Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html"
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {"leaf_size": scipy.stats.randint(5, 50),
                          "metric": ["minkowski"],
                          "p": [1, 2]}

        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature parameters
        with open(file_path + 'sklearn_k_neighbors_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

    def _get_default_model(self):
        return sklearn.neighbors.KNeighborsClassifier()

    def get_name():
        return "SklearnKNeighborsClassifier"