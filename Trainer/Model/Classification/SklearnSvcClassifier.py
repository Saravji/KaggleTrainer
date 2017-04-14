import json
import Model.SklearnModel as SklearnModel
import scipy.stats
import sklearn.svm

class SklearnSvcClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn c-type support vector machine
    """
    def __init__(self, properties_dict):
        super(SklearnSvcClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn C-type Support Vector Machine Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {'C': scipy.stats.expon(scale=100), 
                          'gamma': sp_expon(scale=.1),
                          'kernel': ['linear', 'rbf']}
        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature parameters
        with open(file_path + 'sklearn_svc_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

        #TODO: There are probably more parameters that can be saved here.

    def _get_default_model(self):
        return sklearn.ensemble.RandomForestClassifier()

    def get_name():
        return "SklearnSvcClassifier"