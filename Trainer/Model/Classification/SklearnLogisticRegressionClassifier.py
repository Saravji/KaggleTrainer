import json
import Model.SklearnModel as SklearnModel
import sklearn.linear_model

class SklearnLogisticRegressionClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn Logistic Regression classifier
    """
    def __init__(self, properties_dict):
        super(SklearnLogisticRegressionClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn Logistic Regression Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {"C": [.0001, .001, .01, .1, 1, 10, 100, 1000],
                          "penalty": ["l1", "l2"]}

        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature parameters
        with open(file_path + 'sklearn_logistic_regression_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

        #Save weights
        with open(file_path + 'sklearn_logistic_regression_weights.txt', 'w') as f:
            for column in self._column_names:
                f.write('{0}\t'.format(column))
            f.write('intercept\n')

            weights = self._trained_model.coef_
            intercepts = self._trained_model.intercept_

            for i in range(0, weights.shape[0], 1):
                for j in range(0, weights.shape[1], 1):
                    f.write('{0}\t'.format(weights[i][j]))
                f.write('{0}\n'.format(intercepts[i]))

    def _get_default_model(self):
        return sklearn.linear_model.LogisticRegression()

    def get_name():
        return "SklearnLogisticRegressionClassifier"
