import json
import Model.SklearnModel as SklearnModel
import numpy as np
import pandas as pd
import pydotplus
import scipy.stats
import sklearn.tree

class SklearnDecisionTreeClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn decision tree classifier 
    """
    def __init__(self, properties_dict):
        super(SklearnDecisionTreeClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn Decision Tree Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html" 
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {"max_depth": [3, 5, 7, 9, None],
                          "min_samples_split": scipy.stats.uniform(loc=0.00001, scale=1),
                          "min_samples_leaf": scipy.stats.uniform(loc=0.00001, scale=0.5),
                          "criterion": ["gini", "entropy"],
                          "class_weight": [None, "balanced"]}
        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature importances
        with open(file_path + 'sklearn_decision_tree_feature_importances.txt', 'w') as f:
            indices = np.argsort(self._trained_model.feature_importances_)[::-1]
            for i in range (0, len(self._trained_model.feature_importances_), 1):
                f.write('{0}: feature {1} ({2}) (column number {3})\n'.format(i+1, self._column_names[indices[i]], self._trained_model.feature_importances_[indices[i]], indices[i]))

        #Save graph viz
        try:
            with open(file_path + 'sklearn_decision_tree.dot', 'w') as f:
                sklearn.tree.export_graphviz(self._trained_model, 
                                             out_file=f, 
                                             feature_names = self._column_names,
                                             class_names = self._class_names,
                                             filled = True,
                                             rounded = True,
                                             special_characters = True)

            #Save pdf
            graph = pydotplus.graph_from_dot_file(file_path + 'sklearn_decision_tree.dot')
            graph.write_pdf(file_path + 'sklearn_decision_tree.pdf')
        except:
            print('Warning: unable to save tree visualizations. Do you have GraphViz installed and in your PATH? (http://www.graphviz.org/)')

        #Save feature parameters
        with open(file_path + 'sklearn_decision_tree_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

    def _get_default_model(self):
        return sklearn.tree.DecisionTreeClassifier()

    def get_name():
        return "SklearnDecisionTreeClassifier"
