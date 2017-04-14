import json
import Model.SklearnModel as SklearnModel
import numpy as np
import pandas as pd
import pydotplus
import PyPDF2
import scipy.stats
import sklearn.ensemble

class SklearnRandomForestClassifier(SklearnModel.SklearnModel):
    """
    A wrapper for the sklearn random forest
    """
    def __init__(self, properties_dict):
        super(SklearnRandomForestClassifier, self).__init__(properties_dict)

    def describe():
        description = "The ScikitLearn Random Forrest Classifier. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
        description += SklearnModel.SklearnModel.describe()
        return description

    def get_default_hyperparameter_grid(self):
        param_dist = {    "n_estimators": range(5, 25, 2),
                          "max_depth": [3, 5, 7, 9, None],
                          "min_samples_split": [.00001, .1, .2, .3, .4, .5, .6, .7, .8, .9, .99],
                          "min_samples_leaf": [.00001, .05, .1, .15, .2, .25, .3, .35, .4, .45, .5],
                          "criterion": ["gini", "entropy"]}
        return {'param_dist': param_dist}
        

    def summarize(self, file_path):
        if self._trained_model is None:
            raise ValueError('Cannot summarize an untrained model.')

        #Save feature importances
        with open(file_path + 'sklearn_random_forest_feature_importances.txt', 'w') as f:
            indices = np.argsort(self._trained_model.feature_importances_)[::-1]
            for i in range (0, len(self._trained_model.feature_importances_), 1):
                f.write('{0}: feature {1} ({2}) (column number {3})\n'.format(i+1, self._column_names[indices[i]], self._trained_model.feature_importances_[indices[i]], indices[i]))

        #Save feature parameters
        with open(file_path + 'sklearn_random_forest_parameters.txt', 'w') as f:
            f.write(json.dumps(self._best_parameters))

        #Save graph viz
        try:
            tree_index = 0
            for estimator in self._trained_model.estimators_:
                with open(file_path + 'sklearn_random_forest_{0}.dot'.format(tree_index), 'w') as f:
                    sklearn.tree.export_graphviz(estimator, 
                                                 out_file=f, 
                                                 feature_names = self._column_names,
                                                 class_names = self._class_names,
                                                 filled = True,
                                                 rounded = True,
                                                 special_characters = True)

                #Save pdf
                graph = pydotplus.graph_from_dot_file(file_path + 'sklearn_random_forest_{0}.dot'.format(tree_index))
                graph.write_pdf(file_path + 'sklearn_random_forest_{0}.pdf'.format(tree_index))
                tree_index += 1
            
            merger = PyPDF2.PdfFileMerger()
            for i in range(0, tree_index, 1):
                with open(file_path + 'sklearn_random_forest_{0}.pdf'.format(i), 'rb') as f:
                    merger.append(PyPDF2.PdfFileReader(f))
            merger.write(file_path + 'sklearn_random_forest_all.pdf')
        except:
            print('Warning: unable to save tree visualizations. Do you have GraphViz installed and in your PATH? (http://www.graphviz.org/)')

    def _get_default_model(self):
        return sklearn.ensemble.RandomForestClassifier()

    def get_name():
        return "SklearnRandomForestClassifier"