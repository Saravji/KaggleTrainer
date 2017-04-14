import pandas as pd
import numpy as np

class HeuristicPredictor:
    """
    A class to handle simple if-then rules that are found during model training.
    Models will first attempt to use the heuristics to predict on data points.
    This can help deal with outliers
    """
    def __init__(self):
        self._heuristics = []

    def set_heuristics(self, heuristics):
        """
        Sets the heuristics to use for prediction.

        :param heuristics: The heuristics to use
        """
        self._heuristics = heuristics

    def predict(self, dataframe):
        """
        Performs a prediction on all members of the dataframe.
        This will return a series of labels, in the same order as the dataframe.
        """
        return dataframe.apply(lambda r: self._predict_internal(r), axis=1)

    def _predict_internal(self, row):
        label = None
        for heuristic in self._heuristics:
            label = heuristic(row)
            if label is not None:
                break
        return label

    def description(self):
        """
        Prints a user-friendly description of this module
        """
        des = '''
        The heuristic predictor can be used to define simple if-then rules about your data to make predictions.
        The heuristic predictor is applied before any transforms.
        Data points that can be predicted by the heuristic predictor will not be used to train the model.

        The intent of this module is to help deal with outliers in datasets.
        
        Heuristics are expected to take the form of a list of lambda expressions, which will produce a label.
        If the heuristic does not apply, the lambda should return None.
        The heuristics will be applied in order, with earlier predictions taking precedence over later predictions. 
        
        Example: [lambda r: 1 if r['columnName'] > 1000 else None]
        '''
        return des
