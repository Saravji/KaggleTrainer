import Metrics.MetricsProcessor as MetricsProcessor
import numpy as np
import pandas as pd
import sklearn.metrics

class MulticlassMetricsProcessor(MetricsProcessor.MetricsProcessor):
    def __init__(self):
        pass

    def compute(self, predicted_labels, actual_labels, requested_metrics = None):

        if (requested_metrics == None):
            requested_metrics = MulticlassMetricsProcessor.get_metric_names()

        current_metrics = {}

        if 'neg_log_loss' in requested_metrics:
            current_metrics['neg_log_loss'] = self._compute_log_loss(predicted_labels, actual_labels)

        if 'accuracy' in requested_metrics or 'confusion_matrix' in requested_metrics:
            thresholded_predicted_labels = predicted_labels.apply(lambda r: int(np.argmax(r)), axis=1)
            thresholded_actual_labels = actual_labels.apply(lambda r: int(np.argmax(r)), axis=1)

            if 'accuracy' in requested_metrics:
                current_metrics['accuracy'] = self._compute_accuracy(thresholded_predicted_labels, thresholded_actual_labels)
            if 'confusion_matrix' in requested_metrics:
                current_metrics['confusion_matrix'] = self._compute_confusion_matrix(thresholded_predicted_labels, thresholded_actual_labels)
        return current_metrics

    def get_metric_names():
        return set(['neg_log_loss', 'accuracy'])

    def describe():
        des = '''
        The multiclass metrics processor computes a slew of metrics related to multiclass classification. The metrics computed are the following:
        
        neg_log_loss: from sklearn.metrics. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
        l1: the average l1 error
        l2: the average l2 error
        accuracy: The incoming data points are turned into one-hot vectors. The accuracy is then computed as the trace of the confusion matrix divided by the number of training samples
        confusion_matrix: The incoming data points are turned into one-hot vectors. Then, sklearn's confusion_matrix is used. Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
        '''
        return des

    def compare_metrics(self, first, second, metric):
        #for these metrics, lower is better
        if metric == 'l1' or metric == 'l2' or metric == 'neg_log_loss':
            return first < second
        #for these metrics, higher is better
        elif metric == 'accuracy':
            return first > second
        else:
            raise ValueError('Unrecognized metric {0}. Supported values: "l1", "l2", "neg_log_loss", "accuracy".'.format(metric))

    def get_name():
        return 'MulticlassMetrics'

    def _compute_log_loss(self, predicted_labels, actual_labels):
        return sklearn.metrics.log_loss(actual_labels, predicted_labels)

    def _compute_accuracy(self, predicted_labels, actual_labels):
        confusion_matrix = self._compute_confusion_matrix(predicted_labels, actual_labels)

        trace = 0
        for i in range(0, confusion_matrix.shape[1], 1):
            trace += confusion_matrix[i][i]

        return trace / predicted_labels.shape[0]

    def _compute_confusion_matrix(self, predicted_labels, actual_labels):
        return sklearn.metrics.confusion_matrix(actual_labels, predicted_labels)
