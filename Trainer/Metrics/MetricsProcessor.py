import abc

class MetricsProcessor(object, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def compute(self, predicted_labels, actual_labels, requested_metrics = None):
        """
        Computes the required metrics for a pair of labels

        :param predicted_labels: the labels predicted by the model
        :param actual_labels: the actual labels predicted by the model
        :requested_metrics: the metrics to compute. If None, all metrics are computed
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_metric_names():
        """
        Gets the metrics that can be computed for a given problem type
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def describe():
        """
        Returns a string giving a friendly description of the metric processor
        """
        pass

    @abc.abstractmethod
    def compare_metrics(self, first, second, metric_name):
        """
        returns true if first metric is better than the second, false otherwise
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        returns the name of the processor
        """
        pass