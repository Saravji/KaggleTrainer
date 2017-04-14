import abc

class Transform:
    """
    The base class for all data transforms
    """
    def __init__(self, properties_dict):
        """
        Called when a transform is initialized.

        :param properties_dict: the properties of the transform to initialize 
        """
        if 'apply_to' in properties_dict:
            self.apply_to_training = ('train' in properties_dict['apply_to'])
            self.apply_to_testing = ('test' in properties_dict['apply_to'])
            self.apply_to_prediction = ('predict' in properties_dict['apply_to'])
        else:
            self.apply_to_training = True
            self.apply_to_testing = True
            self.apply_to_prediction = True

    @abc.abstractmethod
    def train_transform(self, dataset, labels):
        """
        Called with the output of the previous phase of the pipeline during training.
        Used with trainers that need to compute some statistic about the dataset.

        :param dataset: the dataset on which to train.
        :param labels: the labels for the dataset.
        """
        pass

    @abc.abstractmethod
    def apply_transform(self, dataset, labels):
        """
        Applies the saved transform to the incoming dataset.
        
        :param dataset: the dataset on which to apply the transform
        :param labels: the labels for the dataset
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def describe():
        """
        Describes the transform and its parameters.
        """
        pass

    @abc.abstractmethod
    def summarize(self, output_path):
        """
        Summarize the transform and its parameters.
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        Gets the name of the transform.
        """
        pass