import abc

class Model(object, metaclass=abc.ABCMeta):
    """
    A wrapper for all models integrated into the system.
    """
    def __init__(self, properties_dict):
        """
        Called whenever a new model is instantiated

        :param properties_dict: the additional properties with which to initialize the model
        """
        self.model_name = None

    @abc.abstractmethod
    def train(self, train_data, train_labels):
        """
        Trains the model on the provided dataset.

        :param train_data: the training data
        :param train_labels: the training labels
        """
        pass

    @abc.abstractmethod
    def predict_proba(self, data):
        """
        Performs a probability prediction on the incoming data

        :param data: the data on which to perform a prediction
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def describe():
        """
        Gives a user-friendly description of the model
        """
        pass

    @abc.abstractmethod
    def get_default_hyperparameter_grid(self):
        """
        Gets the default hyperparameter grid for a grid sweep.
        Can be either a grid search (For models with few hyperparameters) or a random search (for models with a lot of hyperparameters)
        """
        pass

    @abc.abstractmethod
    def get_hyperparameters(self):
        """
        Gets the hyperparameters of the trained model.
        """
        pass

    @abc.abstractmethod
    def get_hyperparameters(self):
        """
        Gets the tunable hyperparameters for the model
        """
        pass

    @abc.abstractmethod
    def summarize(self, file_path):
        """
        Gets a summary of the post-trained model.
        The output depends on the model type. 
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def get_name():
        """
        Gets the name of the model
        """
        pass
