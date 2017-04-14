import inspect
import Metrics.MetricsProcessor as MetricsProcessor
import Metrics.MulticlassMetricsProcessor 

class MetricsProcessorFactory():
    def __init__(self):
        self._metrics_processor_information = []

        #some OO trickery to get all of the concrete implementations of model baseclass
        def get_concrete_subclasses(cls):
            subclasses = []
            for subclass in cls.__subclasses__():
                if not inspect.isabstract(subclass):
                    subclasses.append(subclass)
                subclasses.extend(get_concrete_subclasses(subclass))
            return subclasses

        metrics_information = get_concrete_subclasses(MetricsProcessor.MetricsProcessor)

        for metric_information in metrics_information:
            info = {}
            info['name'] = metric_information.get_name()
            info['description'] = metric_information.describe()
            info['class'] = metric_information
            self._metrics_processor_information.append(info)

    def create(self, metric_processor_specification):
        """
        Creates the metric processor specified by the metric_processor_name.

        :param metric_processor_name: the name of the metric processor to create.
        """
        if metric_processor_specification is None or 'name' not in metric_processor_specification:
            raise ValueError("Metric Processor name not specified. Must be one of the following values: {0}.".format(','.join(map(lambda mpi: mpi['name'], self._metrics_processor_information))))
        
        for metrics_processor_information in self._metrics_processor_information:
            if metrics_processor_information['name'] == metric_processor_specification['name']:
                return metrics_processor_information['class']()

        raise ValueError('Metric Processor name {0} not recognized. Must be one of the following values: {1}'.format(metric_processor_specification['name'], ','.join(map(lambda mpi: mpi['name'], self._metrics_processor_information))))

    def get_metric_processor_names_with_information(self):
        """
        Gets the valid metric processor names that can be created by the factory

        Returns a list of dictionaries with the following properties:
            'name': metric processor name
            'description': a description for the metric processor
            'class': a class reference which can be used to instantiate the model
        """
        return self._metrics_processor_information

