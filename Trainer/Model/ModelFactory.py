import inspect
import Model.Classification.SklearnDecisionTreeClassifier
import Model.Classification.SklearnExtraTreesClassifier
import Model.Classification.SklearnKNeighborsClassifier
import Model.Classification.SklearnLogisticRegressionClassifier
import Model.Classification.SklearnMlpClassifier
import Model.Classification.SklearnRandomForestClassifier
import Model.Classification.SklearnSvcClassifier
import Model.Model as Model
import Model.SklearnModel

class ModelFactory:
    def __init__(self):
        self._model_information = []

        #some OO trickery to get all of the concrete implementations of model baseclass
        def get_concrete_subclasses(cls):
            subclasses = []
            for subclass in cls.__subclasses__():
                if not inspect.isabstract(subclass):
                    subclasses.append(subclass)
                subclasses.extend(get_concrete_subclasses(subclass))
            return subclasses

        base_models = get_concrete_subclasses(Model.Model.Model)

        for model in base_models:
            info = {}
            info['name'] = model.get_name()
            info['description'] = model.describe()
            info['class'] = model
            self._model_information.append(info)

    def create(self, model_specifications):
        """
        Creates a model from the given specifications
        
        :param model_specifications: the specifications of the model to create
        """
        models = []
        for model_specification in model_specifications:
            
            if 'name' not in model_specification:
                raise ValueError('Required parameter "model_name" not in specification {0}'.format(model_specification))

            model_name = model_specification['name']

            matched = False
            for model_information in self._model_information:
                if model_information['name'] == model_name:
                    models.append(model_information['class'](model_specification))
                    matched = True
                    break

            if not matched:
                raise ValueError('Unrecognized model type {0}. Valid types: {1}'.format(model_name, ','.join(map(lambda mi: mi['name'], self._model_information))))  

        return models

    def get_model_names_with_descriptions(self):
        """
        Gets the valid model names that can be created by the factory

        Returns a list of dictionaries with the following properties:
            'name': model name
            'description': a description for the model
            'class': a class reference which can be used to instantiate the model
        """
        return self._model_information
        
