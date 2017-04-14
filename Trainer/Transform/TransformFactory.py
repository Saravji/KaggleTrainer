import inspect
import Transform.ColumnNormalizerTransform
import Transform.DownsampleTransform
import Transform.PcaTransform
import Transform.SmoteTransform
import Transform.Transform as Transform

class TransformFactory():
    def __init__(self):
        self._transform_information = []

        #some OO trickery to get all of the concrete implementations of model baseclass
        def get_concrete_subclasses(cls):
            subclasses = []
            for subclass in cls.__subclasses__():
                if not inspect.isabstract(subclass):
                    subclasses.append(subclass)
                subclasses.extend(get_concrete_subclasses(subclass))
            return subclasses

        base_transformations = get_concrete_subclasses(Transform.Transform)

        for transform in base_transformations:
            info = {}
            info['name'] = transform.get_name()
            info['description'] = transform.describe()
            info['class'] = transform
            self._transform_information.append(info)

    def create(self, transform_specifications):
        transforms = []

        for transform_specification in transform_specifications:
            if 'name' not in transform_specification:
                raise ValueError('Required member "name" not in transform specification {0}.'.format(transform_specification))

            transform_type = transform_specification['name']

            matched = False
            for transform_information in self._transform_information:
                if transform_information['name'] == transform_type:
                    transforms.append(transform_information['class'](transform_specification))
                    matched = True
                    break

            if not matched:
                raise ValueError('Tranform {0} not recognized. Valid transform types: {1}.'.format(transform_type, ','.join(map(lambda ti: ti['name'], self._transform_information))))

        return transforms

    def get_transform_names_with_descriptions(self):
        """
        Gets the valid transform names that can be created by the factory

        Returns a list of dictionaries with the following properties:
            'name': transform name
            'description': a description for the transform
            'class': a class reference which can be used to instantiate the transform
        """
        return self._transform_information