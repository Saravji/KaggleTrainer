import numpy as np
import pandas as pd
import Transform.Transform as Transform

class ColumnNormalizerTransform(Transform.Transform):
    def __init__(self, properties_dict):
        super(ColumnNormalizerTransform, self).__init__(properties_dict)

        #Check provided parameters for validity
        if 'type' not in properties_dict:
            raise ValueError('Missing required configuration parameter "type" for ColumnNormalizerTransform. Valid values: minmax, normal')
        if properties_dict['type'] not in ['minmax', 'normal']:
            raise ValueError('Unrecognized configuration parameter "type" for ColumnNormalizerTransform. Valid values: minmax, normal. Given value: {0}'.format(properties_dict['type']))

        self._type = properties_dict['type']

        self._name_normalization_properties = {}

    def train_transform(self, dataset, labels):
        for column in dataset.columns:
            norm_dict = {}
            if self._type == 'minmax':
                norm_dict['min'] = dataset[column].min()
                norm_dict['max'] = dataset[column].max()

                #If we get min == max, then set denominator to 1 to avoid dbz error
                if (norm_dict['min'] == norm_dict['max']):
                    norm_dict['max'] = norm_dict['min'] + 1
            elif self._type == 'normal':
                norm_dict['mean'] = dataset[column].mean()
                norm_dict['std'] = dataset[column].std()

                #If std deviation is 0, set to 1 to avoid dbz
                if (norm_dict['std'] == 0):
                    norm_dict['std'] == 1
            else:
                raise ValueError('Somehow, ColumNormalizerTransformType is {0}, which is not valid.'.format(self._type))
            self._name_normalization_properties[column] = norm_dict


    def apply_transform(self, dataset, labels):
        normalized_dataset = pd.DataFrame(columns = dataset.columns)
        for column in dataset.columns:
            if self._type == 'minmax':
                normalized_dataset[column] = dataset[column].apply(lambda v: (v-self._name_normalization_properties[column]['min']) / (self._name_normalization_properties[column]['max'] - self._name_normalization_properties[column]['min']))
            elif self._type == 'normal':
                normalized_dataset[column] = dataset[column].apply(lambda v: (v-self._name_normalization_properties[column]['mean']) / (self._name_normalization_properties[column]['std']))
            else:
                raise ValueError('Somehow, ColumnNormalizerTransformType is {0}, which is not valid'.format(self._type))

        return normalized_dataset

    def describe():
        des = '''
        This transform normalizes the columns of the incoming dataset. 
        It takes one parameter, 'type', which describes how the normalization is to be done. Types:

        'minmax': minmax normalization will be applied to the dataset. That is, v = (v - min) / (max - min)
        'normal': the data will be scaled so that it has zero mean and unit standard deviation
        '''
        return des

    def summarize(self, file_path):
        with open(file_path, 'w') as f:
            f.write('Type of normalization: {0}\n'.format(self._type))
            for column in self._name_normalization_properties:
                current_column = self._name_normalization_properties[column]
                f.write('Column {0}:\n'.format(column))
                for c in current_column:
                    f.write('\t{0}: {1}\n'.format(c, current_column[c]))

    def get_name():
        return 'ColumnNormalizerTransform'