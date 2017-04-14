import numpy as np
import pandas as pd
import Transform.Transform as Transform
import sklearn.utils

class DownsampleTransform(Transform.Transform):
    def __init__(self, properties_dict):
        super(DownsampleTransform, self).__init__(properties_dict)

        if 'key_label' not in properties_dict:
            raise ValueError('Required parameter "key_label" not in specification.')

        self._key_label = properties_dict['key_label']
        
        if 'ratios' not in properties_dict:
            raise ValueError('Required parameter "ratios" not in specification.')

        self.ratios = properties_dict['ratios']

        if 'random_state' not in properties_dict:
            self._random_state = 42
        else:
            self._random_state = properties_dict['random_state']


    def train_transform(self, dataset, labels):
        #Nothing to train
        pass

    def apply_transform(self, dataset, labels):

        classes_count = labels.value_counts()
        
        if (self._key_label not in classes_count.index):
            raise ValueError('Key Column {0} not in labels. The Downsample transform needs at least one instance of the key class.'.format(self._key_label))

        key_class_count = classes_count[self._key_label]

        new_dataset = []

        for class_count in classes_count.index:

            label_idx = labels.apply(lambda v: v == class_count)

            current_dataset = dataset[label_idx]

            if class_count == self._key_label:
                max_number = key_class_count
            elif class_count in self.ratios:
                max_number = self.ratios[class_count] * key_class_count
            else:
                raise ValueError('Label {0} does not have ratio specified, and is not a key column.'.format(class_count))

            if (max_number < current_dataset.shape[0]):
                current_dataset = sklearn.utils.resample(current_dataset, replace = False, n_samples = max_number, random_state = self._random_state)

            new_dataset.append(current_dataset)

        output_dataset = pd.concat(new_dataset, axis=0)

        return output_dataset

    def describe():
        des = '''
        This transform downsamples classes in the incoming data. 
        Use this transform to deal with unbalanced classes.
        This function takes the following parameters:

        random_state: the seed for the RNG. If not specified, '42' will be used.
        key_class: all samples in this class will be kept. Denote the number of examples in this class as n_key
        ratios: a dictionary of label: float pairs. For each label, up to n_key * ratio examples will be kept. The data will be randomly sampled afterwards.
        
        For example, consider a dataset with 5 0s, and 50 1s. Applying this transform with the following config:
        key_class: '0'
        ratios:
        {
          '1': '2.0'
        }

        would yield a dataset with 5 0s and 10 1s randomly sampled from the 50 1s.
        '''
        return des

    def summarize(self, file_path):
        with open(file_path, 'w') as f:
            f.write('Nothing to summarize for the Downsample Transform.\n')

    def get_name():
        return 'DownsampleTransform'