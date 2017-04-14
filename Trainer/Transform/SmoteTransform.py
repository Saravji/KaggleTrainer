import numpy as np
import pandas as pd
import random
import sklearn.neighbors
import sklearn.utils
import Transform.Transform as Transform


class SmoteTransform(Transform.Transform):
    def __init__(self, properties_dict):
        super(SmoteTransform, self).__init__(properties_dict)

        if 'label' not in properties_dict:
            raise ValueError('Required parameter "label" not in specification.')

        self._label = properties_dict['label']
        
        if 'upsample_ratio' not in properties_dict:
            raise ValueError('Required parameter "upsample_ratio" not in specification.')

        self._upsample_ratio = float(properties_dict['upsample_ratio'])
        if (self._upsample_ratio < 1.0):
            raise ValueError('SMOTE transform does not support downsampling. Use DownsampleTransform instead (_upsample_ratio was {0}, which is less than 1)'.format(self._upsample_ratio))

        if 'num_nearest_neighbors' not in properties_dict:
            self._num_nearest_neighbors = 4
        else:
            self._num_nearest_neighbors = int(properties_dict['num_nearest_neighbors'])
            if (self._num_nearest_neighbors < 1):
                raise ValueError('num_nearest_neighbors < 1. Value: {0}'.format(self._num_nearest_neighbors))

        if 'max_smote_step' not in properties_dict:
            self._max_smote_step = 1.0
        else:
            self._max_smote_step = float(properties_dict['max_smote_step'])
            if (self._max_smote_step < 0 or self._max_smote_step > 1):
                raise ValueError('max_smote_step not on range [0, 1]. Value: {0}'.format(self._max_smote_step))

        if 'random_state' not in properties_dict:
            self._random_state = 42
        else:
            self._random_state = properties_dict['random_state']

    def train_transform(self, dataset, labels):
        #Nothing to train
        pass

    def apply_transform(self, dataset, labels):

        random.seed(self._random_state)
        minority_datapoints_idx = labels.apply(lambda v: v == self._label)
        minority_dataset = dataset[minority_datapoints_idx]

        if minority_dataset.shape[0] < 2:
            raise ValueError('Minority class {0} does not have at least two samples. SMOTE transform cannot be performed.'.format(self._label))

		#Compute distance matrix
        distance_metric = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
        distances = distance_metric.pairwise(minority_dataset)
        
        row_number_nn = {}

        row_num = 0
        for row in distances:
            row_part = np.argpartition(row, self._num_nearest_neighbors + 1) #need to exclude self
            row_number_nn[row_num] = list(row_part[1:self._num_nearest_neighbors+1])
            row_num += 1

        num_new_samples = int(minority_dataset.shape[0] * (self._upsample_ratio - 1.0))

        new_df = []
        new_index = []

        for i in range(0, num_new_samples, 1):

            random_source = random.randint(0, minority_dataset.shape[0]-1)
            source = minority_dataset.iloc[random_source]

            random_target = random.randint(0, self._num_nearest_neighbors-1)
            target = minority_dataset.iloc[row_number_nn[i][random_target]]

            smote_step = random.random() * self._max_smote_step

            new_row = {}
            for column in dataset.columns:
                new_row[column] = source[column] + ((source[column] - target[column]) * smote_step)
            new_df.append(new_row)
            new_index.append(minority_dataset.index[random_source])

        upsampled_datapoints = pd.DataFrame(data = new_df, index = new_index)

        return pd.concat([dataset, upsampled_datapoints], axis=0)
      
    def describe():
        des = '''
        This transform upsamples classes in the dataset using the SMOTE algorithm. 
		For more information on this algorithm, consult this paper: https://www.jair.org/media/953/live-953-2037-jair.pdf
		
		This transform takes in two required parameters:
			label: the label to upsample
			upsample_ratio: the ratio of starting_count to ending_count. This should be >1. 
		
		There are also the following optional parameters:
			num_nearest_neighbors: the number of nearest neighbors to consider. The default value is 4.
			max_smote_step: the maximal distance from the initial data point for the SMOTE step. Must be a value on the range [0, 1]. Default is 1.0
			random_state: the state of the RNG. The default value is 42.
        '''
        return des

    def summarize(self, file_path):
        with open(file_path, 'w') as f:
            f.write('Nothing to summarize for the SMOTE Transform.\n')

    def get_name():
        return 'SmoteTransform'