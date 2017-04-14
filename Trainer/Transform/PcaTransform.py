import numpy as np
import pandas as pd
import Transform.Transform as Transform
import sklearn.decomposition 

class PcaTransform(Transform.Transform):
    def __init__(self, properties_dict):
        super(PcaTransform, self).__init__(properties_dict)

        if 'min_energy' not in properties_dict:
            raise ValueError('Required parameter "min_energy" not in transform specification. Required to be float on range [0, 1]')
        try:
            min_energy = float(properties_dict['min_energy'])
        except:
            raise ValueError('Parameter "min_energy" could not be converted to float: {0}'.format(properties_dict['min_energy']))

        if (min_energy < 0 or min_energy > 1):
            raise ValueError('Parameter "min_energy" was > 1 or < 0. Requried to be on the range [0, 1]')

        self._pca = None
        self._min_energy = min_energy
        self._new_column_names = None

    def train_transform(self, dataset, labels):
        self._pca = sklearn.decomposition.PCA()
        self._pca.fit(dataset)
            
        #Determine minimum number of dimensions to keep
        cumulative_energy = 0.0
        number_of_dimensions = 0
        for i in range(0, len(self._pca.explained_variance_ratio_), 1):
            if (cumulative_energy > self._min_energy):
                break
            cumulative_energy += self._pca.explained_variance_ratio_[i]
            number_of_dimensions += 1

        self.number_of_dimensions = number_of_dimensions
        self.cumulative_energy = cumulative_energy
        self._original_variance_ratios = self._pca.explained_variance_ratio_
        
        #Only save dimensionality reduction transform if dimensionality can be reduced.
        if (number_of_dimensions < len(self._pca.explained_variance_ratio_)):
            self._pca = sklearn.decomposition.PCA(n_components = number_of_dimensions)
            self._pca.fit(dataset)

            #get new column names
            #TODO: Is this indexing correct?
            new_columns = []
            for j in range(0, number_of_dimensions, 1):
                new_column_name = ''
                for i in range(0, self._pca.components_.shape[0], 1):
                    if abs(self._pca.components_[i][j]) > 0.001:
                        new_column_name += '{0}_{1}_'.format(dataset.columns[i], self._pca.components_[i][j])
                new_columns.append(new_column_name)

            self._new_column_names = new_columns

        else:
            self._pca = None
            self._new_column_names = dataset.columns

    def apply_transform(self, dataset, labels):
        if self._pca is None:
            return dataset
        
        return pd.DataFrame(data = self._pca.transform(dataset), columns = self._new_column_names, index = dataset.index)

    def describe():
        des = '''
        This transformation implements sklearn's Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.
        Columns are renamed based upon which columns contribute to each principal component. 
        Documentation: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
        This function takes one parameter:

        min_energy: a float on the range [0, 1] detailing how much of the explained_variance_ will be retained. Setting min_energy = 1 is the equivalent of having the transform do nothing
        '''
        return des

    def summarize(self, file_path):
        with open(file_path, 'w') as f:
            f.write('Original eigenvalue energies: {0}.\n'.format(self._original_variance_ratios))
            if self._pca is None:
                f.write('No PCA was performed. No low-dimensional substructure was detected in the model.\n')
            else:
                f.write('Number of dimensions retained: {0}\n'.format(self.number_of_dimensions))
                f.write('Cumulative energy retained: {0}\n'.format(self.cumulative_energy))
                f.write('\n')
                f.write('Principal components:\n')
                for i in range(0, len(self._new_column_names), 1):
                    f.write('{0}: {1}\n'.format(i, self._new_column_names[i]))

    def get_name():
        return 'PcaTransform'
