##############################################
#
#  TEST.py
#
#  This is an exaple of how to use the pipeline. 
#
#  TODO: Make a better example
#
##############################################

#This is necessary on Windows :(
if __name__ == "__main__":

    import Trainer.Trainer as Trainer
    import pandas as pd
    import numpy as np
    import dill

    #Here, we perform some domain-specific transformations on our data.
    print('Reading data...')
    train_data = pd.read_json('train_data_augmented.json')

    print('Generating labels...')
    train_labels = train_data['interest_level'].apply(lambda r: 0 if r == 'low' else 1 if r == 'medium' else 2)
    del train_data['interest_level']

    #Make heuristics
    def identify_location_outlier(r):
        if (r['latitude'] < 40 or r['latitude'] > 41.5 or r['longitude'] > -73 or r['longitude'] < -75):
            return 0
        return None
        
    def identify_price_outlier(r):
        if (r['price'] < 400 or r['price'] > 50000):
            return 0
        return None

    heuristics = [lambda r: identify_location_outlier(r), lambda r: identify_price_outlier(r)]

    #Initialize pipeline
    print('Initializing pipeline...')
    t = Trainer.Trainer()

    #Create a sample pipeline and visualize
    t.initialize_pipeline(
        heuristics = heuristics,
        transform_specifications = 
        [
            {
                'name': 'ColumnNormalizerTransform',
                'type': 'normal'
            },
            {
                'name': 'SmoteTransform',
                'label': 2,
                'upsample_ratio': 2.0,
                'max_smote_step': 0.5,
                'apply_to': ['train']
            },
            {
                'name': 'DownsampleTransform',
                'key_label': 2,
                'ratios':
                {
                    1: 1.1,
                    0: 1.5
                },
                'apply_to': ['train']
            }
        ],
        model_specifications = 
        [
             {
                'name': 'SklearnRandomForestClassifier',
                'sweep_parameters': True
            },
            {
                'name': 'SklearnExtraTreesClassifier',
                'sweep_parameters': True
            },
            {
                'name': 'SklearnDecisionTreeClassifier',
                'sweep_parameters': True
            },
            {
                'name': 'SklearnKNeighborsClassifier',
                'sweep_parameters': True
            },
            {
                'name': 'SklearnLogisticRegressionClassifier',
                'sweep_parameters': True
            }
        ],
        metric_processor_specification = {'name': 'MulticlassMetrics'},
        output_directory = 'test/',
        best_model_metric = 'neg_log_loss',
        train_split_ratio = 0.8,
        retrain_on_entire_dataset = False,
        random_state = 42)

    t.visualize_pipeline()

    #Train the pipeline
    print('Training...')
    t.train_pipeline(train_data, train_labels)

    #Normally, we'd perform all the domain-specific transformations above on our test data.
    # To illustrate how to generate predictions, we will predict on train data.
    print('Testing...')
    output_labels = t.predict_pipeline(train_data)

    print('Saving to prediction.csv...')
    output_labels.to_csv('test/prediction.csv', index=False)

    print('Saving pipeline to pipeline.dill...')
    with open('test/pipeline.dill', 'wb') as f:
        dill.dump(t, f)