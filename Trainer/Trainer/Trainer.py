import Heuristics.HeuristicPredictor as HeuristicPredictor
import Model.ModelFactory as ModelFactory
import Metrics.MetricsProcesorFactory as MetricsProcessorFactory
import numpy as np
import os
import pandas as pd
import sklearn.model_selection as model_selection
import sys
import Transform.TransformFactory as TransformFactory

class Trainer:
    def __init__(self, ignore_warnings=True):
        self._model_factory = ModelFactory.ModelFactory()
        self._transform_factory = TransformFactory.TransformFactory()
        self._metrics_processor_factory = MetricsProcessorFactory.MetricsProcessorFactory()
        self._heuristic_predictor = HeuristicPredictor.HeuristicPredictor()
        self._trained_transforms = []
        self._trained_models = []
        self._best_model = None
        self._metrics_processor = None
        self._initialized = False
        self._trained = False

        # When sklearn calls numpy, it triggers some internal warnings.
        #  Prevent those from cluttering the console.
        if (ignore_warnings):
            np.warnings.filterwarnings('ignore')

    def help_components(self):
        """
        Prints all of the valid component names that can be used in the system and a description of the different component types.
        """
        print('-------------------------------------')
        print('-----------Component Types-----------')
        print('-------------------------------------')
        print('Heuristic Predictor: allows injection of simple if-then rules for predicting labels into model pipeline.')
        print('Metrics Processor: computes metrics associated with a model.')
        print('Model: a ML algorithm that predicts labels given data.')
        print('Transform: a data transformation that consumes an input dataset and produces an output dataset')
        print('')
        print('-------------------------------------')
        print("--------Valid component names--------")
        print('-------------------------------------')
        print('Heuristic Predictor')
        print('\tNone. To use, specify heuristics when initializing pipeline.')
        print('Metrics Processor')
        for name in map(lambda mpi: mpi['name'], self._metrics_processor_factory.get_metric_processor_names_with_information()):
            print('\t{0}'.format(name))
        print('Model')
        for name in map(lambda mi: mi['name'], self._model_factory.get_model_names_with_descriptions()):
            print('\t{0}'.format(name))
        print('Transform')
        for name in map(lambda ti: ti['name'], self._transform_factory.get_transform_names_with_descriptions()):
            print('\t{0}'.format(name))
        print('')
        print('For more information about a particular component, call one of the help_x() methods with that particular component name.')

    def help_model(self, model_name):
        """
        Prints the help for a model name
        """
        valid_models = self._model_factory.get_valid_model_names_with_descriptions()
        for valid_model in valid_models:
            if valid_model['name'] == model_name:
                print(valid_model['description'])
                return

        print('Model name {0} unrecognized. Valid names: {1}'.format(model_name, ','.join(map(lambda m: m['name'], valid_models))))

    def help_transform(self, transform_name):
        """
        Prints the help for a transform name
        """
        valid_transforms = self._transform_factory.get_transform_names_with_descriptions()
        for valid_transform in valid_transforms:
            if valid_transform['name'] == transform_name:
                print(valid_transform['description'])
                return

        print('Transform name {0} unrecognized. Valid names: {1}'.format(transform_name, ','.join(map(lambda t: t['name'], valid_transforms))))

    def help_metric_processor(self, metric_name):
        """
        Prints the help for a metric processor name
        """
        valid_metrics_processors = self._metrics_processor_factory.get_metric_processor_name_with_information()
        for valid_metrics_processor in valid_metrics_processors:
            if valid_metrics_processor['name'] == metric_name:
                print(valid_metrics_processor['description'])
                return

        print('Metrics processor name {0} not recognized. Valid names: {1}'.format(metric_name, ','.join(map(lambda mp: mp['name'], valid_metrics_processors))))

    def help_heuristic_predictor(self):
        """
        Prints the help for the heuristic predictor
        """
        print(self._heuristic_predictor.description())


    def initialize_pipeline(self,
                            heuristics,
                            transform_specifications,
                            model_specifications,
                            metric_processor_specification,
                            output_directory,
                            best_model_metric,
                            train_split_ratio,
                            retrain_on_entire_dataset,
                            random_state):
        """
        Initializes an instance of the ML pipeline.

        :param heuristics: The heuristics to use for the heuristic predictor
        :param transform_specifications: A list of dictionaries specifying the transforms. The transforms will be applied in the order specified
        :param model_specifications: A list of specifications for the models being trained. 
        :param metric_processor_specification: The metric processor to use for evaluating the models.
        :param output_directory: The output directory for the log files, computed metrics, prediction results, and per-instance results
        :param best_model_metric: The model metric to use to determine the best model. Must be supported by the chosen metric processor
        :param train_test_split_ratio: The ratio of data to use for training. A float on the range [0,1]. For example, if 0.8, 80% of the data will be used for training.
        :param retrain_on_entire_dataset: If true, then the best model will be retrained on the entire dataset.
        :param random_state: An integer to seed the random number generator.
        """
        self._heuristics = heuristics
        self._transform_specifications = transform_specifications
        self._model_specifications = model_specifications
        self._metric_processor_specification = metric_processor_specification
        self._best_model_metric = best_model_metric
        self._output_directory = output_directory
        self._train_split_ratio = train_split_ratio
        self._retrain_on_entire_datset = retrain_on_entire_dataset
        self._random_state = random_state

        self._initialized = True
        self._trained = False

    def train_pipeline(self, data, labels):
        """
        Trains the pipeline on a set of training data.

        :param data: the data to use for training the pipeline.
        :param labels: the labels to use for the training data.
        """
        if not self._initialized:
            print('Cannot train on an uninitialized pipeline. Call initialize_pipeline() first.')
            return

        if (not os.path.exists(self._output_directory)):
            os.mkdir(self._output_directory)

        print('Creating modules...')
        transforms = self._transform_factory.create(self._transform_specifications)
        models = self._model_factory.create(self._model_specifications)
        self._metrics_processor = self._metrics_processor_factory.create(self._metric_processor_specification)
        self._heuristic_predictor.set_heuristics(self._heuristics)

        print('Done! Removing examples covered by heuristics...')
        idx = self._heuristic_predictor.predict(data).apply(lambda r: pd.isnull(r))
        data = data[idx]
        labels = labels[idx]
        print('Removed {0} points via heuristic prediction.'.format(idx.shape[0] - idx.sum()))

        print('Splitting transform_data...')
        train_data, test_data, train_labels, test_labels = model_selection.train_test_split(data, labels, test_size = 1.0 - self._train_split_ratio, random_state = self._random_state, stratify = labels)

        transform_train_data = train_data.copy(deep=True)
        transform_test_data = test_data.copy(deep=True)
        print('Done! Training transforms...')
        for transform in transforms:
            transform.train_transform(transform_train_data, train_labels)
            self._trained_transforms.append(transform)
            transform.summarize(os.path.join(self._output_directory, '{0}.summary.txt'.format(transform.__class__.get_name())))

            if transform.apply_to_training:
                transform_train_data = transform.apply_transform(transform_train_data, train_labels)
                train_data = train_data.ix[transform_train_data.index]
                train_labels = train_labels.ix[transform_train_data.index]
                train_data.reset_index(drop=True, inplace=True)
                train_labels.reset_index(drop=True, inplace=True)
                transform_train_data.reset_index(drop=True, inplace=True)
            if transform.apply_to_testing:
                transform_test_data = transform.apply_transform(transform_test_data, test_labels)
                test_data = test_data.ix[transform_test_data.index]
                test_labels = test_labels.ix[transform_test_data.index]
                transform_test_data.reset_index(drop=True, inplace=True)
                test_data.reset_index(drop=True, inplace=True)
                test_labels.reset_index(drop=True, inplace=True)
       
        print('Done! Training models...')
        for model in models:
            print('\tTraining model {0}...'.format(model.__class__.get_name()))
            model.train(transform_train_data, train_labels)
            self._trained_models.append(model)
            model.summarize(self._output_directory)

        #TODO: this should be generalized when regression models are introduced
        print('Done! Generating per-instance results and computing aggregate statistics...')
        metrics = []
        for model in self._trained_models:
            print('\tPredicting for model {0}...'.format(model.__class__.get_name()))
            train_predicted_labels = model.predict_proba(transform_train_data)
            test_predicted_labels = model.predict_proba(transform_test_data)

            #TODO: This can probably be more efficient
            train_inst = train_data.reset_index()
            for column in train_predicted_labels.columns:
                train_inst['{0}_probability'.format(column)] = list(train_predicted_labels[column])
            train_inst['predicted_labels'] = train_predicted_labels.apply(lambda r: np.argmax(r), axis=1)
            train_inst['actual_labels'] = list(train_labels)

            test_inst = test_data.reset_index()
            for column in test_predicted_labels.columns:
                test_inst['{0}_probability'.format(column)] = list(test_predicted_labels[column])
            test_inst['predicted_labels'] = test_predicted_labels.apply(lambda r: np.argmax(r), axis=1)
            test_inst['actual_labels'] = list(test_labels)

            train_inst.to_csv(os.path.join(self._output_directory, '{0}.train.inst.tsv'.format(model.__class__.get_name())), sep='\t')
            test_inst.to_csv(os.path.join(self._output_directory, '{0}.test.inst.tsv'.format(model.__class__.get_name())), sep='\t')

            test_aggreagate_metrics = self._metrics_processor.compute(test_predicted_labels, pd.get_dummies(test_labels))
            train_aggregate_metrics = self._metrics_processor.compute(train_predicted_labels, pd.get_dummies(train_labels))
            
            with open(os.path.join(self._output_directory, '{0}.aggregatemetrics.tsv'.format(model.__class__.get_name())), 'w') as f:
                f.write('Train:\n')
                for metric in train_aggregate_metrics:
                    f.write('{0}:\n{1}\n'.format(metric, train_aggregate_metrics[metric]))
                f.write('Test:\n')
                for metric in test_aggreagate_metrics:
                    f.write('{0}:\n{1}\n'.format(metric, test_aggreagate_metrics[metric]))

            metrics.append(test_aggreagate_metrics)

        print('\tGenerating aggregate statistics...')
        with open(os.path.join(self._output_directory, 'summary.tsv'), 'w') as f:
            f.write('\t')
            f.write('{0}\n'.format('\t'.join(map(lambda m: m.__class__.get_name(), self._trained_models))))
            for metric in metrics[0]:
                f.write('{0}\t'.format(metric))
                for model_metric in metrics:
                    f.write('{0}\t'.format(model_metric[metric]))
                f.write('\n')

        print('Done! Determining the best model...')
        best_model_index = 0
        for i in range(1, len(self._trained_models), 1):
            if (not self._metrics_processor.compare_metrics(metrics[best_model_index][self._best_model_metric], metrics[i][self._best_model_metric], self._best_model_metric)):
                best_model_index = i

        self._best_model = self._trained_models[best_model_index]
        print('Done!\nThe best model type is {0} with a {1} of {2} and hyperparameters {3}.'.format(
            self._best_model.__class__.get_name(),
            self._best_model_metric,
            metrics[best_model_index][self._best_model_metric],
            self._best_model.get_hyperparameters()))

        if (self._retrain_on_entire_datset):
            print('Retraining transforms and model on entire dataset...')
            for transform in self._trained_transforms:
                transform.train_transform(data, labels)
                if (transform.apply_to_training):
                    data = transform.apply_transform(data, labels)
                    labels = labels.ix[data.index]
                    data.reset_index(drop=True, inplace=True)
                    labels.reset_index(drop=True, inplace=True)

            self._best_model.train(data, labels)

        print('Results saved to {0}'.format(self._output_directory))
        self._trained = True
        print('The pipeline has been trained. You may now use predict_pipeline() to predict on unseen data.')

    def predict_pipeline(self, data):
        """
        Uses a previously trained pipeline to predict on unseen data.
        Returns the labels. 

        :param data: the data on which to predict.
        """
        if not self._initialized:
            print('Cannot predict on an uninitialized pipeline. Call initialize_pipeline() first.')
            return

        if not self._trained:
            print('Cannot predict with an untrained pipeline. Call train_pipeline() first.')
            return

        heuristic_labels = self._heuristic_predictor.predict(data)

        for transform in self._trained_transforms:
            if transform.apply_to_prediction:
                data = transform.apply_transform(data, None)

        labels = self._best_model.predict_proba(data)
        if (heuristic_labels is not None):
            i = 0
            for value in heuristic_labels:
                if not pd.isnull(value):
                    for j in range(0, labels.shape[1], 1):
                        if j == value:
                            labels.iloc[i][j] = 1.0
                        else:
                            labels.iloc[i][j] = 0
            i += 1

        return labels

    def visualize_pipeline(self):
        """
        Visualizes the configured pipeline.
        """
        if not self._initialized:
            print('Cannot visualize an uninitialized pipeline. Call initialize_pipeline() first.')
            return

        def die_with_error(config):
            print('***Malformed component with specification {0}***'.format(config))
            print('(expected field "name", which was not found)')
            print('Try re-initializing the pipeline with a valid configuration.')
            return

        if self._heuristics is not None and len(self._heuristics) > 0:
            print('Heuristics')
            print('\t|')
            print('\tV')
        for transform in self._transform_specifications:
            try:
                print(transform['name'])
                print('\t|')
                print('\tV')
            except:
                return die_with_error(transform)
        for model in self._model_specifications:
            try:
                print(model['name'])
            except:
                return die_with_error(model) 
        try:
            print('\t|\n\tV\n{0}'.format(self._metric_processor_specification['name']))
        except:
            return die_with_error(self._metric_processor_specification)

        print('')
        print('Output Directory: {0}'.format(self._output_directory))
        print('Train split ratio: {0}'.format(self._train_split_ratio))
        print('Retrain final model on entire dataset: {0}'.format(self._retrain_on_entire_datset))
        print('Random state: {0}'.format(self._random_state))