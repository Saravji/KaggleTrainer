# Kaggle Training Pipeline

### Overview

The Kaggle Training pipeline is a toolset for training Machine Learning models. Although its primary use case at the moment is for [Kaggle competitions](https://www.kaggle.com/), the pipeline is generic enough to be used in a wide variety of applications. The pipeline includes the following capabilities:

* Provide common preprocessing transformations to the data such as column normalization, dimensionality reduction, SMOTE, and downsampling 
* Provide wrappers around multiple popular machine learning algorithms, providing a common interface
* Supports training multiple different models and saving summary statistics, allowing for comparison across different model types
* Supports saving debugging information such as tree visualizations and per-instance results, improving iteration speed

### Using the model
To use the pipeline, you will need to have the python files in this repo in the same directory as your top-level script. Then, import Trainer.

To see the components that are available for use, call trainer.help_components()
To get help on an individual component, call trainer.help_transform(), trainer.help_model(), trainer.help_metric_processor(), or trainer.help_heuristic_predictor().

You will set up your pipeline by calling trainer.initialize_pipeline(). Then, to train, call trainer.train().

Once trained, the pipeline can be saved and loaded using dill. To predict on unseen datapoints, call trainer.predict()

For a full example, see "Test.py" in the root 

### Contributing to the project
Adding transforms and new model types are pretty easy. 

To add a new transform:
* Add a new python class inside the Transform/ folder
* Define your transform class, and inherit from 'Transform'
* Implement the abstract methods inside Transform
* Import your new transform inside TransformFactory.py
* If all goes well, you should see your new transform when you call trainer.help_components()

To add a new model:
* For classification, add a new python file inside of the Model/Classification folder. 
* Define your model class. If the model is not from scikit-learn, inherit from Model.py. Otherwise, inherit from SklearnModel.py. 
* Implement the unimplemented abstract members
* Import your model into ModelFactory.py
* If all goes well, you should see your new model when you call trainer.help_components()

Changing the pipeline, metrics processors, and heuristics generators shouldn't have to be done much. The only major change for those components will be to support regression. 