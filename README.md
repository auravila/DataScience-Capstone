
# Diabetes Classification Project

The following capstone project will showcase many of the key concepts learned through the Machine Learning Engineer with Microsoft Azure Udacity course.

The aim of the project is to provide a walkthrough via Jupyter notebooks and screenshots on the use of concepts such as datasets, Azureml model training, Azureml SDK, AutoML, Hyperdrive tunning and machine learning operations in order to build an end to end solution capable of predicting the probability of diabetes testing for a sample population and based on a finite set of measurements.

## Project Set Up and Installation
SDK version 1.20 version required 

## Dataset

### Overview

#### Summary:

The dataset was obtained from https://datahub.io/machine-learning/diabetes#readme but this dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 

The main goal is to predict diabetes on a patient based on general measurements, body hormones and proteins.

All patients from the sample are females of Pima Indian heritage (Group of Native Americans living in an area consisting of what is now central and southern Arizona, as well as north-western Mexico).

The variables use is explained below:

*Pregnancies: Number of times pregnant

*Glucose: Plasma glucose concentration 2 hours in an oral glucose tolerance test

*Blood Pressure: Diastolic blood pressure (mm Hg)

*Skin Thickness: Triceps skin fold thickness (mm)

*Insulin: 2-Hour serum insulin (mu U/ml)

*BMI: Body mass index (weight in kg/ (height in m) ^2)

*DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)

*Age: Age (years)

*Outcome: Class variable (0 or 1)

### Task
Use the list of measurements (attributes) from the tabular dataset to predict the class variable outcome value of 0 or 1 (tested negative / tested positive) this will indicate if a patient has a probability of developing diabetes.

### Access
The dataset is referenced multiple times during the project. 

####    Manually loaded via AzureML Interface
Initial upload of data, this is done manually using the AzureML Studio GUI

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/14-DatasetManual.png)

####    As training input for Hyperdrive train.py
Reference via python train.py using SDK

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/15-DatasetTrain.png)

####    Via SDK calls on Notebooks
References by the Jupyter notebooks and SDK

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/16-DatasetSDK.png)


## Automated ML
For the first execution of the automl model the settings chosen were to use a classification model with a timeout of 60 minutes and max concurrency 5 running jobs. The primary metric for evaluation was AUC_weighted chosen due to the imbalanced dataset and the potential of getting high error rate on the accuracy metric.   

One of the benefits of utilizing areas under the curve is that they remain the same whether the data is balanced or not.

### Results
Whilst AUC_weighted was considered as one of the best measurements the result of the execution of automl model indicated that AUC_micro provided the best run metric of them all. 

Parameters used for the automl run.

automl_settings = {
    "experiment_timeout_minutes": 60,
    "max_concurrent_iterations": 5,
    "primary_metric" : 'AUC_weighted'
}
automl_config = AutoMLConfig(compute_target=compute_target,
                             task = "classification",
                             training_data=dataset,
                             label_column_name="class",   
                             path = project_folder,
                             enable_early_stopping= True,
                             enable_onnx_compatible_models=True, ******
                             featurization= 'auto',
                             debug_log = "automl_errors.log",
                             **automl_settings
                            )

Automl RunDetails
![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/1-automlrundetails.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/25-AM.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/26-AM.png)

Automl Metrics
![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/1.1-automlmetrics.png)


#### AutoML Parameters

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/19-FittedModel.png)

Parameters used by AutoML execution:
                                                                                                    n_estimators=10,
                                                                                                    n_jobs=1,
                                                                                                    oob_score=False,
                                                                                                    random_state=None,
                                                                                                    verbose=0,
                                                                                                    warm_start=False))],
                                                                     verbose=False))],
                                               flatten_transform=None,
                                               weights=[0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.13333333333333333,
                                                        0.06666666666666667,
                                                        0.06666666666666667,
                                                        0.13333333333333333,
                                                        0.13333333333333333,
                                                        0.13333333333333333,
                                                        0.13333333333333333]))],
         verbose=False)
Y_transformer(['LabelEncoder', LabelEncoder()])


#### AutoML Best Model
VotingClassifier best algorithm used in the best model

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/20-AutoMLBestRun.png)



****** This feature was enabled later in order to convert and register the model as onnx

## Hyperparameter Tuning

### Part 1:
In order to select which is the best model for deployment, I decided to run a parameter sampling configuration hyperdrive over the AUC_weighted metric. Then used two parameters, one for max interactions and other for the regularization strength as input for a logistic regression in order to train the model.

Hyperdrive was configured to run with a max concurrent run of five and over one hundred total runs bounded by a bandit policy to check for thirty percent slack on every fifth execution. (30% was use based on the output of the first experimental run)

Details of the execution can be found listed in the notebook https://github.com/auravila/DataScience-Capstone/blob/main/hyperparameter_tuning-pimadiabetes.ipynb

### Part 2:

In order to validate the results of the parameter sampling best metric performance, I decided to run a grid sampling configuration hyperdrive over the AUC_weighted metric with the same two parameters, one for max interactions and other for the regularization strength as input for a logistic regression in order to train the model.

Hyperdrive was configured to run with a max concurrent run of five and over one hundred total runs bounded by a bandit policy to check for thirty percent slack on every fifth execution.

Details of the execution can be found listed in the notebook https://github.com/auravila/DataScience-Capstone/blob/main/hyperparameter_tuning-GSamp-pimadiabetes.ipynb

### Results

For the parameter sampling best run measure using the AUC_weighted metric a value of 0.71 was obtained

Best run id: HD_3ab2f6ca-c0b2-4906-b3cd-63d96f91146c_1

################################

 AUC_weighted: {'AUC_weighted': 0.7138211382113822}

################################

 Learning rate: ['--C', '1.5', '--max_iter', '30']

################################
{'_aml_system_ComputeTargetStatus': '{"AllocationState":"steady","PreparingNodeCount":0,"RunningNodeCount":0,"CurrentNodeCount":0}'}


#### To validate these results a hyperdrive using grid sampling was also executed which lead to the same results.


Best run id: HD_32d38e02-bbeb-4725-9506-304cb2847450_6

################################

 AUC_weighted: {'AUC_weighted': 0.7138211382113822}

################################

 Learning rate: ['--max_iter', '30', '--C', '1.5']

################################

{'_aml_system_ComputeTargetStatus': '{"AllocationState":"steady","PreparingNodeCount":0,"RunningNodeCount":5,"CurrentNodeCount":5}'}


It is also worth noting that some changes we added to the max iteration parameters since the runs were to small and a message of non-convergence was being raised
by the train.py logfiles. Increasing the iterations helped to provide the best run maximized value. Iterations were increase from 100 to 150 and to 300.

The results of these executions were far less from the results of automl votingclassified metrics therefore automl experiment was chosen as the best model for deployment.

### Parameter Sampling Run


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/5-BestRunHyperPSampv1.png)

### Parameter Sampling RunDetails Progress


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/12-RunDetailsParamSamp.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/24-HD1.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/24-HD2.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/24-HD3.png)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/24-HD4.png)




### Grid Sampling Run


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/6-BestRunHyperGSampv1.png)


### Grid Sampling RunDetails Progress


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/13-RunDetailsGridSamp.png)


#### Future Improvements for the project:

The following list of items could possibly improve the model outcome.
- Prevent over fitting by using more trainning date and the use of fewer features
- Prevent target leakage and simplity the model.
- Run a Bayesian sampling to confirm hyperdrive results
- Develop a more fit for purpose scoring function in the tran.py to train and score the model. (Adjust the function parameters)


## Model Deployment


## HyperDrive Model Registration (Parameter Sampling Run): "ParamSambestmodel.pkl:5" - Not the best model!

The execution results of Grid Sampling and Parameter sampling returned as similar outcome so for this project the parameter sampling
was chosen in order to register the model.
model = best_run.register_model(model_name='ParamSampbestmodel.pkl', model_path='.',
tags={'area': "diabetes", 'type': "Classification"},
description= "Best Model using Hyperdrive Parameter Sampling"
)

print ('Model Name',model.name)
print ('Model Version',model.version)
print ('Model Tags',model.tags) 
print ('Model Description', model.description)
model.properties


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/10-HyperDriveBestModel.png)

## AutomML Best Model Registration & Deployment

Steps take to deploy the model:
1.- Save mode to disk using joblib library.   ['./output/Capstone_automl_best.joblib']
2.- Register the model
3.- Download the best model environment and scoring files: ('conda_env_v_1_0_0.yml', 'scoring_file_v_1_0_0.py') for the inference config 
4.- Create a deployment config file and attach to an ACI deployment
5.- Deploy the Model

In order to query the endpoint

1. Prepared a set of data as inputs and converted them into json format
2. Raised a request to the endpoint scoring uri with the following parameters: resp = requests.post(scoring_uri, input_data, headers=headers)

#### Registration and Deployment Model - Automl: "automlpimadiabetes:3"

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/21-AutoMLBestmodel.png)


#### Deployment

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/22-AutoMLBestModelDep.png)


#### Active Endpoint

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/7.1-ModelEndpointActive.png)

#### Endpoint Call
![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/7.2-ModelEndpointRestCall.png)

#### Scoring File

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/18-ModelDeploymentScoringpy.png)

#### Environment File

bestmodel.download_file('outputs/conda_env_v_1_0_0.yml', 'myenv.yml')  

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/23-MyEnvyml.png)


## Screen Recording
https://youtu.be/8W8EHRs3HdI

## Standout Suggestions
Other Option attempted was to enable logging and ONNX conversion. Both of these were implemented succesfully

## Enable Logging

In order to enable logging, enable_app_insights to the web service were enabled to true. service.update(enable_app_insights=True)

![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/8-LoggingEnabled.png)

## ONNX Conversion

ONNX model conversion required to rerun the Automl experiment configuration setting to enable the compatibility mode to true.
enable_onnx_compatible_models=True, ******


![](https://github.com/auravila/DataScience-Capstone/blob/main/Screenshots/9-ONNXModelConversion.png)
