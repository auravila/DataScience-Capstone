# Diabetes Classification Project

The following capstone project will showcase many of the key concepts learned through the Machine Learning Engineer with Microsoft Azure Udacity course.

The aim of the project is to provide a walkthrough via Jupyter notebooks and screenshots on the use of concepts such as datasets, AzureML model training, AzureML SDK, AutoML, Hyperdrive tunning and machine learning operations in order to build an end to end solution capable of predicting the probability of diabetes testing for a sample population and based on a finite set of measurements.

## Project Set Up and Installation

The list of steps below is required in order to execute the AutoML project which includes the best model and webservice deployment.


||Dependencies Setup and Installation|
|---|---|
|1|Create a compute Instance in AzureML|
|2|Download project files from current repository and Upload Jupyter Notebook to AzureML - File:DataScience-Capstone/automl-pimadiabetes.ipynb |
|3|Upgrade SDK version 1.20|
|4|Install AzureML-sklearn|
|5|Run Notebook|


## Dataset

### Overview

The dataset was obtained from https://datahub.io/machine-learning/diabetes#readme but this dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 

The main goal is to predict diabetes on a patient based on general measurements, body hormones and proteins.

All patients from the sample are females of Pima Indian heritage (Group of Native Americans living in an area consisting of what is now central and southern Arizona, as well as north-western Mexico).

The variables use is explained below:


|Variable|Description|Data type| Unique Value| Mean | Standard Dev|
|---|---|--|--|--|--|
|Pregnancies| Number of times pregnant|Numeric|17|3.8|3.4|
|Plasma glucose concentration|2 hours in an oral glucose tolerance test|Numeric|136|120.9|32|
|Blood Pressure|Diastolic blood pressure (mm Hg)|Numeric|47|69.1|19.4|
|Skin Thickness|Triceps skin fold thickness (mm)|Numeric|51|20.5|16.0|
|Insulin|2-Hour serum insulin (mu U/ml)|Numeric|186|79.8|115.2|
|BMI|Body mass index (weight in kg/ (height in m) ^2)|Numeric|248|32.0|7.9|
|DiabetesPedigreeFunction|Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)|Numeric|517|0.5|0.3|
|Age| Age (years)|Numeric |52|33.2|11.8|
|Outcome| Class variable (0 or 1)|Numeric |2|Values casted to text|N/A|
  

|Total Sample Records|768|
|-|-|


### Task
Use the list of measurements (attributes) from the tabular dataset to predict the class variable outcome value of 0 or 1 (tested negative / tested positive) this will indicate if a patient has a probability of developing diabetes.

### Access
The dataset is referenced multiple times during the project. 

|References|Screenshot|
|-|-|
|Tabular dataset manual upload using AzureML Studio| ![](/Screenshots/14-DatasetManual.png) |
|Via python train.py using SDK call| ![](/Screenshots/15-DatasetTrain.png)|
|Via Jupyter notebooks and SDK call| ![](/Screenshots/16-DatasetSDK.png)|

|Code Lines| 
|-|
|URL = 'https://datahub.io/machine-learning/diabetes/r/diabetes.csv'  
ds = TabularDatasetFactory.from_delimited_files(path=URL)|  



## Automated ML

In this project an initial AutoML experiment was built with the aid of ML Studio in order to obtain some preliminary metrics. And to obtain a view on the confusion matrix for the target variable. For this experiment a classification model was selected since it fits properly to predict small set of values.

* **Classification:** A model that generates output that will be limited to some finite set of discrete values.*

The confusion matrix indicated that the dataset is unbalanced and directed to choose an appropriate best metric indicator. 

| |Tested Negative|Tested Positive|
|-|-|-|
|Tested Negative|451|49|
|Tested Positive|116|152|

Based on this result and the nature of the dataset, it was decided to use area under the curve primary metrics to obtain the most accurate results, one of the benefits of utilizing areas under the curve is that they remain the same whether the data is balanced or not.

The first execution of the automl model the settings chosen were to use a classification model with a timeout of 60 minutes and max concurrency 5 running jobs. The primary metric for evaluation was AUC_weighted chosen due to the imbalanced dataset and the potential of getting high error rate on the accuracy metric.   

List of parameters used for the AutoML config

|Parameter|Value|Rationale|
|-|-|-|
|Experiment Timeout Minutes|60|Maximum time in minutes that each iteration can run for before it terminates|
|Max_Concurrent_iterations|5|Represents the maximum number of iterations, on a cluster the sum of the max_concurrent_iterations values for all experiments should be less than or equal to the maximum number of nodes.|
|Primary Metric|AUC_Weighted|The metric that Automated Machine Learning will optimize for model selection.|
|compute_target|compute_target|The Azure Machine Learning compute target to run the experiment|
|task|classification|This is the nature of the problem to solve|
|label_column_name|dataset|The name of the label column.|
|path|project_folder|path to the Azure Machine Learning project folder|
|enable_early_stopping|True|Default behaviour, early stopping window starts on the 21st iteration and looks for early_stopping_n_iters iterations|
|enable_onnx_compatible_models|True|enable or disable enforcing the ONNX-compatible models. **Enabled for ONNX deployment step**|
|featurization|auto|Automated machine learning featurization steps (feature normalization, handling missing data, converting text to numeric, etc.)|
|debug_log|automl_error.log|log file to write debug information to|


Parameters used for the automl run, code:

|Code Lines| 
|-|  
|automl_settings = {|
|    "experiment_timeout_minutes": 60,|
|    "max_concurrent_iterations": 5,|
|    "primary_metric" : 'AUC_weighted'|
|}|
|automl_config = AutoMLConfig(compute_target=compute_target,|
|                             task = "classification",|
|                             training_data=dataset,|
|                             label_column_name="class",|   
|                             path = project_folder,|
|                             enable_early_stopping= True,|
|                             enable_onnx_compatible_models=True, ******|
|                             featurization= 'auto',|
|                             debug_log = "automl_errors.log",|
|                             **automl_settings|
|                            )|

*Note*: For this particular exercise I also executed another automl experiment with the featurization switch set to off, results were the same however took some extra time to complete, this was somehow expected due to having a clean dataset. This can be found in the otherRuns folder, filename: AutoMLnoFeat.ipynb notebook


### Results
Whilst AUC_weighted was considered as one of the best measurements the result of the execution of automl model indicated that AUC_micro provided the best run metric of them all. 


|AutoML|Results & Screenshots|
|-|-|
|Automl RunDetails Results| ![](/Screenshots/1-automlrundetails.png) |
|RunDetails: Best Algorithm and metric score| ![](/Screenshots/25-AM.png)|
|RunDetails: Primary Metric| ![](/Screenshots/26-AM.png)|
|AutoMl Experiment Metrics| ![](/Screenshots/1.1-automlmetrics.png)|


#### AutoML Parameters

Parameters used by AutoML execution:

datatransformer  
{'enable_dnn': None,   
 'enable_feature_sweeping': None,   
 'feature_sweeping_config': None,   
 'feature_sweeping_timeout': None,   
 'featurization_config': None,   
 'force_text_dnn': None,   
 'is_cross_validation': None,   
 'is_onnx_compatible': None,   
 'logger': None,   
 'observer': None,   
 'task': None,   
 'working_dir': None}  

prefittedsoftvotingclassifier
{'estimators': ['1', '3', '21', '2', '32', '15', '26', '33', '4', '18', '10'],   
 'weights': [0.2857142857142857,   
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142,  
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142,   
             0.07142857142857142]}  

1 - maxabsscaler.  
{'copy': True}.  

1 - xgboostclassifier.  
{'base_score': 0.5,   
 'booster': 'gbtree',   
 'colsample_bylevel': 1,   
 'colsample_bynode': 1,   
 'colsample_bytree': 1,   
 'gamma': 0,   
 'learning_rate': 0.1,   
 'max_delta_step': 0,   
 'max_depth': 3,   
 'min_child_weight': 1,   
 'missing': nan,   
 'n_estimators': 100,   
 'n_jobs': 1,  
 'nthread': None,   
 'objective': 'binary:logistic',   
 'random_state': 0,   
 'reg_alpha': 0,   
 'reg_lambda': 1,   
 'scale_pos_weight': 1,  
 'seed': None,   
 'silent': None,  
 'subsample': 1,  
 'tree_method': 'auto',   
 'verbose': -10,   
 'verbosity': 0}.   

#### AutoML Best Model
VotingClassifier best algorithm used in the best model

|Parameter Sampling|Results & Screenshots|
|-|-|
|AutoML Best Run| ![](/Screenshots/20-AutoMLBestRun.png) |


## Hyperparameter Tuning

In order to validate the results of the previous AutoML experiment run and to select the best model, I decided to run two more experiments using hyperdrive. The first experiment (part 1) uses Parameter Sampling and the second one Grid Sampling (part 2)

The model function is defined in the train.py esklearn estimator script and it is based in a logistic regression:

model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

**Logistic Regression** *is a Machine Learning classification algorithm that is used to predict the probability of a variable. In logistic regression, the variable is a binary variable that contains data coded as 1 or 0. For this use case (tested_positive,tested_negative)*

where the parameters for the search space are represented as

|Parameter|Description|Parameter Expression|
|-|-|-|
|C|Regularization Strength|Choice|
|Max Iter|maximum number of iterations using max_iter parameter, where we typically increase when we have a very large amount of train data|Choice|

Choice: Function to generate a discrete set of values

The experiments were executed with the following combinations of parameters:  
ps = RandomParameterSampling ( {"--max_iter":choice(30,50,100),"--C":choice(0.5,1,1.5)} )    
ps = RandomParameterSampling ( {"--max_iter":choice(30,150,300),"--C":choice(0.5,1,1.5)} )     
ps = RandomParameterSampling ( {"--max_iter":choice(100,500,1000),"--C":choice(0.5,1,1.5,2.0,2.5)} )      

And a termination **bandit policy** defined as follow:
policy = BanditPolicy(slack_factor=0.30,evaluation_interval=1,delay_evaluation=5) 

Runs to be terminated where Bandit rule not met. i.e. If slack factor 30% on every fifth execution then terminate. (30% was use based on the output of the first experimental run)

### Parameter Sampling (part 1):

Experiment Configuration:

est = SKLearn (source_directory = "./",   
               entry_script = 'train.py',  
               compute_target = MYcompute_cluster)  

hyperdrive_config = HyperDriveConfig (   
    estimator=est,  
    hyperparameter_sampling=ps,  
    policy=policy,  
    primary_metric_name=primary_metric_name,  
    primary_metric_goal=primary_metric_goal,  
    max_total_runs=100,  
    max_concurrent_runs=5) 


Details of the execution can be found listed in the notebook https://github.com/auravila/DataScience-Capstone/blob/main/hyperparameter_tuning-pimadiabetes.ipynb

|Parameter Sampling|Results & Screenshots|
|-|-|
|Parameter Sampling Run| ![](/Screenshots/5-BestRunHyperPSampv1.png) |
|RunDetails| ![](/Screenshots/12-RunDetailsParamSamp.png)|
|RunDetails: Primary Metric| ![](/Screenshots/24-HD1.png)|
|AUC_weighted| ![](/Screenshots/24-HD2.png)|
|RunDetails: Parameters| ![](/Screenshots/24-HD3.png)|
|Best Run Id| ![](/Screenshots/24-HD4.png)|

### Grid Sampling (part 2):

In order to validate the results of the parameter sampling best metric performance, I decided to run a grid sampling configuration hyperdrive over the AUC_weighted metric with the same combination of the two parameters and same configuration

Details of the execution can be found listed in the notebook https://github.com/auravila/DataScience-Capstone/blob/main/hyperparameter_tuning-GSamp-pimadiabetes.ipynb

ps = GridParameterSampling ( {"--max_iter":choice(30,50,100),"--C":choice(0.5,1,1.5)} ).   
ps = GridParameterSampling ( {"--max_iter":choice(30,150,100),"--C":choice(0.5,1,1.5)} ).  
ps = GridParameterSampling ( {"--max_iter":choice(100,500,1000),"--C":choice(0.5,1,1.5,2.0,2.5)} )    

primary_metric_name = "AUC_weighted". 
primary_metric_goal = PrimaryMetricGoal.MAXIMIZE 

policy = BanditPolicy(slack_factor=0.30,evaluation_interval=1,delay_evaluation=5)  

est = SKLearn (source_directory = "./",    
               entry_script = 'train.py',   
               compute_target = MYcompute_cluster)  

hyperdrive_config = HyperDriveConfig (     
    estimator=est,    
    hyperparameter_sampling=ps,    
    policy=policy,    
    primary_metric_name=primary_metric_name,    
    primary_metric_goal=primary_metric_goal,    
    max_total_runs=100,   
    max_concurrent_runs=5).  


|Grid Sampling Run |Results & Screenshots|
|-|-|
|Best Run | ![](/Screenshots/6-BestRunHyperGSampv1.png) |
|Run Details| ![](/Screenshots/13-RunDetailsGridSamp.png)|
|Run Details| ![](/Screenshots/28-HDGMet.png)|
|Best Run Details| ![](/Screenshots/29-HDGDetails.png)|

### Result 

|Standout Suggestion |Parameter Sampling|Grid Sampling| 
|-|-|-|
|Best Run Id|HD_3ab2f6ca-c0b2-4906-b3cd-63d96f91146c_1|HD_32d38e02-bbeb-4725-9506-304cb2847450_6|
|AUC_weighted|{'AUC_weighted': 0.7138211382113822}|AUC_weighted: {'AUC_weighted': 0.7138211382113822}|
|Learning Rate| Learning rate: ['--C', '2.5', '--max_iter', '100']|Learning rate: ['--max_iter', '100', '--C', '2.5']|

Both hyperdrive runs provided a similar result

*The results of these executions were far less accurate from the results of automl votingclassified metrics therefore **automl experiment was chosen as the best model** for deployment.*


## Future Improvements for the project:

**Prevent Overfitting** which could occur when the model learns the noise from the training data and it negatively impacts the performance of the model on new data. So, in order to prevent this, we have a couple of options use less features and use more training data.

**Look for the better hyper-parameters**
Test more and different hyper-parameters, expand the grid search and look for better scores and metrics. This search will take time, but it can improve the results.


## AutomML Best Model Registration & Deployment

Steps take to deploy the model:
1.- Save mode to disk using joblib library.   ['./output/Capstone_automl_best.joblib']
2.- Register the model
3.- Download the best model environment and scoring files: ('conda_env_v_1_0_0.yml', 'scoring_file_v_1_0_0.py') for the inference config 
4.- Create a deployment config file and attach to an ACI deployment
5.- Deploy the Model

In order to query the endpoint

1. Obtain the Scoring URI
    service = Webservice(workspace=ws, name='pimadiabetesendpv3').   
    print(service.scoring_uri).   
    scoring_uri = service.scoring_uri.  
2. Prepared a set of data as inputs and converted them into json format
   2.1 In order to easily do this open the swagger link created when registering the endpoint, for this case an in bold:

|Endpoint Swagger URL|Data|
|-|-|
|http://159a4ac6-4925-4238-bd97-dbe0430d3a97.australiaeast.azurecontainer.io/swagger.json|**{"data": [{"preg": 0, "plas": 0, "pres": 0, "skin": 0, "insu": 0, "mass": 0.0, "pedi": 0.0, "age": 0}]}}**|


   
3. Raised a request to the endpoint scoring uri with the following parameters: resp = requests.post(scoring_uri, input_data, headers=headers)


|Model Deployment|Results & Screenshots|
|-|-|
|AutoML Registration and Deployment model: **"automlpimadiabetes:3"** | ![](/Screenshots/21-AutoMLBestmodel.png) |
|Model Deployment| ![](/Screenshots/22-AutoMLBestModelDep.png)|
|Active Endpoint| ![](/Screenshots/7.1-ModelEndpointActive.png)|
|Endpoint Call| ![](/Screenshots/7.2-ModelEndpointRestCall.png)|
|Model Scoring py| ![](/Screenshots/18-ModelDeploymentScoringpy.png)|
|Environment File bestmodel.download_file('outputs/conda_env_v_1_0_0.yml', 'myenv.yml')| ![](/Screenshots/23-MyEnvyml.png)|


#### HyperDrive Model Registration (Parameter Sampling Run): "ParamSambestmodel.pkl:5" 

The execution results of Grid Sampling and Parameter sampling returned a similar outcome so for this project the parameter sampling
was chosen in order to register the model.

model = best_run.register_model(model_name='ParamSampbestmodel.pkl', model_path='.',  
tags={'area': "diabetes", 'type': "Classification"},   
description= "Best Model using Hyperdrive Parameter Sampling"  
)   
 
|Model Registration|Results & Screenshots|
|-|-|
|Hyperdrive Model Registration : **"ParamSambestmodel.pkl:5** | ![](/Screenshots/10-HyperDriveBestModel.png) |



## Screen Recording
https://youtu.be/RnX0XOyPqrU


## Standout Suggestions
Options attempted and successfully implemented were to enable logging and ONNX model conversion. 


|Standout Suggestion |Results & Screenshots|
|-|-|
|Enable Logging | ![](/Screenshots/8-LoggingEnabled.png) |
|ONNX Conversion| ![](/Screenshots/9-ONNXModelConversion.png)|

#### Enable Logging

For a web service to have the logging enable is required to execute the code below:

|Code Lines| 
|-|  
|service.update(enable_app_insights=True)|


#### ONNX Conversion

For a model to be registered and converted to ONNX, the AutoML configuration setting of the experiment needs to be enabled as follow.
enable_onnx_compatible_models=True

Then run the following code snippet

|Code Lines| 
|-|  
|from AzureML.automl.runtime.onnx_convert import OnnxConverter|
||
|best_run, onnx_mdl = remote_run.get_output(return_onnx_model=True)|
||
|onnx_fl_path = "./best_model.onnx"|
|OnnxConverter.save_onnx_model(onnx_mdl, onnx_fl_path)|
||
|model = Model.register(model_path = "./",|
|                       model_name = "best_model.onnx",|
|                       tags = {"onnx": "MyFirstonnx"},|
|                       description = "pimadiabetesonnx",|
|                       workspace = ws)|
