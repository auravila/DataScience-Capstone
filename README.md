
# Diabetes Classification Project

The following project aims to identify the probablity of testing diabetes based on a finite set of measurements.

## Project Set Up and Installation
*OPTIONAL:* If your project has any special installation steps, this is where you should put it. To turn this project into a professional portfolio project, you are encouraged to explain how to set up this project in AzureML.

## Dataset

### Overview

Summary:
The dataset was obtained from https://datahub.io/machine-learning/diabetes#readme but this dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. 

The main goal is to predict diabetes on a patient based on general measurements, body hormones and proteins.

All patients from the sample are females of Pima Indian heritage (Group of Native Americans living in an area consisting of what is now central and southern Arizona, as well as northwestern Mexico).

The variables uses are explaine below:

*Pregnancies: Number of times pregnant

*Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test

*BloodPressure: Diastolic blood pressure (mm Hg)

*SkinThickness: Triceps skin fold thickness (mm)

*Insulin: 2-Hour serum insulin (mu U/ml)

*BMI: Body mass index (weight in kg/(height in m)^2)

*DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)

*Age: Age (years)

*Outcome: Class variable (0 or 1)

### Task
Use the list of measurements (attributes) from the tabular dataset to predict the class variable outcome value of 0 or 1 (tested negative / tested positive)

### Access
The dataset is referenced twice, downloaded and uploaded to a datarouce storage of machine learning studio and also is referenced via the train.py program.

## Automated ML
For the first execution of automl model the settings chosen were to use a classification model with a timeout of 60 minutes and max concurrency 5 running jobs. The primary metric for evaluation was AUC_weighted chosen due to the imabalanced dataset and the potential of getting high error rate on the accuracy metric.   

One of the benefits of utilizing areas under the curve is that they remain the same whether the data is balanced or not.

### Results
Whislt AUC_weighted was considered as one of the best measurements the result of the execution of automl model indicated that AUC_micro provided the best run metric of them all. 



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
