# Module 17: Supervised Machine Learning and Credit Risk

## Overview of the Analysis

### Purpose
The purpose of this analysis was to evaluate the performance of machine learning models for predicting credit risk. The data was oversampled using the **RandomOverSampler** and **SMOTE**, as well as undersampled using the **ClusterCentroids** algorithms. Then, a combination approach of over- and under-sampling using the **SMOTEENN** algorithm was applied to the dataset. A **LogisticRegression** classifier was then applied for each of the previous sampling algorithms. Finally, two machine learning models, **BalancedRandomForestClassifier** and **EasyEnsembleClassifier** were applied to predict credit risk. 

### Resources
* Jupyter Notebook, Python 3.7.13
* Python Libraries: scikit-learn, imbalanced-learn, pandas, numpy, collections
* Data Sources: LoanStats_2019Q1.csv
* Challenge Code: [credit_risk_resampling.ipynb](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/credit_risk_resampling.ipynb), [credit_risk_ensemble.ipynb](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/credit_risk_ensemble.ipynb)

## Results
Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all six machine learning models. Use screenshots of your outputs to support your results.



## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.