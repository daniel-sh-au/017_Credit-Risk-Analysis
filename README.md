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
The balanced accuracy scores and classification reports for each machine learning model are shown below:

### Naive Random Oversampling
Balanced Accuracy Score: 0.6573
![classification_report_random_oversample.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_random_oversample.png)

### SMOTE Oversampling
Balanced Accuracy Score: 0.6259
![classification_report_SMOTE_oversample.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_SMOTE_oversample.png)

### Cluster Centroids Undersampling
Balanced Accuracy Score: 0.5318
![classification_report_ClusterCentroids_undersample.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_ClusterCentroids_undersample.png)

### SMOTEENN Combination Sampling
Balanced Accuracy Score: 0.6585
![classification_report_SMOTEENN_combined.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_SMOTEENN_combined.png)

### Balanced Random Forest Classifier
Balanced Accuracy Score: 0.7885
![classification_report_BalancedRandomForestClassifier.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_BalancedRandomForestClassifier.png)

### Easy Ensemble AdaBoost Classifier
Balanced Accuracy Score: 0.9317
![classification_report_EasyEnsembleClassifier.png](https://github.com/daniel-sh-au/UofT_DataBC_Module17_Credit_Risk_Analysis/blob/main/Resources/classification_report_EasyEnsembleClassifier.png)

### Analysis
* The EasyEnsembleClassifier algorithm had the highest accuracy score (0.9317) and the Cluster Centroids undersampling algorithm had the lowest accuracy score (0.5318).

## Summary
Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. If you do not recommend any of the models, justify your reasoning.