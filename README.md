Executive summary:
The goal of the project is to be able to predict the sub-type of Ovarian Cancer based on the dimensions of the
histopathology images.

Rationale:
Ovarian Cancer is one of the leading causes of deaths among women.
The American Cancer Society estimates for ovarian cancer in the United States for 2024 are:

About 19,680 women will receive a new diagnosis of ovarian cancer.
About 12,740 women will die from ovarian cancer.

A woman's risk of getting ovarian cancer during her lifetime is about 1 in 87. Her lifetime chance of dying from ovarian cancer is about 1 in 130.

There is a high chance of saving a life of the patient with the accurate and early identification of Ovarian Cancer sub-type.

Findings:
Logistic Regression is found to be the best model for predicting the subtype of ovarian cancer, since the Logistic Regression produced best results with both training and test data. The other models that are used are K-Nearest Neighbor, Decision Tree, Support Vector Machines, AdaBoost and RandomForest.

Research Question:
The question this project aims to answer is what is the best classification model for predicting the sub-type of Ovarian Cancer, as well as what the best hyperparameters for this task are.

Data Sources:
The data is available at https://www.kaggle.com/competitions/UBC-OCEAN/overview

Model evaluation and results:
ROC AUC Accuracy score is used to determine the best model.
An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters: True Positive Rate. False Positive Rate.

AUC stands for Area Under the Curve. AUC measures the entire two-dimensional area underneath the entire ROC curve

Model performance is visualized using confusion matrices, which indicate the counts of predicted value and true value for each of the 5 sub-types of Ovarian Cancer.

Model Name - Train Time - Train Accuracy - Test Accuracy

Logistic Regression - Low - 41.86% - 57.13%
K Nearest Neighbor - Low - 42.32% - 48.51%
Decision Tree - Medium - 36.51% - 48.39%
Support Vector Machines - High - 41.86% - 42.88%
AdaBoost - Low - 41.40% - 54.99%
RandomForest - Low - 36.04% - 53.22%

Future research and development:
The data set has 538 entries. This is one of the key reasons why train and test accuracy are on lower side. More data is needed to build better models and improve the prediction accuracy.

Convolutional Neural Networks can be used to build models based using the actual histopathology images to predict he Ovarian Cancer sub-type.

Technical Implementation Steps:

Step 1 : Understanding the data

Downloaded the dataset on my machine and reviewed it by applying various filter.
https://www.kaggle.com/competitions/UBC-OCEAN/overview

Reviewed the documentation provided.
Step 2: Reading the data

Used pd.read_csv to load the document locally into Jupyter notebook.
Used info() function to understand the column names, data types and number of records.
Checked if there are any null values in the dataset.

Step 3: Understanding the features
The dataset has numerical feature columns.

Step 4: Understanding the task
The goal of this exercise is to determine if it is possible to predict the sub-type of Ovarian Cancer accurately.  

Step 5: Engineering Features
After reviewing the dataset, I concluded that image_width and image_height are most the key features and column 'label' is the target column.
It is a multi-class classification as 'label' field has multiple classes.  

Step 6: Train/Test Split
Used train_test_split to split the dataset into 80% for training and 20% for test datasets.

Step 7: Baseline Model
I created baseline models with Logistic Regression, KNN, Decision Tree, Support Vector Machines, AdaBoost and Random Forest classifiers and calculated accuracy score for each model.

Step 8: Data Modelling
I used Logistic Regression, K Nearest Neighbor, Decision Tree, Support Vectore Machines, AdaBoost and Random Forest classifiers.

Logistic Regression:

Created a Pipeline with StandardScaler and LogisticRegression
Used GridSearchCV to get the best hyperparameters namely classifier__solver, classifier__penalty and classifier__C
Used the best Logistic Regression model found to predict the target for test data
Visualized the predictions using Confusion Matrix
Calculated ROC AUC Score for Logistic Regression Model. It came out to 57.14%

K Nearest Neighbor:

Created a StandardScaler.
Fitted and transformed the training data using StandardScaler.
Used Cross validation to determine the best value of k
Plotted Number of Neighbors v/s Cross Validation Accuracy
Best value of k = 51 based on the above plot
Used the optimal KNN model found to predict the target for test data
Visualized the predictions using Confusion Matrix
Calculated ROC AUC Score for KNN Model. It came out to 48.51%

Decision Tree:

Created a Pipeline with StandardScaler() and DecisionTreeClassifier.
Identified the optimal model with criterion and max depth hyperparameters using GridSearchCV.
Used the optimal DecisionTreeClassifier model found to predict the target for test data
Visualized Decision Tree using graphviz
Visualized the predictions using Confusion Matrix
Calculated ROC AUC Score for DecisionTreeClassifier Model. It came out to 48.39%

Support Vector Machines:

Determined the optimal kernel for creating SVC model using cross validation score.
Next, tried to determine the optimal hyperparameters for SVC model. But the code execution took very long time and
it did not return any outcome either.The commented code is available in the jupyter notebook.
Since it was not possible to determine the optimal SVC model with best hyperparameters, I decided to create SVC model with kernel = 'poly' to determine the accuracy for predictions for test data.
Calculated ROC AUC Score for DecisionTreeClassifier Model. It came out to 42.88%

AdaBoost:
Created a Pipeline with StandardScaler() and AdaBoostClassifier.
Identified the optimal model with n_estimators, learning_rate and algorithm hyperparameters using GridSearchCV.
Used the optimal AdaBoostClassifier model found to predict the target for test data
Visualized the predictions using Confusion Matrix
Calculated ROC AUC Score for DecisionTreeClassifier Model. It came out to 54.99%

RandomForest:
Created a Pipeline with StandardScaler() and RandomForestClassifier.
Identified the optimal model with n_estimators, learning_rate and max_depth hyperparameters using GridSearchCV.
Used the optimal RandomForestClassifier model found to predict the target for test data
Visualized the predictions using Confusion Matrix
Calculated ROC AUC Score for DecisionTreeClassifier Model. It came out to 53.22%

Step 8 - Model Comparision

Model Name - Train Time - Train Accuracy - Test Accuracy

Logistic Regression - Low - 41.86% - 57.13%
K Nearest Neighbor - Low - 42.32% - 48.51%
Decision Tree - Medium - 36.51% - 48.39%
Support Vector Machines - High - 41.86% - 42.88%
AdaBoost - Low - 41.40% - 54.99%
RandomForest - Low - 36.04% - 53.22%

Conclusion: Logistic Regression is found to be the best model for predicting the subtype of ovarian cancer, since the Logicstic Regression produced best results with both training and test data.
