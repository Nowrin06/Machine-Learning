# IA651-MACHINE LEARNING PROJECT 

## AUTISM PREDICTION IN ADULTS

#### Team Members: Hemamalini Venkatesh Vanetha & Shifat Nowrin

### Table of Contents

* [Introduction](#Introduction)
* [Abstract](#Abstract)
* [Data Overview](#Data-Overview)
* [Exploratory Data Analysis](#Exploratory-Data-Analysis)
* [Encoding Categorical Variables](#Encoding-Categorical-Variables)
* [Train Test Split](#Train-Test-Split)
* [Scaling](#Scaling)
* [Principal Component Analysis](#Principal-Component-Analysis)
* [SMOTE](#SMOTE)
* [Logistic Regression](#Logistic-Regression)
* [Decision Tree](#Decision-Tree)
* [Random Forest](#Random-Forest)
* [Support Vector Classifier](#Support-Vector-Classifier)
* [Conclusion](#Conclusion)
* [Future Scope](#Future-Scope)



# Introduction

### Autism
Autism, or autism spectrum disorder (ASD), refers to a broad range of conditions characterized by challenges with social skills, repetitive behaviors, speech and nonverbal communication.

### Causes and Challenges
It is mostly influenced by a combination of genetic and environmental factors. Because autism is a spectrum disorder, each person with autism has a distinct set of strengths and challenges. The ways in which people with autism learn, think and problem-solve can range from highly skilled to severely challenged.
Research has made clear that high quality early intervention can improve learning, communication and social skills, as well as underlying brain development. Yet the diagnostic process can take several years.

### The Role of Machine Learning
This dataset is composed of survey results for more than 704 people who filled an app form. There are labels portraying whether the person received a diagnosis of autism, allowing machine learning models to predict the likelihood of having autism, therefore allowing healthcare professionals prioritize their resources.

# Abstract

Predict the likelihood of a person having autism using survey and demographic variables.

Explore Autism across Gender, Age, and other variables




# Data Overview

This dataset has been taken from the Kaggle Website

Dataset Link : [Autism Screening on Adults](https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults)

This dataset originally has 704 rows and 21 columns.

The dataset has the following parameters:

| Variables       | Description                                                                                                         |
|-----------------|---------------------------------------------------------------------------------------------------------------------|
| A1 - A10 score  | These columns represent the individual's responses to  10 different questions on the autism screening application.  |
| age             | The Age of the individual                                                                                           |
| gender          | The gender of the individual                                                                                        |
| ethnicity       | The ethnic backgroud of the individual                                                                              |
| jaundice        | Indicates whether the individual has jaundice                                                                       |
| autism          | Indicates whether the person were already diagnosed with  autism or not                                             |
| country_of_res  | Country of residence of the individual                                                                              |
| used_app_before | Indicates whether the individual has used this screening app before                                                 |
| result          | This is the sum of scores(A1 score - A10 score)                                                                     |
| age_desc        | Description of the age group                                                                                        |
| relation        | The person who completed the screening on the app on  behalf of the individual being assessed                       |
| Class/ASD       | The class label indicates whether the individual is likely to  autism                                               |

Among these 21 variables, 20 are independent variables and 'Class/ASD' will be dependent variable.

Columns A1 Score - A10 score is the response score of the questions and these questions were developed by the National Institute for Health Research.

The 10 questions given in this questionnaire is:

Q1: I often notice small sounds when others do not

Q2: I usually concentrate more on the whole picture, rather than the small details

Q3: I find it easy to do more than one thing at once.

Q4: If there is an interruption, I can switch back to what I was doing very quickly

Q5: I find it easy to 'read between the lines' when someone is talking to me

Q6: I know how to tell if someone is listening to me is getting bored

Q7: When I'm reading a story I find it difficult to work out the characters's intentions

Q8: I like to collect information about categories of things (eg.types of car, types of bird, types of train etc)

Q9: I find it easy to work out what someone is thinking or feeing just by looking at their face

Q10: I find it difficult to work out people's intentions

All these questions can be answered Yes or No, which is converted to 0 as No and 1 as Yes.

Below is a sample of this dataset :

| A1_Score | A2_Score | A3_Score | A4_Score | A5_Score | A6_Score | A7_Score | A8_Score | A9_Score | A10_Score | age | gender | ethnicity      | jaundice | autism | contry_of_res | used_app_before | result | age_desc      | relation | Class/ASD |
|----------|----------|----------|----------|----------|----------|----------|----------|----------|-----------|-----|--------|----------------|----------|--------|---------------|-----------------|--------|---------------|----------|-----------|
| 1        | 1        | 1        | 1        | 0        | 0        | 1        | 1        | 0        | 0         | 26  | f      | White-European | no       | no     | United States | no              | 6      | 18 and more   | Self     | NO        |
| 1        | 1        | 0        | 1        | 0        | 0        | 0        | 1        | 0        | 1         | 24  | m      | Latino         | no       | yes    | Brazil         | no              | 5      | 18 and more   | Self     | NO        |
| 1        | 1        | 0        | 1        | 1        | 0        | 1        | 1        | 1        | 1         | 27  | m      | Latino         | yes      | yes    | Spain          | no              | 8      | 18 and more   | Parent   | YES       |
| 1        | 1        | 0        | 1        | 0        | 0        | 1        | 1        | 0        | 1         | 35  | f      | White-European | no       | yes    | United States | no              | 6      | 18 and more   | Self     | NO        |
| 1        | 0        | 0        | 0        | 0        | 0        | 0        | 1        | 0        | 0         | 40  | f      | ?              | no       | no     | Egypt          | no              | 2      | 18 and more   | ?        | NO        |


# Exploratory Data Analysis

Before implementing any of the models for the dataset, we would like to explore the data, their correlations, distribution among categorical and numerical variables , histograms to have more understanding of the dataset and also to see if there are any interesting parts.

First of all, using python we have obtained the summary of the numerical variables using describe() method:

|       | **A1_Score** | **A2_Score** | **A3_Score** | **A4_Score** | **A5_Score** | **A6_Score** | **A7_Score** | **A8_Score** | **A9_Score** | **A10_Score** | **age**    | **result** |
|-----------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|--------------|---------------|------------|-------------|
| **count** | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000   | 704.000000    | 702.000000 | 704.000000  |
| **mean**  | 0.721591     | 0.453125     | 0.457386     | 0.495739     | 0.498580     | 0.284091     | 0.417614     | 0.649148     | 0.323864     | 0.573864      | 29.698006  | 4.875000    |
| **std**   | 0.448535     | 0.498152     | 0.498535     | 0.500337     | 0.500353     | 0.451301     | 0.493516     | 0.477576     | 0.468281     | 0.494866      | 16.507465  | 2.501493    |
| **min**   | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000      | 17.000000  | 0.000000    |
| **25%**   | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000      | 21.000000  | 3.000000    |
| **50%**   | 1.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 0.000000     | 1.000000     | 0.000000     | 1.000000      | 27.000000  | 4.000000    |
| **75%**   | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000      | 35.000000  | 7.000000    |
| **max**   | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000     | 1.000000      | 383.000000 | 10.000000   |

### Dataset Cleaning

- We tried to check if there are any null values in the dataset.There were 2 null values in the column age, so we dropped it.

- Then, we checked for any outliers and found that age has a outlier and removed it from the dataset.

- Also, removed the duplicate values. Replaced some values with proper notation like '?' with 'others'. Dropped the age_decs column because it has only one value which is 18 and more.

- After all this process, we finally have 696 rows and 20 columns.

### Visualization of Categorical Variables

![Distribution of Gender](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20Gender.png)

![Distribution of Ethnicity](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Ethnicity.png) 

![Distribution of Jaundice](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20jaundice.png)

![Distribution of Autism](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Autism.png)

![Distribution of Used_app_before](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20app%20usedpng.png)

![Distribution of Relation](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20relation.png)

![Distribution of Class/ASD](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20ClassASD.png)

![Distribution of Country of Residence](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20country%20of%20residence.png)

From the plots, we can see the distributions of different categorical variables.

Now, let's analyze the dependent variable plot:

As we can see that there around 200 people with Autism and remaining people doesn't have autism. This is unbalanced.
We will handle this in the later part.

### Visualization of Numerical Variables

![Distribution of Age](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribution%20of%20Age.png)

![Distribution of Result](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Distribtuion%20of%20Result.png)

In this result, if the score is above 6, the individual is mostly predicted to have autism.

![Distribution of A1-A10 score](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Count%20plot%20of%20scores.png)

From this count plot, we can see that for different question the scores are distributed and used to predict the autism of the individual.

![Categorizing Age groups](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Age%20group%20plot.png)

This plot shows age group 21-40 has the highest autism count.

After visualizing all the columns,we are dropping result and age_category column.

```python
df1.drop(['result','age_category'],axis = 1, inplace = True)
```

Here, we drop the result column because this directly predicts the y variable which is not very good to use in the model because this will have direct correlation to the y variable.
After this, we'll have 19 columns and 696 rows.

### Correlation Matrix

![Heatmap of columns](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Correlation%20matrix.png)

From the correlation matrix, A9 score,A6 score and A5 score has highly correlated with the dependent variable.

# Encoding Categorical Variables

```python
label_encoder = LabelEncoder()
df1['gender'] = label_encoder.fit_transform(df1['gender'])
df1['jaundice'] = label_encoder.fit_transform(df1['jaundice'])
df1['autism'] = label_encoder.fit_transform(df1['autism'])
df1['used_app_before'] = label_encoder.fit_transform(df1['used_app_before'])
df1['Class/ASD'] = label_encoder.fit_transform(df1['Class/ASD'])
```
We have label encoded the above five variables because they have only 2 values.

```python
df1 = pd.get_dummies(df1, columns=['ethnicity', 'relation','contry_of_res'])
```
One-hot encoding these three variables because they have 3 or more categories.

# Train Test Split

```python
X = df1.drop('Class/ASD', axis = 1)
y = df1['Class/ASD']

X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

Separating the independent variables and assiging it to X and y.

Splitting the data: 80% of the data will be used for training and 20% of the data will be used for testing the data.

In this way, the model can have best understanding of the predicting variable.

# Scaling

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

StandardScaler - A Standard Scaler is a technique used in data preprocessing, particularly in machine learning, to standardize features by removing the mean and scaling to unit variance. This means that it transforms the data in such a way that the mean of each feature becomes 0 and the standard deviation becomes 1.

Here, we scaled both training and testing data.


# Principal Component Analysis

Principal Component Analysis is done to is a dimensionality reduction technique commonly used in data analysis and machine learning.

We will perform PCA and reduce the dimensionality in our data.

We are considering the first 20 Principal Components which explains 38% of the variance.

![Scree Plot](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/PCA%20Scree%20plot.png)

# SMOTE

```python
from imblearn.pipeline import Pipeline

smote = SMOTE(random_state=42)
```

SMOTE (Synthetic Minority Over-sampling Technique) is used to address class imbalance in datasets, where one class is underrepresented compared to others. This imbalance can lead to biased models that perform poorly on the minority class.

In our dataset, one of the class is underrepresented so we are using the smote to balance the data.

# Logistic Regression

Logistic Regression model is implemented for the data using maximum iteration of 1000.

We first train the model using scaled training data. Then, we predict the training data and also testing data to see how well the model performs.

| **Values**   | **Train** | **Test** |
|--------------|-----------|----------|
| **Accuracy** | 97.6%     | 95.7%    |
| **F1 score** | 97%       | 94.6%    |

### Grid Search

#### Hyperparameter :
```python
lg_param_grid = {
    'lg__C': [0.1, 10, 100, 200, 500],
    'lg__penalty': ['l1', 'l2']
```
Best Parameters: {'lg__C': 10, 'lg__penalty': 'l2'}

Best Score: 0.96

Performing grid search for different hyperparameter helped in selecting the best parameters and best score for the model.

Best score for the Logistic Regression is 96% which is pretty good score.

Now, training and testing the model using these paramaters will give us the final best accuracy for the model.

- Logistic Regression Best Training Accuracy: 0.98

- Logistic Regression Best Testing Accuracy: 0.95

- F1 score:0.91

- ROC AUC score:0.98

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.99      | 0.94   | 0.97     | 103     |
| 1            | 0.86      | 0.97   | 0.91     | 37      |
| accuracy     |           |        | 0.95     | 140     |
| macro avg    | 0.92      | 0.96   | 0.94     | 140     |
| weighted avg | 0.95      | 0.95   | 0.95     | 140     |

- The accuracy after using the best hyperparameters doesn't have much difference in the result. 

- The final Testing Accuracy for the model is 95%.

- The Accuracy is pretty good fo the Logistic Regression.

- F1 Score: 0.911 - The F1 score is the harmonic mean of precision and recall It tells you how the model trades off bias and variance. A score of 0.911 gives a slightly more balanced precision and recall model which is good.

- ROC AUC Score: 0.984 - The ROC AUC score evaluates the model's capacity to predict classes. A score of 0.984 very close to 1 suggests that the model has a fantastic discriminative capability between both classes positive and negative classs is fine tuned.

Confusion Matrix:

![Confusion Matrix](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Logistic%20regression%20Confusion%20Matrix.png)

- True Negatives (TN): 97
The model correctly predicted 97 instances as negative (class 0).

- False Positives (FP): 6
The model incorrectly predicted 6 instances as positive when they were actually negative.

- False Negatives (FN): 1
The model incorrectly predicted 1 instance as negative when it was actually positive.

- True Positives (TP): 36
The model correctly predicted 36 instances as positive (class 1).

The model did a good job identifying most cases of negative and positive instances, making only a few errors. Namely, it had 6 false positive errors and just one false negative. This means that model is very good at predicting each classes and with only few misclassifications.

# Decision Tree

Now, let's see how the decision tree model predicts the Autism.

We first trained the model and predicted accuracy and F1 score for both training and testing.

The values are:

| **Values**   | **Train** | **Test** |
|--------------|-----------|----------|
| **Accuracy** | 100%      | 92.1%    |
| **F1 score** | 100%      | 89.8%    |

From the above values, we can clearly say that the model has done 100% in the trained data and 92% in that testing data which means it doesn't perform as good as training data.

### Grid Search

#### Hyperparameter :
```python
dt_param_grid = {
        'dt__max_depth': [10, 20, 30],
        'dt__min_samples_split': [2, 5, 10],
        'dt__min_samples_leaf': [1, 2, 4]
}
```
Best Parameters: {'dt__max_depth': 30, 'dt__min_samples_leaf': 4, 'dt__min_samples_split': 10}

Best Score: 0.93

Decision Tree with max_depth of 3:

![Decision Tree](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Decision%20Tree%20image.png)

Best score for Decision Tree is 93% which is good but lesser than the Logistic Regression Best score.

Using best parameters, trained and tested the model which gave the final training and testing Accuracy, F1 score, ROC AUC score.

- Decision Tree Best Training Accuracy: 0.98

- Decision Tree Best Testing Accuracy: 0.92

- F1 score:0.85

- ROC AUC score:0.94

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.95      | 0.94   | 0.95     | 103     |
| 1            | 0.84      | 0.86   | 0.85     | 37      |
| accuracy     |           |        | 0.92     | 140     |
| macro avg    | 0.90      | 0.90   | 0.90     | 140     |
| weighted avg | 0.92      | 0.92   | 0.92     | 140     |

- The Final Training Accuracy for the Decision Tree is 98% and final testing accuracy for testing accuracy is 92% which is good model but the accuracy is lesser than the Logistic Regression.

- F1 Score - 0.853: Our model achieve an AUC of 0.853 which points that we have a balance precision and recall in the positive class (1).

- ROC AUC Score- 0.942 : A score of 0.942 indicates a very good model discrimination between the two categories, where higher scores mean better performance

Confusion Matrix:

![Confusion Matrix](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Decision%20Tree%20Confusion%20Matrix.png)

- True Negatives (TN): 97
The model correctly predicted 97 instances as negative (class 0). These are the true negative cases.

- False Positives (FP): 6
The model incorrectly predicted 6 instances as positive when they were actually negative. These are false positive cases.

- False Negatives (FN): 5
The model incorrectly predicted 5 instances as negative when they were actually positive. These are false negative cases.

- True Positives (TP): 32
The model correctly predicted 32 instances as positive (class 1). These are the true positive cases.

The Decision Tree model also has good accuracy overall but not better than Logistic Regression.


# Random Forest

Now, let's see how the Random Forest model predicts the Autism.

| Values   | Train | Test  |
|----------|-------|-------|
| Accuracy | 100%  | 90%   |
| F1 score | 100%  | 87.3% |

Here, the training accuracy is same as decision tree because we know collection of decision tree is random forest. But the testing accuracy is 90% which even lesser than the decision tree.

### Grid Search

#### Hyperparameter :
```python
rf_param_grid = {
        'rf__n_estimators': [10, 20, 30],
        'rf__max_depth' : [10, 20, 30]
}
```
Best Parameters: {'rf__max_depth': 20, 'rf__n_estimators': 20}

Best Score: 0.93

The max depth for random forest is 20 and n_estimators is 20. At this values, the model will perform better than other than values. Now let's try to use these values, train and test the model.

- Random Forest Best Training Accuracy: 1.0

- Random Forest Best Testing Accuracy: 0.9

- F1 score:0.81

- ROC AUC score:0.97

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.94      | 0.92   | 0.93     | 103     |
| 1            | 0.79      | 0.84   | 0.82     | 37      |
| accuracy     |           |        | 0.90     | 140     |
| macro avg    | 0.87      | 0.88   | 0.87     | 140     |
| weighted avg | 0.90      | 0.90   | 0.90     | 140     |

- The final training accuracy is 100% but the testing accuracy is 90% which means the model is overfitting.

- F1 Score-0.816 : An F1 score of 0.816 means the model strikes a good balance between precision and recall for the positive class (1), but there is big room to improve it due to higher value can also achieved from this kind on dataset by just changing classifier.

- ROC AUC Score - 0.972 : A score of 0.972 is quite high which means the model does fantastic job in discriminating between the two classes.

Confusion Matrix:

![Confusion Matrix](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Random%20forest%20confusion%20matrix.png)

- True Negatives (TN): 95
The model correctly predicted 95 instances as negative (class 0).

- False Positives (FP): 8
The model incorrectly predicted 8 instances as positive when they were actually negative.

- False Negatives (FN): 6
The model incorrectly predicted 6 instances as negative when they were actually positive.

- True Positives (TP): 31
The model correctly predicted 31 instances as positive (class 1).

The model performs very well with training data but not with the testing data. There is some trade-off with false positives and false negatives.
Also, the accuracy for the testing data is lesser than both decision tree and logistic regression.

### Feature Importance

Let's try to analyze the feature importance of the Random Forest model and plot them into graph.

A1_Score: 0.445

A2_Score: 0.109

A3_Score: 0.058

A4_Score: 0.036

A5_Score: 0.033

A6_Score: 0.052

A7_Score: 0.031

A8_Score: 0.025

A9_Score: 0.017

A10_Score: 0.021

age: 0.016

gender: 0.016

jaundice: 0.016

autism: 0.015

used_app_before: 0.015

ethnicity_Asian: 0.020

ethnicity_Black: 0.013

ethnicity_Hispanic: 0.017

ethnicity_Latino: 0.011

ethnicity_Middle Eastern : 0.027


![Feature importance random forest](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/Random%20forest%20Feature%20importance.png)

- A1_Score is the most important feature, contributing 44.5% to the model's predictions.

- A2_Score and A3_Score are also important but less so, with 10.9% and 5.8% contributions, respectively.

- Ethnicity features like ethnicity_Asian and ethnicity_Middle Eastern have moderate importance.

- Features such as age, gender, and jaundice have lower importance.

# Support Vector Classifier

SVC is a powerful tool for classifying data by finding the best possible boundary that separates different classes while focusing on the most critical data points near the boundary.

Now let's see how SVC model performs for our data.

| Values   | Train | Test  |
|----------|-------|-------|
| Accuracy | 98.5% | 90.7% |
| F1 score | 98.1% | 88.5% |

The result shows that training accuracy is 98.5% which is pretty good but when it comes to testing accuracy , the model didn't perform good here.Overall, 90% accuracy is good.

### Grid Search

#### Hyperparameter :

```python
svc_param_grid = {
        'svc__C': [0.1, 1, 10],
        'svc__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'svc__degree': [1, 2, 3],
        'svc__gamma': [0.1, 0.01, 0.001]
}
```
Best Parameters: {'svc__C': 10, 'svc__degree': 1, 'svc__gamma': 0.1, 'svc__kernel': 'linear'}

Best Score: 0.96

Here, the best parameter for kernel is linear.The C is 10, degree is 1 and gamma is selected as 0.1. Now's let's train and test these models using best paramters.

- SVC Best Training Accuracy: 0.98

- SVC Forest Best Testing Accuracy: 0.97

- F1 score:0.94

- ROC AUC score:0.98

|              | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| 0            | 0.99      | 0.97   | 0.98     | 103     |
| 1            | 0.92      | 0.97   | 0.95     | 37      |
| accuracy     |           |        | 0.97     | 140     |
| macro avg    | 0.96      | 0.97   | 0.96     | 140     |
| weighted avg | 0.97      | 0.97   | 0.97     | 140     |

- The model achieved a training accuracy of 98.6% on the training data. This indicates that the model performs very well on the data it was trained on.
The model achieved a testing accuracy of 97.1% on the test data. This is also very high, showing that the model generalizes well to new, unseen data.

- F1 Score- 0.947 : The F1 score of 0.947 is a measure of the model's accuracy, balancing precision and recall. A high F1 score indicates that the model is good at identifying positive cases with a balanced precision and recall.

- ROC AUC Score- 0.985 : The ROC AUC score of 0.985 is very high, indicating that the model has an excellent ability to distinguish between different classes.

Confusion Matrix:

![Confusion Matrix](https://github.com/Clarkson-Applied-Data-Science/2024_ia651_hemamalini_shifat/blob/main/SVC%20Confusion%20Matrix.png)

- True Negatives (TN): 100
The model correctly predicted 100 instances as negative (class 0).

- False Positives (FP): 3
The model incorrectly predicted 3 instances as positive when they were actually negative.

- False Negatives (FN): 1
The model incorrectly predicted 1 instance as negative when it was actually positive.

- True Positives (TP): 36
The model correctly predicted 36 instances as positive (class 1).

The model performs exceptionally well with high accuracy, precision, and recall for both classes. It has very few misclassifications, indicating strong performance.

# Conclusion

Let's compare the test accuracy of the models that is the models accuracy on new unseen data.

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | 95%      |
| Decision Tree       | 92%      |
| Random Forest       | 90%      |
| SVC                 | 98%      |

- SVC model has the highest accuracy comparatively.This is particularly important in a critical application like Autism Prediction.

- Logistic Regression is the next highest accuracy model.It provides a clear understanding of the influence of each feature on the prediction.

- Decision Tree is simple to understand and visualize but prone to overfitting, especially with complex or noisy data. May not generalize well if the tree is too deep.

- Random forest reduces overfitting. Robust to noisy data and capable of handling a large number of features but this model is less interpretable than individual decision trees and can be computationally expensive.

SVC with 98% accuracy, making it the top choice if achieving the highest accuracy is the primary goal.
Logistic Regression offers a strong balance between high accuracy (95%) and interpretability, making it a good alternative if understanding the model's decisions is important.
Decision Trees and Random Forests offer lower accuracy compared to SVC but can be useful depending on the need for model interpretability and computational resources. Decision Trees are easier to interpret, while Random Forests offer better performance through ensemble learning.

The model selection can be done by understanding what the data in the real world achieves.So, if the accuracy is the priority we can go with SVC. If the interpretability is important we can priortize Logistic Regression and the same applies for the Decision tree and Random forest when the balance is needed between performance and interpretebility.

# Future Scope

- Enhanced Feature Engineering
- Exploring deep learning techniques with more data
- Combining autism prediction with other health and psychological data to provide a more comprehensive understanding
- Conducting studies that track individuals over time to understand how symptoms and behaviors evolve and to improve early detection

By pursuing these areas, researchers and practitioners can work towards more accurate, personalized, and practical solutions for autism prediction in adults, ultimately improving diagnosis, treatment, and quality of life for individuals with autism.






























             

