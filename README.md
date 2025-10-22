# BLENDED LEARNING
# Implementation of Support Vector Machine for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a Support Vector Machine (SVM) model to classify food items and optimize hyperparameters for better accuracy.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load Data Import and prepare the dataset to initiate the analysis workflow.
   
2.Explore Data Examine the data to understand key patterns, distributions, and feature relationships.

3.Select Features Choose the most impactful features to improve model accuracy and reduce complexity.

4.Split Data Partition the dataset into training and testing sets for validation purposes.

5.Scale Features Normalize feature values to maintain consistent scales, ensuring stability during training.

6.Train Model with Hyperparameter Tuning Fit the model to the training data while adjusting hyperparameters to enhance performance.

7.Evaluate Model Assess the model’s accuracy and effectiveness on the testing set using performance metrics.

## Program:
```
/*
Program to implement SVM for food classification for diabetic patients.
Developed by: Pagadala Mithun Kalyan
RegisterNumber: 212223040142
*/

#import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt

#step-1:Load the dataset from the url
data=pd.read_csv("/content/food_items_binary (1).csv")

#step-2:Data Exploration
#Display the first few rows and the colum names for verification
print(data.head())
print(data.columns)

#step-3:Selection Features and Target
#Define relevant features and target column
features = ['Calories', 'Total Fat', 'Saturated Fat', 'Monounsaturated Fat']
target = 'class'

x = data[features]
y = data[target]

#step-4: Splitting Data
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


#step-5: Feature Scaling
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

#step-6:Model training with Hyperparameter Tuning using GridSearchCV
#Define the SVM model
svm=SVC()

#set up heperparameter grid for tuning
param_grid = {
    'C': [0.1,1,10,100],
    'kernel': ['linear','rbf'],
    'gamma': ['scale','auto']
}

#Initialize GridSearchCV
grid_search=GridSearchCV(svm,param_grid,cv=5)
grid_search.fit(x_train,y_train)

#Extract the best model
best_model=grid_search.best_estimator_
print("Name: MITHUN KALYAN")
print("Register Number: 212223040142")
print("Best Parameter:",grid_search.best_params_)

y_pred=best_model.predict(x_test)

accurancy=accuracy_score(y_test,y_pred)
print("Name: MITHUN KALYAN")
print("Register Number: 212223040142")
print("Accuracy:",accurancy)
print("Classification Report:\n",classification_report(y_test,y_pred))

#confusion matrix
conf_matrix=confusion_matrix(y_test,y_pred)
sns.heatmap(conf_matrix,annot=True,fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Output:
<img width="815" height="688" alt="image" src="https://github.com/user-attachments/assets/51394647-deb9-4af2-9bf7-6b34103b1944" />

<img width="612" height="81" alt="image" src="https://github.com/user-attachments/assets/63267c8b-0a93-41c1-986f-2b48eca8155c" />

<img width="565" height="278" alt="image" src="https://github.com/user-attachments/assets/ec0c8df1-227d-4d1c-9f03-2a4083dd2a25" />

<img width="757" height="573" alt="image" src="https://github.com/user-attachments/assets/287972cf-9663-43f8-8d38-61ab36d07e0c" />

## Result:
Thus, the SVM model was successfully implemented to classify food items for diabetic patients, with hyperparameter tuning optimizing the model's performance.
