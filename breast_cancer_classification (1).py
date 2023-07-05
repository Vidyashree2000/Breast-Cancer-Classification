#Importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Mounting Google drive
from google.colab import drive
drive.mount('/content/drive')

#Loading dataset from drive
breast_cancer_dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data.csv')

#Reading dataset
print(breast_cancer_dataset)

breast_cancer_dataset.head()

breast_cancer_dataset.tail()

#Loading the data to a data frame
df = breast_cancer_dataset

#Exploratory data analysis
df.shape

df.info()

df.isnull().sum()

df.describe()

df['diagnosis'].value_counts()

sns.countplot(df['diagnosis'],label="Count")

df.groupby('diagnosis').mean()

df.dtypes

#Encoding categorical data values
from sklearn.preprocessing import LabelEncoder
labelencoder_Y = LabelEncoder()
df.iloc[:,1]= labelencoder_Y.fit_transform(df.iloc[:,1].values)
print(labelencoder_Y.fit_transform(df.iloc[:,1].values))

#Selecting features and labels
X = df.drop(columns='diagnosis', axis=1)
Y = df['diagnosis']

print(X)

print(Y)

Y=df.diagnosis
drop_cols=['Unnamed: 32','id','diagnosis']
X = df.drop(drop_cols,axis=1)
X.head()
#data visualization
ax=sns.countplot(Y,label='Count',palette='Blues')
B,M=Y.value_counts()
print("Benign Tumours is ", B)
print("Malignant Tumours is ", M)

X.describe()

f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(X.corr(),annot=True,linewidth=0.5,fmt='.1f',ax=ax)
plt.show()

drop_list = ['perimeter_mean','radius_mean','compactness_mean','concave points_mean',
             'radius_se','perimeter_se','radius_worst','perimeter_worst','compactness_worst',
             'concavity_worst','compactness_se','concave points_se','texture_worst','area_worst']
X1 = X.drop(drop_list ,axis = 1 )
X1.head()

f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(X1.corr(),annot=True,linewidth=0.5,fmt='.1f',ax=ax)
plt.show()

#Train and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing Confusion matrix to validate performance of the models
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# Support vector classifier
from sklearn.svm import SVC
svc_classifier = SVC()
svc_classifier.fit(X_train, Y_train)
Y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_scv)

# Train with Standard scaled Data
svc_classifier2 = SVC()
svc_classifier2.fit(X_train_sc, Y_train)
Y_pred_svc_sc = svc_classifier2.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_svc_sc)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_classifier = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier.fit(X_train, Y_train)
Y_pred_lr = lr_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_lr)

# Train with Standard scaled Data
lr_classifier2 = LogisticRegression(random_state = 51, penalty = 'l2')
lr_classifier2.fit(X_train_sc, Y_train)
Y_pred_lr_sc = lr_classifier.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_lr_sc)

# K – Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, Y_train)
Y_pred_knn = knn_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_knn)

# Train with Standard scaled Data
knn_classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
knn_classifier2.fit(X_train_sc, Y_train)
Y_pred_knn_sc = knn_classifier.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_knn_sc)

# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, Y_train)
Y_pred_nb = nb_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_nb)

# Train with Standard scaled Data
nb_classifier2 = GaussianNB()
nb_classifier2.fit(X_train_sc, Y_train)
Y_pred_nb_sc = nb_classifier2.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_nb_sc)

# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier.fit(X_train, Y_train)
Y_pred_dt = dt_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_dt)

# Train with Standard scaled Data
dt_classifier2 = DecisionTreeClassifier(criterion = 'entropy', random_state = 51)
dt_classifier2.fit(X_train_sc, Y_train)
Y_pred_dt_sc = dt_classifier.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_dt_sc)

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 51)
rf_classifier.fit(X_train, Y_train)
Y_pred_rf = rf_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_rf)

# Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
adb_classifier = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier.fit(X_train, Y_train)
Y_pred_adb = adb_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_adb)

# Train with Standard scaled Data
adb_classifier2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 200),
                                    n_estimators=2000,
                                    learning_rate=0.1,
                                    algorithm='SAMME.R',
                                    random_state=1,)
adb_classifier2.fit(X_train_sc, Y_train)
Y_pred_adb_sc = adb_classifier2.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_adb_sc)

# XGBoost Classifier
from xgboost import XGBClassifier
xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, Y_train)
Y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_score(Y_test, Y_pred_xgb)

# Train with Standard scaled Data
xgb_classifier2 = XGBClassifier()
xgb_classifier2.fit(X_train_sc, Y_train)
Y_pred_xgb_sc = xgb_classifier2.predict(X_test_sc)
accuracy_score(Y_test, Y_pred_xgb_sc)

cm = confusion_matrix(Y_test, Y_pred_lr)
plt.title('Heatmap of Confusion Matrix(LR)', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()

#Classification report

print(classification_report(Y_test, Y_pred_lr))

model = LogisticRegression()

# training the Logistic Regression model using Training data

model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)

print('Accuracy on training data = ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)

print('Accuracy on test data = ', test_data_accuracy)

input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Benign')

else:
  print('The Breast Cancer is Malignant')

input_data = (15.34,14.26,102.5,704.4,0.1073,0.2135,0.2077,0.09756,0.2521,0.07032,0.4388,0.7096,3.384,44.91,0.006789,0.05328,0.06446,0.02252,0.03672,0.004394,18.07,19.08,125.1,980.9,0.139,0.5954,0.6305,0.2393,0.4667,0.09946)
    # change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The Breast cancer is Benign')

else:
  print('The Breast Cancer is Malignant')

