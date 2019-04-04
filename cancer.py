#Importing the library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the datasets
#loading the datasets from sklearn
from sklearn.datasets import load_breast_cancer
cancer =  load_breast_cancer()
cancer
cancer.keys()
print(cancer['DESCR'])
print(cancer['target'])

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))

df_cancer.head()
#visualizing the data

sns.pairplot(df_cancer, hue='target', vars =['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])

sns.countplot(df_cancer['target'])

sns.scatterplot(x= "mean area", y= "mean smoothness", hue= "target", data=df_cancer)

plt.figure(figsize = (20, 10))
sns.heatmap(df_cancer.corr(), annot= True)
#Model Training(Finding a problem solution)
x= df_cancer.drop(['target'], axis=1)
y=df_cancer['target']
#Training and Testing of the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

#Support Vector Machine to do Classification
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
svc_model = SVC()
svc_model.fit(x_train, y_train)
#Evaluating the model
y_pred = svc_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)
#Improving the model
min_train = x_train.min()
range_train =(x_train-min_train).max()
x_train_scaled = (x_train - min_train)/range_train


min_test = x_test.min()
range_test=(x_test-min_test).max()
x_test_scaled = (x_test - min_test)/range_test

svc_model.fit(x_train_scaled, y_train)

y_predict = svc_model.predict(x_test_scaled)


cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm, annot=True)


print(classification_report(y_test, y_predict))

#improving the model part2
param_grid = {'c': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose=4)

grid.fit(x_train_scaled, y_train)
y_predict=svc_model.predict(x_test_scaled)
cm=confusion_matrix
grid.best_params_

GridSearchCV.get_params().keys()

cm=confusion_matrix(y_test,grid_predictions)
sns.heatmap(cm,annot=True)