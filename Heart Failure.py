# -*- coding: utf-8 -*-
"""

"""

#data visuklization and Random Forets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 heart=pd.read_csv("C:/Users/ADMIN/Desktop/Siddhi/heart_failure_clinical_records_dataset.csv")

#eda
heart.head(5)
heart.info()
heart.describe()
heart.columns
#checking for outliers
plt.boxplot(heart[['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium','time']])

yes = heart[heart['DEATH_EVENT'] == 1]['DEATH_EVENT']
no = heart[heart['DEATH_EVENT'] == 0]['DEATH_EVENT']

print(len(no))
print(len(yes))

#data visualization
plt.figure(figsize=(16,10))
sns.heatmap(heart.corr(method='pearson'), annot=True)

#distplot of age
sns.distplot(x=heart['age'])

#plotting age vs death
age_yes = heart[heart['DEATH_EVENT'] == 1].age
age = heart.age

plt.figure(figsize=(8,6))
plt.xlabel('Age')
plt.ylabel('Death Event')
plt.hist([age_yes, age], label=['Death Event', 'Total per age'])
plt.legend()

#plotting platelets vs death
plat =heart[heart['DEATH_EVENT'] == 1].platelets

plt.figure(figsize=(8,6))
plt.xlabel('Platelets')
plt.ylabel('Death Event')
plt.hist(plat, label=['Death Event'])
plt.legend()

#checking how manny patients sffering from anemia
sns.countplot(heart['anaemia'])

#machine learning model

x=heart.iloc[:,:-1].values
y=heart.iloc[:,-1].values

#splitting the data
from sklearn.model_selection import  train_test_split
x_train, x_test ,y_train ,y_test = train_test_split(x,y,test_size=0.2)

#Logistic Regression
from sklearn.linear_model import  LogisticRegression
classifier = LogisticRegression()

#model fit
classifier.fit(x_train,y_train)

#prediction
y_pred=classifier.predict(x_test)

#accuracy
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred)

#model accuracy = 0.8870967741935484

#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier as KNC
knnmodel=KNC(n_neighbors=5)

#fitting the model
knnmodel.fit(x_train, y_train)

#predict
predicted=knnmodel.predict(x_test)

#actual
actual_y= y_test

#accuracy
knnmodel.score(x_test, y_test)
#accuracy=0.8870967741935484

#random Forest
from sklearn.ensemble import RandomForestClassifier as RFC
rfmodel=RFC(n_estimators= 250,max_depth= 1)

#FITTING THE MODEL
rfmodel.fit(x_train,y_train)

#predicting
pred=rfmodel.predict(x_test)

#actual value
actualy= y_test

#accuracy of the model
from sklearn import metrics
# Model Accuracy
print("Accuracy:",metrics.accuracy_score(actualy, pred))
#accuracy=0.7903225806451613


#Conclusion - KNN and Logistic Regression have a higher accuracy as compared to random Forest
