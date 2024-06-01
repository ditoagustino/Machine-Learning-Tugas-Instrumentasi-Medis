

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('/content/drive/MyDrive/Data For Machine Learning/diabetes.csv')

dataset.head()

dataset.shape

dataset.info()

dataset.describe().T

dataset.isnull().sum()

sns.countplot(x = 'Outcome',data = dataset)

sns.pairplot(data = dataset, hue = 'Outcome')
plt.show()

sns.heatmap(dataset.corr(), annot = True)
plt.show()

dataset_new = dataset
dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = dataset_new[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)

dataset_new.isnull().sum()

dataset_new["Glucose"].fillna(dataset_new["Glucose"].mean(), inplace = True)
dataset_new["BloodPressure"].fillna(dataset_new["BloodPressure"].mean(), inplace = True)
dataset_new["SkinThickness"].fillna(dataset_new["SkinThickness"].mean(), inplace = True)
dataset_new["Insulin"].fillna(dataset_new["Insulin"].mean(), inplace = True)
dataset_new["BMI"].fillna(dataset_new["BMI"].mean(), inplace = True)

dataset_new.isnull().sum()

y = dataset_new['Outcome']
X = dataset_new.drop('Outcome', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify = dataset_new['Outcome'] )

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)
y_predict = model.predict(X_test)

y_predict

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_predict)
cm

sns.heatmap(pd.DataFrame(cm), annot=True)

from sklearn.metrics import accuracy_score
accuracy =accuracy_score(Y_test, y_predict)
accuracy

#Contoh Untuk Mengecek Seseorang Terkena Diabetes Atau Tidak
y_predict = model.predict([[1,148,72,35,79.799,33.6,0.627,50]])
print(y_predict)
if y_predict==1:
    print("Diabetes")
else:
    print("Tidak Diabetes")
