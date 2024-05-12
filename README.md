# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages.

2.Import the dataset to operate on.

3.Split the dataset.

4.Predict the required output.

5.End the program.
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: KAVI NILAVAN DK
RegisterNumber: 212223230103
*/
import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy
```

## Output:
#### Result Output:
![ml 9 1](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/ab057ba3-d5b7-44b1-938e-f081f301c367)
#### data.head():
![Screenshot 2024-05-12 213322](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/3bd6a4f6-0aa5-400c-9eb0-b8d3126952f3)
#### data.info():
![ML 02](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/ce0b5167-e968-4f22-a560-387963062b5f)
#### data.isnull().sum():
![ML BE SVC](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/ef9a2355-b306-438e-aa38-899b67ef697c)
![ML SVC](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/9070c3f6-16a1-4bb8-a29e-0d782190f2f4)
#### Y_prediction Value:
![Screenshot 2024-05-12 213356](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/63644ff3-fea7-4e71-aba1-f56c217daac2)
#### Accuracy Value:
![Screenshot 2024-05-12 213411](https://github.com/KavinilavanDK/Implementation-of-SVM-For-Spam-Mail-Detection/assets/144870429/5a886bee-b0db-4306-adaf-aef92b02cf5e)
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
