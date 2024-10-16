# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load dataset and split into features (`X`) and target (`y`).
2. Train a Decision Tree Regressor on `X` and `y`.
3. Predict salary values using the trained model.
4. Evaluate model performance using MSE and R² metrics.
5. Plot and visualize the decision tree structure.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SARANYA S
RegisterNumber: 212223110044 
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/757f71d6-4bba-41bb-96b8-6db28b64495f)
```
data.info()
```
## Output:
![image](https://github.com/user-attachments/assets/ef4f8bf6-e0a2-4288-b046-8223347bc768)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
## Output:
![image](https://github.com/user-attachments/assets/7acd4b59-ce40-4e1e-af70-b3f15ce470ed)
```
x=data[["Position","Level"]]
y=data["Salary"]
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print(mse)
```
## Output:
![image](https://github.com/user-attachments/assets/059d95bc-b465-4b84-a02c-dbb0a19afb60)

```
r2=metrics.r2_score(y_test,y_pred)
print(r2)
dt.predict([[5,6]])
```
## Output:
![image](https://github.com/user-attachments/assets/cb1da165-6212-4877-a04d-d64d37d1f092)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
