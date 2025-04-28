# Exp-09: Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Load the Salary dataset, check for missing values, and review the data types.  
2. Encode the 'Position' column using Label Encoding.  
3. Set the input features `x` as 'Position' and 'Level', and the target `y` as 'Salary'.  
4. Split the dataset into training and testing sets with an 80-20 split.  
5. Create a Decision Tree Regressor model and train it on the training set.  
6. Predict the salaries for the testing set, calculate Mean Squared Error (MSE) and R² score, and predict for a given sample input.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SURIYA M
RegisterNumber:  212223110055
*/
print("Name : SURIYA M")
print("Register Number : 212223110055")
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head()
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()
x=df[['Position','Level']]
y=df['Salary']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeRegressor
model=DecisionTreeRegressor()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(y_test,y_pred)
mse
r2=r2_score(y_test,y_pred)
r2
print("Name : SURIYA M")
print("Register Number : 212223110055")
model.predict([[5,6]])
```

## Output:

### Head of Dataset
![image](https://github.com/user-attachments/assets/4573ff9e-3023-4310-98b1-dab95a87656c)

### Dataset Info
![image](https://github.com/user-attachments/assets/fd97a873-4568-4aa5-89aa-8887512ec21d)

### Null Counts
![image](https://github.com/user-attachments/assets/ce4f129d-b3dc-463e-8f4e-867c07d6c896)

### Encoded Data
![Screenshot 2025-04-28 162330](https://github.com/user-attachments/assets/466e046a-8436-4950-902c-b7fc514cdaee)

### MSE Value
![image](https://github.com/user-attachments/assets/9fd29151-1c8a-461b-8309-949bb7ca3d0d)

### R2 Value
![image](https://github.com/user-attachments/assets/2863620e-d3fc-4107-a7af-05d1f3198a62)

### Predicted Value for new data
![image](https://github.com/user-attachments/assets/44d62b53-0b5f-411c-b3d6-fdac4401a338)

## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
