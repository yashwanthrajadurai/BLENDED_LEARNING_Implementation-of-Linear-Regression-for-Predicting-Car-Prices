# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
# Name : YASHWANTH RAJA DURAI.V
# REG NO : 212222040284
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries:
Import necessary libraries such as pandas, numpy, matplotlib, and sklearn.
2. Load Dataset:
Load the dataset containing car prices and relevant features.
3. Data Preprocessing:
Handle missing values and perform feature selection if necessary.
4. Split Data:
Split the dataset into training and testing sets.
5. Train Model:
Create a linear regression model and fit it to the training data.
6. Make Predictions:
Use the model to make predictions on the test set.
7. Evaluate Model:
Assess model performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8. Check Assumptions:
Plot residuals to check for homoscedasticity, normality, and linearity.
9. Output Results:
Display the predictions and evaluation metrics.
Program to implement linear regression model for predicting car prices and test assumptions.


Developed by: S Harish
RegisterNumber:  212223040062
## Program:
```

1.import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("CarPrice_Assignment (1).csv")
df
````

![image](https://github.com/user-attachments/assets/48b27cbf-f555-4656-8897-c5de4544e051)


```
2.x = df[['enginesize','horsepower','citympg','highwaympg']]
y = df['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
df.head()
```
![image](https://github.com/user-attachments/assets/05eb1cdd-2844-4c61-868e-7e09cf796c39)

```

3.df.describe()
```
![image](https://github.com/user-attachments/assets/6833f439-eebb-4aff-a54b-a080393ad617)

```
4.df.head()
```
![image](https://github.com/user-attachments/assets/924d2ee1-b22c-49ba-8f4a-02b82316fcae)

```
5.df.tail()
```
![image](https://github.com/user-attachments/assets/b1bb9a1a-7ed8-448e-bfdb-f27ddfc915c7)

```
6.df.notnull()
```
![image](https://github.com/user-attachments/assets/ef87254c-7dee-4171-b717-935297634614)

```
7.df.isnull()
```
![image](https://github.com/user-attachments/assets/8767a725-faea-448f-972f-cc5ec7518cb6)

```
8.df.info
```
![image](https://github.com/user-attachments/assets/b6b305db-2980-4fc2-ac5f-1f55380f343f)

```
9.scaler= StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model= LinearRegression()
model.fit(x_train_scaled,y_train)

y_pred = model.predict(x_test_scaled)

print("="*50)
print("MODEL COEFFICIENTS")

for feature,coef in zip(x.columns,model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test,y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test,y_pred))}")
print(f"{'R-squared':>12}: {r2_score(y_test,y_pred):>10.2f}")
print("="*50)
```
![image](https://github.com/user-attachments/assets/fdc35c5e-953b-47fa-b685-44d8387c19f0)


```
10.

# 1. Linearity Check
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

# 2. Independence Check (Durbin-Watson Test)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_test:.2f}")
print("(Values close to 2 indicate no autocorrelation)")

# 3. Homoscedasticity Check
plt.figure(figsize=(10, 5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color': 'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/24e24b7b-acb1-4982-9374-d2380920f649)

```
11. # 4. Normality of residuals
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(residuals, kde=True, ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals, line='45', fit=True, ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()

```
![image](https://github.com/user-attachments/assets/1d246029-4e79-4d69-8526-f60f904d9a36)


## Output:
![simple linear regression model for predicting the marks scored](sam.png)


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
