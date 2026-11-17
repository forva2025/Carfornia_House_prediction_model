#Hands on project: predicting house prices using Linear Regression
'''
Steps:
1. Load the dataset 
2. Exploratory data analysis
3. Prepare features and target
4. Train-test split
5. Train the model
6. Evaluate the model
7. Visualize results
8. Model coefficients
9. Model Correction and improvement if any
10. deployment
    - Save the model
    - Load the model
    - Make predictions
11. Conclusion
'''
#import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import joblib


#load data
california=fetch_california_housing(as_frame=True)
data=california.frame
print(data.head())

#Exploratory data analysis
print(data.isnull().sum()) #display all null entries in the dataset
print(data.describe()) #display statistical summary of the dataset
print(data.info())

data.hist(bins=50, figsize=(20,15)) #histogram of the dataset showing all the 9 columns
plt.show()

#prepare features and targets
x=data.drop('MedHouseVal', axis=1) # remove the  column MedHouseVal on features
y= data['MedHouseVal'] #Target is the MedHouseVal.

#Train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.4,random_state=42)
print(f"Training set shape:{x_train.shape}")
print(f"Testing set shape:{x_test.shape}")

#Train the model
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#evaluate the model
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(f"MSE: {mse:4f}")
print(f"R-squared: {r2:4f}")

#Visualize the results
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred, alpha=0.6,color='blue')
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()], 'r--',lw=3)
plt.title("Actual Vs Predicted House prices")
plt.xlabel("Actual prices (in $100,000's)")
plt.ylabel("Predicted Prices(in $100,000's)")
plt.show()

#model coefficients
coefficients=pd.DataFrame(model.coef_,x.columns,columns=['Coefficient'])
print(coefficients )    #display the coefficients of the linear model

#Deploy.
#save the model
joblib.dump(model, "house_price_pipeline.joblib")

