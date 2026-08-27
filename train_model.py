import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score

# Load Dataset
data = pd.read_csv("std_marks.csv")

# input (X) and target(y)
X= data[["Hours"]]
y=data[["Marks"]]

# split data into training and testing
X_train,X_test,y_train,y_test = train_test_split(X ,y,test_size=0.2,random_state=42)

# Create model
model = LinearRegression()

# Train Model
model.fit(X_train,y_train)

# make  prediction
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test,predictions)
r2 = r2_score(y_test,predictions)




print("Actual Marks:")
print (y_test.values)


print ("\nPredicted Marks:")
print(predictions)


print ("\nMean Absolute Error:",mae)
print("R2 Score:",r2)

# # display dataset information
# print("\nDataset information:")
# print(data.info())

import joblib

# save trained model
joblib.dump(model, "model.pkl")

print ("\nModel saved successfully!")