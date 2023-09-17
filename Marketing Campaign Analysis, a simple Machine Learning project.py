import pandas as pd
import sklearn
import numpy as np

# Load data from https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
cmpn = pd.read_csv('./data/customer_churn_dataset-testing-master.csv')

# Identify categorical ("object") and numerical variables
var_char = cmpn.dtypes[cmpn.dtypes == 'object'].index.tolist()
var_num = cmpn.dtypes[cmpn.dtypes != 'object'].index.tolist()
var_key = ['CustomerID']

# Remove duplicate rows from the dataset
cmpn = cmpn.drop_duplicates()

# Separate categorical and numerical variables
cmpn_char = cmpn[var_char]
cmpn_num = cmpn[var_num]
cmpn_key = cmpn[var_key]

# Create a copy of cmpn to avoid modifying the original dataset
df = cmpn.copy()

# Drop columns that are not needed
df = df.drop(columns=['y', 'duration', 'date'])

# One-hot encode categorical variables
df2 = pd.get_dummies(df)
X = df2[['CustomerID','Age','Gender','Tenure','Usage Frequency','Support Calls',
         'Payment Delay','Subscription Type','Contract Length','Total Spend','Last Interaction']]
y = df2.Churn

from sklearn.model_selection import train_test_split
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=824)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
clf = DecisionTreeClassifier()  # Decision Tree classifier
# Train the Decision Tree classifier
clf = clf.fit(X_train, y_train)
# Predict the response for the test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  # Model accuracy calculation，output is:0.8630735615440641
