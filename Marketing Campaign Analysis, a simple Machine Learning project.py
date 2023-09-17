import pandas as pd
import sklearn
import numpy as np

# Load data from www.kaggle.com
cmpn = pd.read_csv('./data/cmpn.csv')

# Identify categorical ("object") and numerical variables
var_char = cmpn.dtypes[cmpn.dtypes == 'object'].index.tolist()
var_num = cmpn.dtypes[cmpn.dtypes != 'object'].index.tolist()
var_key = ['cust_id', 'date']

# Remove key variables from the lists
for i in var_key:
    if (i in var_num):
        var_num.remove(i)
    if (i in var_char):
        var_char.remove(i)

# Remove duplicate rows from the dataset
cmpn = cmpn.drop_duplicates()

# Fit the imputer on a sample data (Learning)
imp.fit([[1, 2], [np.nan, 3], [7, 6], [1, 4]])
X = [[np.nan, 2], [6, np.nan], [7, 6], [3, np.nan]]     # X is indicater

# Separate categorical and numerical variables
cmpn_char = cmpn[var_char]
cmpn_num = cmpn[var_num]
cmpn_key = cmpn[var_key]

# Create a copy of cmpn to avoid modifying the original dataset
df = cmpn_impute.copy()

# Convert the outcome variable to binary (0 or 1)
df.loc[df.y == 'yes', "y_int"] = 1
df.loc[df.y == 'no', "y_int"] = 0

# Drop columns that are not needed
df = df.drop(columns=['y', 'duration', 'date'])

# One-hot encode categorical variables
df2 = pd.get_dummies(df)
X = df2[['age', 'contact_num', 'p_days', 'duration_mins',
         'p_outcome_failure', 'p_outcome_nonexistent', 'p_outcome_success', 'p_outcome_unknown']]
y = df2.y_int

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
