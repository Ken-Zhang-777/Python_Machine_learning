import pandas as pd
import numpy as np

#download from Kaggle.com (https://www.kaggle.com/datasets/tawfikelmetwally/advertising-dataset)
df=pd.read_csv('./data/Advertising.csv')  
data_original=df.values   # Convert the DataFrame into a NumPy array
data = data_original[:, 1:]     # delete the first useless column
def lin_reg(data):
    # Create a matrix 'X' with a column of ones added to the input
    X = np.concatenate((np.full((data.shape[0],1),1), data[:,:-1]), axis=1)
    y=data[:,-1]
    # Calculate the coefficients (beta_hat) using the least squares formula for linear regression
    beta_hat = np.linalg.inv(X.T@X)@X.T@y
    return beta_hat
print(lin_reg(data)) # Print the result of the linear regression
#by now, coefficient for each channel generated
