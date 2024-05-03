# Load the Boston Housing dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through correlation matrix
# Build the Linear Regression Model and find its accuracy score
# Remove outliers and again see the accuracy of the model


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def RemoveOutlier(df,var):
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)
    df = df[((df[var] >= low) & (df[var] <= high))]
    return df

def BuildModel(X, Y):
    #1. divide the dataset into training and testing 80% train 20%testing
    #2. Choose the model (linear regression)
    #3. Train the model using training data #4. Test the model using testing data
    #5. Improve the performance of the model
    # Training and testing data
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.20, random_state=0)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model = model.fit(xtrain,ytrain) #Training
    ypred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    print('MAE:',mean_absolute_error(ytest,ypred))
    print("Model Score:",model.score(xtest,ytest))


df = pd.read_csv('./Boston.csv')
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)

print('Statistical information of Numerical Columns: \n',df.describe().T)

print('Total Number of Null Values in Dataset:', df.isna().sum())


sns.heatmap(df.corr(),annot=True)
plt.show()


X = df[['ptratio','lstat']] #input variables
Y = df['medv'] #output variable
BuildModel(X, Y)

fig, axes = plt.subplots(1,2)
sns.boxplot(data = df, x ='ptratio', ax=axes[0])
sns.boxplot(data = df, x ='lstat', ax=axes[1])
fig.tight_layout()
plt.show()
df = RemoveOutlier(df, 'ptratio')
df = RemoveOutlier(df, 'lstat')

X = df[['ptratio','lstat']]
Y = df['medv']
BuildModel(X, Y)

# fig, axes = plt.subplots(1,2)
# sns.boxplot(data = df, x ='ptratio', ax=axes[0])
# sns.boxplot(data = df, x ='lstat', ax=axes[1])
# fig.tight_layout()
# plt.show()

X = df[['rm','lstat', 'ptratio']]
Y = df['medv']
BuildModel(X, Y)

