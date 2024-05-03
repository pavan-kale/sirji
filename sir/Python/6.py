# Load the Iris dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Feature Engineering through confusion matrix
# Build the Naive Based Regression Model and find its accuracy score
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
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest =train_test_split(X,Y,test_size= 0.25, random_state=0)
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model = model.fit(xtrain,ytrain)
    ypred = model.predict(xtest)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(ytest, ypred)
    sns.heatmap(cm, annot=True)
    plt.show()
    from sklearn.metrics import classification_report
    print(classification_report(ytest, ypred))


df = pd.read_csv('iris.csv')
df = df.drop('Id', axis=1)
df.columns = ('SL', 'SW', 'PL', 'PW', 'Species')
print('Information of Dataset:\n', df.info)
print('Shape of Dataset (row x column): ', df.shape)
print('Columns Name: ', df.columns)
print('Total elements in dataset:', df.size)
print('Datatype of attributes (columns):', df.dtypes)
print('First 5 rows:\n', df.head().T)
print('Last 5 rows:\n',df.tail().T)
print('Any 5 rows:\n',df.sample(5).T)
print('Statistical information of Numerical Columns: \n',df.describe())

print('Total Number of Null Values in Dataset:', df.isna().sum())

df['Species']=df['Species'].astype('category')
df['Species']=df['Species'].cat.codes
sns.heatmap(df.corr(),annot=True)
plt.show()
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)

fig, axes = plt.subplots(2,2)
sns.boxplot(data = df, x ='SL', ax=axes[0,0])
sns.boxplot(data = df, x ='SW', ax=axes[0,1])
sns.boxplot(data = df, x ='PL', ax=axes[1,0])
sns.boxplot(data = df, x ='PW', ax=axes[1,1])
plt.show()
df = RemoveOutlier(df, 'SW')
X = df[['SL','SW', 'PL', 'PW']]
Y = df['Species']
BuildModel(X, Y)
