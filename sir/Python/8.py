# Load the titanic dataset
# Display basic information
# Display statistical information
# Display null values
# Fill the null values
# Disp and interpret histogram of 1 var , 2 var
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('titanic.csv')

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
df['Age'].fillna(df['Age'].median(), inplace=True)
print('Total Number of Null Values in Dataset:', df.isna().sum())

fig, axis = plt.subplots(1,3)
sns.histplot(ax = axis[0], data = df, x='Sex', hue = 'Sex', multiple = 'dodge', shrink = 0.8)
sns.histplot(ax = axis[1], data = df, x='Pclass', hue = 'Pclass',multiple = 'dodge', shrink = 0.8)
sns.histplot(ax = axis[2], data = df, x='Survived', hue = 'Survived', multiple = 'dodge', shrink = 0.8)
plt.show()
fig, axis = plt.subplots(1,2)
sns.histplot(ax = axis[0], data = df, x='Age', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[1], data = df, x='Fare', multiple = 'dodge', shrink = 0.8, kde = True)
plt.show()
fig, axis = plt.subplots(2,2)
sns.histplot(ax = axis[0,0], data = df, x='Age', hue = 'Sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax = axis[0,1], data = df, x='Fare', hue = 'Sex', multiple = 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,0], data=df, x='Age', hue = 'Survived', multiple = 'dodge',shrink=0.8, kde= True)
sns.histplot(ax = axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink=0.8, kde = True)
plt.show()
fig, axis = plt.subplots(2,2)
sns.histplot(ax=axis[0,0], data=df, x='Sex', hue='Survived', multiple= 'dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[0,1], data=df, x='Pclass', hue='Survived', multiple='dodge', shrink=0.8, kde= True)
sns.histplot(ax=axis[1,0], data=df, x='Age', hue='Survived', multiple='dodge', shrink = 0.8, kde = True)
sns.histplot(ax=axis[1,1], data=df, x='Fare', hue='Survived', multiple='dodge', shrink = 0.8, kde = True)
plt.show()