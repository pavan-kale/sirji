import pandas as pd
import numpy as np
df = pd.read_csv('Placement.csv')
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

df['gender'].fillna(df['gender'].mode()[0], inplace=True)
df['ssc_p'].fillna(df['ssc_p'].mean(), inplace=True)
print('Mode of ssc_b: ', df['ssc_b'].mode())
df['ssc_b'].fillna(df['ssc_b'].mode()[0], inplace=True)
print('Total Number of Null Values in Dataset:', df.isna().sum())

df['sl_no']=df['sl_no'].astype('int8')
print('Change in datatype: ', df['sl_no'].dtypes)


df['gender'].replace(['M','F'],[0,1],inplace=True)

df['ssc_b']=df['ssc_b'].astype('category')
df['ssc_b']=df['ssc_b'].cat.codes

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
df[['hsc_b']]=enc.fit_transform(df[['hsc_b']])
print('After converting categorical variable to numeric variable: ')
print(df.head().T)

df['salary']=(df['salary']-df['salary'].min())/(df['salary'].max()-df['salary'].min())

from sklearn.preprocessing import MaxAbsScaler
abs_scaler=MaxAbsScaler()
df[['mba_p']]=abs_scaler.fit_transform(df[['mba_p']])

print(df.head().T)