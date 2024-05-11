# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df1=pd.read_csv("/content/bmi.csv")
df2=pd.read_csv("/content/bmi.csv")
df3=pd.read_csv("/content/bmi.csv")
df4=pd.read_csv("/content/bmi.csv")
df5=pd.read_csv("/content/bmi.csv")
df1.head()
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/4e2d865a-7d2e-4ccc-80e1-55002097a870)

```
df.dropna()
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/e619dc74-496a-4a31-9cb8-3f3df94333e5)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/f3f2098b-5f85-48e0-8fe1-8c19cda1982b)

```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/8244aa02-3e32-4496-b6e8-92f0ee417958)

```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df2[['Height','Weight']]=sc.fit_transform(df2[['Height','Weight']])
df2.head(10)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/c5a4880e-2151-41b3-bf37-b0ed745d6e8e)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/53af0b2b-bdb9-4406-a4c9-b30dcd947dab)

```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/affe1f97-5dc7-4a98-a699-88f66cb002df)

```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df5[['Height','Weight']]=scaler.fit_transform(df2[['Height','Weight']])
df5.head()
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/de359605-bb34-492a-ad7f-dae37f270270)

# FEATURE SELECTION :

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

data=pd.read_csv('/content/income.csv',na_values=[" ?"])
data
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/f0124d92-09dd-436f-b98a-50668b19d2f1)

```
data.isnull().sum()
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/898b5a6a-52e5-4b76-b3e6-a0fbfbc82361)

```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/1b376755-969f-4ed7-82f1-6daffed3daf1)

```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/e8aea04e-27c3-4381-be95-2fd018335c53)

```
sal=data['SalStat']
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,'greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/17a3cbf1-0af6-40c1-a979-5e6dffcebc2b)

```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/c280dcb9-40c1-4c42-883f-030bd126784f)

```
new_data=pd.get_dummies(data2,drop_first=True)
new_data
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/3c75e643-5fff-44fa-845c-572ffe6de4be)

# FEATURE SELECTION METHOD IMPLEMENTATION :
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/7226be10-5ce2-487e-acde-a3c6d2d88f4f)

```
#seperating the input names from data
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/7d6aec00-e12d-469f-8c16-e3db35b572c3)

```
#storing the output values in y
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/15d147cd-08a3-4699-b7a0-6999e451a622)

```
x = new_data[features].values
print(x)
```
![image](https://github.com/SanjithaBolisetti/EXNO-4-DS/assets/119393633/64ec62e3-91b0-4331-8fad-1726711e5b7d)


# RESULT:
Thus, Feature selection and Feature scaling has been used on thegiven dataset.
