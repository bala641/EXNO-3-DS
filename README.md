## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
```

![image](https://github.com/user-attachments/assets/fe3052d7-0299-4e80-926a-03053e0796ef)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```

![image](https://github.com/user-attachments/assets/e59a9646-2aa6-4174-a808-9417f46f2b37)

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```

![image](https://github.com/user-attachments/assets/68d45b92-f38e-491f-a02c-0e9a903648a5)

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```

![image](https://github.com/user-attachments/assets/31c05a6f-b11e-4c15-8ad5-c3db74a039da)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False) # Change 'sparse' to 'sparse_output'
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```

![image](https://github.com/user-attachments/assets/890af855-6c4e-4118-aa48-11da45ab2141)

```
pd.get_dummies(df2,columns=["nom_0"])
```

![image](https://github.com/user-attachments/assets/8e29d63c-ac2f-487b-978d-f270f8fe638f)
```
!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

dfb=pd.concat([df,nd],axis=1)
dfb
```

![image](https://github.com/user-attachments/assets/912fed4b-db07-4a46-b7ec-330e761f0162)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```

![image](https://github.com/user-attachments/assets/82881ca1-aec1-40fb-9f4d-6bd4c9e64791)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
```
![image](https://github.com/user-attachments/assets/512705c8-0549-4018-bc84-d538eaa99418)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/4be5cd39-c571-4472-b09a-a6d84145689e)
```
np.log(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/9b31a3ee-fe16-4cb3-b079-9abbc7d8486f)
```
 np.reciprocal(df["Moderate Positive Skew"])
```

![image](https://github.com/user-attachments/assets/2482bf04-6a00-458a-99da-653e9cd095bf)
```
np.sqrt(df["Highly Positive Skew"])
```

![image](https://github.com/user-attachments/assets/8a11d631-c681-414c-9272-0d860693a37c)
```
np.square(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/44aa1785-0589-47b9-afec-e49421cb2f2d)

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

![image](https://github.com/user-attachments/assets/323f1a32-8d94-4214-891f-071080cd83bb)
```
df.skew()
```

![image](https://github.com/user-attachments/assets/3b7ac30c-cfbe-4507-a1aa-f9363a327ad0)
```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

![image](https://github.com/user-attachments/assets/7e45634e-0763-44c8-a135-068cb8a0f17a)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```
![image](https://github.com/user-attachments/assets/80015fa6-d3ae-4dd4-a5f6-52b05f605d4d)

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/5204d59e-5114-4de7-ba57-b6cc6968f99d)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/639acb38-bcf0-4239-ae04-8af7a504e56f)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/8277718a-8a13-49bf-9b6b-ae32d4040626)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/1dd9e4e1-501e-4bdf-b85d-031a9e12d7e9)
```
dt=pd.read_csv("/titanic_dataset (1).csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```

![image](https://github.com/user-attachments/assets/7b3bdd3d-8e71-4173-b8ac-295f24a5096c)

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

![image](https://github.com/user-attachments/assets/6346fd30-6cd9-4b28-8da9-b44e652f3239)
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
