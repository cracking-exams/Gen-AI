import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


data = {
    'image_id':[1,2,3,4],
    'description':[
        'sunset over mountain',
        'cat on sofa',
        np.nan,
        'dog at night'
    ],
    'width':[1920,1024,800,np.nan],
    'height':[1080,768,768,600],
    'category':['landscape','animal','animal',np.nan]
}

df = pd.DataFrame(data)
print(df)
 

df['description'] = df['description'] .fillna('no description')

df["width"] = df["width"].fillna(df['width'].mean())

df["category"] = df["category"].fillna(df['category'].mode()[0])

print('After handling missing values:')
print(df)


scaler = MinMaxScaler()
df[['width','height']] = scaler.fit_transform(df[['width','height']])


print(df)
