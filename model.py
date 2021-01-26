import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

df=pd.DataFrame({"experience":[np.nan,np.nan,5,2,7,3,10,11],'test_score':[8,8,6,10,9,7,np.nan,7],"interview_score":[9,6,7,10,6,10,7,8],"salery":[50000,45000,60000,65000,70000,62000,72000,80000]},dtype=int)

df['experience'].fillna(0,inplace=True)
df['test_score'].fillna(df['test_score'].mean(),inplace=True)

x=df.iloc[:,:3]
y=df.iloc[:,-1]

regressor=LinearRegression()

regressor.fit(x,y)

pickle.dump(regressor,open('model.pkl','wb'))

model=pickle.load(open('model.pkl','rb'))

print(model.score(x,y))

print(model.predict([[0,9,8]]))