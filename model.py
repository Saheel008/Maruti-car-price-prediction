# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from word2number import w2n

df = pd.read_csv("car_data_SWIFT.csv")

x=df[['km_driven','engine_size','car_age']]
y=df['selling_price']
#x=df.loc[:, df.columns != "selling_price"].to_numpy()
#y = np.squeeze(df.loc[:, "selling_price"].to_numpy())

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fitting model with trainig data
model.fit(x, y) 

#y=m1x1+m2x2+c

# Saving model to disk
pickle.dump(model, open('lin_reg_model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''