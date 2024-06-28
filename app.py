import numpy as np
import pandas as pd
import yfinance as yf 
from keras.models  import load_model
import streamlit as st 
import matplotlib.pyplot as plt

model = load_model('C:\\Users\\tyagi\\OneDrive\\Desktop\\jupyter notebook\\stock price prediction model.keras')
st.header('Stock Market Predictor')

stock = st.text_input('Enter stock symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock,start,end)
st.subheader('Stock Data')
st.write(data)
data_train = pd.DataFrame(data.Close[0:int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):len(data)])

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
pass_100_days = data_train.tail(100)
data_test = pd.concat([pass_100_days,data_train],ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price Vs MAS50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price Vs MAS50 Vs MAS100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8,6))
plt.plot(ma_50_days,'r')
plt.plot(ma_100_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig2)

st.subheader('price Vs MAS100 Vs MAS200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days,'r')
plt.plot(ma_200_days,'b')
plt.plot(data.Close,'g')
plt.show()
st.pyplot(fig3)

x= []
y= []
for i in range(100,data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x,y = np.array(x),np.array(y)

predict = model.predict(x)
scale = 1/scaler.scale_
predict = predict*scale
y= y*scale

st.subheader('Orginal Price Vs Predicted Price')
fig4 = plt.figure(figsize=(8,6))
plt.plot(predict,'g',label = 'Orginal Price')
plt.plot(y,'r',label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
st.pyplot(fig4)
