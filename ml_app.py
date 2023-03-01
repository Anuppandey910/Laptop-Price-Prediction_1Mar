import pandas as pd
import numpy as np
from sklearn import *
import pickle
import streamlit as st

df = pickle.load(open('df.pkl','rb'))
model = pickle.load(open('rf.pkl','rb'))

st.title('Laptop Price Prediction')
st.header('Fill the details to Predict the Laptop Price')

# Features
# ['Company', 'TypeName', 'Ram', 'Weight', 'Price', 'Touchscreen', 'Ips',
    #    'Cpu brand', 'HDD', 'SSD', 'Gpu brand', 'os']
    
company = st.selectbox('Company',df['Company'].unique())
typename = st.selectbox('TypeName',df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)',[ 8, 16,  4,  2, 12,  6, 32, 24, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen',['Yes','No'])
ips = st.selectbox('Ips',['Yes','No'])
cpu = st.selectbox('Cpu brand',df['Cpu brand'].unique())
hdd = st.selectbox('HDD',[0,500,1000,2000,32,128])
ssd = st.selectbox('SDD',[128,0,256,512,32,64,1000,1024,16,768,180,240,8])
gpu = st.selectbox('Gpu Brand',df['Gpu brand'].unique())
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Laptop Price'):
    if touchscreen=="Yes":
        touchscreen=1
    else:
        touchscreen=0
    if ips=="Yes":
        ips=1
    else:
        ips=0
    test_data = np.array([company,typename,ram,weight,touchscreen,ips,cpu,hdd,ssd,gpu,os])
    test_data = test_data.reshape([1,11])

    st.success(model.predict(test_data)[0])      
