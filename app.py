import streamlit as st
import tensorflow as tf
from tensorflow import *
import pandas as pd
import numpy as np


df = pd.DataFrame({'Parameter': ['Modulation Format', 'OSNR', 'Fibre Link Length', 'Pulse Peak Power', 'Modulation Frequency'],
                   'Value(s) used': ['OOK, BPSK, QPSK, PAM4, QAM8, QAM16', '8 to 25 dB', '0 to 80 km (step size of 5 km)', '0 dBm', '10 GBaud']})
# df.set_index('Parameter', inplace=True)

st.title("Parameter estimation in Optical fibre system using Cascaded Neural Network")
st.write("\n")
st.subheader("Use this webapp for the estimation of modulation format, OSNR and fibre link length of an optical fibre system.")
st.write(" There are three models, arranged in a cascaded structure for the prediction of modulation format, OSNR and fibre link length respectively. The models were trained on MATLAB based simulations of an optical fibre network for the following parameters.")
st.dataframe(df.assign(hack='').set_index('hack'))

st.write("\n The input should be a csv file containing 500 I/Q pairs, first the inphase values followed by the quadrature phase values and it should be of csv extension. Refer the demo file link below for better understanding of the format.")
# df = pd.DataFrame(
#     ['\nreal_1 | real_2 | ... | real_500 | imag_1 | imag_2 | imag _3 | ... | imag_500'], columns=[''])
# # df = pd.DataFrame([['real_1'], ['real_2'], ['...'],
# #                            ['real_500'], ['imag_1'], ['imag_2'], ['...'], ['imag_500'])
# st.dataframe(df.assign(hack='').set_index('hack'))
st.write("Once the dataset is ready, upload the csv file below and wait for the results to show up.")
# st.write("Make sure you record first and then predict, else it is bound to show some error")
st.write(
    "You can use these file for trial purpose as well [demo file](https://drive.google.com/drive/folders/1VafrhbivpjEMhsvidPNITiB1gUKNib4s?usp=sharing)")

uploaded_file = st.file_uploader(
    "Upload your csv file for prediction", type=['csv'])
if uploaded_file is not None:
    mf = 'pend'
    while mf == 'pend':
        with st.spinner('Waiting for results...'):
            df = pd.read_csv(uploaded_file, header=None)
            mfi = tf.keras.models.load_model('model_mfi_crx.h5')
            osnr = tf.keras.models.load_model('model_osnr_crx.h5')
            smf = tf.keras.models.load_model('model_smf_crx.h5')
            x_test = df
            x_test = np.array(x_test).reshape(-1,  x_test.shape[1], 1)
            t = mfi.predict(x_test)
            t = np.array(t).reshape(-1,  t.shape[1], 1)
            x_test = np.append(x_test, t, axis=1)
            eos = osnr.predict(x_test)
            eos = np.array(eos).reshape(-1,  eos.shape[1], 1)
            x_test = np.append(x_test, eos, axis=1)
            esm = smf.predict(x_test)
            if t[0][0] > 0.7:
                mf = 'OOK'
            elif t[0][1] > 0.7:
                mf = 'BPSK'
            elif t[0][2] > 0.7:
                mf = 'QPSK'
            elif t[0][3] > 0.7:
                mf = 'PAM4'
            elif t[0][4] > 0.7:
                mf = 'QAM8'
            elif t[0][5] > 0.7:
                mf = 'QAM16'
            else:
                mf = 'ambiguous result obtained'
            break
    st.success("Estimation Successful!")
    df = pd.DataFrame({'Modulation Format': [mf], 'OSNR': [
                      eos[0][0][0]], 'Fibre link length': [esm[0][0]]})

    st.dataframe(df.assign(hack='').set_index('hack'))
