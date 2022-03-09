import streamlit as st
from streamlit import components
import tensorflow as tf
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
from streamlit import components

# Load model
load_model = tf.keras.models.load_model("D:/PyCharm/water_streamlit/Water.h5")
df = pd.read_csv("D:/PyCharm/water_streamlit/latest_water1.csv")
X = df.drop(['is_safe'], 1)

def main():
  st.title("Water Prediction")
  #	aluminium	arsenic	cadmium	chloramine	chromium
  #aluminium - dangerous if greater than 2.8
  #arsenic - dangerous if greater than 0.01
  # cadmium - dangerous if greater than 0.005
  # chloramine - dangerous if greater than 4
  # chromium - dangerous if greater than 0.1
  aluminium = st.slider("Number of Aluminium",min_value=0.0,max_value=10.0,step=0.01)
  st.write("The value of Aluminium is ", aluminium)
  arsenic = st.text_input("Number of Arsenic")
  st.write("The value of Arsenic is ", arsenic)
  cadmium = st.text_input("Number of Cadmium")
  st.write("The value of Cadmium is ", cadmium)
  chloramine = st.slider("Number of Chloramine",min_value=0.0,max_value=10.0,step=0.01)
  st.write("The value of Chloramine is ", chloramine)
  chromium = st.text_input("Number of Chromium")
  st.write("The value of Chromium is ", chromium)

    
  if st.button('Test Result'): 
    if aluminium <= 0 and len(arsenic) <= 0 and len(cadmium) <= 0 and chloramine <= 0 and len(chromium) <= 0:
      st.write("No Input")
    else:
      input_data = ([[aluminium,float(arsenic),float(cadmium),chloramine,float(chromium)]])   
      def prob(data):
        y_pred=load_model.predict(data).reshape(-1, 1)
        y_pred =(y_pred>0.65)
        return np.hstack((1-y_pred,y_pred))

      prediction = load_model.predict(input_data)    
      if (prediction <= 0.70):
        st.success('Not Safe')
      else:
        st.success('Safe') 

      #Convert Input testing data to pandas series
      lst = ['Aluminium','Arsenic','Cadmium','Chloramine','Chromium']
      ss = pd.Series(input_data[0], lst)
      columns = ['aluminium', 'arsenic',  'cadmium',  'chloramine', 'chromium']
      explainer = lime.lime_tabular.LimeTabularExplainer(X[list(X.columns)].astype(int).values, mode='classification',class_names=['Not Safe', 'Safe'],feature_names=list(X.columns))
      exp = explainer.explain_instance(ss, prob)
      exp.show_in_notebook(show_table=True)
      html = exp.as_html()
      import streamlit.components.v1 as components
      components.html(html, height=800)





if __name__ == '__main__':
  main()