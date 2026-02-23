import streamlit as st
import pandas as pd

st.title('🎈 Machine Learning App')

st.info('This app builds a machine learning model')

with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/master/penguins_cleaned.csv")
  df

  st.write("**X**")
  X = df.drop("species", axis = 1)
  X

  st.write("**Y**")
  Y = df.species
  Y
with st.expander("Data Visualization"):
  st.scatter_chart(data=df, x = "bill_length_mm", y = "body_mass_g", color = "species")

with st.sidebar:
  st.header("Input features")
  island = st.selectbox("Island", ("Biscoe", "Dream", "Togersen"))
  gender = st.selectbox("Gender", ("male", "female"))
  bill_length_mm = st.slider("Bill length (mm)", 32.1, 60.0, 45.0)
  bill_depth_mm = st.slider("Bill depth (mm)", 13.0, 22.0, 17.0)
  flicker_length_mm = st.slider("Flicker length (mm)", 172.0, 231.0, 201.0)
  body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4700.0 )
  data = {"island": island,
         "bill_length_mm": bill_length_mm,
         "bill_depth_mm": bill_depth_mm,
         "flicker_length_mm": flicker_length_mm,
         "body_mass_g": body_mass_g,
         "sex": gender}
  input_df = pd.DataFrame(data, index = [0])
  input_penguins = pd.concat([input_df, X], axis = 0)

  encode = ["island", "sex"]
  df_penguins = pd.get_dummies(input_penguins, prefix = encode)
  input_row = df_penguins[:1]
