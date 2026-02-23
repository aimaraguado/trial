import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

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

    with st.form("penguin_form"):
        island = st.selectbox("Island", ("Biscoe", "Dream", "Togersen"))
        gender = st.selectbox("Gender", ("male", "female"))

        bill_length_mm = st.slider("Bill length (mm)", 32.1, 60.0, 45.0)
        bill_depth_mm = st.slider("Bill depth (mm)", 13.0, 22.0, 17.0)
        flipper_length_mm = st.slider("Flipper length (mm)", 172.0, 231.0, 201.0)
        body_mass_g = st.slider("Body mass (g)", 2700.0, 6300.0, 4700.0)

        submitted = st.form_submit_button("🔮 Predict species")
  if submitted:
    data = {
        "island": island,
        "bill_length_mm": bill_length_mm,
        "bill_depth_mm": bill_depth_mm,
        "flipper_length_mm": flipper_length_mm,
        "body_mass_g": body_mass_g,
        "sex": gender
    }

    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X], axis = 0)

  encode = ["island", "sex"]
  df_penguins = pd.get_dummies(input_penguins, prefix = encode)
  input_row = df_penguins[:1]

target_mapper = {"Adelie": 0,
                 "Chinstrap": 1,
                 "Gentoo": 2
                }
def encode_value(val):
  return target_mapper[val]

y = Y.apply(encode_value)

with st.expander("Data preparation"):
  st.write("**Encoded input penguin X**")
  input_row
  st.write("**Encoded output penguin Y**")
  y

X = df_penguins[1:]
clf = RandomForestClassifier()
clf.fit(X,y)

predict = clf.predict(input_row)
predict_proba = clf.predict_proba(input_row)
st.subheader("Predicted Species")
st.dataframe(predict_proba,
            column_config = {
              "Adelie": st.column_config.ProgressColumn(
                "Adelie",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1
              ),
              "Chinstrap": st.column_config.ProgressColumn(
                "Chinstrap",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1
              ),
              "Gentoo": st.column_config.ProgressColumn(
                "Gentoo",
                format="%f",
                width="medium",
                min_value=0,
                max_value=1
              )
            })

penguins_species = np.array(["Adelie", "Chinstrap", "Gentoo"])
st.success(penguins_species[predict][0])
