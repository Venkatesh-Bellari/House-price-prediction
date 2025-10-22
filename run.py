import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the model
loaded_model = joblib.load('your_model.pkl')

# If your model uses LabelEncoder for 'city', load it
city_encoder = joblib.load('city_encoder.pkl')  # Save this during training

st.title("House Price Prediction")

uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        selected_index = st.selectbox("Select a row for prediction:", data.index)

        input_data = {}
        numeric_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                        'floors', 'view', 'condition', 'sqft_above', 'sqft_basement',
                        'yr_built']
        categorical_cols = ['city']

        st.write("Enter or edit the input data for prediction:")

        # Numeric inputs
        for col in numeric_cols:
            input_data[col] = st.number_input(
                f"{col}:",
                value=float(data.loc[selected_index, col]),
                step=0.01,
                format="%.5f",
                key=col
            )

        # Categorical inputs
        for col in categorical_cols:
            input_data[col] = st.selectbox(
                f"{col}:",
                options=data[col].unique(),
                index=list(data[col].unique()).tolist().index(data.loc[selected_index, col])
            )

        # Encode categorical columns
        for col in categorical_cols:
            input_data[col] = city_encoder.transform([input_data[col]])[0]

        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Predict button
        if st.button("Predict"):
            prediction = loaded_model.predict(input_df)
            st.success(f"Predicted House Price: {prediction[0]:.2f}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV file or manually enter data for prediction.")
