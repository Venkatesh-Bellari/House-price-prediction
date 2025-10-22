import streamlit as st
import pandas as pd
import joblib

# Load the model
loaded_model = joblib.load('your_model.pkl')

# Title of the app
st.title("House Price Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file for input data", type=["csv"])

if uploaded_file:
    try:
        # Read the uploaded file
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:")
        st.dataframe(data)

        # Dropdown to select a row
        selected_index = st.selectbox("Select a row for prediction:", data.index)

        # Pre-fill inputs based on the selected row or manual entry
        input_data = {}
        numeric_columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                           'floors', 'view', 'condition', 'sqft_above', 'sqft_basement',
                           'yr_built']

        st.write("Enter or edit the input data for prediction:")

        # Numeric columns
        for col in numeric_columns:
            if selected_index is not None:
                input_data[col] = st.number_input(
                    f"{col}:", 
                    value=float(data.loc[selected_index, col]), 
                    step=0.00001,
                    format="%.5f",
                    key=col
                )
            else:
                input_data[col] = st.number_input(f"{col}:", value=0.0, step=0.00001, format="%.5f", key=col)

        # City column as dropdown
        cities = data['city'].unique().tolist()
        if selected_index is not None:
            default_city = data.loc[selected_index, 'city']
        else:
            default_city = cities[0]
        input_data['city'] = st.selectbox("City:", options=cities, index=cities.index(default_city))

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert city to dummies (one-hot)
        input_df = pd.get_dummies(input_df)
        # Make sure the columns match the model's training columns
        model_columns = loaded_model.feature_names_in_  # This requires scikit-learn >=1.0
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]  # reorder columns

        # Predict button
        if st.button("Predict"):
            prediction = loaded_model.predict(input_df)
            st.write(f"Predicted House Price: {prediction[0]}")

    except Exception as e:
        st.error(f"Error processing the file: {e}")

else:
    st.write("Please upload a CSV file or manually enter data for prediction.")
