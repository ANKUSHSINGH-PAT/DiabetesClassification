import streamlit as st
import pickle
import os
import numpy as np

# Function to load the pickled model
def load_model():
    try:
        model_path = os.path.join(os.getcwd(), 'model.pkl')
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Error: 'model.pkl' not found in directory: {os.getcwd()}")
        return None
    except Exception as e:
        print(f"Error loading 'model.pkl': {e}")
        return None

# Function to predict diabetes
def predict_diabetes(model, input_data):
    # Reshape input data for prediction
    input_data = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    # Load the model
    model = load_model()
        # Title of the web app
    st.title('Diabetes Prediction')

        # Input features for prediction
    Pregnancies = st.number_input('Pregnancies (number of times pregnant)', min_value=0, step=1)
    Glucose = st.number_input('Glucose (plasma glucose concentration)', min_value=0.0)
    BloodPressure = st.number_input('BloodPressure (diastolic blood pressure)', min_value=0.0)
    SkinThickness = st.number_input('SkinThickness (triceps skinfold thickness)', min_value=0.0)
    Insulin = st.number_input('Insulin (2-hour serum insulin)', min_value=0.0)
    BMI = st.number_input('BMI (body mass index)', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction', min_value=0.0)
    Age = st.number_input('Age (in years)', min_value=0, step=1)

        # Predict button
    if st.button('Predict'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        prediction = predict_diabetes(model, input_data)

            # Display prediction
        threshold = 0.5  # You can adjust the threshold as needed
        if prediction > threshold:
            prediction_output = 'Person diagnosed with diabetes'
        else:
            prediction_output = 'Person not diagnosed with diabetes'

        st.success(f'Prediction: {prediction_output}')

if __name__ == '__main__':
    main()
