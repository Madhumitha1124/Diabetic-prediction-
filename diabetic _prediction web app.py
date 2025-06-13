'''import numpy as np
import pickle
import streamlit as st

# Load trained model and scaler
model_path = r"C:\\Users\\Madhumitha S\\Downloads\\trained_model (1).sav"
scaler_path = r"C:\\Users\\Madhumitha S\\Downloads\\scaler.sav"

loaded_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Prediction function
def diabetic_prediction(input_data):
    input_data_np = np.asarray(input_data, dtype=float).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_np)
    prediction = loaded_model.predict(input_data_scaled)
    return prediction[0]

# Recommendation system
def recommend_plan(prediction):
    if prediction == 1:
        st.info("Diabetes Detected! Suggested Food & Workout Plan:")
        st.write("- Eat more fiber-rich foods (vegetables, fruits, whole grains).")
        st.write("- Reduce intake of refined carbs and sugar.")
        st.write("- Increase lean protein (chicken, fish, tofu).")
        st.write("- Stay hydrated and drink more water.")
        st.write("- 30 minutes of light walking or yoga daily.")
        st.write("- Strength training (light weights) 2-3 times a week.")
    else:
        st.info("No Diabetes Detected! Suggested Healthy Lifestyle Plan:")
        st.write("- Maintain a balanced diet (carbs, protein, fats).")
        st.write("- Include fruits, vegetables, whole grains, and lean proteins.")
        st.write("- Drink plenty of water and avoid excess sugar.")
        st.write("- 30-45 minutes of moderate exercise (jogging, cycling, swimming).")

# Main app
def main():
    st.title('Diabetic Prediction Web Application')

    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose level")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

  if st.button('Diabetic Test Result'):
       input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                     Insulin, BMI, DiabetesPedigreeFunction, Age]

       result = diabetic_prediction(input_data)
       diagnosis = 'The person is diabetic' if result == 1 else 'The person is not diabetic'
       st.success(diagnosis)
       recommend_plan(result)
        
        if st.button('Diabetic Test Result'):
    input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                  Insulin, BMI, DiabetesPedigreeFunction, Age]

    result = diabetic_prediction(input_data)

    # Color-coded output
    if result == 1:
        st.markdown("<h3 style='color:green;'>The person is diabetic</h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='color:red;'>The person is not diabetic</h3>", unsafe_allow_html=True)

    recommend_plan(result)
    


if __name__ == '__main__':
    main()'''


import numpy as np
import pickle
import streamlit as st

# Load trained model and scaler
model_path = r"C:\\Users\\Madhumitha S\\Downloads\\trained_model (1).sav"
scaler_path = r"C:\\Users\\Madhumitha S\\Downloads\\scaler.sav"

loaded_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

# Prediction function
def diabetic_prediction(input_data):
    input_data_np = np.asarray(input_data, dtype=float).reshape(1, -1)
    input_data_scaled = scaler.transform(input_data_np)
    prediction = loaded_model.predict(input_data_scaled)
    return prediction[0]

# Recommendation system
def recommend_plan(prediction):
    if prediction == 1:
        st.info("Diabetes Detected! Suggested Food & Workout Plan:")
        st.write("- Eat more fiber-rich foods (vegetables, fruits, whole grains).")
        st.write("- Reduce intake of refined carbs and sugar.")
        st.write("- Increase lean protein (chicken, fish, tofu).")
        st.write("- Stay hydrated and drink more water.")
        st.write("- 30 minutes of light walking or yoga daily.")
        st.write("- Strength training (light weights) 2-3 times a week.")
    else:
        st.info("No Diabetes Detected! Suggested Healthy Lifestyle Plan:")
        st.write("- Maintain a balanced diet (carbs, protein, fats).")
        st.write("- Include fruits, vegetables, whole grains, and lean proteins.")
        st.write("- Drink plenty of water and avoid excess sugar.")
        st.write("- 30-45 minutes of moderate exercise (jogging, cycling, swimming).")

# Main app
def main():
    st.title('Diabetic Prediction Web Application')

    Pregnancies = st.number_input("Number of Pregnancies")
    Glucose = st.number_input("Glucose level")
    BloodPressure = st.number_input("Blood Pressure")
    SkinThickness = st.number_input("Skin Thickness")
    Insulin = st.number_input("Insulin")
    BMI = st.number_input("BMI")
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function")
    Age = st.number_input("Age")

    if st.button('Diabetic Test Result'):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness,
                      Insulin, BMI, DiabetesPedigreeFunction, Age]

        result = diabetic_prediction(input_data)

        # Color-coded output
        if result == 1:
            st.markdown("<h3 style='color:green;'>The person is diabetic</h3>", unsafe_allow_html=True)
        else:
            st.markdown("<h3 style='color:red;'>The person is not diabetic</h3>", unsafe_allow_html=True)

        recommend_plan(result)

if __name__ == '__main__':
    main()

