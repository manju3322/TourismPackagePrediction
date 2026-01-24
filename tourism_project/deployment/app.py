import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="manjushs/tourism-model", filename="best_tourism_model.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of opting tourism package by a customer based on the parameters given.
Please enter the details below to get a prediction.
""")

# User input
#Type = st.selectbox("Machine Type", ["Self Enquiry","Company Invited"])
age = st.number_input("Age of customer", min_value=18, max_value=61, value=43, step=1)
typeofContact = st.selectbox("Type Of Contact", ["Self Enquiry","Company Invited"])
cityTier = st.selectbox("City Tier", ["Tier 1","Tier 2","Tier 3"])
durationOfPitch = st.number_input("Duration of Pitch", min_value=5, max_value=127, value=20, step=1)
occupation = st.selectbox("Occupation", ["Salaried","Small Business","Large Business","Free Lancer"])
gender = st.selectbox("Gender", ["Male","Female"])
numberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=5, value=3, step=1)
numberOfFollowups = st.number_input("Number Of Followups", min_value=1, max_value=6, value=4, step=1)
productPitched = st.selectbox("Product Pitched", ["Basic","Deluxe","Standard","Super Deluxe","King"])
preferredPropertyStar = st.number_input("Preferred Property Star", min_value=3, max_value=5, value=4, step=1)
maritalStatus = st.selectbox("Marital Status",["Married","Divorced","Unmarried","Single"])
numberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=22, value=4, step=1)
passport = st.selectbox("Passport",["Yes","No"])
pitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=4, step=1)
ownCar = st.selectbox("OwnCar",["Yes","No"])
numberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=3, value=1, step=1)
designation = st.selectbox("Designation",["Executive","Manager","Senior Manager","AVP","VP"])
monthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=99999, value=25000, step=100)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeofContact,
    'CityTier': cityTier,
    'DurationOfPitch': durationOfPitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': numberOfPersonVisiting,
    'NumberOfFollowups' : numberOfFollowups,
    'ProductPitched' : productPitched,
    'PreferredPropertyStar': preferredPropertyStar,
    'MaritalStatus' : maritalStatus,
    'NumberOfTrips' : numberOfTrips,
    'Passport' : passport,
    'PitchSatisfactionScore' : pitchSatisfactionScore,
    'OwnCar' : ownCar,
    'NumberOfChildrenVisiting' : numberOfChildrenVisiting,
    'Designation' : designation,
    'MonthlyIncome' : monthlyIncome
}])


if st.button("Predict Package Opted or Not"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Package Opted" if prediction == 1 else "Tourism Package Not Opted"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
