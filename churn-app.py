import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder

# load the model from disk
import joblib
model = joblib.load("Churn Prediction")

def main():
    #Setting Application title
    st.title('Churn Prediction App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict customer churn in a ficitional Service Providing use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default

    #add_selectbox = st.sidebar.selectbox("Mode of Prediction?", ("Online", "Batch"))
    #st.sidebar.info('This app is created to predict Customer Churn')
    # st.sidebar.image(image)

    #if add_selectbox == "Online":
    st.info("Input data below")
    #Based on our optimal features selection
    st.subheader("Demographic data")
    gender = st.selectbox('Gender:', ('Male', 'Female'))
    dependents = st.selectbox('Dependent:', ('Yes', 'No'))


    st.subheader("Watch Data")
    tenure = st.slider('Number of days subscribed', min_value=1, max_value=3655, value=0)
    WeeklyMinimum  = st.slider('Weekly Minimum Watch (Minutes)', min_value=0, max_value=600, value=0)
    DailyMinimum  = st.slider('Daily Minimum Watch (Minutes)', min_value=0, max_value=150, value=0)
    DailyMaximum  = st.slider('Daily Maximum Watch (Minutes)', min_value=0, max_value=200, value=0)
    WeeklyMaxNights = st.slider('Weekly Maximum Night Watch (Minutes)', min_value=40, max_value=360, value=0)
    VideosWatched  = st.slider('Videos Watched', min_value=0, max_value=200, value=0)
    DaysInactive  = st.slider('Number of Dayes Inactive', min_value=0, max_value=100, value=0)
    #PaymentMethod = st.selectbox('Payment Method',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))

    #totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

    st.subheader("Services signed up for")
    MultiScreening = st.selectbox('Multi-Screening', ('Yes', 'No'))
    MailSubscription = st.selectbox('Subscribed to Mail', ('Yes', 'No'))
    CustomerCareCall = st.number_input('Customer Support Calls', min_value=0, max_value=10, value=0)

    data = {
            'gender': gender,
            'Dependents': dependents,
            'no_of_days_subscribed':tenure,
            'weekly_mins_watched': WeeklyMinimum,
            'minimum_daily_mins': DailyMinimum,
            'maximum_daily_mins':  DailyMaximum,
            'weekly_max_night_mins':  WeeklyMaxNights,
            'videos_watched': VideosWatched,
            'maximum_days_inactive': DaysInactive,
            #'PaymentMethod':PaymentMethod,
            #'TotalCharges': totalcharges,
            'MultiScreening': MultiScreening,
            'MailSubscription': MailSubscription,
            #'CustomerCareCall': CustomerCareCall
            }
    features_dataset = pd.DataFrame.from_dict([data])
    features_dataset = pd.get_dummies(features_dataset, columns = ['gender', 'Dependents', 'MultiScreening', 'MailSubscription'])
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.write('Overview of input is shown below')
    st.markdown("<h3></h3>", unsafe_allow_html=True)
    st.dataframe(features_dataset)


    if st.button('Predict'):
        if model.predict(features_dataset) == 1:
            #st.warning('Yes, the customer will terminate the service.')
            st.success('No, the customer is happy with the Services.')
        else:
            #st.success('No, the customer is happy with the Services.')
            st.warning('Yes, the customer will terminate the service.')


if __name__ == '__main__':
        main()