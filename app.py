import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
feature_cols = joblib.load('features.pkl')  

st.set_page_config(page_title="Netflix Churn Prediction", layout="centered")

st.title("🎬 Netflix Churn Prediction App")
st.markdown("Enter customer data below to predict whether they will churn or not.")

with st.form("input_form"):
    age = st.number_input("Age", min_value=10, max_value=100, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    subscription_type = st.selectbox("Subscription Type", ['Basic', 'Standard', 'Premium'])
    watch_hours = st.slider("Watch Hours (per month)", 0, 300, 100)
    last_login_days = st.slider("Days Since Last Login", 0, 100, 5)
    region = st.selectbox("Region", ['North America', 'Europe', 'Asia', 'South America', 'Africa'])
    device = st.selectbox("Device", ['Mobile', 'TV', 'Desktop', 'Tablet'])
    monthly_fee = st.number_input("Monthly Fee (INR)", min_value=100, max_value=2000, value=500)
    payment_method = st.selectbox("Payment Method", ['Credit Card', 'Debit Card', 'PayPal', 'UPI'])
    number_of_profiles = st.slider("Number of Profiles", 1, 6, 1)
    avg_watch_time_per_day = st.slider("Average Watch Time per Day (hours)", 0.0, 24.0, 2.0, step=0.5)
    favorite_genre = st.selectbox("Favorite Genre", ['Drama', 'Action', 'Comedy', 'Romance', 'Documentary', 'Horror'])

    submit = st.form_submit_button("Predict Churn")

if submit:
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'subscription_type': subscription_type,
        'watch_hours': watch_hours,
        'last_login_days': last_login_days,
        'region': region,
        'device': device,
        'monthly_fee': monthly_fee,
        'payment_method': payment_method,
        'number_of_profiles': number_of_profiles,
        'avg_watch_time_per_day': avg_watch_time_per_day,
        'favorite_genre': favorite_genre
    }])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=feature_cols, fill_value=0)
    input_scaled = scaler.transform(input_aligned)
    prediction = model.predict(input_scaled)[0]
    churn_prob = model.predict_proba(input_scaled)[0][1]  # assuming 1 = churn
    stay_prob = 1 - churn_prob

    st.subheader("Churn Probability")
    st.progress(int(churn_prob * 100))  
    if churn_prob > 0.5:
        st.error(f"❌ Customer likely to leave (Churn: {churn_prob:.2%})")
    else:
        st.success(f"✅ Customer likely to stay (Stay: {1 - churn_prob:.2%})")
    

    labels = ['Staying', 'Churning']
    sizes = [1 - churn_prob, churn_prob]
    colors = ['#66b3ff', '#ff6666']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  
    st.pyplot(fig)
