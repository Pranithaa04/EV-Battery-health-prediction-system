import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("battery_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔋 EV Battery Health Checker")

# User inputs via sliders
avg_temp = st.slider("Average outdoor temperature (°C)", 10, 60, 30)
charge_cycles = st.slider("Charging cycles completed", 0, 2000, 500)
daily_km = st.slider("Daily distance driven (km)", 0, 300, 100)
fast_charging_ratio = st.slider("Percentage of fast charging", 0.0, 1.0, 0.3)
idle_days = st.slider("Idle days per month", 0, 30, 2)
battery_age = st.slider("Battery age (months)", 0, 120, 24)

# Predict button
if st.button("Check Battery Health"):
    input_data = pd.DataFrame([{
        "avg_temperature": avg_temp,
        "charge_cycles": charge_cycles,
        "daily_distance_km": daily_km,
        "fast_charging_ratio": fast_charging_ratio,
        "idle_days": idle_days,
        "battery_age_months": battery_age
    }])

    prediction = model.predict(input_data)[0]

    if prediction == 0:
        st.success("✅ Your battery is Healthy.")
    elif prediction == 1:
        st.warning("⚠️ Your battery is Degrading. Plan a checkup soon.")
    else:
        st.error("🚨 Critical: Battery failure predicted! Take action.")
