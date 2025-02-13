import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_folium import st_folium
import folium

@st.cache_resource
def load_model():
    try:
        return joblib.load("xgboost_model.pkl") 
    except FileNotFoundError:
        st.error("🚨 Model file not found. Please check the path.")
        return None

def predict_output(model, input_data):
    if not model:
        return "No model loaded"
    prediction = model.predict([input_data])
    return prediction[0]  # Extract single value from array

def generate_energy_demand():
    hours = np.arange(24)
    demand = np.random.randint(100, 1000, size=24) 
    return pd.DataFrame({"Hour": hours, "Demand (MW)": demand})

def main():
    st.title("⚡ Smart Grid Management")
    st.markdown("### Distribution of the energy based on prediction")
    
    user_input_demand = st.slider("Adjust Demand (MW)", min_value=0, max_value=1000, value=500)
    energy_demand_df = generate_energy_demand()
    
    if st.button("📊 Show Hourly Energy Demand Graph"):
        st.line_chart(energy_demand_df.set_index("Hour"))
    
    st.subheader("🗺️ Select Power Plant Location")
    m = folium.Map(location=[12.5, 79.5], zoom_start=6)
    map_click = st_folium(m, height=400, width=700)
    
    model = load_model()
    
    st.subheader("🔮 Provide Dynamic Input")
    temperature_setting_C = st.number_input("Temperature Setting (°C)", min_value=-10.0, max_value=50.0, value=22.0, step=0.1)
    occupancy_status = st.selectbox("Occupancy Status", options=["Occupied", "Unoccupied"])
    appliance = st.selectbox("Appliance", options=["AC", "Heater", "Washing Machine", "Lighting", "Refrigerator"])
    usage_duration_minutes = st.number_input("Usage Duration (minutes)", min_value=1, max_value=180, value=60)
    season = st.selectbox("Season", options=["Winter", "Spring", "Summer", "Autumn"])
    day_of_week = st.selectbox("Day of Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    
    # Encoding categorical inputs
    occupancy_status = 1 if occupancy_status == "Occupied" else 0
    appliance_mapping = {"AC": 1, "Heater": 2, "Washing Machine": 3, "Lighting": 4, "Refrigerator": 5}
    appliance = appliance_mapping[appliance]
    season_mapping = {"Winter": 1, "Spring": 2, "Summer": 3, "Autumn": 4}
    season = season_mapping[season]
    day_of_week_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    day_of_week = day_of_week_mapping[day_of_week]
    
    user_input = [temperature_setting_C, occupancy_status, appliance, usage_duration_minutes, season, day_of_week]
    
    if "predicted_output" not in st.session_state:
        st.session_state.predicted_output = None
    
    if st.button("🚀 Predict Energy Output"):
        with st.spinner("🔄 Generating prediction..."):
            time.sleep(2)  # Simulate processing time
            st.session_state.predicted_output = predict_output(model, user_input)
        st.success(f"✅ Prediction: {st.session_state.predicted_output:.2f} kWh")
    
    if st.session_state.predicted_output is not None:
        st.subheader("📊 Display Feature Graphs")
        
        features = [
            ("Temperature Setting vs Energy Consumption", temperature_setting_C),
            ("Occupancy Status vs Energy Consumption", occupancy_status),
            ("Appliance vs Energy Consumption", appliance),
            ("Usage Duration vs Energy Consumption", usage_duration_minutes),
            ("Season vs Energy Consumption", season),
            ("Day of Week vs Energy Consumption", day_of_week)
        ]
        
        for title, feature in features:
            fig, ax = plt.subplots()
            sns.scatterplot(x=[feature], y=[st.session_state.predicted_output], ax=ax)
            ax.set_title(title)
            st.pyplot(fig)
    
if __name__ == "__main__":
    main()

