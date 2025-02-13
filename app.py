import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_folium import st_folium
import folium

def generate_energy_demand():
    hours = np.arange(24)
    demand = np.random.randint(100, 1000, size=24) 
    return pd.DataFrame({"Hour": hours, "Demand (MW)": demand})
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
    if not input_data:
        return "Invalid input data"
    prediction = model.predict([input_data])
    return prediction
def get_user_input_from_dataset(dataset_path):
    try:
        df = pd.read_csv(dataset_path)
    except FileNotFoundError:
        st.error("🚨 Dataset file not found. Please check the path.")
        return []
    columns = ['temperature_setting_C', 'occupancy_status', 'appliance', 'usage_duration_minutes', 'season', 'day_of_week', 'energy_consumption_kWh']
    if df.empty:
        st.warning("⚠️ Dataset is empty!")
        return []
    input_values = df[columns].iloc[0].tolist()
    st.write("🔍 Raw input values:", input_values)
    cleaned_values = []
    for i, value in enumerate(input_values):
        try:
            cleaned_values.append(float(value))
        except (ValueError, TypeError):
            st.warning(f"⚠️ Invalid data in column '{columns[i]}': {value}. Replacing with 0.0")
            cleaned_values.append(0.0)
    return cleaned_values
def main():
    st.title("⚡ Smart Grid Management")
    st.markdown("### Distribution of the energy based on prediction")
    user_input_demand = st.slider("Adjust Demand (MW)", min_value=0, max_value=1000, value=500)
    energy_demand_df = generate_energy_demand()
    st.subheader("📊 Hourly Energy Demand")
    st.line_chart(energy_demand_df.set_index("Hour"))
    st.write(energy_demand_df)
    st.sidebar.header("⚙️ Settings")
    user_input_demand = st.sidebar.slider("Set Demand (MW)", min_value=100, max_value=1500, value=500)
    uploaded_file = st.sidebar.file_uploader("📂 Upload CSV Dataset", type=["csv"])
    
    st.subheader("🗺️ Select Power Plant Location")
    m = folium.Map(location=[12.5, 79.5], zoom_start=6)
    map_click = st_folium(m, height=400, width=700)
    if map_click and "last_clicked" in map_click:
        lat, lon = map_click["last_clicked"]["lat"], map_click["last_clicked"]["lng"]
    else:
        lat, lon = None, None 
        model = load_model()
    if st.button("🔄 Refresh Data"):
        energy_demand_df = generate_energy_demand()
        st.success("✅ Data refreshed successfully!")
        st.line_chart(energy_demand_df.set_index("Hour"))

    if st.button("🚀 Predict Energy Output"):
        with st.spinner("🔄 Generating prediction..."):
            time.sleep(2)  # Simulate processing time
            predicted_output = predict_output(model, user_input)
        st.success(f"✅ Prediction: {predicted_output} MW")

    if model is None:
        return
    dataset_path = "/home/imayavan/Downloads/smart-grid-management-main/models/smart_home_energy_usage_dataset.csv"
    user_input = get_user_input_from_dataset(dataset_path)
    st.subheader("🔮 Predicted Output")
    if st.button("Predict"):
        predicted_output = predict_output(model, user_input)
        st.write(f"**Predicted Output:** {predicted_output}")
if __name__ == "__main__":
    main()

