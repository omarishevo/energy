import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load dataset from uploaded file
def load_data(uploaded_file):
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Ensure the 'Timestamp' column is properly converted to datetime format
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')  # Using 'coerce' to handle invalid date formats
    
    return df

# Streamlit file uploader widget
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load data from the uploaded file
    df = load_data(uploaded_file)

    # Sidebar
    st.sidebar.title("Filter Data")
    min_occupancy = st.sidebar.slider("Minimum Occupancy", 0, int(df['Occupancy_Count'].max()), 0)
    max_temp = st.sidebar.slider("Max Internal Temperature (°C)", 0, 50, 50)
    df_filtered = df[(df["Occupancy_Count"] >= min_occupancy) & (df["Temperature_C"] <= max_temp)]

    # Handle potential missing data in the filtered dataset
    df_filtered = df_filtered.dropna(subset=["Temperature_C", "Occupancy_Count", "HVAC_Power_Consumption_kWh", "Energy_Efficiency_%"])

    # Main UI
    st.title("🏠 HVAC Energy Optimization Dashboard")

    st.subheader("📊 Dataset Overview")
    st.write(df_filtered.head())

    st.line_chart(df_filtered[['HVAC_Power_Consumption_kWh', 'Energy_Efficiency_%']].reset_index(drop=True))

    # Model Training
    st.subheader("🔧 Predict Energy Efficiency")

    features = [
        "Temperature_C", "Humidity_%", "CO2_ppm", "Occupancy_Count",
        "External_Temperature_C", "Kp", "Ki", "Kd", "Fuzzy_Adjustment_Factor",
        "ISA_Optimization_Score", "HVAC_Power_Consumption_kWh", "Cooling_Heating_Output_C",
        "Response_Time_s"
    ]
    X = df_filtered[features]
    y = df_filtered["Energy_Efficiency_%"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    st.write(f"✅ Mean Absolute Error on Test Data: {mae:.2f}")

    # User Input Prediction
    st.subheader("📈 Predict Efficiency for Custom Input")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Internal Temp (°C)", value=24.0)
            humidity = st.number_input("Humidity (%)", value=45.0)
            co2 = st.number_input("CO2 (ppm)", value=700.0)
            occupancy = st.number_input("Occupancy Count", value=10)
            external_temp = st.number_input("External Temp (°C)", value=15.0)
            kp = st.number_input("Kp", value=1.0)
            ki = st.number_input("Ki", value=0.5)
            kd = st.number_input("Kd", value=0.3)
        with col2:
            fuzzy = st.number_input("Fuzzy Factor", value=1.0)
            isa_score = st.number_input("ISA Score", value=70.0)
            hvac_power = st.number_input("HVAC Power (kWh)", value=6.0)
            output_c = st.number_input("Heating/Cooling Output", value=0.5)
            response_time = st.number_input("Response Time (s)", value=10.0)

        submitted = st.form_submit_button("Predict Efficiency")

        if submitted:
            input_data = pd.DataFrame([[temperature, humidity, co2, occupancy, external_temp, kp, ki, kd, fuzzy, isa_score, hvac_power, output_c, response_time]], columns=features)

            prediction = model.predict(input_data)[0]
            st.success(f"🔋 Predicted Energy Efficiency: {prediction:.2f}%")

    # Footer
    st.markdown("---")
    st.caption("Built for smart energy optimization using HVAC operational data.")

else:
    st.warning("Please upload a CSV file to get started.")
