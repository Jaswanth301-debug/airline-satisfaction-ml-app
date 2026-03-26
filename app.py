import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load model files
# ---------------------------

model = joblib.load("model.pkl")
preprocess = joblib.load("preprocessing.pkl")

encoder = preprocess["encoder"]
scaler = preprocess["scaler"]
columns = preprocess["columns"]
cat_cols = preprocess["cat_cols"]
num_cols = preprocess["num_cols"]

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Airline Predictor", layout="wide")

# ---------------------------
# Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(to right, #1f77b4, #4facfe);
    border-radius: 10px;
    color: white;
}
.card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.stButton>button {
    background: linear-gradient(to right, #1f77b4, #4facfe);
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header
# ---------------------------
st.markdown("""
<div class="header">
    <h1>✈ Airline Satisfaction Predictor</h1>
    <p>Smart AI system to predict passenger experience</p>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Layout (Cards)
# ---------------------------
col1, col2 = st.columns(2)

# ---------------------------
# Passenger Card
# ---------------------------
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧍 Passenger Details")

    age = st.number_input("Age", 7, 85, 30)
    flight_distance = st.number_input("Flight Distance", 50, 5000, 500)

    gender = st.selectbox("Gender", ["Male", "Female"])
    travel_type = st.radio("Travel Type", ["Business travel", "Personal Travel"])
    class_type = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Service Card
# ---------------------------
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🛫 Service Experience")

    wifi = st.slider("Wifi Service", 0, 5, 3)
    online_boarding = st.slider("Online Boarding", 0, 5, 3)
    seat_comfort = st.slider("Seat Comfort", 0, 5, 3)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# Input Data
# ---------------------------
input_data = pd.DataFrame({
    "Age": [age],
    "Flight Distance": [flight_distance],
    "Gender": [gender],
    "Type of Travel": [travel_type],
    "Class": [class_type],
    "Inflight wifi service": [wifi],
    "Online boarding": [online_boarding],
    "Seat comfort": [seat_comfort]
})

# ---------------------------
# Predict Button
# ---------------------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 Predict Satisfaction", use_container_width=True):

    # Fill missing columns
    for col in num_cols:
        if col not in input_data.columns:
            input_data[col] = 0

    for col in cat_cols:
        if col not in input_data.columns:
            input_data[col] = "Unknown"

    input_data = input_data.reindex(columns=list(num_cols) + list(cat_cols), fill_value=0)

    # Encode & Scale
    X_cat = encoder.transform(input_data[cat_cols])
    X_cat = pd.DataFrame(X_cat)

    X_num = scaler.transform(input_data[num_cols])
    X_num = pd.DataFrame(X_num, columns=num_cols)

    X_final = pd.concat([X_num, X_cat], axis=1)
    X_final = X_final.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(X_final)[0]
    probability = model.predict_proba(X_final)[0][1]

    # ---------------------------
    # Result Card
    # ---------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.success("😊 Passenger is Satisfied")
        st.progress(int(probability * 100))
    else:
        st.error("😞 Passenger is Dissatisfied")
        st.progress(int((1 - probability) * 100))

    st.metric("Confidence Score", f"{probability:.2f}")

    st.markdown("</div>", unsafe_allow_html=True)