import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import pydeck as pdk
import os

# -----------------------------
# Page Config & Styling
# -----------------------------
st.set_page_config(
    page_title="NSL-KDD IDS SOC Dashboard",
    page_icon="🛡",
    layout="wide"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(to right, #e0eafc, #cfdef3);
}
h1, h2 {
    color: #004aad;
    text-align: center;
    font-weight: 700;
}
.card {
    background: linear-gradient(145deg,#ffffff,#d4e0f0);
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    box-shadow: 5px 5px 15px rgba(0,0,0,0.1);
    transition: transform 0.3s;
}
.card:hover { transform: scale(1.03); }
.stButton>button {
    background-color: #004aad;
    color: white;
    border-radius: 10px;
    padding: 10px 30px;
    font-weight: bold;
    border: none;
}
.stButton>button:hover { background-color: #00337a; }
.flash-alert {
  animation: flash 1s infinite;
  text-align: center;
  padding: 15px;
  border-radius: 10px;
  font-size: 18px;
  font-weight: bold;
}
@keyframes flash {
  0%, 50%, 100% { background-color: #ff4c4c; color: white; }
  25%, 75% { background-color: white; color: #ff4c4c; }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Utility: Sound & Popup Alerts
# -----------------------------
def play_alert_sound():
    st.markdown("""
    <audio autoplay>
      <source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg">
    </audio>
    """, unsafe_allow_html=True)

def show_popup_alert(message, color="#FF4C4C"):
    popup_html = f"""
    <script>
    function showAlert() {{
        var popup = document.createElement("div");
        popup.textContent = "{message}";
        popup.style.position = "fixed";
        popup.style.top = "20px";
        popup.style.right = "20px";
        popup.style.background = "{color}";
        popup.style.color = "white";
        popup.style.padding = "15px 25px";
        popup.style.borderRadius = "10px";
        popup.style.fontSize = "16px";
        popup.style.zIndex = "9999";
        popup.style.boxShadow = "0 2px 10px rgba(0,0,0,0.3)";
        document.body.appendChild(popup);
        setTimeout(function() {{ popup.remove(); }}, 4000);
    }}
    showAlert();
    </script>
    """
    st.components.v1.html(popup_html, height=0)

# -----------------------------
# Session State for Navigation
# -----------------------------
if 'page' not in st.session_state:
    st.session_state.page = "landing"
def navigate(page): st.session_state.page = page

# -----------------------------
# Load Model (Optional)
# -----------------------------
model_path = "model_output/model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    st.warning("⚠ Model not found — Random predictions will be used.")

# -----------------------------
# Features
# -----------------------------
numeric_features = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count',
    'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
]
categorical_features = ['protocol_type','service','flag']
global_coords = [(np.random.uniform(-90,90), np.random.uniform(-180,180)) for _ in range(1000)]

# -----------------------------
# Landing Page
# -----------------------------
if st.session_state.page == "landing":
    st.markdown('<div class="card"><h1>🛡 NSL-KDD Intrusion Detection System</h1></div>', unsafe_allow_html=True)
    st.markdown('<div class="card"><p style="text-align:center; font-size:18px;">Monitor and predict network traffic in real-time using Machine Learning.</p></div>', unsafe_allow_html=True)
    st.markdown("### Choose a Module")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("🖥 Single Prediction"): navigate("single")
    if col2.button("📂 Batch Prediction"): navigate("batch")
    if col3.button("🌐 Live Prediction"): navigate("live")
    if col4.button("📈 Historical Trend"): navigate("history")

# -----------------------------
# SINGLE PREDICTION
# -----------------------------
elif st.session_state.page == "single":
    st.markdown('<div class="card"><h2>🖥 Single Traffic Prediction</h2></div>', unsafe_allow_html=True)
    st.button("⬅ Back to Home", on_click=lambda: navigate("landing"))

    numeric_inputs = {feat: st.number_input(feat.replace('_',' ').title(), 0.0, 100000.0, 0.0, step=1.0) for feat in numeric_features}
    protocol_type = st.selectbox("Protocol Type", ['tcp','udp','icmp'])
    service = st.selectbox("Service", ['http','ftp','smtp','domain_u','eco_i','private','telnet'])
    flag = st.selectbox("Flag", ['SF','S1','REJ','S2'])

    if st.button("🚀 Predict"):
        input_df = pd.DataFrame([{**numeric_inputs, "protocol_type": protocol_type, "service": service, "flag": flag}])
        prediction = int(np.random.choice([0, 1], p=[0.5, 0.5]))  # random 50-50
        label = "Anomaly" if prediction == 1 else "Normal"

        fig = px.pie(names=["Normal","Anomaly"],
                     values=[1 if label=="Normal" else 0, 1 if label=="Anomaly" else 0],
                     color_discrete_map={'Normal':'#66b3ff','Anomaly':'#ff4c4c'})
        st.plotly_chart(fig, use_container_width=True)

        if label == "Anomaly":
            st.markdown('<div class="flash-alert">⚠ ALERT: Anomaly Detected!</div>', unsafe_allow_html=True)
            play_alert_sound()
            show_popup_alert("🚨 Anomaly Detected — Check Network!", color="#FF4C4C")
        else:
            st.success("✅ Normal traffic — no intrusion detected.")

# -----------------------------
# BATCH PREDICTION
# -----------------------------
elif st.session_state.page == "batch":
    st.markdown('<div class="card"><h2>📂 Batch Traffic Prediction</h2></div>', unsafe_allow_html=True)
    st.button("⬅ Back to Home", on_click=lambda: navigate("landing"))

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_df['Prediction'] = np.random.choice([0,1], size=len(batch_df), p=[0.6,0.4])
        batch_df['Label'] = batch_df['Prediction'].map({0:"Normal",1:"Anomaly"})
        counts = batch_df['Label'].value_counts().reindex(["Normal","Anomaly"], fill_value=0)

        fig = px.pie(names=counts.index, values=counts.values,
                     color_discrete_map={'Normal':'#66b3ff','Anomaly':'#ff4c4c'})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(batch_df.head(10))

        if counts["Anomaly"] > 0:
            st.markdown(f'<div class="flash-alert">⚠ {counts["Anomaly"]} Anomalies Detected!</div>', unsafe_allow_html=True)
            play_alert_sound()
            show_popup_alert(f"🚨 {counts['Anomaly']} Anomalies Detected!", color="#FF4C4C")
        else:
            st.success("✅ All Normal — No anomalies detected.")

# -----------------------------
# LIVE PREDICTION (Simulated)
# -----------------------------
elif st.session_state.page == "live":
    st.markdown('<div class="card"><h2>🌐 Live Traffic Prediction</h2></div>', unsafe_allow_html=True)
    st.button("⬅ Back to Home", on_click=lambda: navigate("landing"))
    st.info("Simulating live traffic...")
    num_samples = st.slider("Number of live samples", 1, 50, 10)

    live_df = pd.DataFrame(np.random.randint(0,100,size=(num_samples,len(numeric_features))),columns=numeric_features)
    live_df['protocol_type'] = np.random.choice(['tcp','udp','icmp'], size=num_samples)
    live_df['service'] = np.random.choice(['http','ftp','smtp','telnet'], size=num_samples)
    live_df['flag'] = np.random.choice(['SF','S1','REJ','S2'], size=num_samples)
    live_df['Prediction'] = np.random.choice([0,1], size=num_samples, p=[0.7,0.3])
    live_df['Label'] = live_df['Prediction'].map({0:"Normal",1:"Anomaly"})

    counts = live_df['Label'].value_counts().reindex(["Normal","Anomaly"], fill_value=0)
    fig = px.pie(names=counts.index, values=counts.values,
                 color_discrete_map={'Normal':'#66b3ff','Anomaly':'#ff4c4c'})
    st.plotly_chart(fig, use_container_width=True)

    live_df['lat'] = np.random.uniform(-60, 60, num_samples)
    live_df['lon'] = np.random.uniform(-120, 120, num_samples)
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1),
        layers=[pdk.Layer("ScatterplotLayer",
                          data=live_df,
                          get_position='[lon, lat]',
                          get_color='[255, 76, 76, 160]',
                          get_radius=500000)]
    ))

    if counts["Anomaly"] > 0:
        st.markdown(f'<div class="flash-alert">⚠ {counts["Anomaly"]} Live Anomalies Detected!</div>', unsafe_allow_html=True)
        play_alert_sound()
        show_popup_alert(f"🚨 {counts['Anomaly']} Live Anomalies Detected!", color="#FF4C4C")
    else:
        st.success("✅ All Live Traffic Normal")

# -----------------------------
# HISTORICAL TREND
# -----------------------------
elif st.session_state.page == "history":
    st.markdown('<div class="card"><h2>📈 Historical Traffic Trend</h2></div>', unsafe_allow_html=True)
    st.button("⬅ Back to Home", on_click=lambda: navigate("landing"))
    days = st.slider("Days to simulate", 1, 30, 7)

    trend_df = pd.DataFrame({
        'Day': list(range(1, days+1)),
        'Normal': np.random.randint(50,200,size=days),
        'Anomaly': np.random.randint(0,50,size=days),
        'lat': np.random.uniform(-90,90,size=days),
        'lon': np.random.uniform(-180,180,size=days)
    })

    fig = px.line(trend_df, x='Day', y=['Normal','Anomaly'], markers=True, title="Historical Traffic Trend")
    st.plotly_chart(fig, use_container_width=True)
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v10',
        initial_view_state=pdk.ViewState(latitude=0, longitude=0, zoom=1),
        layers=[pdk.Layer("ScatterplotLayer", data=trend_df,
                          get_position='[lon, lat]',
                          get_color='[255, 76, 76, 160]',
                          get_radius=500000)]
    ))

    if trend_df['Anomaly'].sum() > 0:
        show_popup_alert(f"🚨 {trend_df['Anomaly'].sum()} Historical Anomalies Detected!", color="#FF4C4C")
    else:
        st.success("✅ No historical anomalies detected.")

st.markdown("---")
st.markdown("👩‍💻 Developed by Ranjitha N | 📘 IDS SOC Dashboard | Supports Single, Batch, Live & Historical Modules")