import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="Diamond Price Predictor - Deep Learning",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# CUSTOM CSS FOR BETTER UI
# ===========================
st.markdown("""
<style>
    .main-title {
        font-size: 3em;
        font-weight: bold;
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .section-divider {
        margin: 40px 0;
        border-top: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ===========================
# LOAD DATA & PREPROCESS
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("cubic_zirconia.csv")
    
    if "Unnamed: 0" in df.columns:
        df.drop("Unnamed: 0", axis=1, inplace=True)
    
    df.dropna(inplace=True)
    
    df["cut"] = df["cut"].map({
        "Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4
    })
    df["color"] = df["color"].map({
        "J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6
    })
    df["clarity"] = df["clarity"].map({
        "I1": 0, "SI2": 1, "SI1": 2, "VS2": 3,
        "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7
    })
    
    X = df.drop("price", axis=1)
    y = df["price"]
    
    return df, X, y

# ===========================
# TRAIN DEEP LEARNING MODEL (ANN)
# ===========================
@st.cache_resource
def train_dl_model():
    df, X, y = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    
    # Feature Scaling (Important for Deep Learning / Neural Networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Building Deep Learning Model (Multi-Layer Perceptron Neural Network)
    # Architecture: Input -> Hidden Layer 1 (64 neurons) -> Hidden Layer 2 (32 neurons) -> Output
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    prediction = model.predict(X_test_scaled)
    score = r2_score(y_test, prediction)
    
    return model, scaler, score

# Load data and model
df, X, y = load_data()
model, scaler, score = train_dl_model()

# ===========================
# HEADER SECTION
# ===========================
st.markdown('<div class="main-title">💎 Deep Learning Diamond Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by Artificial Neural Networks (ANN) & Deep Regression</div>', unsafe_allow_html=True)

# ===========================
# KEY METRICS
# ===========================
col1, col2, col3, col4 = st.columns(4, gap="medium")

metrics = [
    (col1, "📊 Dataset", f"{len(df):,}", "Records"),
    (col2, "🧠 Architecture", "64-32 Neurons", "Dense Layers"),
    (col3, "✅ ANN R² Score", f"{score:.3f}", "Accuracy"),
    (col4, "💰 Avg Price", f"${df['price'].mean():,.0f}", "USD")
]

for col, icon, value, label in metrics:
    with col:
        st.metric(icon, value, label)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ===========================
# MAIN INTERFACE WITH TABS
# ===========================
tab1, tab2, tab3, tab4 = st.tabs([
    "💎 DL Price Predictor",
    "📊 Data Analysis",
    "🧠 Neural Network Insights",
    "📄 Dataset"
])

# ========================
# TAB 1: PRICE PREDICTOR
# ========================
with tab1:
    st.subheader("Enter Diamond Details for Neural Network Prediction")
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.write("#### Quality Attributes")
        cut = st.selectbox("💎 Cut Quality", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        color = st.selectbox("🎨 Color Grade", ["D", "E", "F", "G", "H", "I", "J"], index=3)
        clarity = st.selectbox("✨ Clarity", ["IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"])
    
    with col2:
        st.write("#### Weight & Dimensions")
        carat = st.slider("⚖️ Carat Weight", 0.20, 5.00, 1.00, 0.01)
        depth = st.slider("📏 Depth %", 40.0, 80.0, 61.5, 0.1)
        table = st.slider("📐 Table %", 40.0, 100.0, 57.0, 0.1)
    
    st.write("#### Physical Dimensions (mm)")
    col_x, col_y, col_z = st.columns(3)
    with col_x:
        x = st.slider("Length (X)", 0.00, 15.00, 5.50, 0.01)
    with col_y:
        y = st.slider("Width (Y)", 0.00, 15.00, 5.50, 0.01)
    with col_z:
        z = st.slider("Height (Z)", 0.00, 10.00, 3.50, 0.01)
    
    cut_map = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
    color_map = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_map = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}
    
    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut_map[cut]],
        "color": [color_map[color]],
        "clarity": [clarity_map[clarity]],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })
    
    predict_button = st.button("🔮 PREDICT WITH DEEP LEARNING", use_container_width=True, type="primary")
    
    if predict_button:
        # Scale input data before passing to Neural Network
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        st.balloons()
        st.markdown(f'<div class="prediction-result">${prediction:,.2f}</div>', unsafe_allow_html=True)
        st.success(f"✅ Deep Neural Network prediction completed successfully! (Model Accuracy: {score:.1%})")

# ========================
# TAB 2: DATA ANALYSIS
# ========================
with tab2:
    st.subheader("Exploratory Data Analysis")
    fig1 = px.histogram(df, x="price", nbins=40, title="Distribution of Diamond Prices", color_discrete_sequence=["#ff416c"])
    st.plotly_chart(fig1, use_container_width=True)

# ========================
# TAB 3: MODEL INSIGHTS
# ========================
with tab3:
    st.subheader("Deep Learning Model Architecture & Loss Insights")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Neural Network Type", "Multi-Layer Perceptron")
    with col2:
        st.metric("Activation Function", "ReLU")
    with col3:
        st.metric("Optimizer", "Adam Optimizer")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.write("#### 📉 Training Loss Curve Trend")
    st.write("The network successfully minimized loss across iterations using backpropagation.")
    
    # Plotting training loss curve if available in model
    if hasattr(model, "loss_curve_"):
        loss_df = pd.DataFrame({"Iteration": range(len(model.loss_curve_)), "Loss": model.loss_curve_})
        fig_loss = px.line(loss_df, x="Iteration", y="Loss", title="ANN Training Loss Reduction over Iterations")
        st.plotly_chart(fig_loss, use_container_width=True)

# ========================
# TAB 4: DATASET PREVIEW
# ========================
with tab4:
    st.subheader("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

# ===========================
# FOOTER
# ===========================
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em;'>
💎 Deep Learning Diamond Pricing System | Built with Streamlit & Artificial Neural Networks (ANN)
</div>
""", unsafe_allow_html=True)
