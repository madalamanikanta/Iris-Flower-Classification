import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
st.set_page_config(page_title="Iris Flower Classifier", layout="centered")
st.title("Iris Flower Classification (KNN Model)")
st.markdown("""
This app predicts the **type of Iris flower** using your trained **KNN (k=5)** model.  
The model achieves about **97% accuracy** on the test data.
""")
MODEL_PATH = "knn_model.pkl"
SCALER_PATH = "scaler.pkl"
@st.cache_data
def load_data():
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["target_name"] = df["target"].map({i: n for i, n in enumerate(iris.target_names)})
    return df, iris

df, iris = load_data()
def train_model():
    X = df[iris.feature_names]
    y = df["target"]
    scaler=StandardScaler()
    X=scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train , y_train)

    acc = knn.score(X_test, y_test)

    # Save model and scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(knn, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    return knn, scaler, acc

def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = pickle.load(open(MODEL_PATH, "rb"))
        scaler = pickle.load(open(SCALER_PATH, "rb"))
        return model, scaler
    else:
        return None, None
st.sidebar.header("Options")
retrain = st.sidebar.button("Retrain Model")
show_data = st.sidebar.checkbox("Show Dataset")

if retrain:
    model, scaler, acc = train_model()
    st.sidebar.success(f"Model retrained successfully! Accuracy: {acc*100:.2f}%")
else:
    model, scaler = load_model()
    if model is None:
        st.sidebar.warning("Model not found. Training a new one...")
        model, scaler, acc = train_model()
        st.sidebar.success(f"Model trained. Accuracy:{acc*100:.2f}%")
    else:
        st.sidebar.info("Loaded existing trained model.")

if show_data:
    st.subheader("Complete Iris Dataset")
    st.dataframe(df)
st.subheader("Make a Prediction")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)
with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

if st.button("Predict Species"):
    X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    X_scaled = scaler.transform(X_input)
    pred_idx = model.predict(X_scaled)[0]
    pred_name = iris.target_names[pred_idx]
    proba = model.predict_proba(X_scaled)[0]

    st.success(f"### Predicted Species: **{pred_name.capitalize()}**")
    st.write("#### Prediction Probabilities:")
    proba_df = pd.DataFrame({
        "Species": iris.target_names,
        "Probability": np.round(proba, 3)
    }).sort_values("Probability", ascending=False)
    st.table(proba_df)
st.markdown("---")
st.subheader("Check Model Accuracy")

if st.button("Show Test Accuracy"):
    model, scaler, acc = train_model()
    st.success(f"Model retrained successfully! Accuracy: {acc*100:.2f}%")
st.markdown("""
---
### 👨‍💻 About This App   
Built with **Streamlit + scikit-learn (KNN)**  
""")
