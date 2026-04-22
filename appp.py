import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Churn App", layout="wide")

# -------------------------
# LOAD MODEL & DATA
# -------------------------
model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))
df = pd.read_csv("C:/Users/Ashwini/Downloads/archive (1)/Customer-Churn.csv")

# -------------------------
# TITLE
# -------------------------
st.title("📊 Customer Churn Prediction App")
st.write("🔍 Predict whether a customer will stay or leave")

# -------------------------
# KPI
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Churn Rate", f"{df['Churn'].value_counts(normalize=True)['Yes']*100:.1f}%")
col3.metric("Avg Monthly Charges", f"{df['MonthlyCharges'].mean():.2f}")

st.markdown("---")

# -------------------------
# DATA PREVIEW
# -------------------------
st.subheader("📊 Dataset Overview")
st.dataframe(df.head())

# -------------------------
# CHURN PIE
# -------------------------
st.subheader("📊 Churn Distribution")

fig1, ax1 = plt.subplots()
df["Churn"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax1)
st.pyplot(fig1)

# -------------------------
# BOXPLOT
# -------------------------
st.subheader("📈 Monthly Charges vs Churn")

fig2, ax2 = plt.subplots()
df.boxplot(column="MonthlyCharges", by="Churn", ax=ax2)
st.pyplot(fig2)

# -------------------------
# FEATURE IMPORTANCE
# -------------------------
st.subheader("⭐ Feature Importance")

importance = model.coef_[0]

imp_df = pd.DataFrame({
    "Feature": columns,
    "Importance": importance
}).sort_values(by="Importance", key=abs, ascending=False).head(10)

st.write("Top factors influencing churn")
st.bar_chart(imp_df.set_index("Feature"))

st.markdown("---")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("Enter Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.number_input("Monthly Charges", value=50.0)
total = st.sidebar.number_input("Total Charges", value=1000.0)

# -------------------------
# INPUT PREP
# -------------------------
input_dict = {col: 0 for col in columns}

input_dict["tenure"] = tenure
input_dict["MonthlyCharges"] = monthly
input_dict["TotalCharges"] = total

if gender == "Male" and "gender_Male" in input_dict:
    input_dict["gender_Male"] = 1

if partner == "Yes" and "Partner_Yes" in input_dict:
    input_dict["Partner_Yes"] = 1

if dependents == "Yes" and "Dependents_Yes" in input_dict:
    input_dict["Dependents_Yes"] = 1

if phone == "Yes" and "PhoneService_Yes" in input_dict:
    input_dict["PhoneService_Yes"] = 1

if internet == "Fiber optic" and "InternetService_Fiber optic" in input_dict:
    input_dict["InternetService_Fiber optic"] = 1
elif internet == "DSL" and "InternetService_DSL" in input_dict:
    input_dict["InternetService_DSL"] = 1

if contract == "One year" and "Contract_One year" in input_dict:
    input_dict["Contract_One year"] = 1
elif contract == "Two year" and "Contract_Two year" in input_dict:
    input_dict["Contract_Two year"] = 1

input_df = pd.DataFrame([input_dict])

# -------------------------
# PREDICTION
# -------------------------
st.subheader("🔮 Prediction")

if st.button("🔍 Predict Churn"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer will LEAVE\nConfidence: {prob:.2f}")
    else:
        st.success(f"✅ Customer will STAY\nConfidence: {1-prob:.2f}")