import joblib
import pandas as pd
import streamlit as st

model = joblib.load("student_performance_model.joblib")

st.title("Student Performance Prediction App")

st.write("Nicher information diye predict korbo je student final exam e pass korbe naki fail.")


sex = st.selectbox("Sex", ["M", "F"])
age = st.slider("Age", 15, 22, 17)
studytime = st.selectbox(
    "Weekly study time (code)",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "<2 hours",
        2: "2–5 hours",
        3: "5–10 hours",
        4: ">10 hours"
    }[x]
)
failures = st.slider("Past class failures", 0, 3, 0)
absences = st.slider("Number of absences", 0, 93, 4)
goout = st.slider("Going out with friends (1–5)", 1, 5, 3)
health = st.slider("Health status (1–5)", 1, 5, 3)

if st.button("Predict"):
    
    data = {
        "sex": [sex],
        "age": [age],
        "studytime": [studytime],
        "failures": [failures],
        "absences": [absences],
        "goout": [goout],
        "health": [health],
    }

    input_df = pd.DataFrame(data)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success(f"Prediction: PASS (probability ≈ {proba*100:.1f}%)")
    else:
        st.error(f"Prediction: FAIL (probability ≈ {proba*100:.1f}%)")
