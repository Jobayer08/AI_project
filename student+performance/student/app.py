import joblib
import pandas as pd
import streamlit as st


model = joblib.load("student_performance_model.joblib")

st.title("Student Performance Prediction App")
st.write(
    "Nicher information diye predict korbo je student final exam e pass korbe naki fail."
)



school = st.selectbox("School", ["GP", "MS"])
sex = st.selectbox("Sex", ["M", "F"])
address = st.selectbox("Address", ["U", "R"])  
famsize = st.selectbox("Family size", ["LE3", "GT3"])
Pstatus = st.selectbox("Parent cohabitation status", ["T", "A"])

Mjob = st.selectbox(
    "Mother's job",
    ["teacher", "health", "services", "at_home", "other"]
)
Fjob = st.selectbox(
    "Father's job",
    ["teacher", "health", "services", "at_home", "other"]
)

reason = st.selectbox(
    "Reason to choose school",
    ["home", "reputation", "course", "other"]
)

guardian = st.selectbox("Guardian", ["mother", "father", "other"])

schoolsup = st.selectbox("Extra educational support (schoolsup)", ["yes", "no"])
famsup = st.selectbox("Family educational support (famsup)", ["yes", "no"])
paid = st.selectbox("Extra paid classes (Math)", ["yes", "no"])
activities = st.selectbox("Extra-curricular activities", ["yes", "no"])
nursery = st.selectbox("Attended nursery school", ["yes", "no"])
higher = st.selectbox("Wants higher education", ["yes", "no"])
internet = st.selectbox("Internet access at home", ["yes", "no"])
romantic = st.selectbox("In a romantic relationship", ["yes", "no"])



age = st.slider("Age", 15, 22, 17)
Medu = st.slider("Mother's education (0-4)", 0, 4, 2)
Fedu = st.slider("Father's education (0-4)", 0, 4, 2)
traveltime = st.slider("Home to school travel time (1-4)", 1, 4, 1)
studytime = st.slider("Weekly study time (1-4)", 1, 4, 2)
failures = st.slider("Past class failures (0-4)", 0, 4, 0)
famrel = st.slider("Family relationship quality (1-5)", 1, 5, 4)
freetime = st.slider("Free time after school (1-5)", 1, 5, 3)
goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
Dalc = st.slider("Workday alcohol consumption (1-5)", 1, 5, 1)
Walc = st.slider("Weekend alcohol consumption (1-5)", 1, 5, 2)
health = st.slider("Current health status (1-5)", 1, 5, 3)
absences = st.slider("Number of school absences", 0, 93, 4)


if st.button("Predict"):
    data = {
    
        "age": [age],
        "Medu": [Medu],
        "Fedu": [Fedu],
        "traveltime": [traveltime],
        "studytime": [studytime],
        "failures": [failures],
        "famrel": [famrel],
        "freetime": [freetime],
        "goout": [goout],
        "Dalc": [Dalc],
        "Walc": [Walc],
        "health": [health],
        "absences": [absences],

        
        "school": [school],
        "sex": [sex],
        "address": [address],
        "famsize": [famsize],
        "Pstatus": [Pstatus],
        "Mjob": [Mjob],
        "Fjob": [Fjob],
        "reason": [reason],
        "guardian": [guardian],
        "schoolsup": [schoolsup],
        "famsup": [famsup],
        "paid": [paid],
        "activities": [activities],
        "nursery": [nursery],
        "higher": [higher],
        "internet": [internet],
        "romantic": [romantic],
    }

    
    input_df = pd.DataFrame(data)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    fail_prob = proba[0]
    pass_prob = proba[1]

    if pred == 1:
        st.success("Final prediction: PASS")
    else:
        st.error("Final prediction: FAIL")

    
    st.write(f"Pass probability: **{pass_prob*100:.2f}%**")
    st.write(f"Fail probability: **{fail_prob*100:.2f}%**")
