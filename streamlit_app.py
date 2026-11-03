
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Smart Outcome Predictor", page_icon="🎓", layout="centered")
st.title("🎓 Smart Outcome Predictor")
st.write("Estimate course completion and final performance from student engagement details.")

completion_model = joblib.load("best_completion_model.joblib")
score_model = joblib.load("best_final_score_model.joblib")

st.subheader("Enter student details")
age = st.number_input("Age", min_value=10, max_value=100, value=25)
country_region = st.selectbox("Country / region", ["Africa", "Americas", "Asia", "Europe", "Oceania"])
device_type = st.selectbox("Device type", ["Laptop", "Mobile", "Tablet", "Desktop"])
education_background = st.selectbox("Education background", ["HighSchool", "Undergrad", "Graduate", "WorkingPro"])
course_level = st.selectbox("Course level", ["Beginner", "Intermediate", "Advanced"])
course_category = st.selectbox("Course category", ["Business", "Data", "Design", "Marketing", "Programming"])
week_of_year = st.slider("Week of year", 1, 53, 20)
sessions = st.number_input("Sessions", min_value=0, value=8)
time_spent_hours = st.number_input("Time spent (hours)", min_value=0.0, value=15.0)
videos_watched = st.number_input("Videos watched", min_value=0, value=10)
quiz_attempts = st.number_input("Quiz attempts", min_value=0, value=4)
assignments_submitted = st.number_input("Assignments submitted", min_value=0, value=4)
forum_posts = st.number_input("Forum posts", min_value=0, value=2)
avg_quiz_score = st.slider("Average quiz score", 0.0, 100.0, 70.0)
attendance_rate = st.slider("Attendance rate", 0.0, 1.0, 0.8)

if st.button("Predict outcome"):
    student = pd.DataFrame([{
        "age": age, "country_region": country_region, "device_type": device_type,
        "education_background": education_background, "course_level": course_level,
        "course_category": course_category, "week_of_year": week_of_year,
        "sessions": sessions, "time_spent_hours": time_spent_hours,
        "videos_watched": videos_watched, "quiz_attempts": quiz_attempts,
        "assignments_submitted": assignments_submitted, "forum_posts": forum_posts,
        "avg_quiz_score": avg_quiz_score, "attendance_rate": attendance_rate
    }])
    completion = int(completion_model.predict(student)[0])
    score = float(score_model.predict(student)[0])
    st.metric("Predicted course completion", "Completed" if completion == 1 else "Not completed")
    st.metric("Predicted final score", f"{score:.1f} / 100")
    st.caption("Predictions are estimates, not guarantees. Use them to guide supportive interventions.")
