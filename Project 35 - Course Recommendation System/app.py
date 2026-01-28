import streamlit as st
import pandas as pd
import warnings

warnings.simplefilter("ignore")

data = pd.read_csv('traineddata.csv')

st.title("Course Recommendation System")
st.write("**Enter A Course Name To Get Recommendations:**")
st.caption(
    "***This is a keyword-based recommendation system built using pre-trained course data — "
    "no Machine Learning model is used. Enter keywords like Python, Excel, JavaScript, etc. "
    "to discover relevant courses.***"
)

course_name = st.text_input("Course Name Input", placeholder="Type Your Course Name Here", key="course_input", label_visibility="hidden")

if st.button("Recommend Course"):
    if course_name.strip() == "":
        st.error("**Please Enter A Course Name To Get Recommendations.**")
    else:
        recommended_courses = data[data['Clean_Title'].str.contains(course_name, case=False, na=False)]
        
        recommended_courses = recommended_courses.sort_values(by='num_reviews', ascending=False)
        
        if not recommended_courses.empty:
            st.write(f"**Top Course Recommendations For '{course_name}':**")
            for index, row in recommended_courses.head(6).iterrows():
                st.markdown(
                    f""" 
                    <div style="border: 2px solid black; border-radius: 10px; padding: 10px; margin: 10px 0; background-color: #000000;">
                    <h3 style="margin-bottom: 5px;">{row['Clean_Title']}</h3>
                    <p><strong>Subscribers:</strong> {row['num_subscribers']} | <strong>Reviews:</strong> {row['num_reviews']}</p>
                    <p><strong>Level:</strong> {row['level']} | <strong>Duration:</strong> {row['content_duration']}</p>
                    <p><strong>Price:</strong> {'Free' if not row['is_paid'] else f'${row["price"]}'}</p>
                    <p><strong>Subject:</strong> {row['subject']}</p>
                    <a href="{row['url']}" target="_blank" style="color: yellow; background-color: green; padding: 8px 12px; text-decoration: none; border-radius: 5px;">View Course</a>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
        else:
            st.error(f"**Sorry, No Recommended Course Found For '{course_name}'**")

