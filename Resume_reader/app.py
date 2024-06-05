import streamlit as st
import uuid


if 'unique_id' not in st.session_state:
    st.session_state['unique_id'] = ''

#def main():

st.set_page_config(page_title="Resume Screening Assistance")
st.title("HR - Resume Screening Assistance...")
st.subheader("I can help you in resume screening process")

job_description = st.text_area("Please paste the 'JOB DESCRIPTION' here...",
                                   key="1")
document_count = st.text_input("No.of 'RESUMES' to return", key="2")
pdf = st.file_uploader("Upload resumes here, only PDF files allowed",
                           type=["pdf"],
                           accept_multiple_files=True)
    
submit = st.button("Help me with the analysis")

if submit:
    with st.spinner('Wait for it...'):

        st.write("Our process")

        st.session_state['unique_id']=uuid.uuid4().hex
        st.write(st.session_state['unique_id'])

st.success("Hope I was able to save your time")


