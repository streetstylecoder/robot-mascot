import streamlit as st
import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to perform similarity analysis
def perform_similarity_analysis(resume_text, jd_text):
    text = [resume_text, jd_text]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(text)

    # Calculate cosine similarity
    match_percentage = cosine_similarity(count_matrix)[0][1] * 100
    match_percentage = round(match_percentage, 2)

    return match_percentage

# Function to get text from an uploaded file
def get_text_from_uploaded_file(uploaded_file):
    reader = PdfReader(uploaded_file)
    page = reader.pages[0]
    text = page.extract_text()
    return text

# Function to get text from a PDF file given its path
def get_text_from_pdf_path(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        page = reader.pages[0]
        text = page.extract_text()
    return text

def generate_tile(jd_name, similarity_score, jd_path):
    # Apply CSS styling to the tile
    tile_style = """
        border: 1px solid #ccc;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #252730;
        border-radius: 10px;
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: space-between;
    """
    
    # Display a rectangular tile with updated CSS
    st.markdown(
        f'<div style="{tile_style}">'
        f'<div style="color: white;"><strong>JD:</strong> {jd_name}</div>'
        f'<div style="color: white;"><strong>Similarity Score:</strong> {similarity_score}%</div>'
        f'<a style="color: #3498db;" href="{jd_path}" download="JD_{jd_name}"><strong>Download JD</strong></a>'
        f'<a href="https://mail.google.com/" target="_blank"><button style="background-color: #4CAF50; color: white; padding: 5px 10px; border: none; border-radius: 5px;">Apply</button></a>'

        f'</div>',
        unsafe_allow_html=True
    )

    # Add some spacing
    st.write("")  # Empty line for spacing

def compare_resume_jd():
    st.title("Resume-JD Comparator")
    st.write("This tool helps compare resumes with job descriptions.")

    # File picker for the resume
    uploaded_resume = st.file_uploader("Upload Resume (PDF)", type="pdf")

    # Local folder for JDs
    jd_folder = "jd_uploads"
    jd_files = [file for file in os.listdir(jd_folder) if file.endswith(".pdf")]

    # Submit button
    if st.button("Compare with the Job Listings"):
        if uploaded_resume is not None:
            # Perform comparison with each JD and display rectangular tiles
            with st.spinner("Comparing..."):
                resume_text = get_text_from_uploaded_file(uploaded_resume)
                for jd_file in jd_files:
                    print("working")
                    jd_path = os.path.join(jd_folder, jd_file)
                    jd_text = get_text_from_pdf_path(jd_path)
                    similarity_score = perform_similarity_analysis(resume_text, jd_text)
                    generate_tile(jd_file, similarity_score, jd_path)


compare_resume_jd()