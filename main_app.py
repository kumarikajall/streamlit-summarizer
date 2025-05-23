import streamlit as st
from models.t5_summarizer import T5FileSummarizer
from models.bart_summarizer import BARTFileSummarizer
from models.pegasus_summarizer import PegasusFileSummarizer
import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(page_title="Multi-Model Summarizer", layout="wide")
st.title("📄 Multi-Model File Summarizer (T5, BART, Pegasus)")

model_choice = st.selectbox("Choose a summarization model", ["T5", "BART", "Pegasus"])
uploaded_file = st.file_uploader("Upload a file (PDF, DOCX, PPTX, or TXT)", type=["pdf", "docx", "pptx", "txt"])

max_length = st.slider("Max Summary Length", min_value=50, max_value=500, value=150)
min_length = st.slider("Min Summary Length", min_value=10, max_value=100, value=30)

if uploaded_file:
    upload_path = os.path.join("uploads", uploaded_file.name)
    with open(upload_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded {uploaded_file.name}")

    if st.button("Summarize"):
        try:
            if model_choice == "T5":
                summarizer = T5FileSummarizer()
            elif model_choice == "BART":
                summarizer = BARTFileSummarizer()
            else:
                summarizer = PegasusFileSummarizer()

            with st.spinner("Summarizing..."):
                summary = summarizer.summarize_file(upload_path, max_length=max_length, min_length=min_length)
                st.subheader("📝 Summary:")
                st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
