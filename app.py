import streamlit as st
import pickle
import docx
import PyPDF2
import re

# ---------------- LOAD MODEL ----------------
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# ---------------- CLEAN TEXT ----------------
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.lower()
    return text

# ---------------- EXTRACT TEXT ----------------
def extract_text(file):
    file_type = file.name.split('.')[-1]

    if file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])

    elif file_type == "docx":
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])

    elif file_type == "txt":
        try:
            return file.read().decode("utf-8")
        except:
            return file.read().decode("latin-1")

    return ""

# ---------------- PREDICT ----------------
def predict_resume_category(text):
    cleaned = clean_resume(text)
    vectorized = tfidf.transform([cleaned])
    prediction = clf.predict(vectorized)[0]

    # 🔥 FIX (NO MORE ADVOCATE BUG)
    category = le.inverse_transform([prediction])[0]
    return category

# ---------------- UI ----------------
st.set_page_config(page_title="Resume Analyzer", layout="centered")

st.title("📄 AI Resume Analyzer")
st.write("Upload your resume and get predicted job category")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    resume_text = extract_text(uploaded_file)

    if not resume_text.strip():
        st.error("❌ Could not read file properly")
    else:
        st.success("✅ Resume loaded successfully")

        if st.checkbox("Show Resume Text"):
            st.text_area("Resume Content", resume_text, height=300)

        predicted_category = predict_resume_category(resume_text)

        st.subheader("🎯 Predicted Category")
        st.success(predicted_category)
