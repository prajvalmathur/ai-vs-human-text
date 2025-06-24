import streamlit as st
import pickle
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document
import pdfplumber

# Load models
model_files = {
    'SVM': 'models/svm_model.pkl',
    'Decision Tree': 'models/decision_tree_model.pkl',
    'AdaBoost': 'models/adaboost_model.pkl',
}
models = {name: pickle.load(open(path, 'rb')) for name, path in model_files.items()}
vectorizer = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))

# text Input
def extract_text(file):
    if file.name.endswith('.pdf'):
        with pdfplumber.open(file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return file.read().decode('utf-8')

def predict(text, model):
    tfidf_input = vectorizer.transform([text])
    prob = model.predict_proba(tfidf_input)[0]
    pred = model.predict(tfidf_input)[0]
    return pred, prob

# Streamlit UI
st.title("AI vs Human Text Detector")
st.write("Upload a file or paste text below to check if it's AI or human written.")

input_method = st.radio("Input Method", ["Upload File", "Type/Paste Text"])

text = ""
if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload .txt, .pdf, or .docx", type=["txt", "pdf", "docx"])
    if uploaded_file:
        text = extract_text(uploaded_file)
        st.text_area("Extracted Text", text, height=200)
else:
    text = st.text_area("Enter text here", height=200)

model_name = st.selectbox("Choose a model", list(models.keys()))

if st.button("Predict"):
    if text.strip():
        pred, prob = predict(text, models[model_name])
        label = "AI" if pred == 1 else "Human"
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"**Confidence:** AI: {prob[1]*100:.2f}% | Human: {prob[0]*100:.2f}%")
    else:
        st.warning("Please input some text.")
